"""Verify recovered factual-review versions and deterministic final-item selection.

Uses the Python standard library. Does not alter review files or run models.
Factual claims in review reasons are recorded, not independently re-reviewed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path


NAMES = [
    "csra_v08_blinded_common_review.csv",
    "csra_v08_blinded_common_review_SEALED.csv",
    "csra_v08_blinded_common_review_AUDITED.csv",
    "csra_v08_blinded_common_review (1).csv",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return reader.fieldnames, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tables = {name: read_csv(args.reviews / name) for name in NAMES}
    columns, review = tables[NAMES[1]]
    review_sha = sha(args.reviews / NAMES[1])
    seal = json.loads((args.records / "CSRA_v08_REVIEW_SEAL_R4_A2.json").read_text())
    gate = json.loads((args.records / "CSRA_v08_POSTREVIEW_SELECTION_GATE_R4_A2.json").read_text())
    lock = json.loads((args.records / "CSRA_v08_POSTREVIEW_ANALYSIS_LOCK_R4_A2.json").read_text())
    spec = json.loads((args.repo / "protocol/csra_v08_analysis_spec.json").read_text())
    final_path = args.repo / "data/final_common_items.csv"
    final_columns, released = read_csv(final_path)

    substantive = [k for k in columns if k not in {"review_status", "review_reason"}]
    base_columns, base = tables[NAMES[0]]
    profiles = {}
    same_fields = True
    for name, (fields, rows) in tables.items():
        fields_equal = fields == base_columns
        content_equal = len(rows) == len(base) and all(
            all(a[k] == b[k] for k in substantive) for a, b in zip(base, rows)
        )
        same_fields = same_fields and fields_equal and content_equal
        profiles[name] = {
            "sha256": sha(args.reviews / name),
            "rows": len(rows),
            "status_counts": dict(collections.Counter(r["review_status"] for r in rows)),
            "same_nonreview_fields_as_blank_template": fields_equal and content_equal,
            "changed_fields_vs_blank_template": dict(collections.Counter(
                k for a, b in zip(base, rows) for k in base_columns if a[k] != b[k]
            )),
        }

    relations = sorted({r["relation"] for r in review})
    approvals = dict(collections.Counter(r["relation"] for r in review if r["review_status"] == "APPROVE"))
    exclusions = dict(collections.Counter(r["relation"] for r in review if r["review_status"] == "EXCLUDE"))
    cap = int(spec["selection"]["cap_per_original_relation"])
    minimum = int(spec["selection"]["minimum_common_items"])
    selected = []
    for relation in relations:
        approved = []
        for row in review:
            if row["relation"] == relation and row["review_status"] == "APPROVE":
                key = f"20260812|v08|select|{relation}|{row['item_id']}"
                approved.append({**row, "selection_hash": hashlib.sha256(key.encode()).hexdigest()})
        approved.sort(key=lambda row: (row["selection_hash"], row["item_id"]))
        selected.extend({**row, "selection_rank": str(rank)} for rank, row in enumerate(approved[:cap], 1))

    selected_counts = dict(collections.Counter(r["relation"] for r in selected))
    # The released final CSV uses UTF-8 without a BOM and CRLF record endings.
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=final_columns, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(selected)
    reconstructed = buffer.getvalue().encode("utf-8")
    country = [r for r in review if r["relation"] == "country_of_origin"]
    checks = {
        "sealed_sha_matches_review_seal": review_sha == seal["sha256"],
        "sealed_sha_matches_selection_gate": review_sha == gate["sealed_review_sha256"],
        "sealed_sha_matches_analysis_lock": review_sha == lock["review_and_selection"]["sealed_review_sha256"],
        "audited_and_sealed_byte_identical": (args.reviews / NAMES[1]).read_bytes() == (args.reviews / NAMES[2]).read_bytes(),
        "review_row_count_matches_seal": len(review) == int(seal["row_count"]) == 240,
        "unique_review_item_ids": len({r["item_id"] for r in review}) == len(review),
        "all_review_statuses_valid": all(r["review_status"] in {"APPROVE", "EXCLUDE"} for r in review),
        "all_exclusions_have_reasons": all(r["review_reason"].strip() for r in review if r["review_status"] == "EXCLUDE"),
        "versions_preserve_all_nonreview_fields": same_fields,
        "blank_copy_has_no_decisions": all(not r["review_status"] and not r["review_reason"] for r in base),
        "numbered_copy_is_all_approve": all(r["review_status"] == "APPROVE" and not r["review_reason"] for r in tables[NAMES[3]][1]),
        "approval_counts_match_gate": approvals == gate["approval_counts_by_relation"],
        "exclusion_counts_match_gate": exclusions == gate["exclusion_counts_by_relation"],
        "selection_counts_match_gate": selected_counts == gate["selected_counts_by_relation"],
        "information_gate_matches": len(selected) >= minimum and bool(gate["information_gate_pass"]),
        "final_item_count_matches": len(selected) == int(gate["final_common_item_count"]) == 100,
        "reconstructed_final_rows_match_release": selected == released,
        "reconstructed_final_bytes_match_release": reconstructed == final_path.read_bytes(),
        "released_final_sha_matches_gate": sha(final_path) == gate["final_common_items_sha256"],
    }
    result = {
        "checks": checks,
        "version_profiles": profiles,
        "review_status_counts": dict(collections.Counter(r["review_status"] for r in review)),
        "by_relation": {rel: {"reviewed": sum(r["relation"] == rel for r in review), "approved": approvals.get(rel, 0), "excluded": exclusions.get(rel, 0), "selected": selected_counts.get(rel, 0)} for rel in relations},
        "exclusion_reason_categories": dict(collections.Counter(r["review_reason"].split(":", 1)[0] for r in review if r["review_status"] == "EXCLUDE")),
        "country_of_origin": {"reviewed": len(country), "irrelevant_answer_counts": dict(collections.Counter(r["irrelevant_answer"] for r in country)), "review_reason_counts": dict(collections.Counter(r["review_reason"] for r in country))},
        "deterministic_selection": {"rule": gate["selection_rule"], "cap_per_relation": cap, "minimum_common_items": minimum, "selected_items": len(selected), "reconstructed_final_sha256": hashlib.sha256(reconstructed).hexdigest()},
        "limits": [
            "Review reasons are supplied records; factual correctness and reviewer independence were not independently established.",
            "Names and hashes establish content identity, not reviewer identity, chronology between copies, or external pre-outcome timing.",
            "The 240 review rows do not recover the full 600-candidate pool, raw retrieval snapshot, or common-eligibility screening computations.",
            "No language-model scoring or statistical outcome analysis was rerun.",
        ],
        "scientific_result_change": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if not all(checks.values()):
        raise RuntimeError("Failed checks: " + ", ".join(k for k, v in checks.items() if not v))
    summary_path = args.out.parent / "audit_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["recovered_factual_review"] = result
        old_limit = "Individual label-permutation scores, the completed candidate/review pool, activation record, and external timing evidence remain unavailable in the supplied materials."
        new_limit = "Individual label-permutation scores, the full 600-candidate pool and raw source snapshot, screening outputs, activation record, final R4/A2 amendment documents, and external timing evidence remain unavailable. The sealed 240-row review and final-selection reconstruction are now verified."
        review_limit = "Recorded review reasons and content identity are verified; factual correctness, reviewer identity and independence, and chronology between review copies are not independently established."
        summary["limits"] = [new_limit if x == old_limit else x for x in summary["limits"]]
        if review_limit not in summary["limits"]:
            summary["limits"].append(review_limit)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"PASS: {len(checks)} checks; 240 reviewed, 171 approved, 69 excluded; final 100-item CSV reproduced byte for byte.")


if __name__ == "__main__":
    main()
