#!/usr/bin/env python3
"""Verify the two CSRA lock documents in the recovered answer package.

python audit_lock_lineage.py --package CSRA_v08_ANSWER_EXECUTION_R4_A2_TR2_PACKAGE.zip \
    --repo /path/to/CSRA-8000 --out results/lock_lineage_verification.json
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(package, repo):
    with zipfile.ZipFile(package) as z:
        files = {i.filename: z.read(i) for i in z.infolist() if not i.is_dir()}
    checksum_checks = []
    for line in files["SHA256SUMS.txt"].decode().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        checksum_checks.append(name in files and digest(files[name]) == expected)
    assert all(checksum_checks), "Answer-package checksum failure"
    metadata_name = "CSRA_v08_POSTREVIEW_ANALYSIS_LOCK_R4_A2.json"
    canonical_name = "postreview/postreview_analysis_lock.json"
    completion_name = "CSRA_v08_POSTREVIEW_TECHNICAL_COMPLETION_R4_A2.json"
    canonical = json.loads(files[canonical_name])
    completion = json.loads(files[completion_name])
    metadata_sha = digest(files[metadata_name])
    canonical_sha = digest(files[canonical_name])
    runner_ast = ast.parse((repo / "code/run_csra_v08_analysis_phase_R4_A2_TR4.py").read_text())
    expected = next(ast.literal_eval(n.value) for n in runner_ast.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "EXPECTED" for t in n.targets))
    checks = {
        "metadata_matches_review_release": files[metadata_name] == (repo / "protocol/postreview_analysis_lock.json").read_bytes(),
        "canonical_matches_execution_runner": canonical_sha == expected["postreview_sha"],
        "completion_binds_metadata": completion["upstream"]["prior_chat_metadata_lock_sha256"] == metadata_sha,
        "completion_binds_canonical": completion["outputs"]["postreview_analysis_lock.json"] == canonical_sha,
        "selected_items_match_review_release": files["postreview/final_common_items.csv"] == (repo / "data/final_common_items.csv").read_bytes(),
        "review_seal_binding_matches": digest(files["CSRA_v08_REVIEW_SEAL_R4_A2.json"]) == canonical["review_seal_sha256"],
    }
    bindings = {
        "alternate_wording_ids_sha256": "alternate_wording_item_ids.json",
        "analysis_relation_map_sha256": "analysis_relation_map.json",
        "empty_result_shells_sha256": "empty_result_shells.json",
        "fold_assignments_sha256": "fold_assignments.json",
        "selected_items_sha256": "final_common_items.csv",
    }
    for key, name in bindings.items():
        checks[f"canonical_binding_{name}"] = digest(files["postreview/" + name]) == canonical[key]
    # Independent reconstruction of all six relation-stratified hash fold maps.
    import csv
    import io
    from collections import defaultdict
    selected = list(csv.DictReader(io.StringIO(files["postreview/final_common_items.csv"].decode())))
    relation_items = defaultdict(list)
    for row in selected:
        relation_items[row["relation"]].append(row["item_id"])
    folds = {}
    for key, prefix in [("primary", "fold")] + [(f"sensitivity_{j}", f"fold_sens{j}") for j in range(1, 6)]:
        mapping = {}
        for relation, ids in sorted(relation_items.items()):
            ranked = sorted((digest(f"20260812|v08|{prefix}|{relation}|{item}".encode()), item) for item in ids)
            mapping.update({item: k % 5 for k, (_, item) in enumerate(ranked)})
        folds[key] = mapping
    checks["all_six_fold_maps_match_frozen_rules"] = folds == json.loads(files["postreview/fold_assignments.json"])
    assert all(checks.values()), checks
    return {
        "answer_package_SHA256": digest(Path(package).read_bytes()),
        "package_checksum_files_verified": len(checksum_checks),
        "metadata_lock_file": metadata_name,
        "metadata_lock_SHA256": metadata_sha,
        "canonical_lock_file": canonical_name,
        "canonical_lock_SHA256": canonical_sha,
        "checks": checks,
        "interpretation": "The review release contains the expanded design-metadata lock. The execution runner references a separate compact canonical lock. The recovered technical-completion record explicitly links both documents, and all checked bindings match.",
        "timing_limit": "The recovered files explain document identity and internal lineage; they do not independently establish an external pre-outcome timestamp.",
        "scientific_result_change": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", default=Path("results/lock_lineage_verification.json"), type=Path)
    args = parser.parse_args()
    result = verify(args.package, args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    summary_path = args.out.parent / "audit_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["recovered_lock_lineage"] = result
        summary["postreview_lock_mismatch_explained"] = True
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("PASS: both lock documents and their documented lineage verified.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
