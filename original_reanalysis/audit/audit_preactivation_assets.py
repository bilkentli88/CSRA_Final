#!/usr/bin/env python3
"""Verify recovered prompts and controls against the execution commitments.

python audit_preactivation_assets.py --package CSRA_v08_PREACTIVATION_COMPLETE_PACKAGE.zip \
    --out results/preactivation_asset_verification.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

# Expected values are recorded in the previously audited R4/A2 report runner.
EXPECTED = {
    "step2/prompts/csra_v08_prompt_bundle.json": "1c392ee6b2c396fcaee029411d3a428f8364223bd2e6f90f8c0b7db7ec922f09",
    "step2/CSRA_v08_PROMPT_TEMPLATE_BUNDLE_FROZEN.zip": "c2f09968a3112952558081c086acc17043ee5d2b3d36f481b19b8796808c61f9",
    "step2/CSRA_v08_ANALYSIS_NOTEBOOK_FROZEN.zip": "9edc3257f4567c0da8d4ce16292aa8cef89168adb73f5bac19adddacc183524f",
    "step2/inputs/csra_v05_fictional_controls.jsonl": "84677b428c921cb46429b80f7071f10874621b115ce8037f668a4f56c899e679",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--out", default=Path("results/preactivation_asset_verification.json"), type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.package) as z:
        files = {i.filename: z.read(i) for i in z.infolist() if not i.is_dir()}
    commitments = {name: {"sha256": digest(files[name]), "matches_execution_commitment": digest(files[name]) == expected} for name, expected in EXPECTED.items()}
    assert all(row["matches_execution_commitment"] for row in commitments.values())
    manifest = json.loads(files["step2/manifests/CSRA_v08_STEP2_COMPONENT_MANIFEST.json"])
    component_checks = {name: digest(files["step2/" + name]) == expected for name, expected in manifest["components"].items()}
    assert all(component_checks.values())
    freeze = json.loads(files["step2/CSRA_v08_PREACTIVATION_STEP2_FREEZE_RECORD.json"])
    protocol_checks = {name: digest(files[freeze["paths"][name]]) == freeze["scientific_hashes"][name] for name in ("protocol_docx", "approval_record")}
    assert all(protocol_checks.values())
    prompts = json.loads(files["step2/prompts/csra_v08_prompt_bundle.json"])
    controls = [json.loads(line) for line in files["step2/inputs/csra_v05_fictional_controls.jsonl"].decode().splitlines() if line.strip()]
    assert len(controls) == 10
    result = {
        "package_SHA256": digest(args.package.read_bytes()),
        "execution_commitments": commitments,
        "component_checks": component_checks,
        "protocol_and_approval_checks": protocol_checks,
        "fictional_control_count": len(controls),
        "principal_SELF_stem": prompts["report_stems"]["principal"]["SELF"],
        "principal_PEER_stem": prompts["report_stems"]["principal"]["PEER"],
        "wording_observation": "The prompts differ in ownership and epistemic wording: PEER includes 'most likely'; SELF does not. This is a possible confound, not an experimentally isolated explanation of the observed effects.",
        "recovered": ["Exact system, answer, screening, SELF/PEER and alternate report templates", "All answer/report role mappings", "Ten original fictional-control item definitions", "Original R2 protocol and approval record"],
        "still_not_in_these_packages": ["Completed 240-row factual-review CSV and exclusion reasons", "Official final R4/A2 600-candidate JSONL and raw source snapshot", "Screening outputs", "Actual activation manifest", "Individual model-scoring checkpoints"],
        "scientific_result_change": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    summary_path = args.out.parent / "audit_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["recovered_preactivation_assets"] = result
        summary["limits"] = ["Individual label-permutation scores, the completed candidate/review pool, activation record, and external timing evidence remain unavailable in the supplied materials." if "Exact prompt text" in line else line for line in summary["limits"]]
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("PASS: exact prompts, controls, protocol and all 14 Step-2 components verified.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
