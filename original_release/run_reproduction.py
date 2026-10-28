#!/usr/bin/env python3
"""Reproduce the principal CSRA v0.8 R4/A2 statistical results.

Default mode recomputes point estimates and robustness summaries on CPU.
Optional flags rerun the 2,000-replicate primary or alignment bootstrap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import patsy

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
DATA = ROOT / "data"
SUPPORT = DATA / "supporting"
EXPECTED = ROOT / "expected_results"
OUT = ROOT / "reproduced_results"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(CODE))
import csra_v08_analysis as ana  # noqa: E402
from csra_v08_common import PRIMARY_MODELS  # noqa: E402


def _sha_rank(prefix: str, analysis_relation: str, item_id: str) -> str:
    return hashlib.sha256(f"20260812|v08|{prefix}|{analysis_relation}|{item_id}".encode()).hexdigest()


def build_lock_material(selected: pd.DataFrame, target_dir: Path) -> tuple[dict, dict]:
    counts = selected.groupby("relation")["item_id"].nunique().to_dict()
    relation_map = {r: (r if int(n) >= 5 else "OTHER_RARE") for r, n in counts.items()}
    work = selected[["item_id", "relation"]].drop_duplicates().copy()
    work["analysis_relation"] = work["relation"].map(relation_map)

    folds: dict[str, dict[str, int]] = {}
    for key, prefix in [("primary", "fold")] + [(f"sensitivity_{j}", f"fold_sens{j}") for j in range(1, 6)]:
        mapping: dict[str, int] = {}
        for rel, frame in work.groupby("analysis_relation", sort=True):
            ranked = sorted((_sha_rank(prefix, str(rel), str(iid)), str(iid)) for iid in frame["item_id"])
            for rank, (_, iid) in enumerate(ranked):
                mapping[iid] = rank % 5
        folds[key] = mapping

    (target_dir / "analysis_relation_map.json").write_text(json.dumps(relation_map, indent=2), encoding="utf-8")
    (target_dir / "fold_assignments.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    return relation_map, folds


def tr4_design_vector(fit, model_key: str, cie: float, self_value: int) -> np.ndarray:
    frame = fit.model.data.frame

    def observed(col: str) -> str:
        vals = sorted(set(map(str, frame[col].dropna().tolist())))
        if not vals:
            raise RuntimeError(f"No observed level for {col}")
        return vals[0]

    row = pd.DataFrame([{
        "VSA_z": 0.0,
        "CIE_z": float(cie),
        "SELF": int(self_value),
        "model_key": model_key,
        "frozen_answer_role": observed("frozen_answer_role"),
        "baseline_margin_z": 0.0,
        "strength": observed("strength"),
        "analysis_relation": observed("analysis_relation"),
    }])
    matrix = patsy.build_design_matrices([fit.model.data.design_info], row, return_type="dataframe")[0]
    return matrix.loc[:, fit.fe_params.index].to_numpy(float)[0]


def load_inputs():
    selected = pd.read_csv(DATA / "final_common_items.csv")
    answer = {m: pd.read_csv(SUPPORT / f"{m}_answer_condition_summary.csv") for m in PRIMARY_MODELS}
    main = {m: pd.read_csv(SUPPORT / f"{m}_main_report_results.csv") for m in PRIMARY_MODELS}
    alt = {m: pd.read_csv(SUPPORT / f"{m}_alternate_report_results.csv") for m in PRIMARY_MODELS}
    fa = {m: pd.read_csv(SUPPORT / f"{m}_fiction_answer_summary.csv") for m in PRIMARY_MODELS}
    fr = {m: pd.read_csv(SUPPORT / f"{m}_fiction_report_results.csv") for m in PRIMARY_MODELS}
    spec = json.loads((ROOT / "protocol" / "csra_v08_analysis_spec.json").read_text(encoding="utf-8"))
    return selected, answer, main, alt, fa, fr, spec


def assert_close(name: str, actual: float, expected: float, atol: float = 1e-9):
    if not np.isclose(float(actual), float(expected), atol=atol, rtol=0):
        raise AssertionError(f"{name}: actual={actual}, expected={expected}")


def compare_frame(actual: pd.DataFrame, expected_path: Path, keys: list[str], cols: list[str], atol: float = 1e-8):
    exp = pd.read_csv(expected_path)
    a = actual.sort_values(keys).reset_index(drop=True)
    e = exp.sort_values(keys).reset_index(drop=True)
    if len(a) != len(e):
        raise AssertionError(f"{expected_path.name}: row count {len(a)} != {len(e)}")
    for col in cols:
        if pd.api.types.is_numeric_dtype(e[col]):
            if not np.allclose(a[col].astype(float), e[col].astype(float), atol=atol, rtol=0, equal_nan=True):
                raise AssertionError(f"{expected_path.name}: mismatch in {col}")
        else:
            if a[col].astype(str).tolist() != e[col].astype(str).tolist():
                raise AssertionError(f"{expected_path.name}: mismatch in {col}")


def run_quick():
    selected, answer, main, alt, fa, fr, spec = load_inputs()
    with tempfile.TemporaryDirectory() as td_s:
        td = Path(td_s)
        build_lock_material(selected, td)
        variances = ana.compute_cie_variance_constants(answer, selected["item_id"].tolist())
        units = ana.assemble_primary_units(answer, main, selected, td)
        point, item_effects = ana.primary_point_estimate(units, variances, alpha=1.0)
        sensitivity = ana.fixed_alpha_sensitivity(units, variances, spec, td)

        # Apply the documented TR4 compatibility repair without modifying frozen module bytes.
        ana._design_vector = tr4_design_vector
        alignment_data = ana.assemble_alignment_data(answer, main, selected, td)
        alignment_point, deltas = ana.alignment_point(alignment_data)
        wording = ana.wording_robustness(main, alt)
        fictional = ana.fictional_summary(fa, fr)

    point_expected = json.loads((EXPECTED / "primary_point_estimate.json").read_text(encoding="utf-8"))
    assert_close("theta_SELF", point["theta_SELF"], point_expected["theta_SELF"])
    assert_close("theta_PEER", point["theta_PEER"], point_expected["theta_PEER"])
    for m in PRIMARY_MODELS:
        for k in ("X1", "X2", "X3"):
            for metric in ("R2", "MAE", "normalized_MSE"):
                assert_close(f"{m}.{k}.{metric}", point["model_metrics"][m][k][metric], point_expected["model_metrics"][m][k][metric], 1e-8)

    compare_frame(sensitivity, EXPECTED / "primary_sensitivity_summary.csv", ["analysis", "setting"], ["analysis", "setting", "theta_SELF", "theta_PEER"], 1e-8)
    compare_frame(wording, EXPECTED / "wording_robustness_summary.csv", ["model_key"], ["model_key", "n", "spearman", "mean_shift"], 1e-8)
    compare_frame(fictional, EXPECTED / "fictional_control_summary.csv", ["model_key"], ["model_key", "context_answer_share", "document_report_share", "mean_VSA"], 1e-8)

    align_expected = json.loads((EXPECTED / "alignment_results.json").read_text(encoding="utf-8"))
    for key in ("H", "H_SE", "H_p", "heterogeneity_LR", "heterogeneity_p", "common_interaction", "common_interaction_SE", "common_interaction_p"):
        assert_close(f"alignment.{key}", alignment_point[key], align_expected[key], 1e-8)
    delta_expected = pd.read_csv(EXPECTED / "model_specific_alignment_deltas.csv")
    compare_frame(deltas, EXPECTED / "model_specific_alignment_deltas.csv", ["model_key"], ["model_key", "Delta", "SE", "p_value", "p_BH"], 1e-8)

    item_effects.to_csv(OUT / "primary_item_effects.csv", index=False)
    sensitivity.to_csv(OUT / "primary_sensitivity_summary.csv", index=False)
    wording.to_csv(OUT / "wording_robustness_summary.csv", index=False)
    fictional.to_csv(OUT / "fictional_control_summary.csv", index=False)
    deltas.to_csv(OUT / "model_specific_alignment_point_estimates.csv", index=False)
    (OUT / "primary_point_estimate.json").write_text(json.dumps(point, indent=2), encoding="utf-8")
    (OUT / "alignment_point_estimate.json").write_text(json.dumps(alignment_point, indent=2), encoding="utf-8")

    print("PASS: all quick-reproduction checks match the sealed v0.8 expected results.")
    print(f"Outputs written to {OUT}")


def run_primary_bootstrap():
    selected, answer, main, alt, fa, fr, spec = load_inputs()
    with tempfile.TemporaryDirectory() as td_s:
        td = Path(td_s)
        build_lock_material(selected, td)
        variances = ana.compute_cie_variance_constants(answer, selected["item_id"].tolist())
        units = ana.assemble_primary_units(answer, main, selected, td)
        point, _ = ana.primary_point_estimate(units, variances, alpha=1.0)
        checkpoint = OUT / "primary_full_refit_bootstrap.csv"
        boot = ana.bootstrap_primary(units, selected, variances, reps=int(spec["bootstrap_reps"]), seed=int(spec["bootstrap_seed"]), checkpoint_path=checkpoint)
        summary = ana.bootstrap_summary(point, boot, int(spec["minimum_valid_bootstrap"]), float(spec["practical_delta"]))
    expected = json.loads((EXPECTED / "primary_validation_result.json").read_text(encoding="utf-8"))
    for key in ("theta_SELF", "ci_low", "ci_high", "theta_PEER", "peer_ci_low", "peer_ci_high"):
        assert_close(f"primary_bootstrap.{key}", summary[key], expected[key], 1e-8)
    if summary["classification"] != expected["classification"] or int(summary["valid_bootstrap"]) != int(expected["valid_bootstrap"]):
        raise AssertionError("Primary bootstrap classification/count mismatch")
    (OUT / "primary_validation_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("PASS: full primary bootstrap matches the sealed v0.8 result.")


def run_alignment_bootstrap():
    selected, answer, main, alt, fa, fr, spec = load_inputs()
    with tempfile.TemporaryDirectory() as td_s:
        td = Path(td_s)
        build_lock_material(selected, td)
        ana._design_vector = tr4_design_vector
        data = ana.assemble_alignment_data(answer, main, selected, td)
        point, deltas = ana.alignment_point(data)
        checkpoint = OUT / "alignment_common_item_bootstrap.csv"
        boot = ana.alignment_bootstrap(data, selected, reps=int(spec["alignment"]["bootstrap_reps"]), seed=int(spec["alignment"]["bootstrap_seed"]), checkpoint_path=checkpoint)
        summary, deltas = ana.alignment_bootstrap_summary(point, deltas, boot, int(spec["alignment"]["minimum_valid_bootstrap"]))
    expected = json.loads((EXPECTED / "alignment_results.json").read_text(encoding="utf-8"))
    for key in ("H", "H_SE", "H_p", "H_bootstrap_ci_low", "H_bootstrap_ci_high", "common_interaction", "common_interaction_SE", "common_interaction_p", "common_bootstrap_ci_low", "common_bootstrap_ci_high", "heterogeneity_LR", "heterogeneity_p"):
        assert_close(f"alignment_bootstrap.{key}", summary[key], expected[key], 1e-8)
    compare_frame(deltas, EXPECTED / "model_specific_alignment_deltas.csv", ["model_key"], ["model_key", "Delta", "SE", "p_value", "p_BH", "bootstrap_ci_low", "bootstrap_ci_high"], 1e-8)
    (OUT / "alignment_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    deltas.to_csv(OUT / "model_specific_alignment_deltas.csv", index=False)
    print("PASS: full alignment bootstrap matches the sealed v0.8 result.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-primary", action="store_true", help="Rerun the 2,000-replicate primary predictive bootstrap")
    parser.add_argument("--bootstrap-alignment", action="store_true", help="Rerun the 2,000-replicate mixed-model alignment bootstrap (slow)")
    args = parser.parse_args()
    run_quick()
    if args.bootstrap_primary:
        run_primary_bootstrap()
    if args.bootstrap_alignment:
        run_alignment_bootstrap()


if __name__ == "__main__":
    main()
