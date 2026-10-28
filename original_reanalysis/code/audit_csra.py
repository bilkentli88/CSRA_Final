#!/usr/bin/env python3
"""Independent numerical audit of the released CSRA v0.8 R4/A2 summaries.

Usage: python audit_csra.py --repo /path/to/extracted/CSRA-8000 --out results
Uses the repository's five folds, Ridge alpha=1, fixed population variances,
and 2,000 relation-stratified paired item draws. Bootstrap Ridge fits are
computed from weighted sufficient statistics, with train-fold numeric
standardization refitted in every draw. No language models are downloaded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

MODELS = ("qwen", "granite", "phi")
CONDITIONS = ("C_SKEPTICAL", "C_PROVISIONAL", "C_BARE")
ROLES = ("document", "prior", "both", "uncertain")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def fold_map(selected):
    result = {}
    for relation, frame in selected.groupby("relation", sort=True):
        ranked = sorted((hashlib.sha256(f"20260812|v08|fold|{relation}|{i}".encode()).hexdigest(), i) for i in frame.item_id)
        result.update({item: k % 5 for k, (_, item) in enumerate(ranked)})
    return result


def bootstrap_weights(selected, reps):
    ids = sorted(selected.item_id)
    position = {item: i for i, item in enumerate(ids)}
    seeds = np.random.default_rng(20260813).integers(0, 2**63-1, size=reps, dtype=np.int64)
    weights = np.zeros((reps, len(ids)), float)
    for b, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        for _, frame in selected.groupby("relation", sort=True):
            sampled = rng.choice(frame.item_id.tolist(), size=len(frame), replace=True)
            weights[b] += np.bincount([position[i] for i in sampled], minlength=len(ids))
    return ids, weights


def ridge_loss(frame, numeric, categorical, weights, item_ids):
    """Exact alpha=1 Ridge with an unpenalized intercept, for all draws."""
    lookup = {item: i for i, item in enumerate(item_ids)}
    item_index = frame.item_id.map(lookup).to_numpy(int)
    numeric_matrix = frame[numeric].to_numpy(float)
    category_matrix = pd.get_dummies(frame[categorical].astype(str), dtype=float).to_numpy()
    design = np.concatenate((numeric_matrix, category_matrix), axis=1)
    if not np.isfinite(design).all():
        raise ValueError("Audit assumes the released predictor values are finite")
    y = frame.CIE.to_numpy(float)
    folds = frame.fold_id.to_numpy(int)
    n_features = design.shape[1]
    squared_errors = np.zeros(len(weights))
    for fold in range(5):
        train = folds != fold
        test = ~train
        # Per-item moments preserve all three strength rows together.
        n = np.zeros(len(item_ids))
        sx = np.zeros((len(item_ids), n_features))
        sxx = np.zeros((len(item_ids), n_features, n_features))
        sy = np.zeros(len(item_ids))
        sxy = np.zeros((len(item_ids), n_features))
        for i in np.flatnonzero(train):
            j = item_index[i]
            n[j] += 1
            sx[j] += design[i]
            sxx[j] += np.outer(design[i], design[i])
            sy[j] += y[i]
            sxy[j] += design[i] * y[i]
        total_n = weights @ n
        if (total_n <= 0).any():
            raise ValueError("Empty training fold")
        sum_x = weights @ sx
        mean_x = sum_x / total_n[:, None]
        mean_y = (weights @ sy) / total_n
        sum_xx = np.einsum("bi,ijk->bjk", weights, sxx, optimize=True)
        centered_xx = sum_xx - total_n[:, None, None] * mean_x[:, :, None] * mean_x[:, None, :]
        scale = np.ones_like(mean_x)
        variance = np.diagonal(centered_xx, axis1=1, axis2=2)[:, :len(numeric)] / total_n[:, None]
        scale[:, :len(numeric)] = np.sqrt(np.maximum(variance, 0))
        scale[scale < 1e-12] = 1
        gram = centered_xx / (scale[:, :, None] * scale[:, None, :])
        gram += np.eye(n_features)[None, :, :]  # frozen Ridge alpha=1
        rhs = ((weights @ sxy) - total_n[:, None] * mean_x * mean_y[:, None]) / scale
        coef = np.linalg.solve(gram, rhs[:, :, None])[:, :, 0] / scale
        intercept = mean_y - np.sum(mean_x * coef, axis=1)
        prediction = coef @ design[test].T + intercept[:, None]
        residual = prediction - y[test][None, :]
        squared_errors += np.sum(residual**2 * weights[:, item_index[test]], axis=1)
    variance_constant = np.var(y, ddof=0)
    return squared_errors / (3 * weights.sum(axis=1) * variance_constant)


def weighted_slope(frame, x_name, y_name, weights, item_ids):
    position = {item: i for i, item in enumerate(item_ids)}
    moments = np.zeros((len(item_ids), 5))
    for row in frame.itertuples():
        x = getattr(row, x_name)
        y = getattr(row, y_name)
        moments[position[row.item_id]] += [1, x, y, x*x, x*y]
    n, sx, sy, sxx, sxy = (weights @ moments).T
    denominator = sxx - sx*sx/n
    numerator = sxy - sx*sy/n
    return numerator / denominator, numerator, denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", default=Path("results"), type=Path)
    parser.add_argument("--reps", default=2000, type=int)
    args = parser.parse_args()
    root, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(root / "data/final_common_items.csv")
    alignment = pd.read_csv(root / "data/alignment_analysis_dataset.csv")
    assert len(selected) == selected.item_id.nunique() == 100
    assert len(alignment) == 1800
    item_ids, boot_weights = bootstrap_weights(selected, args.reps)
    weights = np.concatenate((np.ones((1, 100)), boot_weights), axis=0)
    folds = fold_map(selected)
    units = []
    answer_tables, report_tables = {}, {}
    reconstruction = {}
    for model in MODELS:
        answer = pd.read_csv(root / f"data/supporting/{model}_answer_condition_summary.csv")
        report = pd.read_csv(root / f"data/supporting/{model}_main_report_results.csv")
        answer_tables[model], report_tables[model] = answer, report
        primary = answer[answer.condition.isin(CONDITIONS)].copy()
        paired_reference = answer[answer.condition.isin([f"P_{s}" for s in ("SKEPTICAL", "PROVISIONAL", "BARE")])]
        p_lookup = paired_reference.set_index(["item_id", "strength"]).mean_L_C_minus_P
        reference = np.array([p_lookup.loc[(i, s)] for i, s in zip(primary.item_id, primary.strength)])
        calculated_cie = primary.mean_logp_C - primary.mean_logp_P - reference
        reconstruction[f"{model}_CIE_max_abs_error"] = float(np.max(np.abs(primary.CIE - calculated_cie)))
        reconstruction[f"{model}_VSA_max_abs_error"] = float(np.max(np.abs(report.VSA - (report.logp_report_document - report.logp_report_prior))))
        assert reconstruction[f"{model}_CIE_max_abs_error"] < 1e-10
        assert reconstruction[f"{model}_VSA_max_abs_error"] < 1e-10
        pivot = report[report.condition.isin(CONDITIONS)].pivot(index=["item_id", "condition"], columns="frame", values="VSA").reset_index()
        u = primary.merge(pivot, on=["item_id", "condition"], validate="one_to_one")
        u["model_key"] = model
        u["analysis_relation"] = u.relation
        u["fold_id"] = u.item_id.map(folds)
        units.append(u.sort_values(["item_id", "condition"]).reset_index(drop=True))
    unit = pd.concat(units, ignore_index=True)
    unit.to_csv(out / "reconstructed_primary_units.csv", index=False)
    comparison = alignment.merge(unit[["model_key", "item_id", "condition", "CIE", "SELF", "PEER"]].rename(columns={"CIE": "computed_CIE", "SELF": "computed_SELF", "PEER": "computed_PEER"}), on=["model_key", "item_id", "condition"], validate="many_to_one")
    reconstruction["stored_alignment_CIE_max_abs_error"] = float(np.max(np.abs(comparison.CIE-comparison.computed_CIE)))
    reconstruction["stored_alignment_VSA_max_abs_error"] = float(np.max(np.abs(comparison.VSA-np.where(comparison.frame.eq("SELF"),comparison.computed_SELF,comparison.computed_PEER))))
    assert max(reconstruction.values()) < 1e-10

    # Independent point estimates and full-refit bootstrap, including answer ablation.
    point_rows, interval_rows, draws = [], [], {}
    prediction_losses = {}
    for case in ("with_recorded_answer", "without_recorded_answer"):
        case_self, case_peer = [], []
        for model in MODELS:
            frame = unit[unit.model_key.eq(model)].copy()
            cat = ["strength", "analysis_relation"]
            if case == "with_recorded_answer":
                cat = ["frozen_answer_role"] + cat
            losses = []
            for extra in ([], ["PEER"], ["PEER", "SELF"]):
                losses.append(ridge_loss(frame, ["baseline_margin"] + extra, cat, weights, item_ids))
            prediction_losses[(case, model)] = losses
            peer, self_increment = losses[0] - losses[1], losses[1] - losses[2]
            case_peer.append(peer)
            case_self.append(self_increment)
            point_rows.append({"case": case, "model_key": model, **{f"R2_X{k+1}": float(1-v[0]) for k, v in enumerate(losses)}, "theta_PEER": float(peer[0]), "theta_SELF": float(self_increment[0]), "relative_SSE_SELF_reduction": float(self_increment[0] / losses[1][0])})
            draws[f"{case}_{model}_SELF"] = self_increment[1:]
            draws[f"{case}_{model}_PEER"] = peer[1:]
        draws[f"{case}_mean_SELF"] = np.mean(case_self, axis=0)[1:]
        draws[f"{case}_mean_PEER"] = np.mean(case_peer, axis=0)[1:]
        subset = [r for r in point_rows if r["case"] == case]
        point_rows.append({"case": case, "model_key": "equal_model_mean", **{key: float(np.mean([r[key] for r in subset])) for key in subset[0] if key not in ("case", "model_key")}})
        print(f"Completed predictive analysis: {case}", flush=True)
    for row in point_rows:
        key_model = "mean" if row["model_key"] == "equal_model_mean" else row["model_key"]
        for effect in ("SELF", "PEER"):
            vals = draws[f"{row['case']}_{key_model}_{effect}"]
            lo, hi = np.quantile(vals, [0.025, 0.975])
            interval_rows.append({"case": row["case"], "model_key": row["model_key"], "effect": effect, "estimate": row[f"theta_{effect}"], "ci_low": float(lo), "ci_high": float(hi), "valid_draws": int(np.isfinite(vals).sum()), "threshold": 0.01, "classification": "meaningful_positive" if lo > 0.01 else ("below_practical_threshold" if hi < 0.01 else "indeterminate_practical_magnitude")})
    pd.DataFrame(point_rows).to_csv(out / "predictive_point_estimates.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(out / "predictive_intervals.csv", index=False)
    pd.DataFrame(draws).to_csv(out / "predictive_bootstrap_draws.csv", index=False)

    # Verify optimized implementation against the released sklearn implementation.
    sys.path.insert(0, str(root / "code"))
    import csra_v08_analysis as author
    variances = {m: float(np.var(unit[unit.model_key.eq(m)].CIE, ddof=0)) for m in MODELS}
    expected = json.loads((root / "expected_results/primary_point_estimate.json").read_text())
    validation = {"point_max_abs_error": 0.0, "bootstrap_max_abs_error": 0.0, "bootstrap_draws_checked_against_author": 3}
    original_rows = {r["model_key"]: r for r in point_rows if r["case"] == "with_recorded_answer"}
    for m in MODELS:
        for k in ("X1", "X2", "X3"):
            error = abs(original_rows[m][f"R2_{k}"] - expected["model_metrics"][m][k]["R2"])
            validation["point_max_abs_error"] = max(validation["point_max_abs_error"], error)
    for b in range(3):
        copies = []
        for j, item in enumerate(item_ids):
            for duplicate in range(int(boot_weights[b, j])):
                part = unit[unit.item_id.eq(item)].copy()
                part["item_id"] = f"{item}|copy{duplicate}"
                copies.append(part)
        resampled = pd.concat(copies, ignore_index=True)
        point, _ = author.primary_point_estimate(resampled, variances)
        for effect in ("SELF", "PEER"):
            fast = draws[f"with_recorded_answer_mean_{effect}"][b]
            validation["bootstrap_max_abs_error"] = max(validation["bootstrap_max_abs_error"], abs(fast - point[f"theta_{effect}"]))
        # Also validate the answer-omitted branch with the released sklearn helper.
        effects = []
        for model in MODELS:
            frame = resampled[resampled.model_key.eq(model)].copy()
            categorical = ["strength", "analysis_relation"]
            predictions = [author._fit_predict_fixed_fold(frame, ["baseline_margin"] + extra + categorical, categorical, 1.0) for extra in (["PEER"], ["PEER", "SELF"])]
            y = frame.CIE.to_numpy(float)
            effects.append(float(np.mean((y-predictions[0])**2 - (y-predictions[1])**2) / variances[model]))
        validation["bootstrap_max_abs_error"] = max(validation["bootstrap_max_abs_error"], abs(float(np.mean(effects)) - draws["without_recorded_answer_mean_SELF"][b]))
    assert validation["point_max_abs_error"] < 1e-8, validation
    assert validation["bootstrap_max_abs_error"] < 1e-8, validation
    sealed = json.loads((root / "expected_results/primary_validation_result.json").read_text())
    original_mean = next(r for r in interval_rows if r["case"] == "with_recorded_answer" and r["model_key"] == "equal_model_mean" and r["effect"] == "SELF")
    validation["sealed_primary_CI_max_abs_error"] = max(abs(original_mean["ci_low"] - sealed["ci_low"]), abs(original_mean["ci_high"] - sealed["ci_high"]))
    if args.reps == 2000:
        assert validation["sealed_primary_CI_max_abs_error"] < 1e-8
    write_json(out / "implementation_validation.json", validation)

    # Paired-difference OLS is algebraically the mixed-model frame interaction.
    alignment_rows, align_draws, alignment_intervals = [], {}, []
    numerators, denominators = {"pooled": [], "within_frame": []}, {"pooled": [], "within_frame": []}
    paired_all = []
    for model in MODELS:
        a = alignment[alignment.model_key.eq(model)]
        wide = a.pivot(index=["item_id", "condition", "strength", "frozen_answer_role", "baseline_margin_z", "analysis_relation", "CIE_z"], columns="frame", values=["VSA", "VSA_z"]).reset_index()
        wide.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in wide.columns]
        pooled_diff = wide.VSA_z_SELF - wide.VSA_z_PEER
        self_z = (wide.VSA_SELF - wide.VSA_SELF.mean()) / wide.VSA_SELF.std(ddof=0)
        peer_z = (wide.VSA_PEER - wide.VSA_PEER.mean()) / wide.VSA_PEER.std(ddof=0)
        wide["difference_pooled"] = pooled_diff
        wide["difference_within_frame"] = self_z - peer_z
        wide["model_key"] = model
        paired_all.append(wide)
        r_peer = pearsonr(wide.CIE_z, wide.VSA_PEER).statistic
        r_self = pearsonr(wide.CIE_z, wide.VSA_SELF).statistic
        row = {"model_key": model, "r_CIE_PEER": float(r_peer), "r_CIE_SELF": float(r_self), "r_SELF_PEER": float(pearsonr(wide.VSA_SELF, wide.VSA_PEER).statistic), "SD_SELF_div_PEER": float(wide.VSA_SELF.std(ddof=0) / wide.VSA_PEER.std(ddof=0)), "CIE_SD_nats": float(unit[unit.model_key.eq(model)].CIE.std(ddof=0)), "spearman_CIE_PEER": float(spearmanr(wide.CIE_z, wide.VSA_PEER).statistic), "spearman_CIE_SELF": float(spearmanr(wide.CIE_z, wide.VSA_SELF).statistic)}
        for scale in ("pooled", "within_frame"):
            values, numerator, denominator = weighted_slope(wide, "CIE_z", f"difference_{scale}", weights, item_ids)
            row[f"Delta_{scale}"] = float(values[0])
            align_draws[f"{scale}_Delta_{model}"] = values[1:]
            numerators[scale].append(numerator)
            denominators[scale].append(denominator)
            lo, hi = np.quantile(values[1:], [0.025, 0.975])
            alignment_intervals.append({"scale": scale, "estimand": f"Delta_{model}", "estimate": float(values[0]), "ci_low": float(lo), "ci_high": float(hi)})
        alignment_rows.append(row)
    for scale in ("pooled", "within_frame"):
        common = np.sum(numerators[scale], axis=0) / np.sum(denominators[scale], axis=0)
        individual = [numerators[scale][i] / denominators[scale][i] for i in range(3)]
        contrast = individual[0] - (individual[1] + individual[2]) / 2
        for name, values in (("common", common), ("H", contrast)):
            align_draws[f"{scale}_{name}"] = values[1:]
            lo, hi = np.quantile(values[1:], [0.025, 0.975])
            alignment_intervals.append({"scale": scale, "estimand": name, "estimate": float(values[0]), "ci_low": float(lo), "ci_high": float(hi)})
    pd.DataFrame(alignment_rows).to_csv(out / "alignment_scale_comparison.csv", index=False)
    pd.DataFrame(alignment_intervals).to_csv(out / "alignment_intervals.csv", index=False)
    pd.DataFrame(align_draws).to_csv(out / "alignment_bootstrap_draws.csv", index=False)
    # Properly conditional paired differences. Separate fits allow nuisance
    # effects to differ by both reporting frame and model.
    import statsmodels.formula.api as smf
    paired = pd.concat(paired_all, ignore_index=True)
    conditional_rows = []
    for m in MODELS:
        frame = paired[paired.model_key.eq(m)]
        conditional = smf.ols("difference_within_frame ~ CIE_z + C(frozen_answer_role) + baseline_margin_z + C(strength) + C(analysis_relation)", frame).fit(cov_type="cluster", cov_kwds={"groups": frame.item_id})
        conditional_rows.append({"model_key": m, "conditional_Delta": float(conditional.params["CIE_z"]), "item_cluster_SE": float(conditional.bse["CIE_z"]), "p_value": float(conditional.pvalues["CIE_z"])})
    pd.DataFrame(conditional_rows).to_csv(out / "conditional_alignment.csv", index=False)
    # The original adjusted per-frame slopes use the fitted mixed model.
    import patsy
    original_alignment = alignment.copy()
    original_alignment["model_key"] = pd.Categorical(original_alignment.model_key, categories=list(MODELS), ordered=True)
    original_formula = "VSA_z ~ CIE_z * SELF * C(model_key) + C(frozen_answer_role) + baseline_margin_z + C(strength) + C(analysis_relation)"
    fit, _ = author.fit_mixedlm_fallback(original_formula, original_alignment)
    slope_rows = []
    for m in MODELS:
        row = {"model_key": m}
        for label, f in (("PEER", 0), ("SELF", 1)):
            template = original_alignment.iloc[[0]].copy()
            template["model_key"] = m
            template["SELF"] = f
            vectors = []
            for cie in (0, 1):
                template["CIE_z"] = cie
                vectors.append(np.asarray(patsy.build_design_matrices([fit.model.data.design_info],template)[0])[0])
            row[f"{label}_slope"] = float((vectors[1]-vectors[0]) @ fit.fe_params.to_numpy())
        slope_rows.append(row)
    pd.DataFrame(slope_rows).to_csv(out / "per_frame_adjusted_slopes.csv", index=False)
    alignment_expected = json.loads((root / "expected_results/alignment_results.json").read_text())
    model_expected = pd.read_csv(root / "expected_results/model_specific_alignment_deltas.csv").set_index("model_key")
    alignment_errors = []
    for row in alignment_intervals:
        if row["scale"] != "pooled":
            continue
        if row["estimand"].startswith("Delta_"):
            model = row["estimand"].removeprefix("Delta_")
            expected_point = model_expected.loc[model, "Delta"]
            expected_low = model_expected.loc[model, "bootstrap_ci_low"]
            expected_high = model_expected.loc[model, "bootstrap_ci_high"]
        else:
            key = "H" if row["estimand"] == "H" else "common_interaction"
            prefix = "H" if key == "H" else "common"
            expected_point = alignment_expected[key]
            expected_low = alignment_expected[f"{prefix}_bootstrap_ci_low"]
            expected_high = alignment_expected[f"{prefix}_bootstrap_ci_high"]
        alignment_errors.append(abs(row["estimate"] - expected_point))
        if args.reps == 2000:
            alignment_errors.extend([abs(row["ci_low"] - expected_low), abs(row["ci_high"] - expected_high)])
    validation["sealed_alignment_point_and_CI_max_abs_error"] = float(max(alignment_errors))
    assert validation["sealed_alignment_point_and_CI_max_abs_error"] < 1e-8
    write_json(out / "implementation_validation.json", validation)
    print("Completed alignment scaling and conditional analyses", flush=True)

    # Full-category diagnostics, including the previously omitted P/I conditions.
    distributions, answer_stats, conditional_reports, fictional = [], [], [], []
    for m in MODELS:
        report, answer = report_tables[m], answer_tables[m]
        for label, fr in [("ALL_FIVE", report), ("PRIMARY_C", report[report.condition.isin(CONDITIONS)])] + list(report.groupby("condition", sort=True)):
            for frame, subset in fr.groupby("frame", sort=True):
                counts = subset.modal_report.value_counts()
                for role in ROLES:
                    distributions.append({"model_key": m, "condition_group": label, "frame": frame, "report_role": role, "count": int(counts.get(role, 0)), "total": len(subset), "share": float(counts.get(role, 0) / len(subset))})
        for (frame, answer_role), fr in report[report.condition.isin(CONDITIONS)].groupby(["frame", "frozen_answer_role"]):
            for role in ROLES:
                conditional_reports.append({"model_key": m, "frame": frame, "frozen_answer_role": answer_role, "report_role": role, "count": int(fr.modal_report.eq(role).sum()), "total": len(fr), "share": float(fr.modal_report.eq(role).mean())})
        for condition, fr in answer[answer.condition.isin(CONDITIONS)].groupby("condition"):
            answer_stats.append({"model_key": m, "condition": condition, "n_items": len(fr), "CIE_mean": float(fr.CIE.mean()), "CIE_SD": float(fr.CIE.std(ddof=0)), "answer_C_share": float(fr.frozen_answer_role.eq("C").mean()), "answer_P_share": float(fr.frozen_answer_role.eq("P").mean()), "answer_I_share": float(fr.frozen_answer_role.eq("I").mean())})
        fiction = pd.read_csv(root / f"data/supporting/{m}_fiction_report_results.csv")
        for frame, fr in fiction.groupby("frame"):
            fictional.append({"model_key": m, "frame": frame, "n": len(fr), "document_report_count": int(fr.modal_report.eq("document").sum()), "document_report_share": float(fr.modal_report.eq("document").mean()), "mean_VSA": float(fr.VSA.mean())})
    pd.DataFrame(distributions).to_csv(out / "report_role_distributions.csv", index=False)
    pd.DataFrame(conditional_reports).to_csv(out / "reports_conditional_on_recorded_answer.csv", index=False)
    pd.DataFrame(answer_stats).to_csv(out / "answer_descriptives.csv", index=False)
    pd.DataFrame(fictional).to_csv(out / "fictional_controls_by_frame.csv", index=False)

    listed = []
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        expected_hash, name = line.split(maxsplit=1)
        file = root / name.lstrip("*")
        if file.is_file():
            listed.append({"file": name, "sha256": sha(file), "matches": sha(file) == expected_hash})
    assert all(r["matches"] for r in listed)
    lock_hash = sha(root / "protocol/postreview_analysis_lock.json")
    runner_expected = "92e7c26fd9f7a0782b309c419d9ad2b69c8fa5e84d1677a32cee77b54edb9bd2"
    summary = {"study": "CSRA v0.8 R4/A2", "audit_date": "2026-10-15", "selected_common_items": 100, "models": list(MODELS), "unique_answer_triples": int(len(selected[["parametric_answer", "counterfactual_answer", "irrelevant_answer"]].drop_duplicates())), "alignment_rows": len(alignment), "predictive_rows": len(unit), "bootstrap_replicates": args.reps, "all_available_manifest_files_match": True, "available_manifest_files_checked": len(listed), "postreview_lock_SHA256": lock_hash, "execution_runner_expected_SHA256": runner_expected, "postreview_lock_matches_execution_runner": lock_hash == runner_expected, "reconstruction": reconstruction, "implementation_validation": validation, "environment": {name: importlib.import_module(name).__version__ for name in ("numpy", "pandas", "scipy", "sklearn", "statsmodels")}, "limits": ["No model scoring rerun; this audit starts from released aggregate scores.", "Exact prompt text, individual label-permutation scores, full candidate/review pool, and external timing evidence are not present.", "All additional analyses are exploratory, following peer review.", "Fixed original normalizations and folds are retained in bootstrap resamples.", "The original alignment bootstrap is reproduced via its algebraically equivalent paired-difference regression, rather than repeated mixed-model optimization."]}
    write_json(out / "audit_summary.json", summary)
    pd.DataFrame(listed).to_csv(out / "verified_file_checksums.csv", index=False)
    print(pd.DataFrame(point_rows).to_string(index=False), flush=True)
    print(pd.DataFrame(alignment_intervals).to_string(index=False), flush=True)
    print("PASS: independent audit completed", flush=True)


if __name__ == "__main__":
    main()
