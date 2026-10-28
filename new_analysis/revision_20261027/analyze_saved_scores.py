#!/usr/bin/env python3
"""Exploratory target heterogeneity and factorial readouts from saved CSRA scores.

No model calls, predictor fitting, tuning, or changes to archived study files.
Run with Python 3.12 and the repository requirements-analysis.txt environment.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd

TARGETS = ('granite', 'phi', 'qwen')
CONTRASTS = {
    'source_over_visible': ('visible', 'source'),
    'source_over_observer': ('observer', 'source'),
    'source_added_to_probe': ('probe', 'probe_source'),
    'observer_added_to_probe': ('probe', 'probe_observer'),
    'probe_source_over_probe_observer': ('probe_observer', 'probe_source'),
    'source_alone_over_probe': ('probe', 'source'),
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

def interval(values):
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    if valid.mean() < .95:
        return dict(ci_low=None, ci_high=None, valid_bootstrap_draws=int(valid.sum()))
    lo, hi = np.quantile(values[valid], [.025, .975])
    return dict(ci_low=float(lo), ci_high=float(hi), valid_bootstrap_draws=int(valid.sum()))

def check_close(a, b, message, tol=1e-10):
    difference = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
    if difference > tol:
        raise AssertionError((message, difference))
    return difference

def fresh40(repo, out):
    source = repo / 'new_analysis/posthoc_20261026/results'
    data = pd.read_csv(source / 'per_case_predictions.csv')
    existing = pd.read_csv(source / 'contrasts.csv')
    groups = sorted(data.group.unique())
    assert len(groups) == 40 and len(data) == 2880
    samples = np.random.default_rng(20261020).integers(0, 40, (10000, 40))
    pooled, targets, leave_target, group_rows, strata, diagnostics = [], [], [], [], [], []
    errors = []
    scenario = ['outcome', 'variant', 'algorithm']
    for keys, frame in data.groupby(scenario, sort=True):
        common = dict(zip(scenario, keys))
        assert len(frame) == 360
        assert frame.groupby(['group', 'target']).size().eq(3).all()
        assert frame.groupby('target').development_variance.nunique().eq(1).all()
        for policy in ('visible', 'source', 'observer', 'probe', 'probe_source', 'probe_observer'):
            errors.append(check_close((frame.Y-frame[policy+'_prediction'])**2,
                                      frame[policy+'_squared_error'], 'saved prediction error'))
        for name, (baseline, added) in CONTRASTS.items():
            gain = (frame[baseline+'_squared_error'] - frame[added+'_squared_error']) / frame.development_variance
            matrix = gain.groupby([frame.group, frame.target]).mean().unstack().reindex(index=groups, columns=TARGETS).to_numpy()
            assert matrix.shape == (40, 3) and np.isfinite(matrix).all()
            group_mean = matrix.mean(axis=1)
            draws = group_mean[samples].mean(axis=1)
            mean = float(group_mean.mean())
            row = dict(**common, contrast=name, estimate=mean, **interval(draws))
            pooled.append(row)
            old = existing[(existing.outcome == keys[0]) & (existing.variant == keys[1]) &
                           (existing.algorithm == keys[2]) & (existing.contrast == name)].iloc[0]
            errors.append(check_close([mean, row['ci_low'], row['ci_high']],
                                      [old.gain, old.ci_low, old.ci_high], 'V4 pooled estimate and interval'))
            for j, target in enumerate(TARGETS):
                values = matrix[:, j]
                targets.append(dict(**common, contrast=name, target=target, estimate=float(values.mean()),
                                    **interval(values[samples].mean(axis=1)), n_groups=40, n_cases=120))
                remaining = np.delete(matrix, j, axis=1).mean(axis=1)
                leave_target.append(dict(**common, contrast=name, omitted_target=target,
                                         estimate=float(remaining.mean()), **interval(remaining[samples].mean(axis=1))))
                for i, group in enumerate(groups):
                    group_rows.append(dict(**common, contrast=name, target=target, group=group,
                                           group_gain=float(values[i]),
                                           leave_group_out_gain=float((values.sum()-values[i])/39)))
            for field in ('relation', 'strength', 'agreement'):
                for value, part in frame.assign(gain=gain).groupby(field):
                    for target, g in part.groupby('target'):
                        strata.append(dict(**common, contrast=name, target=target, stratum=field, value=str(value),
                                           n_cases=len(g), n_groups=g.group.nunique(), estimate=float(g.gain.mean())))
        for target, g in frame.groupby('target'):
            for policy in ('visible', 'source', 'observer', 'probe', 'probe_source', 'probe_observer'):
                mse = float(g[policy+'_squared_error'].mean())
                diagnostics.append(dict(**common, target=target, policy=policy, n_cases=len(g), n_groups=g.group.nunique(),
                                        agreement_share=float(g.agreement.mean()), development_variance=float(g.development_variance.iloc[0]),
                                        fresh_variance=float(np.var(g.Y)), fresh_min=float(g.Y.min()), fresh_max=float(g.Y.max()),
                                        MSE=mse, MAE=float(np.abs(g.Y-g[policy+'_prediction']).mean()),
                                        normalized_MSE=mse/float(g.development_variance.iloc[0]), R2_fresh=1-mse/float(np.var(g.Y))))
    for name, rows in [('fresh_pooled', pooled), ('fresh_by_target', targets), ('fresh_leave_one_target_out', leave_target),
                       ('fresh_group_influence', group_rows), ('fresh_stratum_gains', strata), ('fresh_diagnostics', diagnostics)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'), index=False)
    return dict(rows=len(data), scenarios=8, existing_pooled_comparisons_reproduced=len(pooled),
                target_intervals=len(targets), leave_one_target_out_intervals=len(leave_target),
                max_verification_difference=max(errors))

def factorial(repo, out):
    root = repo / 'studies/factorial/run1/frozen_package'
    sys.path.insert(0, str(root/'code'))
    from analyze_pilot import bootstrap_weights
    selected = pd.read_csv(root/'inputs/items.csv')
    folds = json.loads((root/'protocol/folds.json').read_text())
    weights, rejected = bootstrap_weights(selected, selected.item_id.map(folds).to_numpy(), 10000, 20261016)
    assert weights.shape == (10001, 100)
    reports = pd.read_csv(repo/'studies/factorial/run1/primary_analysis/aggregated_reports.csv')
    answers = pd.read_csv(root/'inputs/answer_targets.csv')
    reports = reports.merge(answers[['model_key', 'item_id', 'condition', 'frozen_answer_role']],
                            on=['model_key', 'item_id', 'condition'], validate='many_to_one')
    assert len(reports) == 7200 and reports.groupby(['model_key', 'item_id', 'condition']).size().eq(8).all()
    logp = reports[['logp_document','logp_prior','logp_both','logp_uncertain']].to_numpy()
    prob = np.exp(logp-logp.max(axis=1, keepdims=True)); prob /= prob.sum(axis=1, keepdims=True)
    errors = [check_close(prob, reports[['q_document','q_prior','q_both','q_uncertain']], 'four-role normalization')]
    reports['document_minus_prior'] = reports.q_document - reports.q_prior
    reports['document_prior_logratio'] = reports.logp_document - reports.logp_prior
    assert reports.document_minus_prior.between(-1, 1).all()
    original = pd.read_csv(repo/'studies/factorial/run1/primary_analysis/primary.csv').set_index('contrast')
    original_target = pd.read_csv(repo/'studies/factorial/run1/primary_analysis/per_model_category_contrasts.csv').set_index(['model','contrast'])
    labels = {'ownership':('OWN','OTHER'), 'hedging':('hedged','certain'), 'visibility':('shown','hidden')}
    contrasts, profiles, associations, balance_counts = [], [], [], []
    for model in (*TARGETS, 'equal_target'):
        frame = reports if model == 'equal_target' else reports[reports.model_key == model]
        for outcome in ('q_both','document_minus_prior','document_prior_logratio'):
            for factor, (positive, negative) in labels.items():
                means = frame.groupby(['item_id', factor])[outcome].mean().unstack().reindex(selected.item_id)
                values = means[positive].to_numpy()-means[negative].to_numpy()
                draws = weights@values / weights.sum(axis=1)
                contrasts.append(dict(target=model, outcome=outcome, contrast=factor, estimate=float(draws[0]),
                                      **interval(draws[1:]), n_items=100, n_groups=57))
                if outcome == 'q_both':
                    if model == 'equal_target':
                        errors.append(check_close(draws[0], original.loc[factor, 'estimate'], 'original pooled factorial effect'))
                        errors.append(check_close(np.quantile(draws[1:],[.00625,.99375]),
                                                  original.loc[factor,['family_ci_low','family_ci_high']], 'original adjusted interval'))
                    else:
                        old = original_target.loc[(model, factor)]
                        errors.append(check_close([draws[0],np.quantile(draws[1:],.025),np.quantile(draws[1:],.975)],
                                                  old[['estimate','ci_low','ci_high']], 'original per-target factorial interval'))
        for visibility in ('shown', 'hidden'):
            for outcome in ('q_document','q_prior','q_both','q_uncertain','document_minus_prior','document_prior_logratio'):
                means = frame[frame.visibility==visibility].groupby('item_id')[outcome].mean().reindex(selected.item_id).to_numpy()
                draws = weights@means / weights.sum(axis=1)
                profiles.append(dict(target=model, visibility=visibility, outcome=outcome, estimate=float(draws[0]), **interval(draws[1:])))
        if model == 'equal_target':
            continue
        vis_association = {}
        for visibility in ('shown', 'hidden'):
            g = frame[frame.visibility == visibility]
            means = {}
            for answer in ('C', 'P'):
                rows = g[g.frozen_answer_role==answer]
                sums = rows.groupby('item_id').document_minus_prior.sum().reindex(selected.item_id, fill_value=0).to_numpy()
                counts = rows.groupby('item_id').size().reindex(selected.item_id, fill_value=0).to_numpy()
                numerator, denominator = weights@sums, weights@counts
                means[answer] = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator>0)
                balance_counts.append(dict(target=model, visibility=visibility, answer_role=answer,
                                           n_report_rows=len(rows), n_items=rows.item_id.nunique(),
                                           n_item_strength_cases=len(rows[['item_id','condition']].drop_duplicates())))
            difference = means['C']-means['P']
            vis_association[visibility] = difference
            associations.append(dict(target=model, visibility=visibility, estimate=float(difference[0]), **interval(difference[1:])))
        difference = vis_association['shown']-vis_association['hidden']
        associations.append(dict(target=model, visibility='shown_minus_hidden', estimate=float(difference[0]), **interval(difference[1:])))
        for visibility, g in frame.groupby('visibility'):
            for role, rows in g.groupby('frozen_answer_role'):
                if role in ('C', 'P'):continue
                balance_counts.append(dict(target=model, visibility=visibility, answer_role=role,
                                           n_report_rows=len(rows), n_items=rows.item_id.nunique(),
                                           n_item_strength_cases=len(rows[['item_id','condition']].drop_duplicates())))
    for name, rows in [('factorial_contrasts', contrasts),('factorial_visibility_profiles',profiles),
                       ('factorial_answer_association',associations),('factorial_answer_counts',balance_counts)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'), index=False)
    return dict(report_rows=len(reports), items=100, ordered_triple_groups=57, bootstrap_draws=10000,
                technical_redraws=rejected, original_per_target_intervals_reproduced=9,
                original_adjusted_category_intervals_reproduced=3, max_verification_difference=max(errors))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out', type=Path)
    args=parser.parse_args()
    repo=args.repo.resolve(); out=(args.out or Path(__file__).resolve().parent/'results').resolve()
    out.mkdir(parents=True, exist_ok=True)
    started=time.perf_counter()
    inputs=[repo/'new_analysis/posthoc_20261026/results'/name for name in ('per_case_predictions.csv','contrasts.csv')]
    inputs += [repo/'studies/factorial/run1'/name for name in ('primary_analysis/aggregated_reports.csv','primary_analysis/primary.csv','primary_analysis/per_model_category_contrasts.csv','frozen_package/inputs/items.csv','frozen_package/inputs/answer_targets.csv','frozen_package/protocol/folds.json')]
    before={str(p.relative_to(repo)):sha(p) for p in inputs}
    result=dict(fresh40=fresh40(repo,out), factorial=factorial(repo,out))
    assert before == {str(p.relative_to(repo)):sha(p) for p in inputs}
    result.update(status='PASSED', analysis_status='POST_HOC_EXPLORATORY', inference_calls=0, predictor_refits=0,
                  source_inputs_unchanged=True, source_sha256=before, script_sha256=sha(Path(__file__)),
                  specification_sha256=sha(Path(__file__).with_name('analysis_spec.json')),
                  python=platform.python_version(), packages={k:importlib.metadata.version(k) for k in ('numpy','pandas','scipy','scikit-learn')},
                  runtime_seconds=time.perf_counter()-started)
    save_json(out/'VALIDATION.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
