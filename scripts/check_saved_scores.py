"""Reconstruct later-study inputs from every committed scoring row on CPU.

Each study runs in a separate process to keep its frozen imports isolated.
No tokenizer/model downloads, predictor unpickling, or GPU execution occur.
"""
from pathlib import Path
import argparse,csv,json,sys,hashlib
from datetime import datetime
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(path.read_text())
def close(actual,expected,tol=1e-9):
    delta=float(np.max(np.abs(np.asarray(actual,float)-np.asarray(expected,float))))
    if not delta<tol:raise AssertionError((delta,tol))
    return delta
def raw_counts(run,pattern):
    result={}
    for path in sorted(run.glob(pattern)):
        ids=[]
        for line in path.read_text().splitlines():
            rows=json.loads(line)['rows'];ids.extend(r['job_id'] for r in rows)
        if len(ids)!=len(set(ids)):raise AssertionError('Duplicate jobs: '+path.name)
        result[path.name]=len(ids)
    return result
def compare_frame(actual,path,keys,columns):
    stored=pd.read_csv(path);actual=actual.sort_values(keys).reset_index(drop=True);stored=stored.sort_values(keys).reset_index(drop=True)
    if len(actual)!=len(stored):raise AssertionError('Row count: '+str(path))
    for key in keys:
        if actual[key].astype(str).tolist()!=stored[key].astype(str).tolist():raise AssertionError('Identity mismatch: '+key)
    return close(actual[columns].to_numpy(),stored[columns].to_numpy())

def factorial(run):
    import analyze_pilot as frozen
    agg,items,bad=frozen.load_data(run)
    if bad or len(items)!=100 or len(agg)!=7200:raise AssertionError('Incomplete factorial run')
    error=compare_frame(agg,run/'primary_analysis/aggregated_reports.csv',['model_key','item_id','condition','cell'],['logp_'+r for r in frozen.ROLES]+['q_'+r for r in frozen.ROLES])
    counts=raw_counts(run,'*_checkpoint.jsonl')
    if sum(counts.values())!=28800 or set(counts.values())!={9600}:raise AssertionError(counts)
    groups={frozen.triple_key(row) for _,row in items.iterrows()}
    if len(groups)!=57:raise AssertionError('Unexpected factorial grouping')
    status=read(run/'primary_analysis/analysis_status.json')
    if status['study_mode']!='exploratory' or status['nonlinear_replicates']!=0:raise AssertionError('Status differs from manuscript')
    return {'jobs':sum(counts.values()),'raw_job_counts':counts,'items':100,'groups':57,'aggregated_rows':7200,'max_aggregation_error':error,'prompt_identities_checked':True,'nonlinear_bootstrap_completed':False}

def development(run):
    import common as frozen
    from predictive import check_cases,group_diagnostics
    lock=frozen.verify_package()
    targets=frozen.aggregate_targets(run,lock)
    if targets!=read(run/'targets.json')['targets']:raise AssertionError('Development target reconstruction differs')
    cases=frozen.aggregate_reports(run,targets,lock)
    if cases!=read(run/'analysis_inputs.json')['cases']:raise AssertionError('Development report reconstruction differs')
    check_cases(cases);counts=raw_counts(run,'scores/*.jsonl')
    if sum(counts.values())!=43200:raise AssertionError(counts)
    diagnostics={}
    for algorithm,folder in [('ridge',run/'analysis/ridge'),('gradient_boosting',ROOT/'studies/development/secondary_refit/gradient_boosting')]:
        errors=pd.read_csv(folder/'heldout_errors.csv');summary=read(folder/'development_summary.json')
        if len(errors)!=900 or errors.case_index.nunique()!=900:raise AssertionError('Development prediction count')
        for row in errors.itertuples():
            c=cases[row.case_index]
            if (row.item_id,row.target,row.strength)!=(c['item_id'],c['target'],c['strength']):raise AssertionError('Development case identity')
            close(row.Y,c['Y'])
            close(row.paired_error_improvement,row.external_squared_error-row.native_squared_error)
        values=group_diagnostics(cases,errors.to_dict('records'))
        max_error=0.
        for actual,expected in zip(values,summary['known_target_diagnostics']):
            if actual['target']!=expected['target']:raise AssertionError('Development target order')
            for name in ('Y_variance','R2_native','R2_second_observer','R2_tfidf','R2_embedding','delta_R2'):
                max_error=max(max_error,close(actual[name],expected[name]))
        close(np.mean([x['delta_R2'] for x in values]),summary['equal_target_mean_delta_R2'])
        for fold in range(5):
            record=read(folder/f'outer_fold_{fold}.json')
            heldout={r['group'] for r in record['results']}
            if heldout&set(record['training_groups']):raise AssertionError('Development fold leakage')
        diagnostics[algorithm]={'equal_target_mean_delta_R2':summary['equal_target_mean_delta_R2'],'max_summary_error':max_error}
    return {'jobs':sum(counts.values()),'raw_job_counts':counts,'items':100,'groups':len({c['group'] for c in cases}),'cases':len(cases),'prompt_identities_checked':True,'diagnostics':diagnostics,'mistral_target_outcomes_used':False}

def fresh40(run):
    import core as frozen
    freeze=frozen.verify_freeze(run);record=read(run/'pre_outcome_freeze.json')
    if record['technical_fixture_only'] or record['fresh_outcomes_at_freeze']!=0:raise AssertionError('Not the real pre-outcome freeze')
    targets=frozen.aggregate_targets(run,freeze)
    if targets!=read(run/'targets.json')['cases']:raise AssertionError('Fresh targets differ')
    cases=frozen.aggregate_reports(run,targets,freeze)
    if cases!=read(run/'analysis/fresh_analysis_inputs.json')['cases']:raise AssertionError('Fresh reports differ')
    counts=raw_counts(run,'scores/*.jsonl')
    if sum(counts.values())!=17280 or len(cases)!=360 or len({c['group'] for c in cases})!=40:raise AssertionError('Fresh run incomplete')
    first=None
    for path in (run/'scores').glob('*.jsonl'):
        for line in path.read_text().splitlines():
            batch=json.loads(line);stamp=datetime.fromisoformat(batch['committed_utc'])
            first=min(stamp,first) if first else stamp
            if stamp<=datetime.fromisoformat(record['created_utc']):raise AssertionError('Local commit precedes freeze')
            runtime=batch['runtime']
            if runtime['model_revision']!=frozen.plan()['models'][runtime['model']]['revision']:raise AssertionError('Model revision differs')
            if runtime['precision']!='unquantized_FP32' or runtime['tf32'] is not False:raise AssertionError('Precision differs')
    from learning import check_probe_invariance
    check_probe_invariance(cases)
    if {c['group'] for c in cases}&{c['group'] for c in frozen.old_cases()}:raise AssertionError('Training/test group overlap')
    # These fixed-policy comparisons were added after inspection, not promoted
    # into the frozen primary contrast family.
    frame=pd.read_csv(run/'analysis/per_case_predictions.csv');frame=frame[frame.algorithm.eq('gradient_boosting')]
    groups=sorted(frame.group.unique());samples=np.random.default_rng(20261020).integers(0,40,(10000,40))
    audit=read(ROOT/'manuscript_support/checks/fresh_audit.json');additional={}
    for name,a,b in [('observer_added_to_probe','probe','probe_observer'),('self_plus_probe_over_observer_plus_probe','probe_observer','probe_self')]:
        delta=(frame[a+'_squared_error']-frame[b+'_squared_error'])/frame.old100_Y_variance
        grouped=delta.groupby(frame.group).mean().loc[groups].to_numpy();expected=audit['additional_postrun_comparisons'][name]
        close(grouped.mean(),expected['estimate']);close(np.quantile(grouped[samples].mean(1),[.025,.975]),expected['descriptive_95pct_interval'])
        additional[name]=float(grouped.mean())
    from analyze import cost_summary
    if cost_summary(run)!=read(run/'analysis/actual_costs.json'):raise AssertionError('Fresh cost summaries differ')
    return {'jobs':sum(counts.values()),'raw_job_counts':counts,'items':40,'groups':40,'cases':360,'prompt_identities_checked':True,'probe_reference_swap_invariance':True,'first_local_commit_after_local_freeze':True,'external_timestamp_established':False,'mistral_target_outcomes_used':False,'additional_postrun_fixed_policy_checks':additional,'successful_forward_costs_recomputed':True}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('study',choices=['factorial','development','fresh40']);args=parser.parse_args()
    run=ROOT/'studies'/args.study/'run1'
    sys.path.insert(0,str(run/'frozen_package/code'))
    result=globals()[args.study](run);result.update(status='PASSED',scope='Saved raw scoring consistency and aggregation; model inference not repeated.')
    out=ROOT/'reproduced_results'/f'{args.study}_score_checks.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
