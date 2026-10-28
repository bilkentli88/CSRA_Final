"""CPU checks for the V1-based revision; no model downloads or inference.

Recompute Fresh40 errors, eight contrasts and bootstrap intervals from saved
predictions; check factorial category contrasts and stored bootstrap draws;
independently refit V1 orthonormal point estimates with scikit-learn.
Run from any directory with the analysis packages installed.
"""
from pathlib import Path
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import csv,json,sys,hashlib,re
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
checks={}

def close(actual,expected,tol=1e-10):
    error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    assert error<tol,(actual,expected,error)
    return error

fresh=pd.read_csv(HERE/'fresh_predictions.csv')
summary=json.loads((HERE/'fresh_summary.json').read_text())
assert len(fresh)==720
contrasts={'self_over_visible':('visible','self'),
 'self_over_selected_observer':('observer','self'),
 'self_added_to_probe':('probe','probe_self'),
 'self_over_probe_cost_comparison':('probe','self')}
groups=sorted(fresh.group.unique())
assert len(groups)==40
samples=np.random.default_rng(20261020).integers(0,40,size=(10000,40))
for algorithm,frame in fresh.groupby('algorithm'):
    assert len(frame)==360 and frame.groupby('group').size().eq(9).all()
    for policy in ['visible','self','observer','probe','probe_self','probe_observer','reference_ceiling']:
        error=(frame.Y-frame[policy+'_prediction'])**2
        checks[algorithm+'_'+policy+'_saved_error']=close(error,frame[policy+'_squared_error'])
    for name,(a,b) in contrasts.items():
        values=(frame[a+'_squared_error']-frame[b+'_squared_error'])/frame.old100_Y_variance
        pergroup=values.groupby(frame.group).mean().loc[groups].to_numpy()
        expected=summary['results'][algorithm]['contrasts'][name]
        checks[algorithm+'_'+name+'_estimate']=close(pergroup.mean(),expected['normalized_MSE_reduction'])
        checks[algorithm+'_'+name+'_interval']=close(np.quantile(pergroup[samples].mean(1),[.025,.975]),expected['descriptive_95pct_cluster_bootstrap_interval'])
        for target,target_frame in frame.groupby('target'):
            checks[algorithm+'_'+name+'_'+target]=close(values.loc[target_frame.index].mean(),expected['by_target'][target])
    for target,target_frame in frame.groupby('target'):
        variance=np.var(target_frame.Y)
        for policy in ['visible','self','observer','probe','probe_self']:
            actual=1-target_frame[policy+'_squared_error'].mean()/variance
            checks[algorithm+'_'+target+'_'+policy+'_R2']=close(actual,summary['results'][algorithm]['target_metrics'][target][policy]['R2_fresh'])

factor=pd.read_csv(HERE/'factorial_aggregated_reports.csv')
primary=pd.read_csv(HERE/'factorial_primary.csv').set_index('contrast')
assert len(factor)==7200 and factor.groupby('model_key').size().eq(2400).all()
logs=factor[['logp_document','logp_prior','logp_both','logp_uncertain']].to_numpy()
q=np.exp(logs-logs.max(1)[:,None]);q/=q.sum(1)[:,None]
checks['factorial_normalized_support']=close(q,factor[['q_document','q_prior','q_both','q_uncertain']].to_numpy())
for name,column,a,b in [('ownership','ownership','OWN','OTHER'),('hedging','hedging','hedged','certain'),('visibility','visibility','shown','hidden')]:
    actual=factor.loc[factor[column].eq(a),'q_both'].mean()-factor.loc[factor[column].eq(b),'q_both'].mean()
    checks['factorial_'+name]=close(actual,primary.loc[name,'estimate'])
draws=pd.read_csv(HERE/'factorial_primary_draws.csv')
assert len(draws)==10000
for key in primary.index:
    checks['factorial_'+key+'_interval']=close(np.quantile(draws[key],[.00625,.99375]),primary.loc[key,['family_ci_low','family_ci_high']].to_numpy(float))

# Retain V1's exact orthonormal representation; this is not an ALR substitution.
sys.path.insert(0,str(PROJECT/'original_reanalysis/code'))
import extend_analysis as extension
unit=pd.read_csv(PROJECT/'original_reanalysis/audit/results/reconstructed_primary_units.csv')
expected=pd.read_csv(PROJECT/'original_reanalysis/results/full_role_prediction.csv')
helmert=np.array([[1,-1,0,0],[1,1,-2,0],[1,1,1,-3]],float)/np.sqrt([2,6,12])[:,None]
for model in ['qwen','granite','phi']:
    frame=unit[unit.model_key.eq(model)].copy().reset_index(drop=True)
    reports=pd.read_csv(PROJECT/f'original_release/data/supporting/{model}_main_report_results.csv')
    reports=reports[reports.condition.isin(['C_SKEPTICAL','C_PROVISIONAL','C_BARE'])].copy()
    scores=reports[['logp_report_'+role for role in ['document','prior','both','uncertain']]].to_numpy()@helmert.T
    for j in range(3):reports['b'+str(j+1)]=scores[:,j]
    for cue in ['PEER','SELF']:
        cols=['b1','b2','b3']
        features=reports[reports.frame.eq(cue)][['item_id','condition']+cols].rename(columns={c:cue+'_'+c for c in cols})
        frame=frame.merge(features,on=['item_id','condition'],validate='one_to_one')
    for case,cat in [('with_recorded_answer',['frozen_answer_role','strength','analysis_relation']),('without_recorded_answer',['strength','analysis_relation'])]:
        peer=['baseline_margin']+['PEER_b'+str(j) for j in [1,2,3]]
        both=peer+['SELF_b'+str(j) for j in [1,2,3]]
        actual=extension.sklearn_loss(frame,peer,cat)-extension.sklearn_loss(frame,both,cat)
        target=expected[expected.case.eq(case)&expected.model_key.eq(model)].iloc[0].estimate
        checks[case+'_'+model+'_orthonormal_point']=close(actual,target,tol=1e-8)

report={'status':'PASSED','numeric_checks':len(checks),'max_absolute_difference':max(checks.values()),
 'fresh_cases_per_algorithm':360,'fresh_bootstrap_draws':10000,'factorial_rows':7200,
 'factorial_bootstrap_draws':10000,'orthonormal_point_estimates_independently_refitted':6,
 'scope':'Saved score/prediction consistency and CPU statistical checks; model inference not rerun.'}
output=Path(os.environ.get('CSRA_VALIDATION_OUTPUT',PROJECT/'reproduced_results/manuscript_checks.json'))
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
