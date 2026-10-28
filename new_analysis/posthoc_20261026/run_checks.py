#!/usr/bin/env python3
"""Post hoc CPU checks; read-only access to an extracted CSRA R4 repository."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

POLICIES = ('visible', 'source', 'observer', 'probe', 'probe_source', 'probe_observer')
CONTRASTS = {
    'source_over_visible': ('visible', 'source'),
    'source_over_observer': ('observer', 'source'),
    'source_added_to_probe': ('probe', 'probe_source'),
    'source_alone_over_probe': ('probe', 'source'),
    'observer_added_to_probe': ('probe', 'probe_observer'),
    'probe_source_over_probe_observer': ('probe_observer', 'probe_source'),
}
OLD_POLICY = dict(source='self', probe_source='probe_self', **{k:k for k in ('visible','observer','probe','probe_observer')})

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_json(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def normalize(s):
    return unicodedata.normalize('NFKC', s).strip().casefold()

def raw_outcomes(run, cases):
    """Reconstruct both contrasts directly from saved six-permutation scores."""
    values = {}
    orders = list(itertools.permutations(('P', 'C', 'I')))
    for target in sorted({c['target'] for c in cases}):
        support = {}
        counts = {}
        for line in (run/'scores'/f'answer_{target}.jsonl').read_text().splitlines():
            for row in json.loads(line)['rows']:
                tag, model, item, condition, permutation = row['job_id'].split('|')
                if tag != 'answer' or model != target:
                    raise ValueError('Wrong raw answer identity')
                order = orders[int(permutation[1:])]
                key = (item, condition)
                d = support.setdefault(key, {r:[] for r in ('P','C','I')})
                counts[key] = counts.get(key,0)+1
                for label, role in zip('ABC', order):
                    d[role].append(float(row['logp'][label]))
        assert set(counts.values()) == {6}
        for c in cases:
            if c['target'] != target:
                continue
            dc = support[(c['item_id'], 'C_'+c['strength'])]
            dp = support[(c['item_id'], 'P_'+c['strength'])]
            log_y = (np.mean(dc['C'])-np.mean(dc['P']))-(np.mean(dp['C'])-np.mean(dp['P']))
            probability_y = (np.exp(dc['C']).mean()-np.exp(dc['P']).mean())-(np.exp(dp['C']).mean()-np.exp(dp['P']).mean())
            assert abs(log_y-c['Y']) < 1e-10
            assert -2-1e-12 <= probability_y <= 2+1e-12
            values[(c['item_id'],target,c['strength'])] = float(probability_y)
    assert len(values) == len(cases)
    return values

def agreement(c, item_lookup):
    # Role C always denotes the displayed document entity; no factual P identity enters.
    item = item_lookup[c['item_id']]
    value = float(normalize(c['answer']) == normalize(item['counterfactual_answer']))
    assert value == float(c['answer_role']=='C')
    assert item['counterfactual_answer'] in c['document']
    return value

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'results')
    args = parser.parse_args()
    repo = args.repo.resolve(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if out == repo or repo in out.parents:
        raise ValueError('Output must be outside the frozen repository')
    started = time.perf_counter()
    inventory_count=0
    for line in (repo/'SHA256SUMS.txt').read_text().splitlines():
        expected,name=line.split('  ',1)
        if digest(repo/name)!=expected:
            raise ValueError('Release checksum mismatch: '+name)
        inventory_count+=1
    print('Verified release inventory:',inventory_count,'files',flush=True)
    run = repo/'studies/fresh40/run1'
    sys.path.insert(0, str(run/'frozen_package/code'))
    import learning as L
    import core
    L.check_environment(); core.verify_freeze(run)
    old = L.old_cases()
    fresh = json.loads((run/'analysis/fresh_analysis_inputs.json').read_text())['cases']
    assert len(old)==900 and len(fresh)==360
    assert len({c['group'] for c in old})==57 and len({c['group'] for c in fresh})==40
    assert not {c['group'] for c in old}&{c['group'] for c in fresh}
    L.check_probe_invariance(old+fresh)
    item_lookup = {x['item_id']:x for x in core.old_items()+core.items()}
    for c in old+fresh:
        agreement(c,item_lookup)
    Eold = core.load_embeddings(run,old,'old')
    Efresh = core.load_embeddings(run,fresh,'fresh')
    original = pd.read_csv(run/'analysis/per_case_predictions.csv')
    original = original.set_index(['algorithm','item_id','target','strength'])
    selected = json.loads((run/'observer_selection.json').read_text())['selected']
    frozen = joblib.load(run/'frozen_models.joblib')
    assert selected==frozen['selected']
    bounded_old = raw_outcomes(repo/'studies/development/run1',old)
    bounded_fresh = raw_outcomes(run,fresh)
    source_files = [Path(__file__),Path(__file__).with_name('analysis_plan.json'),run/'frozen_models.joblib',run/'embeddings.npy',run/'embedding_catalog.json',run/'observer_selection.json',run/'analysis/fresh_analysis_inputs.json',run/'frozen_package/development/analysis_inputs.json',run/'analysis/per_case_predictions.csv']
    source_files += list((repo/'studies/development/run1/scores').glob('answer_*.jsonl')) + list((run/'scores').glob('answer_*.jsonl'))
    source_hashes = {str(p.relative_to(repo)) if p.is_relative_to(repo) else p.name:digest(p) for p in source_files}
    write_json(out/'run_manifest.json',dict(status='POST_HOC_CHECKS_STARTED',environment=L.environment(),python=platform.python_version(),source_hashes=source_hashes,selected_observers=selected,plan=json.loads(Path(__file__).with_name('analysis_plan.json').read_text())))
    all_rows=[]; summaries=[]; verification={'release_inventory_files_checked':inventory_count}; diagnostics=[]
    groups=sorted({c['group'] for c in fresh})
    samples=np.random.default_rng(core.cfg()['seed']).integers(0,40,size=(10000,40))
    for outcome in ('log_support','bounded_probability'):
        train=copy.deepcopy(old); test=copy.deepcopy(fresh)
        if outcome=='bounded_probability':
            for cases,lookup in ((train,bounded_old),(test,bounded_fresh)):
                for c in cases:c['Y']=lookup[(c['item_id'],c['target'],c['strength'])]
        for target in core.cfg()['target_models']:
            for split,cc in [('development',train),('fresh40',test)]:
                rr=[c for c in cc if c['target']==target]
                diagnostics.append(dict(outcome=outcome,split=split,target=target,n=len(rr),variance=float(np.var([c['Y'] for c in rr])),minimum=float(min(c['Y'] for c in rr)),maximum=float(max(c['Y'] for c in rr)),agreement_share=float(np.mean([agreement(c,item_lookup) for c in rr]))))
        print('Nested development-only text forecasts:',outcome,flush=True)
        tp,textfits=L.legacy.text_stack(train,Eold,np.arange(len(train)))
        vp=L.text_predictions(textfits,test,Efresh,np.arange(len(test)))
        np.savez_compressed(out/(outcome+'_text_forecasts.npz'),development_nested_oof=tp,fresh_predictions=vp)
        if outcome=='log_support':
            frozen_vp=L.text_predictions(frozen['text_models'],test,Efresh,np.arange(len(test)))
            verification['max_original_text_forecast_difference']=float(np.max(np.abs(vp-frozen_vp)))
            assert verification['max_original_text_forecast_difference'] < 1e-8
        for variant in ('original','agreement'):
            def make_features(c,text,policy):
                reporter = c['target'] if policy in ('source','probe_source') else selected[c['target']] if policy in ('observer','probe_observer') else None
                tier='probe' if policy in ('probe','probe_source','probe_observer') else 'visible'
                x=L.features(c,text,tier,reporter)
                if variant=='agreement':x=x+[agreement(c,item_lookup)]
                return x
            for algorithm in L.ALGORITHMS:
                print('Fitting fixed decoders:',outcome,variant,algorithm,flush=True)
                current=[]
                max_diff=0.0
                for target in core.cfg()['target_models']:
                    itrain=[i for i,c in enumerate(train) if c['target']==target]
                    itest=[i for i,c in enumerate(test) if c['target']==target]
                    ytrain=np.array([train[i]['Y'] for i in itrain]); variance=float(np.var(ytrain))
                    assert variance>1e-12
                    rows={i:dict(outcome=outcome,variant=variant,algorithm=algorithm,item_id=test[i]['item_id'],group=test[i]['group'],target=target,strength=test[i]['strength'],relation=test[i]['relation'],Y=test[i]['Y'],development_variance=variance,selected_observer=selected[target],agreement=agreement(test[i],item_lookup)) for i in itest}
                    for policy in POLICIES:
                        X=np.array([make_features(train[i],tp[i],policy) for i in itrain],float)
                        V=np.array([make_features(test[i],vp[i],policy) for i in itest],float)
                        scaler=StandardScaler(); X=scaler.fit_transform(X)
                        if algorithm=='ridge':model=Ridge(alpha=1,solver='cholesky')
                        else:model=GradientBoostingRegressor(n_estimators=100,learning_rate=.05,max_depth=2,loss='squared_error',random_state=core.cfg()['seed'])
                        model.fit(X,ytrain); predictions=model.predict(scaler.transform(V))
                        assert np.isfinite(predictions).all()
                        for i,p in zip(itest,predictions):
                            rows[i][policy+'_prediction']=float(p)
                            rows[i][policy+'_squared_error']=float((test[i]['Y']-p)**2)
                            if outcome=='log_support' and variant=='original':
                                expected=original.loc[(algorithm,test[i]['item_id'],target,test[i]['strength']),OLD_POLICY[policy]+'_prediction']
                                max_diff=max(max_diff,abs(float(p)-float(expected)))
                    current.extend(rows.values())
                if outcome=='log_support' and variant=='original':
                    verification[algorithm+'_max_original_policy_prediction_difference']=max_diff
                    assert max_diff<1e-8,(algorithm,max_diff)
                all_rows.extend(current)
                data=pd.DataFrame(current)
                for contrast,(baseline,new) in CONTRASTS.items():
                    gains=(data[baseline+'_squared_error']-data[new+'_squared_error'])/data.development_variance
                    bygroup=gains.groupby(data.group).mean().reindex(groups).to_numpy()
                    estimate=float(bygroup.mean()); lo,hi=np.quantile(bygroup[samples].mean(axis=1),[.025,.975])
                    loo=(bygroup.sum()-bygroup)/39
                    summary=dict(outcome=outcome,variant=variant,algorithm=algorithm,contrast=contrast,gain=estimate,ci_low=float(lo),ci_high=float(hi),leave_one_group_out_min=float(loo.min()),leave_one_group_out_max=float(loo.max()))
                    summary.update({t:float(gains[data.target==t].mean()) for t in core.cfg()['target_models']})
                    summaries.append(summary)
                    if outcome=='log_support' and variant=='original' and contrast=='source_added_to_probe':
                        expected=json.loads((run/'analysis/qualification_summary.json').read_text())['results'][algorithm]['contrasts']['self_added_to_probe']
                        assert abs(estimate-expected['normalized_MSE_reduction'])<1e-10
                        assert np.max(np.abs(np.array([lo,hi])-expected['descriptive_95pct_cluster_bootstrap_interval']))<1e-10
                pd.DataFrame(all_rows).to_csv(out/'per_case_predictions.csv',index=False)
                pd.DataFrame(summaries).to_csv(out/'contrasts.csv',index=False)
    data=pd.DataFrame(all_rows); absolute=[]
    for keys,d in data.groupby(['outcome','variant','algorithm','target'],sort=False):
        variance=float(np.var(d.Y))
        for policy in POLICIES:
            mse=float(d[policy+'_squared_error'].mean())
            absolute.append(dict(zip(['outcome','variant','algorithm','target'],keys),policy=policy,MSE=mse,R2_fresh=1-mse/variance))
    pd.DataFrame(absolute).to_csv(out/'absolute_performance.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'outcome_diagnostics.csv',index=False)
    write_json(out/'verification.json',dict(status='PASSED',**verification,original_policy_predictions_compared=4320,original_intervals_reproduced=True,all_source_hashes_unchanged=all(digest(p)==source_hashes[str(p.relative_to(repo)) if p.is_relative_to(repo) else p.name] for p in source_files)))
    manifest=json.loads((out/'run_manifest.json').read_text());manifest.update(status='POST_HOC_CHECKS_COMPLETED',runtime_seconds=time.perf_counter()-started,original_verification=verification)
    write_json(out/'run_manifest.json',manifest)
    print(pd.DataFrame(summaries).query("algorithm=='gradient_boosting' and contrast=='source_added_to_probe'").to_string(index=False),flush=True)
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
