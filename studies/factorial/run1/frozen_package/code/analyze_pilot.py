"""Paired-cluster primary analysis and prespecified descriptive analyses."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,importlib.metadata
from pathlib import Path
import numpy as np,pandas as pd
from pilot_common import ROOT,ROLES,LABELS,config,rows,triple_key,jobs,read_json,recover,verify_lock,check_registration,atomic_json
from ridge_batch import ridge_r2_batch
from execution_context import validate_header

def load_data(run_dir,root=ROOT,synthetic=False):
    root=Path(root);run_dir=Path(run_dir);lock_sha=verify_lock(root);cfg=config(root);scored=[];bad=set();modes=set()
    for model in cfg['models']:
        header=read_json(run_dir/f'{model}_run_header.json')
        expected_kind='SYNTHETIC_TEST_ONLY' if synthetic else 'REAL_PILOT_REPORTS'
        if header.get('data_kind')!=expected_kind:raise RuntimeError('Checkpoint data kind does not match the requested analysis mode.')
        if header['package_lock_sha256']!=lock_sha:raise RuntimeError('Run header uses a different package lock.')
        if header.get('model')!=model:raise RuntimeError('Run header model mismatch.')
        if not synthetic:
            modes.add(validate_header(run_dir,model,root))
            if len(modes)>1:raise RuntimeError('Exploratory and registered runs cannot be pooled.')
        js=jobs(root,model);expected={j['job_id']:j for j in js}
        saved=recover(run_dir/f'{model}_checkpoint.jsonl',expected,lock_sha,repair_tail=False)
        if len(saved)!=len(js):raise RuntimeError(f'{model}: scoring is incomplete; no interim analysis is allowed.')
        for jid,value in saved.items():
            spec=expected[jid]
            if value['status']!='ok':bad.add(spec['item_id']);continue
            r={k:spec[k] for k in ('model_key','item_id','condition','cell','ownership','hedging','visibility','rotation')}
            for lab,role in spec['label_to_role'].items():r['logp_'+role]=float(value['logp'][lab])
            scored.append(r)
    raw=pd.DataFrame(scored);raw=raw[~raw.item_id.isin(bad)].copy()
    keys=['model_key','item_id','condition','cell','ownership','hedging','visibility']
    counts=raw.groupby(keys).rotation.agg(['count','nunique'])
    if not counts['count'].eq(4).all() or not counts['nunique'].eq(4).all():raise RuntimeError('Every complete cell must contain the four unique rotations.')
    agg=raw.groupby(keys,as_index=False)[['logp_'+r for r in ROLES]].mean()
    logits=agg[['logp_'+r for r in ROLES]].to_numpy();p=np.exp(logits-logits.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
    for j,role in enumerate(ROLES):agg['q_'+role]=p[:,j]
    agg['modal_role']=np.array(ROLES)[np.argmax(p,axis=1)]
    for role in ('document','both','uncertain'):agg['alr_'+role]=agg['logp_'+role]-agg.logp_prior
    agg['VSA']=agg.alr_document
    selected=pd.read_csv(root/'inputs/items.csv');selected=selected[~selected.item_id.isin(bad)].reset_index(drop=True)
    if len(selected)<20:raise RuntimeError('Insufficient complete items even for descriptive analysis.')
    assert len(agg)==len(selected)*3*8*3
    return agg,selected,sorted(bad)

def bootstrap_weights(selected,folds,reps,seed):
    keys=selected.apply(lambda r:triple_key(r),axis=1)
    lookup={key:j for j,key in enumerate(sorted(set(keys)))};cluster=keys.map(lookup).to_numpy()
    strata=[np.unique(cluster[g.index]) for _,g in selected.groupby('relation',sort=True)]
    rng=np.random.default_rng(seed);result=[];rejected=0
    for b in range(reps):
        for attempt in range(1000):
            sampled=np.concatenate([rng.choice(v,len(v),replace=True) for v in strata])
            w=np.bincount(sampled,minlength=len(lookup))[cluster]
            # Predeclared technical redraw rule ensures all fixed CV folds exist.
            if all(w[folds==f].sum()>0 for f in range(5)):break
            rejected+=1
        else:raise RuntimeError('Could not obtain an estimable fixed-fold bootstrap draw.')
        result.append(w)
    return np.vstack((np.ones(len(selected)),np.array(result))),rejected

def regression_units(agg,selected,root=ROOT):
    target=pd.read_csv(Path(root)/'inputs/answer_targets.csv');target=target[target.item_id.isin(selected.item_id)].copy()
    foldmap=read_json(Path(root)/'protocol/folds.json');index={v:j for j,v in enumerate(selected.item_id)}
    units={}
    for model in config(root)['models']:
        base=target[target.model_key.eq(model)].copy();base['analysis_relation']=base.relation
        base['fold_id']=base.item_id.map(foldmap);base['item_index']=base.item_id.map(index)
        for hedge in ('certain','hedged'):
            for visibility in ('shown','hidden'):
                a=agg[(agg.model_key==model)&(agg.hedging==hedge)&(agg.visibility==visibility)]
                p=a.pivot(index=['item_id','condition'],columns='ownership',values=['alr_document','alr_both','alr_uncertain','VSA'])
                p.columns=['_'.join(v) for v in p.columns]
                u=base.merge(p.reset_index(),on=['item_id','condition'],validate='one_to_one').sort_values(['item_id','condition']).reset_index(drop=True)
                for c in ['CIE','baseline_margin']+[v for v in u if v.startswith(('alr_','VSA_'))]:
                    if not np.isfinite(u[c]).all():raise RuntimeError(f'Missing/nonfinite regression input: {c}')
                units[(model,hedge,visibility)]=u
    return units

def score_linear(u,w,representation='four_role',answer=False,baseline=False):
    cats=['strength','analysis_relation']+(['frozen_answer_role'] if answer else [])
    peer=['alr_'+r+'_OTHER' for r in ('document','both','uncertain')] if representation=='four_role' else ['VSA_OTHER']
    own=['alr_'+r+'_OWN' for r in ('document','both','uncertain')] if representation=='four_role' else ['VSA_OWN']
    sets=[[],peer,peer+own] if baseline else [peer,peer+own]
    return np.column_stack([ridge_r2_batch(u,['baseline_margin']+cols,cats,w) for cols in sets])

def nonlinear_metrics(u,w,answer=False):
    from sklearn.ensemble import GradientBoostingRegressor
    cats=['strength','analysis_relation']+(['frozen_answer_role'] if answer else [])
    num=['baseline_margin']+['alr_'+r+'_'+frame for frame in ('OTHER','OWN') for r in ('document','both','uncertain')]
    x=np.column_stack((u[num].to_numpy(),pd.get_dummies(u[cats].astype(str),dtype=float).to_numpy()))
    y=u.CIE.to_numpy();fold=u.fold_id.to_numpy();variance=np.var(y);output=[]
    # All bootstrap refits use duplicated rows, including tree split-count rules.
    for wi in w:
        scores=[]
        dup=np.repeat(np.arange(len(u)),wi.astype(int))
        for k in (1,4,7):
            cols=list(range(k))+list(range(7,x.shape[1]));pred=np.empty(len(u))
            for f in range(5):
                train=dup[fold[dup]!=f];test=fold==f
                fit=GradientBoostingRegressor(n_estimators=100,learning_rate=.05,max_depth=2,loss='squared_error',subsample=1,random_state=20261016)
                fit.fit(x[train][:,cols],y[train]);pred[test]=fit.predict(x[test][:,cols])
            scores.append(1-np.average((y-pred)**2,weights=wi)/variance)
        output.append(scores)
    return np.array(output)

def answer_association(agg,root=ROOT):
    """Uncorrected Cramer's V, descriptive within strength and relation only."""
    targets=pd.read_csv(Path(root)/'inputs/answer_targets.csv')
    data=agg.merge(targets[['model_key','item_id','condition','relation','frozen_answer_role']],on=['model_key','item_id','condition'],validate='many_to_one')
    output=[]
    for key,g in data.groupby(['model_key','condition','relation','cell'],sort=True):
        table=pd.crosstab(g.frozen_answer_role,g.modal_role).to_numpy(dtype=float)
        denominator=min(table.shape)-1
        if denominator>0:
            expected=np.outer(table.sum(axis=1),table.sum(axis=0))/table.sum()
            value=float(np.sqrt(np.sum((table-expected)**2/expected)/(table.sum()*denominator)))
            status='descriptive_only'
        else:value=None;status='undefined_constant_category'
        output.append(dict(zip(['model_key','condition','relation','cell'],key),n=len(g),cramers_v=value,status=status))
    return pd.DataFrame(output)

def analyze(agg,selected,root=ROOT,reps=10000,secondary=False,secondary_reps=2000):
    cfg=config(root);foldmap=read_json(Path(root)/'protocol/folds.json');folds=selected.item_id.map(foldmap).to_numpy()
    weights,rejected=bootstrap_weights(selected,folds,reps,cfg['bootstrap_seed']);units=regression_units(agg,selected,root)
    effects={};per_model=[];labels={'ownership':('OWN','OTHER'),'hedging':('hedged','certain'),'visibility':('shown','hidden')}
    for label,(positive,negative) in labels.items():
        p=agg.groupby(['item_id',label]).q_both.mean().unstack().reindex(selected.item_id)
        itemcontrast=(p[positive]-p[negative]).to_numpy()
        effects[label]=(weights@itemcontrast)/weights.sum(axis=1)
        for model in cfg['models']:
            pm=agg[agg.model_key.eq(model)].groupby(['item_id',label]).q_both.mean().unstack().reindex(selected.item_id)
            values=(weights@(pm[positive]-pm[negative]).to_numpy())/weights.sum(axis=1)
            per_model.append(dict(model=model,contrast=label,estimate=values[0],ci_low=float(np.quantile(values[1:],.025)),ci_high=float(np.quantile(values[1:],.975))))
    predicted={};points=[]
    specs=[('four_role',False)]
    if secondary:specs += [('four_role',True),('VSA',False),('VSA',True)]
    for representation,answer in specs:
        block={}
        for (model,hedge,visibility),u in units.items():
            metrics=[]
            for start in range(0,len(weights),128):
                w=weights[start:start+128,u.item_index.to_numpy()]
                metrics.append(score_linear(u,w,representation,answer,baseline=True))
            met=np.vstack(metrics);block[(model,hedge,visibility)]=met[:,2]-met[:,1]
            points.append(dict(learner='Ridge',representation=representation,answer_included=answer,model=model,hedging=hedge,visibility=visibility,X1_R2=met[0,0],X2_R2=met[0,1],X3_R2=met[0,2],theta_OTHER=met[0,1]-met[0,0],theta_OWN=met[0,2]-met[0,1],ci_low=float(np.quantile((met[:,2]-met[:,1])[1:],.025)),ci_high=float(np.quantile((met[:,2]-met[:,1])[1:],.975))))
        contrast=np.mean([block[(m,h,'shown')]-block[(m,h,'hidden')] for m in cfg['models'] for h in ('certain','hedged')],axis=0)
        predicted[(representation,answer)]=contrast
    effects['predictive_visibility']=predicted[('four_role',False)]
    summary=[]
    for name,values in effects.items():
        lo,hi=np.quantile(values[1:],[.00625,.99375]);l95,u95=np.quantile(values[1:],[.025,.975])
        summary.append(dict(contrast=name,estimate=values[0],family_ci_low=lo,family_ci_high=hi,nominal95_low=l95,nominal95_high=u95,excludes_zero=bool(lo>0 or hi<0),replicates=reps))
    secondary_contrasts=[]
    for (representation,answer),v in predicted.items():
        secondary_contrasts.append(dict(learner='Ridge',representation=representation,answer_included=answer,estimate=v[0],ci_low=float(np.quantile(v[1:],.025)),ci_high=float(np.quantile(v[1:],.975)),replicates=reps))
    for name,v in [('hedging_minus_ownership',effects['hedging']-effects['ownership']),('absolute_hedging_minus_absolute_ownership',np.abs(effects['hedging'])-np.abs(effects['ownership']))]:
        secondary_contrasts.append(dict(learner='category_contrast',representation=name,answer_included=False,estimate=v[0],ci_low=float(np.quantile(v[1:],.025)),ci_high=float(np.quantile(v[1:],.975)),replicates=reps))
    if secondary:
        for answer in (False,True):
            block={}
            for (model,h,v),u in units.items():
                w=weights[:secondary_reps+1,u.item_index.to_numpy()]
                met=nonlinear_metrics(u,w,answer);block[(model,h,v)]=met[:,2]-met[:,1]
                points.append(dict(learner='GradientBoosting',representation='four_role',answer_included=answer,model=model,hedging=h,visibility=v,X1_R2=met[0,0],X2_R2=met[0,1],X3_R2=met[0,2],theta_OTHER=met[0,1]-met[0,0],theta_OWN=met[0,2]-met[0,1],ci_low=float(np.quantile((met[:,2]-met[:,1])[1:],.025)),ci_high=float(np.quantile((met[:,2]-met[:,1])[1:],.975))))
            contrast=np.mean([block[(m,h,'shown')]-block[(m,h,'hidden')] for m in cfg['models'] for h in ('certain','hedged')],axis=0)
            secondary_contrasts.append(dict(learner='GradientBoosting',representation='four_role',answer_included=answer,estimate=contrast[0],ci_low=float(np.quantile(contrast[1:],.025)),ci_high=float(np.quantile(contrast[1:],.975)),replicates=secondary_reps))
    distribution=agg.groupby(['model_key','condition','cell','ownership','hedging','visibility','modal_role']).size().rename('count').reset_index()
    probabilities=agg.groupby(['model_key','condition','cell','ownership','hedging','visibility'],as_index=False)[['q_'+r for r in ROLES]].mean()
    interactions=[]
    for a,b,c in [('ownership','hedging','visibility'),('ownership','visibility','hedging'),('hedging','visibility','ownership')]:
        pa,na=labels[a];pb,nb=labels[b]
        p=agg.groupby(['item_id',a,b]).q_both.mean().unstack([a,b]).reindex(selected.item_id)
        value=p[(pa,pb)]-p[(pa,nb)]-p[(na,pb)]+p[(na,nb)]
        v=(weights@value.to_numpy())/weights.sum(axis=1)
        interactions.append(dict(factor_a=a,factor_b=b,estimate=v[0],ci_low=float(np.quantile(v[1:],.025)),ci_high=float(np.quantile(v[1:],.975))))
    p=agg.groupby(['item_id','ownership','hedging','visibility']).q_both.mean().unstack(['ownership','hedging','visibility']).reindex(selected.item_id)
    value=sum((1 if o=='OWN' else -1)*(1 if h=='hedged' else -1)*(1 if vis=='shown' else -1)*p[(o,h,vis)] for o in ('OWN','OTHER') for h in ('hedged','certain') for vis in ('shown','hidden'))
    v=(weights@value.to_numpy())/weights.sum(axis=1)
    interactions.append(dict(factor_a='ownership',factor_b='hedging_by_visibility',estimate=v[0],ci_low=float(np.quantile(v[1:],.025)),ci_high=float(np.quantile(v[1:],.975))))
    return dict(primary=pd.DataFrame(summary),primary_draws=pd.DataFrame({k:v[1:] for k,v in effects.items()}),per_model_category_contrasts=pd.DataFrame(per_model),predictive_fits=pd.DataFrame(points),secondary_contrasts=pd.DataFrame(secondary_contrasts),category_counts=distribution,category_probabilities=probabilities,answer_association=answer_association(agg,root),interactions=pd.DataFrame(interactions),bootstrap_redraws=rejected)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--secondary',action='store_true');p.add_argument('--allow-synthetic-test-data',action='store_true');p.add_argument('--smoke-reps',type=int)
    args=p.parse_args();cfg=config()
    analysis_versions={k:importlib.metadata.version(k) for k in cfg['analysis_versions']}
    for k,v in cfg['analysis_versions'].items():
        if analysis_versions[k]!=v:raise RuntimeError(f'Pinned analysis environment mismatch: {k} {analysis_versions[k]} != {v}')
    if args.smoke_reps is not None and not args.allow_synthetic_test_data:raise RuntimeError('Reduced bootstrap counts are restricted to synthetic tests.')
    reps=args.smoke_reps or cfg['primary_bootstrap_reps'];sreps=min(reps,2) if args.smoke_reps else cfg['secondary_bootstrap_reps']
    agg,items,bad=load_data(args.run_dir,synthetic=args.allow_synthetic_test_data)
    result=analyze(agg,items,reps=reps,secondary=args.secondary,secondary_reps=sreps)
    args.out.mkdir(parents=True,exist_ok=True)
    for key,value in result.items():
        if isinstance(value,pd.DataFrame):value.to_csv(args.out/f'{key}.csv',index=False)
    agg.to_csv(args.out/'aggregated_reports.csv',index=False)
    kind='SYNTHETIC_TEST_ONLY' if args.allow_synthetic_test_data else 'REAL_PILOT_REPORTS'
    mode='synthetic_test_only' if args.allow_synthetic_test_data else validate_header(args.run_dir,cfg['models'][0])
    status=dict(data_kind=kind,study_mode=mode,interpretation='Existing-item exploratory pilot; internally frozen, not preregistered.' if mode=='exploratory' else ('Synthetic software validation only.' if mode=='synthetic_test_only' else 'Author-attested registered pilot on previously used items; registry not verified remotely.'),external_registration_author_attested=mode=='registered',independent_timestamp_author_attested=mode=='registered',complete_items=len(items),excluded_items=bad,attrition_fraction=len(bad)/100,evaluation_incomplete=len(bad)>5,secondary_run=args.secondary,primary_replicates=reps,nonlinear_replicates=sreps if args.secondary else 0,bootstrap_redraws=result['bootstrap_redraws'],registration_verified_remotely=False)
    status['analysis_versions']=analysis_versions
    atomic_json(args.out/'analysis_status.json',status)
    print(json.dumps(status,indent=2));print(result['primary'].to_string(index=False))

if __name__=='__main__':main()
