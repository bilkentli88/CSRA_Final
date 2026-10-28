"""Old100-only training and observer selection; no fitting to Fresh40 outcomes."""
import os,sys,importlib.metadata,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
from core import *
sys.path.insert(0,str(LEGACY/'code'))
try:
    import predictive as legacy
finally:
    sys.path.remove(str(LEGACY/'code'))
import numpy as np,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
ALGORITHMS=('gradient_boosting','ridge');TIERS=('visible','probe','reference_ceiling')
def environment():return {k:importlib.metadata.version(k) for k in ('numpy','pandas','scikit-learn','scipy','joblib')}
def check_environment():
    expected={'numpy':'2.3.5','pandas':'2.2.3','scikit-learn':'1.8.0','scipy':'1.17.0','joblib':'1.5.3'}
    if environment()!=expected:raise RuntimeError('Analysis environment differs from package pins.')
def probe_features(c):
    # Symmetric in the two alternatives other than the document entity. No P identity is exposed.
    d=c['no_document_support'];actual=c['answer_role'];doc='C';values=np.array(list(d.values()),float);p=np.exp(values-values.max());p/=p.sum()
    return [d[actual]-np.mean([v for k,v in d.items() if k!=actual]),d[doc]-np.mean([v for k,v in d.items() if k!=doc]),-float(np.sum(p*np.log(p.clip(1e-300))))]
def report_features(c,reporter):return np.mean([c['reports'][reporter][cue]['alr'] for cue in cfg()['cues']],axis=0).tolist()
def features(c,text_prediction,tier,reporter):
    values=[float(c['relation']==r) for r in plan()['relations']]+[float(c['strength']==s) for s in cfg()['strengths']]+list(text_prediction)
    if tier=='probe':values+=probe_features(c)
    elif tier=='reference_ceiling':values+=[float(c['answer_role']==r) for r in ANSWER_ROLES]+[float(c['baseline_margin'])]
    elif tier!='visible':raise ValueError(tier)
    if reporter is not None:values+=report_features(c,reporter)
    return values
def fit_decoders(cases,indices,textpred,algorithm):
    models={}
    for target in cfg()['target_models']:
        rows=[j for j,i in enumerate(indices) if cases[i]['target']==target];cc=[cases[indices[j]] for j in rows];y=np.array([c['Y'] for c in cc])
        for tier,reporter in itertools.product(TIERS,[None]+cfg()['reporter_models']):
            X=np.array([features(c,textpred[j],tier,reporter) for c,j in zip(cc,rows)],float)
            scale=StandardScaler();X=scale.fit_transform(X)
            if algorithm=='ridge':model=Ridge(alpha=1,solver='cholesky')
            else:model=GradientBoostingRegressor(n_estimators=100,learning_rate=.05,max_depth=2,loss='squared_error',random_state=cfg()['seed'])
            model.fit(X,y);models[(target,tier,reporter)]=(scale,model)
    return models
def predict_decoders(models,cases,indices,textpred):
    predictions={}
    for (target,tier,reporter),(scale,model) in models.items():
        rows=[j for j,i in enumerate(indices) if cases[i]['target']==target]
        X=np.array([features(cases[indices[j]],textpred[j],tier,reporter) for j in rows])
        pred=model.predict(scale.transform(X))
        for j,v in zip(rows,pred):predictions[(int(indices[j]),tier,reporter)]=float(v)
    return predictions
def text_predictions(fits,cases,E,indices):return np.column_stack([legacy.predict_text(fits[k],cases,E,indices) for k in ('tfidf','embedding')])
def check_probe_invariance(cases):
    for c in cases:
        d=dict(c);d['no_document_support']={({'P':'I','I':'P'}.get(k,k)):v for k,v in c['no_document_support'].items()};d['answer_role']={'P':'I','I':'P'}.get(c['answer_role'],c['answer_role'])
        if not np.allclose(probe_features(c),probe_features(d),atol=1e-12,rtol=0):raise RuntimeError('Reference label leaked into public probe.')
def train(run):
    run=Path(run);require_before_outcomes(run);check_environment();start=time.perf_counter();cases=old_cases();legacy.check_cases(cases);check_probe_invariance(cases);E=load_embeddings(run,cases,'old');indices=np.arange(len(cases));oof=[]
    # Selection uses group-held-out error on old100, never fresh observations.
    for fold in range(cfg()['outer_folds']):
        tr=np.array([i for i,c in enumerate(cases) if c['fold']!=fold]);va=np.array([i for i,c in enumerate(cases) if c['fold']==fold])
        if {cases[i]['group'] for i in tr}&{cases[i]['group'] for i in va}:raise RuntimeError('Selection group leakage.')
        print(f'Old100 selection fold {fold+1}/5: nested text forecasts',flush=True)
        tp,fits=legacy.text_stack(cases,E,tr);vp=text_predictions(fits,cases,E,va)
        # Primary GB selection is frozen once and reused by the Ridge sensitivity.
        models=fit_decoders(cases,tr,tp,'gradient_boosting');pred=predict_decoders(models,cases,va,vp)
        for i in va:
            for reporter in cfg()['reporter_models']:
                c=cases[i];oof.append(dict(item_id=c['item_id'],group=c['group'],target=c['target'],strength=c['strength'],fold=fold,reporter=reporter,squared_error=(c['Y']-pred[(int(i),'visible',reporter)])**2))
    selected={};losses={}
    for target in cfg()['target_models']:
        losses[target]={r:float(np.mean([e['squared_error'] for e in oof if e['target']==target and e['reporter']==r])) for r in cfg()['reporter_models'] if r!=target}
        selected[target]=min(losses[target],key=lambda r:(losses[target][r],r))
    atomic_json(run/'old100_oof_errors.json',oof);atomic_json(run/'observer_selection.json',dict(selection='Lowest old100 group-heldout visible-tier GB MSE; source excluded; lexical tie-break',selected=selected,old100_MSE=losses,used_fresh_outcomes=False))
    print('Training-selected observers:',selected,flush=True);print('Final old100 nested text forecasts and decoder fits',flush=True)
    tp,text_models=legacy.text_stack(cases,E,indices)
    fitted={a:fit_decoders(cases,indices,tp,a) for a in ALGORITHMS};variances={t:float(np.var([c['Y'] for c in cases if c['target']==t])) for t in cfg()['target_models']}
    if min(variances.values())<=1e-12:raise RuntimeError('Degenerate old100 outcome variance.')
    model=dict(text_models=text_models,decoders=fitted,selected=selected,old100_Y_variance=variances,training_groups=sorted({c['group'] for c in cases}),training_target_models=cfg()['target_models'])
    joblib.dump(model,run/'frozen_models.joblib',compress=3)
    atomic_json(run/'training_manifest.json',dict(environment=environment(),old100_cases=900,old100_groups=len(model['training_groups']),old100_inputs_sha256=file_sha(ROOT/'development/analysis_inputs.json'),candidate_aware_text=True,old100_Y_variance=variances,selected_observers=selected,source_report_features='Mean of three ALRs over both ownership cues, four label rotations per cue',feature_tiers={'visible':9,'probe':12,'reference_ceiling':13},report_added_features=3,algorithms=list(ALGORITHMS),primary='gradient_boosting',text_forecasters='Nested grouped TFIDF Ridge and candidate-aware MPNet Ridge, pooled across known targets; target-specific decoders',fresh_outcomes_used=0,training_seconds=time.perf_counter()-start))
