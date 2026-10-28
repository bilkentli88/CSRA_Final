"""Nested group-held-out text forecasters and an equal-cost common decoder."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from common import *

def check_cases(cases):
    expected={(x['item_id'],t,s) for x in items() for t in cfg()['target_models'] for s in cfg()['strengths']}
    actual={(c['item_id'],c['target'],c['strength']) for c in cases}
    if actual!=expected or len(actual)!=len(cases):raise RuntimeError('Development cases must exactly cover the historical items, known targets and strengths.')
    g=grouping(items());fold=folds(items())
    for c in cases:
        if c['group']!=g[c['item_id']] or c['fold']!=fold[c['item_id']]:raise RuntimeError('Group/fold identity differs.')
        if set(c['reports'])!=set(cfg()['reporter_models']):raise RuntimeError('Missing reporter.')
        for reporter in c['reports'].values():
            if set(reporter)!=set(cfg()['cues']):raise RuntimeError('Missing ownership cue.')
        if not math.isfinite(c['Y']):raise RuntimeError('Nonfinite development target.')

def group_splits(cases,indices,n=3):
    groups=np.array([cases[i]['group'] for i in indices]);n=min(n,len(set(groups)))
    if n<2:raise RuntimeError('At least two independent groups are required in every training split.')
    for a,b in GroupKFold(n_splits=n).split(indices,groups=groups):
        tr=np.asarray(indices)[a];va=np.asarray(indices)[b]
        assert not {cases[i]['group'] for i in tr}&{cases[i]['group'] for i in va}
        yield tr,va

def fit_text(cases,E,indices,kind,alpha):
    if kind=='tfidf':
        transform=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=10000,sublinear_tf=True);X=transform.fit_transform([cases[i]['text'] for i in indices])
    else:transform=StandardScaler();X=transform.fit_transform(E[indices])
    model=Ridge(alpha=alpha,solver='lsqr',tol=1e-8,max_iter=5000);model.fit(X,[cases[i]['Y'] for i in indices]);return dict(kind=kind,transform=transform,model=model,alpha=alpha)
def predict_text(fit,cases,E,indices):
    X=fit['transform'].transform([cases[i]['text'] for i in indices]) if fit['kind']=='tfidf' else fit['transform'].transform(E[indices]);return fit['model'].predict(X)
def choose_alpha(cases,E,indices,kind):
    losses={a:0.0 for a in cfg()['text_alpha_grid']};count=0
    for train,valid in group_splits(cases,indices,cfg()['inner_folds']):
        for alpha in losses:
            fit=fit_text(cases,E,train,kind,alpha);p=predict_text(fit,cases,E,valid);losses[alpha]+=float(np.sum((np.array([cases[i]['Y'] for i in valid])-p)**2))
        count+=len(valid)
    selected=min(losses,key=lambda a:(losses[a]/count,-a));return selected

def text_stack(cases,E,indices):
    """Each decoder-training baseline prediction uses nested, group-disjoint training."""
    position={int(i):p for p,i in enumerate(indices)};oof=np.empty((len(indices),2));fits={}
    for k,kind in enumerate(('tfidf','embedding')):
        for train,valid in group_splits(cases,indices,cfg()['inner_folds']):
            alpha=choose_alpha(cases,E,train,kind);fit=fit_text(cases,E,train,kind,alpha)
            oof[[position[int(i)] for i in valid],k]=predict_text(fit,cases,E,valid)
        alpha=choose_alpha(cases,E,indices,kind);fits[kind]=fit_text(cases,E,indices,kind,alpha)
    return oof,fits

def onehot(value,values):return [float(value==x) for x in values]
def base_features(c,text_predictions):
    return [float(c['baseline_margin'])]+onehot(c['relation'],plan()['relations'])+onehot(c['strength'],cfg()['strengths'])+onehot(c['answer_role'],ANSWER_ROLES)+onehot(c['target'],cfg()['target_models'])+list(text_predictions)

def views(c,b):
    """Same cue on both reports; all anchors and independent second reporters."""
    source=c['target'];others=[r for r in cfg()['reporter_models'] if r!=source];X=[];meta=[]
    for cue in cfg()['cues']:
        for anchor in others:
            for second in others:
                if second==anchor:continue
                a=c['reports'][anchor][cue]['alr']
                for arm,reporter in (('native',source),('external',second)):
                    v=c['reports'][reporter][cue]['alr']
                    for order in (0,1):
                        X.append(b+(a+v if order==0 else v+a));meta.append((cue,anchor,second,arm,order))
    return np.asarray(X,float),meta

def fit_stack(cases,E,indices,decoder='ridge'):
    p,fits=text_stack(cases,E,indices);X=[];y=[];weights=[]
    for row,i in enumerate(indices):
        xx,_=views(cases[i],base_features(cases[i],p[row]));X.append(xx);y.extend([cases[i]['Y']]*len(xx));weights.extend([1/len(xx)]*len(xx))
    X=np.concatenate(X);y=np.asarray(y);weights=np.asarray(weights);scale=StandardScaler();X=scale.fit_transform(X,sample_weight=weights)
    if decoder=='ridge':model=Ridge(alpha=cfg()['decoder_alpha'],fit_intercept=True,solver='cholesky')
    elif decoder=='gradient_boosting':model=GradientBoostingRegressor(n_estimators=100,learning_rate=.05,max_depth=2,loss='squared_error',random_state=cfg()['seed'])
    else:raise ValueError('Unknown decoder.')
    model.fit(X,y,sample_weight=weights)
    return dict(text_models=fits,scale=scale,model=model,decoder=decoder,training_groups=sorted({cases[i]['group'] for i in indices}),training_targets=sorted({cases[i]['target'] for i in indices}),native_identity_feature=False,reporter_identity_feature=False,same_cue_both_reports=True,feature_count=X.shape[1])

def evaluate_stack(fit,cases,E,indices,check_group_holdout=True):
    if check_group_holdout and {cases[i]['group'] for i in indices}&set(fit['training_groups']):raise RuntimeError('Training/evaluation groups overlap.')
    bp=np.column_stack([predict_text(fit['text_models'][kind],cases,E,indices) for kind in ('tfidf','embedding')]);result=[]
    for pos,i in enumerate(indices):
        c=cases[i];X,meta=views(c,base_features(c,bp[pos]));pred=fit['model'].predict(fit['scale'].transform(X));grouped={}
        for key,value in zip(meta,pred):grouped.setdefault(key[:4],[]).append(value)
        errors={'native':[],'external':[]}
        for key,values in grouped.items():
            # Average slot-order predictions before computing each view's squared error.
            errors[key[3]].append((c['Y']-float(np.mean(values)))**2)
        native=float(np.mean(errors['native']));external=float(np.mean(errors['external']))
        result.append(dict(case_index=int(i),item_id=c['item_id'],relation=c['relation'],group=c['group'],fold=c['fold'],target=c['target'],strength=c['strength'],Y=c['Y'],native_squared_error=native,external_squared_error=external,paired_error_improvement=external-native,tfidf_squared_error=(c['Y']-float(bp[pos,0]))**2,embedding_squared_error=(c['Y']-float(bp[pos,1]))**2))
    return result

def group_diagnostics(cases,results):
    out=[]
    for model in cfg()['target_models']:
        r=[x for x in results if x['target']==model];variance=float(np.var([x['Y'] for x in r]))
        if variance<=1e-12:raise RuntimeError('Degenerate development outcome variance for '+model)
        for x in r:x['normalized_improvement']=x['paired_error_improvement']/variance
        out.append(dict(target=model,Y_variance=variance,n_cases=len(r),R2_native=1-np.mean([x['native_squared_error'] for x in r])/variance,R2_second_observer=1-np.mean([x['external_squared_error'] for x in r])/variance,R2_tfidf=1-np.mean([x['tfidf_squared_error'] for x in r])/variance,R2_embedding=1-np.mean([x['embedding_squared_error'] for x in r])/variance,delta_R2=np.mean([x['normalized_improvement'] for x in r])))
    return out
