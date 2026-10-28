"""Exploratory revision checks using released scores; no model inference.

Re-estimates frame scales in paired item-bootstrap draws for correlation and
conditional alignment. Tests three orthonormal four-role log-score contrasts
using the existing Ridge specification. All changes postdate peer review.
"""
from pathlib import Path
import argparse, json, os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import audit_csra as audit


def interval(values, **labels):
    lo, hi = np.quantile(values[1:], [0.025, 0.975])
    return {**labels, "estimate":float(values[0]), "ci_low":float(lo), "ci_high":float(hi)}


def alignment_stat(frame, row_weights, dummy):
    total = row_weights.sum()
    raw = frame[["CIE", "SELF", "PEER", "baseline_margin"]].to_numpy(float)
    mu = np.sum(raw * row_weights[:, None], axis=0) / total
    sd = np.sqrt(np.sum((raw-mu)**2 * row_weights[:, None], axis=0)/total)
    z = (raw-mu)/sd
    difference = z[:,1]-z[:,2]
    corr_difference = np.sum(row_weights*z[:,0]*difference)/total
    x = np.column_stack((np.ones(len(frame)), z[:,0], z[:,3], dummy))
    root_weight = np.sqrt(row_weights)
    coef = np.linalg.lstsq(x*root_weight[:,None], difference*root_weight, rcond=None)[0]
    return corr_difference, coef[1]


def sklearn_loss(frame, numeric, cat, variance=None):
    design=ColumnTransformer([("numeric", StandardScaler(), numeric), ("categorical", OneHotEncoder(handle_unknown="ignore",sparse_output=False), cat)])
    prediction=np.zeros(len(frame))
    for fold in range(5):
        train=frame.fold_id.ne(fold).to_numpy(); test=~train
        model=Pipeline([("design",design),("ridge",Ridge(alpha=1))])
        model.fit(frame.loc[train],frame.loc[train,"CIE"])
        prediction[test]=model.predict(frame.loc[test])
    return np.mean((frame.CIE-prediction)**2)/(np.var(frame.CIE,ddof=0) if variance is None else variance)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    selected=pd.read_csv(args.repo/'data/final_common_items.csv')
    unit=pd.read_csv(args.audit/'results/reconstructed_primary_units.csv')
    ids,bw=audit.bootstrap_weights(selected,2000)
    weights=np.vstack((np.ones((1,len(ids))),bw));position={iid:k for k,iid in enumerate(ids)}
    alignment=[];draws={};all_corr=[];all_cond=[];full=[];validation={}
    for model in audit.MODELS:
        frame=unit[unit.model_key.eq(model)].copy().reset_index(drop=True)
        dummy=pd.get_dummies(frame[["frozen_answer_role","strength","analysis_relation"]].astype(str),drop_first=True,dtype=float).to_numpy()
        ix=frame.item_id.map(position).to_numpy(int)
        stats=np.array([alignment_stat(frame,w[ix],dummy) for w in weights])
        all_corr.append(stats[:,0]);all_cond.append(stats[:,1])
        alignment += [interval(stats[:,k], model_key=model, estimand=name) for k,name in enumerate(["correlation_difference","conditional_frame_contrast"])]
        draws[model+'_correlation_difference']=stats[1:,0]
        draws[model+'_conditional_frame_contrast']=stats[1:,1]
        for table,key,actual in [("alignment_scale_comparison.csv","Delta_within_frame",stats[0,0]),("conditional_alignment.csv","conditional_Delta",stats[0,1])]:
            expected=pd.read_csv(args.audit/'results'/table).set_index('model_key').loc[model,key]
            validation[model+'_'+key+'_point_error']=float(abs(actual-expected))
        report=pd.read_csv(args.repo/f'data/supporting/{model}_main_report_results.csv')
        report=report[report.condition.isin(audit.CONDITIONS)].copy()
        raw=report[[f'logp_report_{r}' for r in audit.ROLES]].to_numpy(float)
        helmert=np.array([[1,-1,0,0],[1,1,-2,0],[1,1,1,-3]],float)/np.sqrt([2,6,12])[:,None]
        assert np.allclose(helmert@helmert.T,np.eye(3))
        contrasts=raw@helmert.T
        assert np.allclose(contrasts[:,0]*np.sqrt(2),report.VSA)
        for j in range(3): report[f'b{j+1}']=contrasts[:,j]
        for observer in ['PEER','SELF']:
            cols=[f'b{j+1}' for j in range(3)]
            features=report[report.frame.eq(observer)][['item_id','condition']+cols].rename(columns={c:observer+'_'+c for c in cols})
            frame=frame.merge(features,on=['item_id','condition'],validate='one_to_one')
        full.append(frame)
        print('Completed restandardized alignment:',model,flush=True)
    for name,values in [('correlation_difference',all_corr),('conditional_frame_contrast',all_cond)]:
        matrix=np.array(values)
        for label,stat in [('equal_model_mean',matrix.mean(axis=0)),('H',matrix[0]-(matrix[1]+matrix[2])/2)]:
            alignment.append(interval(stat,model_key=label,estimand=name))
    pd.DataFrame(alignment).to_csv(args.out/'alignment_restandardized_bootstrap.csv',index=False)
    pd.DataFrame(draws).to_csv(args.out/'alignment_restandardized_draws.csv',index=False)
    full=pd.concat(full,ignore_index=True)
    role_results=[];role_draws={}
    for case in ['with_recorded_answer','without_recorded_answer']:
        effects=[]
        for model in audit.MODELS:
            frame=full[full.model_key.eq(model)].copy().reset_index(drop=True)
            cat=['strength','analysis_relation']
            if case=='with_recorded_answer':cat=['frozen_answer_role']+cat
            numeric_sets=[['baseline_margin'],['baseline_margin']+['PEER_b'+str(j) for j in range(1,4)],['baseline_margin']+['PEER_b'+str(j) for j in range(1,4)]+['SELF_b'+str(j) for j in range(1,4)]]
            losses=[audit.ridge_loss(frame,cols,cat,weights,ids) for cols in numeric_sets]
            for k,cols in enumerate(numeric_sets):
                validation[f'{case}_{model}_full_role_X{k+1}_sklearn_loss_error']=float(abs(losses[k][0]-sklearn_loss(frame,cols,cat)))
            checked_draws={1,1000,int(np.argmax(abs(losses[1]-losses[2])))}
            item_ix=frame.item_id.map(position).to_numpy(int)
            for draw in checked_draws:
                repeated=frame.iloc[np.repeat(np.arange(len(frame)),weights[draw,item_ix].astype(int))].reset_index(drop=True)
                for k in [1,2]:
                    error=abs(losses[k][draw]-sklearn_loss(repeated,numeric_sets[k],cat,variance=np.var(frame.CIE,ddof=0)))
                    validation[f'{case}_{model}_bootstrap_{draw}_X{k+1}_error']=float(error)
            peer=losses[0]-losses[1];effect=losses[1]-losses[2];effects.append(effect)
            role_results.append({**interval(effect,case=case,model_key=model), 'R2_X1':float(1-losses[0][0]),'R2_X2':float(1-losses[1][0]),'R2_X3':float(1-losses[2][0]),'theta_PEER':float(peer[0])})
            role_draws[case+'_'+model]=effect[1:]
            print('Completed four-role prediction:',case,model,flush=True)
        mean=np.mean(effects,axis=0)
        role_results.append(interval(mean,case=case,model_key='equal_model_mean'))
    pd.DataFrame(role_results).to_csv(args.out/'full_role_prediction.csv',index=False)
    pd.DataFrame(role_draws).to_csv(args.out/'full_role_prediction_draws.csv',index=False)

    target_rows=[];desc=[]
    for model in audit.MODELS:
        answer=pd.read_csv(args.repo/f'data/supporting/{model}_answer_condition_summary.csv')
        frame=unit[unit.model_key.eq(model)].copy().reset_index(drop=True)
        lookup=answer.set_index(['item_id','condition'])
        nd=[];pc=[]
        for row in frame.itertuples():
            base=lookup.loc[(row.item_id,'NO_DOC')]
            matching=lookup.loc[(row.item_id,'P_'+row.strength)]
            nd.append(row.mean_L_C_minus_P-base.mean_L_C_minus_P)
            pc.append((row.mean_prob_C-row.mean_prob_P)-(matching.mean_prob_C-matching.mean_prob_P))
        cat=['frozen_answer_role','strength','analysis_relation']
        for name,target in [('C_vs_no_document',np.array(nd)),('probability_contrast',np.array(pc))]:
            copy=frame.copy();copy['CIE']=target
            losses=[audit.ridge_loss(copy,['baseline_margin']+extra,cat,weights[:1],ids)[0] for extra in [[],['PEER'],['PEER','SELF']]]
            target_rows.append({'model_key':model,'target':name,'correlation_with_original':float(np.corrcoef(frame.CIE,target)[0,1]),'theta_SELF':float(losses[1]-losses[2]),'theta_PEER':float(losses[0]-losses[1])})
        for strength,g in frame.groupby('strength'):
            desc.append({'model_key':model,'strength':strength,'n':len(g),'CIE_mean':g.CIE.mean(),'CIE_sd':g.CIE.std(ddof=0),'CIE_median':g.CIE.median(),'CIE_q25':g.CIE.quantile(.25),'CIE_q75':g.CIE.quantile(.75),'answer_C_share':g.frozen_answer_role.eq('C').mean(),'answer_I_share':g.frozen_answer_role.eq('I').mean()})
    pd.DataFrame(target_rows).to_csv(args.out/'alternative_targets.csv',index=False)
    pd.DataFrame(desc).to_csv(args.out/'descriptive_statistics.csv',index=False)
    assert max(validation.values())<1e-8,validation
    (args.out/'validation.json').write_text(json.dumps({'checks':validation,'max_error':max(validation.values()),'bootstrap_draws':2000,'analysis_status':'exploratory_after_peer_review','normalization':'Re-estimated inside each alignment bootstrap draw; original target-variance constants retained for predictive bootstrap.','full_role_definition':'Three orthonormal contrasts: (document-prior)/sqrt(2), (document+prior-2*both)/sqrt(6), (document+prior+both-3*uncertain)/sqrt(12). These preserve all relative log-score information, and the first is the original VSA up to scale.'},indent=2)+'\n')

if __name__=='__main__':main()
