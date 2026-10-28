from __future__ import annotations
import json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats
from csra_v08_common import *


def _preprocessor(features,categorical):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder,StandardScaler
    numeric=[c for c in features if c not in categorical]
    return ColumnTransformer([("numeric",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),("categorical",OneHotEncoder(handle_unknown="ignore"),categorical)])


def _fit_predict_fixed_fold(frame, features, categorical, alpha, fold_col="fold_id", target="CIE"):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    pred=np.full(len(frame),np.nan,float)
    for fold in range(5):
        train=frame[fold_col].to_numpy()!=fold; test=~train
        if not train.any() or not test.any(): raise RuntimeError(f"Empty train/test fold {fold}")
        pipe=Pipeline([("features",_preprocessor(features,categorical)),("ridge",Ridge(alpha=float(alpha)))])
        pipe.fit(frame.loc[train,features],frame.loc[train,target].to_numpy(float)); pred[test]=pipe.predict(frame.loc[test,features])
    if not np.isfinite(pred).all(): raise RuntimeError("Non-finite OOF prediction")
    return pred


def assemble_primary_units(answer_summaries: dict[str,pd.DataFrame], report_summaries: dict[str,pd.DataFrame], selected: pd.DataFrame, postreview_lock_dir, constants=None):
    relation_map=load_json(Path(postreview_lock_dir)/'analysis_relation_map.json'); folds=load_json(Path(postreview_lock_dir)/'fold_assignments.json')["primary"]
    selected_ids=set(selected["item_id"]); parts=[]
    for m in PRIMARY_MODELS:
        a=answer_summaries[m]; r=report_summaries[m]
        a=a[a["item_id"].isin(selected_ids)&a["condition"].isin(PRIMARY_CONDITIONS)][["item_id","relation","condition","strength","baseline_margin","CIE","frozen_answer_role"]].drop_duplicates(["item_id","condition"])
        p=r[r["item_id"].isin(selected_ids)&r["condition"].isin(PRIMARY_CONDITIONS)].pivot_table(index=["item_id","condition"],columns="frame",values="VSA",aggfunc="first").reset_index()
        u=a.merge(p,on=["item_id","condition"],validate="one_to_one"); u["model_key"]=m; u["analysis_relation"]=u["relation"].map(relation_map); u["fold_id"]=u["item_id"].map(folds).astype(int); parts.append(u)
    return pd.concat(parts,ignore_index=True)


def compute_cie_variance_constants(answer_summaries: dict[str,pd.DataFrame], complete_item_ids: list[str]):
    out={}
    ids=set(complete_item_ids)
    for m in PRIMARY_MODELS:
        x=answer_summaries[m]; vals=x[x["item_id"].isin(ids)&x["condition"].isin(PRIMARY_CONDITIONS)]["CIE"].dropna().to_numpy(float)
        if len(vals)!=3*len(ids): raise RuntimeError(f"Unexpected primary CIE row count for {m}")
        var=float(np.var(vals,ddof=0))
        if not np.isfinite(var) or var<=0: raise RuntimeError(f"Nonpositive CIE variance for {m}")
        out[m]=var
    return out


def _model_point(frame, model, variance, alpha=1.0, fold_col="fold_id"):
    x=frame[frame["model_key"].eq(model)].copy(); cat=["frozen_answer_role","strength","analysis_relation"]
    f1=["baseline_margin","frozen_answer_role","strength","analysis_relation"]; f2=f1+["PEER"]; f3=f2+["SELF"]
    preds={"X1":_fit_predict_fixed_fold(x,f1,cat,alpha,fold_col),"X2":_fit_predict_fixed_fold(x,f2,cat,alpha,fold_col),"X3":_fit_predict_fixed_fold(x,f3,cat,alpha,fold_col)}
    y=x["CIE"].to_numpy(float)
    metrics={}; losses={}
    for k,p in preds.items():
        err=(y-p)**2; metrics[k]={"R2":1-float(err.mean()/variance),"MAE":float(np.mean(np.abs(y-p))),"normalized_MSE":float(err.mean()/variance)}; losses[k]=err/variance
    tmp=x[["item_id","relation"]].copy(); tmp["d_self"]=losses["X2"]-losses["X3"]; tmp["d_peer"]=losses["X1"]-losses["X2"]
    item=tmp.groupby(["item_id","relation"],as_index=False).agg(D_SELF=("d_self","mean"),D_PEER=("d_peer","mean"))
    return metrics,item


def primary_point_estimate(units, variances, alpha=1.0, fold_col="fold_id"):
    model_metrics={}; items=[]
    for m in PRIMARY_MODELS:
        met,it=_model_point(units,m,variances[m],alpha,fold_col); model_metrics[m]=met; it["model_key"]=m; items.append(it)
    itemdf=pd.concat(items,ignore_index=True)
    summaries=itemdf.groupby("model_key").agg(theta_self=("D_SELF","mean"),theta_peer=("D_PEER","mean")).to_dict("index")
    theta_self=float(np.mean([summaries[m]["theta_self"] for m in PRIMARY_MODELS])); theta_peer=float(np.mean([summaries[m]["theta_peer"] for m in PRIMARY_MODELS]))
    return {"theta_SELF":theta_self,"theta_PEER":theta_peer,"model_metrics":model_metrics,"model_effects":summaries},itemdf


def _bootstrap_sample(selected: pd.DataFrame, rng):
    rows=[]
    for rel,fr in selected[["item_id","relation"]].drop_duplicates().groupby("relation",sort=True):
        ids=fr["item_id"].tolist(); draws=rng.choice(ids,size=len(ids),replace=True)
        for j,iid in enumerate(draws): rows.append({"source_item_id":iid,"boot_item_id":f"{rel}|{j:04d}|{iid}","relation":rel})
    return pd.DataFrame(rows)


def bootstrap_primary(units, selected, variances, reps=2000, seed=20260813, checkpoint_path=None):
    checkpoint_path=Path(checkpoint_path) if checkpoint_path else None
    if checkpoint_path and checkpoint_path.exists(): done=pd.read_csv(checkpoint_path); have=set(done["replicate"].astype(int))
    else: done=pd.DataFrame(); have=set()
    rows=[]; base_rng=np.random.default_rng(int(seed))
    # Pre-generate deterministic replicate seeds so restart position never changes later replicates.
    seeds=base_rng.integers(0,2**63-1,size=int(reps),dtype=np.int64)
    for b in range(int(reps)):
        if b in have: continue
        rng=np.random.default_rng(int(seeds[b])); samp=_bootstrap_sample(selected,rng)
        rep=units.merge(samp,left_on="item_id",right_on="source_item_id",how="inner",validate="many_to_many",suffixes=("","_boot")); rep["item_id_orig"]=rep["item_id"]; rep["item_id"]=rep["boot_item_id"]
        valid=True; reason=""; vals={}
        try:
            for fold in range(5):
                if not (rep["fold_id"].eq(fold)).any(): raise RuntimeError(f"fold {fold} has zero test weight")
            pt,_=primary_point_estimate(rep,variances,alpha=1.0,fold_col="fold_id"); vals={"theta_SELF":pt["theta_SELF"],"theta_PEER":pt["theta_PEER"]}
        except Exception as exc: valid=False; reason=f"{type(exc).__name__}: {exc}"; vals={"theta_SELF":np.nan,"theta_PEER":np.nan}
        row={"replicate":b,"valid":int(valid),"reason":reason,**vals}; rows.append(row)
        if checkpoint_path and (len(rows)>=10 or b==reps-1):
            add=pd.DataFrame(rows); done=add if done.empty else pd.concat([done,add],ignore_index=True); done=done.drop_duplicates("replicate",keep="last").sort_values("replicate"); atomic_write_csv(done,checkpoint_path); rows=[]
    if checkpoint_path: out=pd.read_csv(checkpoint_path)
    else: out=pd.DataFrame(rows)
    return out


def classify(theta, lo, hi, delta=0.01):
    if lo>delta: return "meaningful_positive_SELF_specific_increment"
    if hi<delta: return "no_practically_meaningful_positive_increment"
    return "indeterminate_practical_magnitude"


def bootstrap_summary(point, boot, minimum_valid=1900, delta=0.01):
    valid=boot[boot["valid"].eq(1)&boot["theta_SELF"].notna()]
    if len(valid)<minimum_valid: return {"classification":"technical_indeterminate","valid_bootstrap":int(len(valid))}
    lo,hi=np.quantile(valid["theta_SELF"],[0.025,0.975]); plo,phi=np.quantile(valid["theta_PEER"],[0.025,0.975])
    return {"theta_SELF":float(point["theta_SELF"]),"ci_low":float(lo),"ci_high":float(hi),"classification":classify(point["theta_SELF"],lo,hi,delta),"theta_PEER":float(point["theta_PEER"]),"peer_ci_low":float(plo),"peer_ci_high":float(phi),"valid_bootstrap":int(len(valid)),"delta":float(delta),"zero_comparison":{"lower_gt_zero":bool(lo>0),"upper_lt_zero":bool(hi<0)},"delta_0_02_classification":classify(point["theta_SELF"],lo,hi,0.02)}


def fixed_alpha_sensitivity(units, variances, analysis_spec, postreview_lock_dir):
    rows=[]
    for alpha in [analysis_spec["ridge_alpha"]]+analysis_spec["ridge_alpha_sensitivity"]:
        p,_=primary_point_estimate(units,variances,float(alpha),"fold_id"); rows.append({"analysis":"alpha","setting":alpha,"theta_SELF":p["theta_SELF"],"theta_PEER":p["theta_PEER"]})
    folds=load_json(Path(postreview_lock_dir)/'fold_assignments.json')
    for name in sorted(k for k in folds if k.startswith('sensitivity_')):
        work=units.copy(); work["fold_sens"]=work["item_id"].map(folds[name]).astype(int); p,_=primary_point_estimate(work,variances,1.0,"fold_sens"); rows.append({"analysis":"fold","setting":name,"theta_SELF":p["theta_SELF"],"theta_PEER":p["theta_PEER"]})
    return pd.DataFrame(rows)


def historical_nested_grid(answer_summaries,report_summaries,selected_ids):
    from sklearn.model_selection import GroupKFold,GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score,mean_absolute_error
    rows=[]; ids=set(selected_ids)
    for m in PRIMARY_MODELS:
        a=answer_summaries[m]; r=report_summaries[m]; a=a[a["item_id"].isin(ids)&a["condition"].isin(REPORT_CONDITIONS)][["item_id","relation","condition","strength","baseline_margin","CIE","frozen_answer_role","document_role"]].drop_duplicates(["item_id","condition"]); a["answer_matches_document"]=(a["frozen_answer_role"]==a["document_role"]).astype(int)
        p=r[r["item_id"].isin(ids)&r["condition"].isin(REPORT_CONDITIONS)].pivot_table(index=["item_id","condition"],columns="frame",values="VSA",aggfunc="first").reset_index(); u=a.merge(p,on=["item_id","condition"],validate="one_to_one")
        cat=["strength","relation"]; base=["baseline_margin","answer_matches_document","strength","relation"]
        for name,features in [("visible_only",base),("visible_plus_peer",base+["PEER"]),("visible_plus_peer_and_self",base+["PEER","SELF"])]:
            y=u["CIE"].to_numpy(float); groups=u["item_id"].to_numpy(); pred=np.full(len(u),np.nan); outer=GroupKFold(5)
            for tr,te in outer.split(u,y,groups):
                pipe=Pipeline([("features",_preprocessor(features,cat)),("ridge",Ridge())]); inner=GroupKFold(4); search=GridSearchCV(pipe,{"ridge__alpha":[0.01,0.1,1.0,10.0,100.0]},cv=inner); search.fit(u.iloc[tr][features],y[tr],groups=groups[tr]); pred[te]=search.predict(u.iloc[te][features])
            rows.append({"model_key":m,"feature_set":name,"R2":float(r2_score(y,pred)),"MAE":float(mean_absolute_error(y,pred)),"n":len(u)})
    return pd.DataFrame(rows)


def assemble_alignment_data(answer_summaries,report_summaries,selected,postreview_lock_dir):
    relmap=load_json(Path(postreview_lock_dir)/'analysis_relation_map.json'); ids=set(selected["item_id"]); parts=[]
    for m in PRIMARY_MODELS:
        a=answer_summaries[m]; r=report_summaries[m]; a=a[a["item_id"].isin(ids)&a["condition"].isin(PRIMARY_CONDITIONS)][["item_id","relation","condition","strength","baseline_margin","CIE","frozen_answer_role"]].drop_duplicates(["item_id","condition"]); rr=r[r["item_id"].isin(ids)&r["condition"].isin(PRIMARY_CONDITIONS)].copy(); x=rr.merge(a,on=["item_id","relation","condition","strength","frozen_answer_role"],how="left",validate="many_to_one"); x["model_key"]=m; x["SELF"]=(x["frame"]=="SELF").astype(int); x["analysis_relation"]=x["relation"].map(relmap); x["model_item_id"]=m+"|"+x["item_id"]; parts.append(x)
    data=pd.concat(parts,ignore_index=True); data["model_key"]=pd.Categorical(data["model_key"],categories=list(PRIMARY_MODELS),ordered=True)
    for v in ("VSA","CIE","baseline_margin"):
        mean=data.groupby("model_key",observed=True)[v].transform("mean"); sd=data.groupby("model_key",observed=True)[v].transform(lambda z:z.std(ddof=0)); data[v+"_z"]=(data[v]-mean)/sd
    return data


def fit_mixedlm_fallback(formula,data):
    import statsmodels.formula.api as smf
    errs={}
    for method in ("lbfgs","powell","bfgs","cg","nm"):
        try:
            with warnings.catch_warnings(): warnings.simplefilter("ignore"); fit=smf.mixedlm(formula,data,groups=data["model_item_id"]).fit(reml=False,method=method,maxiter=1000,disp=False)
            if bool(fit.converged) and np.isfinite(fit.params.to_numpy()).all() and np.isfinite(fit.bse.to_numpy()).all(): return fit,method
            errs[method]="nonconverged/nonfinite"
        except Exception as exc: errs[method]=repr(exc)
    raise RuntimeError(errs)


def _design_vector(fit,model_key,cie,self_value):
    import patsy
    row=pd.DataFrame([{"VSA_z":0.0,"CIE_z":float(cie),"SELF":int(self_value),"model_key":model_key,"frozen_answer_role":"P","baseline_margin_z":0.0,"strength":"PROVISIONAL","analysis_relation":"country_of_origin"}]); mat=patsy.build_design_matrices([fit.model.data.design_info],row,return_type="dataframe")[0]; return mat.loc[:,fit.fe_params.index].to_numpy(float)[0]


def _delta_vec(fit,m): return _design_vector(fit,m,1,1)-_design_vector(fit,m,0,1)-_design_vector(fit,m,1,0)+_design_vector(fit,m,0,0)

def _contrast(fit,v):
    names=list(fit.fe_params.index); b=fit.fe_params.loc[names].to_numpy(float); cov=fit.cov_params().loc[names,names].to_numpy(float); est=float(v@b); se=float(np.sqrt(max(v@cov@v,0.0))); p=float(2*scipy.stats.norm.sf(abs(est/se))) if se>0 else np.nan; return est,se,p


def alignment_point(data):
    formula="VSA_z ~ CIE_z * SELF * C(model_key) + C(frozen_answer_role) + baseline_margin_z + C(strength) + C(analysis_relation)"; fit,opt=fit_mixedlm_fallback(formula,data); vec={m:_delta_vec(fit,m) for m in PRIMARY_MODELS}; rows=[]
    for m in PRIMARY_MODELS:
        e,se,p=_contrast(fit,vec[m]); rows.append({"model_key":m,"Delta":e,"SE":se,"p_value":p})
    h=vec["qwen"]-0.5*(vec["granite"]+vec["phi"]); he,hse,hp=_contrast(fit,h)
    common_formula="VSA_z ~ CIE_z * SELF + C(model_key) * CIE_z + C(model_key) * SELF + C(frozen_answer_role) + baseline_margin_z + C(strength) + C(analysis_relation)"; cfit,copt=fit_mixedlm_fallback(common_formula,data); lr=2*(fit.llf-cfit.llf); hetero_p=float(scipy.stats.chi2.sf(lr,2)); common_vec=_design_vector(cfit,"qwen",1,1)-_design_vector(cfit,"qwen",0,1)-_design_vector(cfit,"qwen",1,0)+_design_vector(cfit,"qwen",0,0); ce,cse,cp=_contrast(cfit,common_vec)
    out=pd.DataFrame(rows); from statsmodels.stats.multitest import multipletests; out["p_BH"]=multipletests(out["p_value"].to_numpy(float),method="fdr_bh")[1]
    return {"optimizer":opt,"H":he,"H_SE":hse,"H_p":hp,"heterogeneity_LR":float(lr),"heterogeneity_p":hetero_p,"common_interaction":ce,"common_interaction_SE":cse,"common_interaction_p":cp,"common_optimizer":copt},out


def alignment_bootstrap(data, selected, reps=2000, seed=20260813, checkpoint_path=None):
    checkpoint_path=Path(checkpoint_path) if checkpoint_path else None
    if checkpoint_path and checkpoint_path.exists(): done=pd.read_csv(checkpoint_path); have=set(done["replicate"].astype(int))
    else: done=pd.DataFrame(); have=set()
    base_rng=np.random.default_rng(int(seed)); seeds=base_rng.integers(0,2**63-1,size=int(reps),dtype=np.int64); pending=[]
    for b in range(int(reps)):
        if b in have: continue
        rng=np.random.default_rng(int(seeds[b])); samp=_bootstrap_sample(selected,rng)
        rep=data.merge(samp,left_on="item_id",right_on="source_item_id",how="inner",validate="many_to_many",suffixes=("","_boot")); rep["item_id"]=rep["boot_item_id"]; rep["model_item_id"]=rep["model_key"].astype(str)+"|"+rep["boot_item_id"]
        valid=True; reason=""
        try:
            point,deltas=alignment_point(rep); row={"replicate":b,"valid":1,"reason":"","H":point["H"],"common_interaction":point["common_interaction"]}
            for m in PRIMARY_MODELS: row[f"Delta_{m}"]=float(deltas.loc[deltas["model_key"].eq(m),"Delta"].iloc[0])
        except Exception as exc:
            row={"replicate":b,"valid":0,"reason":f"{type(exc).__name__}: {exc}","H":np.nan,"common_interaction":np.nan,**{f"Delta_{m}":np.nan for m in PRIMARY_MODELS}}
        pending.append(row)
        if checkpoint_path and (len(pending)>=10 or b==reps-1):
            add=pd.DataFrame(pending); done=add if done.empty else pd.concat([done,add],ignore_index=True); done=done.drop_duplicates("replicate",keep="last").sort_values("replicate"); atomic_write_csv(done,checkpoint_path); pending=[]
    return pd.read_csv(checkpoint_path) if checkpoint_path else pd.DataFrame(pending)


def alignment_bootstrap_summary(point, delta_table, boot, minimum_valid=1900):
    valid=boot[boot["valid"].eq(1)].copy()
    if len(valid)<minimum_valid: return {"classification":"technical_indeterminate","valid_bootstrap":int(len(valid))},delta_table
    out=delta_table.copy()
    for m in PRIMARY_MODELS:
        vals=valid[f"Delta_{m}"].dropna().to_numpy(float); lo,hi=np.quantile(vals,[0.025,0.975]); out.loc[out["model_key"].eq(m),"bootstrap_ci_low"]=lo; out.loc[out["model_key"].eq(m),"bootstrap_ci_high"]=hi
    h_lo,h_hi=np.quantile(valid["H"].dropna(),[0.025,0.975]); c_lo,c_hi=np.quantile(valid["common_interaction"].dropna(),[0.025,0.975])
    summary={**point,"H_bootstrap_ci_low":float(h_lo),"H_bootstrap_ci_high":float(h_hi),"common_bootstrap_ci_low":float(c_lo),"common_bootstrap_ci_high":float(c_hi),"valid_bootstrap":int(len(valid))}
    return summary,out


def wording_robustness(main_reports,alt_reports):
    rows=[]
    from scipy.stats import spearmanr
    for m in PRIMARY_MODELS:
        a=main_reports[m]; b=alt_reports[m]; j=a.merge(b,on=["item_id","condition","frame"],suffixes=("_main","_alt")); rows.append({"model_key":m,"n":len(j),"spearman":float(spearmanr(j["VSA_main"],j["VSA_alt"]).statistic),"mean_shift":float((j["VSA_alt"]-j["VSA_main"]).mean())})
    return pd.DataFrame(rows)


def fictional_summary(fiction_answers,fiction_reports):
    rows=[]
    for m in PRIMARY_MODELS:
        a=fiction_answers[m]; r=fiction_reports[m]; rows.append({"model_key":m,"context_answer_share":float((a["frozen_answer_role"]=="C").mean()),"document_report_share":float((r["modal_report"]=="document").mean()),"mean_VSA":float(r["VSA"].mean())})
    return pd.DataFrame(rows)
