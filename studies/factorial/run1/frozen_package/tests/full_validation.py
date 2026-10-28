"""Full 10,000-draw synthetic check; no real model calls, no scientific results."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,tempfile,time,json,importlib.util
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'code'))
import numpy as np,pandas as pd
from pilot_common import *
from analyze_pilot import load_data,analyze,regression_units
from ridge_batch import ridge_r2_batch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
spec=importlib.util.spec_from_file_location('selftest',root/'tests/selftest.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
start=time.perf_counter();sha=verify_lock()
with tempfile.TemporaryDirectory() as folder:
 folder=Path(folder);backend=t.SyntheticBackend()
 for m in config()['models']:
  js=jobs(model=m)
  out=[dict(job_id=j['job_id'],prompt_sha256=j['prompt_sha256'],status='ok',logp=s) for j,s in zip(js,backend.score(js))]
  atomic_json(folder/f'{m}_run_header.json',dict(data_kind='SYNTHETIC_TEST_ONLY',package_lock_sha256=sha,model=m))
  append_batch(folder/f'{m}_checkpoint.jsonl',out,sha,{'kind':'SYNTHETIC_TEST_ONLY'})
 agg,selected,bad=load_data(folder,synthetic=True)
 u=next(iter(regression_units(agg,selected).values()))
 num=['baseline_margin','alr_document_OTHER','alr_both_OTHER','alr_uncertain_OTHER'];cats=['strength','analysis_relation']
 rng=np.random.default_rng(13);w=rng.integers(0,4,len(u));y=u.CIE.to_numpy();fold=u.fold_id.to_numpy();pred=np.empty(len(u));cat=pd.get_dummies(u[cats].astype(str),dtype=float).to_numpy()
 for f in range(5):
  ix=np.repeat(np.where(fold!=f)[0],w[fold!=f]);test=fold==f
  scale=StandardScaler().fit(u[num].iloc[ix]);xtr=np.column_stack((scale.transform(u[num].iloc[ix]),cat[ix]));xte=np.column_stack((scale.transform(u[num].loc[test]),cat[test]))
  fit=Ridge(alpha=1,solver='cholesky').fit(xtr,y[ix]);pred[test]=fit.predict(xte)
 expected=1-np.average((y-pred)**2,weights=w)/np.var(y)
 observed=ridge_r2_batch(u,num,cats,w)[0];error=abs(expected-observed);assert error<1e-8,(expected,observed)
 print('Independent weighted Ridge equivalence passed:',error,flush=True)
 result=analyze(agg,selected,reps=10000,secondary=False)
 assert result['primary_draws'].shape==(10000,4)
 assert np.isfinite(result['primary_draws'].to_numpy()).all()
 info=dict(data_kind='SYNTHETIC_TEST_ONLY',logical_jobs=28800,aggregated_rows=len(agg),complete_items=len(selected),primary_bootstrap_draws=10000,primary_contrasts=4,finite_draws=True,bootstrap_redraws=result['bootstrap_redraws'],ridge_sklearn_absolute_error=error,elapsed_seconds=time.perf_counter()-start,package_lock_sha256=sha)
 print(json.dumps(info,indent=2))
