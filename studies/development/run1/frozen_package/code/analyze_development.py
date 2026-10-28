"""Development diagnostics and a draft frozen stack; no fresh-set inference."""
import argparse,os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
import importlib.metadata,csv
from common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--decoder',choices=['ridge','gradient_boosting'],default='ridge');args=ap.parse_args();run=args.run_dir;lock=verify_package()
    actual={k:importlib.metadata.version(k) for k in plan()['analysis_versions']}
    if actual!=plan()['analysis_versions']:raise RuntimeError('Analysis environment differs from pins: '+str(actual))
    import numpy as np,joblib
    from predictive import check_cases,fit_stack,evaluate_stack,group_diagnostics
    from run_development import get_targets
    targets=get_targets(run,lock);inputs=read_json(run/'analysis_inputs.json');current=aggregate_reports(run,targets,lock)
    expected=dict(package_lock_sha256=lock,targets_sha256=file_sha(run/'targets.json'),report_file_hashes={m:file_sha(run/'scores'/f'report_{m}.jsonl') for m in cfg()['reporter_models']},cases=current)
    if inputs!=expected:raise RuntimeError('Analysis inputs differ from committed scores or their manifests.')
    cases=inputs['cases'];check_cases(cases);enc=read_json(run/'analysis/encoder_manifest.json');embedding_path=run/'analysis/embeddings.npy'
    if enc['analysis_inputs_sha256']!=file_sha(run/'analysis_inputs.json') or enc['embedding_sha256']!=file_sha(embedding_path) or enc['encoder']!=plan()['encoder']:raise RuntimeError('Pinned embedding manifest mismatch.')
    E=np.load(embedding_path,allow_pickle=False)
    if E.shape!=(len(cases),768) or not np.isfinite(E).all():raise RuntimeError('Invalid embeddings.')
    out=run/'analysis'/args.decoder;out.mkdir(parents=True,exist_ok=True)
    identity=dict(package_lock_sha256=lock,analysis_inputs_sha256=file_sha(run/'analysis_inputs.json'),embedding_sha256=file_sha(embedding_path),decoder=args.decoder,packages=actual)
    immutable_json(out/'analysis_identity.json',identity);results=[]
    for fold in range(cfg()['outer_folds']):
        path=out/f'outer_fold_{fold}.json'
        if path.exists():
            value=read_json(path)
            if value['identity']!=identity:raise RuntimeError('Cached outer fold identity differs.')
            results.extend(value['results']);print('Using completed outer fold',fold,flush=True);continue
        train=np.array([i for i,c in enumerate(cases) if c['fold']!=fold]);valid=np.array([i for i,c in enumerate(cases) if c['fold']==fold])
        if not len(valid):raise RuntimeError('Empty frozen outer fold.')
        print('Fitting nested group-held-out fold',fold+1,'of',cfg()['outer_folds'],flush=True)
        fit=fit_stack(cases,E,train,args.decoder);pred=evaluate_stack(fit,cases,E,valid)
        value=dict(identity=identity,training_targets=fit['training_targets'],training_groups=fit['training_groups'],text_alphas={k:v['alpha'] for k,v in fit['text_models'].items()},results=pred);atomic_json(path,value);results.extend(pred)
    if len(results)!=len(cases) or len({x['case_index'] for x in results})!=len(cases):raise RuntimeError('Incomplete or duplicate out-of-group predictions.')
    results=sorted(results,key=lambda x:x['case_index']);diagnostics=group_diagnostics(cases,results)
    with (out/'heldout_errors.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    summary=dict(status='EXPLORATORY_DEVELOPMENT_DIAGNOSTICS_NOT_FRESH_CONFIRMATION',identity=identity,known_target_diagnostics=diagnostics,equal_target_mean_delta_R2=float(np.mean([d['delta_R2'] for d in diagnostics])),n_items=100,n_groups=len({c['group'] for c in cases}),fresh_outcomes=0,held_out_target='mistral',held_out_target_outcomes_used=False,human_re_review_completed=False,normalization='Each known target uses variance over all its out-of-group development Y values; these variances are development calibration quantities only.')
    summary['heldout_errors_sha256']=file_sha(out/'heldout_errors.csv')
    atomic_json(out/'development_summary.json',summary)
    stack_path=out/'draft_development_stack.joblib'
    if not stack_path.exists():
        print('Fitting full old-100 development stack.',flush=True);fit=fit_stack(cases,E,np.arange(len(cases)),args.decoder);temporary=stack_path.with_suffix('.tmp');joblib.dump(fit,temporary);temporary.replace(stack_path)
    fit=joblib.load(stack_path)
    if 'mistral' in fit['training_targets']:raise RuntimeError('Held-out target leaked into training.')
    manifest=dict(identity=identity,stack_sha256=file_sha(stack_path),fitted_targets=fit['training_targets'],n_training_groups=len(fit['training_groups']),text_alphas={k:v['alpha'] for k,v in fit['text_models'].items()},feature_count=fit['feature_count'],frozen_training_artifact=True,confirmatory_release_ready=False,human_re_review_completed=False,reason='Draft development artifact. Factual re-review, final fresh-set selection, precision/resource decision and the main execution/analysis freeze remain outstanding. No fresh data have been evaluated.')
    immutable_json(out/'draft_stack_manifest.json',manifest);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
