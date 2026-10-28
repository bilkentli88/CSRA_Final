#!/usr/bin/env python3
"""CPU reproduction entry points. Frozen inputs are never overwritten."""
from pathlib import Path
import argparse,os,shutil,subprocess,sys,tempfile,json

ROOT=Path(__file__).resolve().parent
ENV=os.environ.copy();ENV['PYTHONDONTWRITEBYTECODE']='1'
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):ENV[name]='1'
def run(script,*args,cwd=ROOT):
    print('Running:',Path(script).name,*map(str,args),flush=True)
    subprocess.run([sys.executable,'-B',str(script),*map(str,args)],cwd=cwd,env=ENV,check=True)
def working_copy(study,include_analysis=True):
    out=ROOT/'reproduced_results';out.mkdir(exist_ok=True)
    destination=Path(tempfile.mkdtemp(prefix=study+'_',dir=out))/'run1'
    source=ROOT/'studies'/study/'run1'
    shutil.copytree(source,destination,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    if not include_analysis:
        # Remove only the new working copy's cached fitted outputs, preserving
        # the embeddings and encoder manifest required for an actual refit.
        for algorithm in ('ridge','gradient_boosting'):
            p=destination/'analysis'/algorithm
            if p.exists():shutil.rmtree(p)
    return destination
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='quick',choices=['integrity','quick','original','original-full','original-reanalysis','factorial','development','fresh40','figures','posthoc','revision'])
    args=p.parse_args()
    run(ROOT/'scripts/verify_integrity.py')
    if args.stage=='integrity':return
    if args.stage=='quick':
        for study in ('factorial','development','fresh40'):run(ROOT/'scripts/check_saved_scores.py',study)
        run(ROOT/'manuscript_support/checks/validate_results.py')
        out=ROOT/'reproduced_results';out.mkdir(exist_ok=True)
        folder=Path(tempfile.mkdtemp(prefix='revision_checks_',dir=out))
        run(ROOT/'new_analysis/revision_20261027/analyze_saved_scores.py','--repo',ROOT,'--out',folder)
        print('PASS: integrity, all later raw score aggregates, 98 historical manuscript checks, 48 V4 sensitivity contrasts, and the new target/factorial analyses.',flush=True)
    elif args.stage=='revision':
        out=ROOT/'reproduced_results';out.mkdir(exist_ok=True)
        folder=Path(tempfile.mkdtemp(prefix='revision_checks_',dir=out))
        run(ROOT/'new_analysis/revision_20261027/analyze_saved_scores.py','--repo',ROOT,'--out',folder)
        print('Exploratory saved-score results:',folder,flush=True)
    elif args.stage=='posthoc':
        # The preserved October 26 script requires outputs outside the input
        # repository. Use a temporary directory, then retain its returned results.
        with tempfile.TemporaryDirectory(prefix='csra_posthoc_refit_') as temp:
            run(ROOT/'new_analysis/posthoc_20261026/run_checks.py','--repo',ROOT,'--out',temp)
            import numpy as np
            import pandas as pd
            expected=ROOT/'new_analysis/posthoc_20261026/results'
            comparisons={
                'per_case_predictions.csv':['outcome','variant','algorithm','item_id','target','strength'],
                'contrasts.csv':['outcome','variant','algorithm','contrast'],
            }
            maximum=0.0
            for name,keys in comparisons.items():
                actual=pd.read_csv(Path(temp)/name).sort_values(keys).reset_index(drop=True)
                reference=pd.read_csv(expected/name).sort_values(keys).reset_index(drop=True)
                if list(actual.columns)!=list(reference.columns) or not actual[keys].equals(reference[keys]):
                    raise AssertionError('Post hoc result identities differ: '+name)
                cols=list(actual.select_dtypes(include='number').columns)
                delta=float(np.max(np.abs(actual[cols].to_numpy()-reference[cols].to_numpy())))
                if delta>1e-8:raise AssertionError('Post hoc refit differs: '+name+' '+str(delta))
                maximum=max(maximum,delta)
            (Path(temp)/'refit_comparison.json').write_text(json.dumps({
                'status':'PASSED','saved_prediction_rows_compared':2880,
                'saved_contrast_rows_compared':48,'maximum_absolute_numeric_difference':maximum,
                'training_data':'original100 only','new_model_inference':False},indent=2)+'\n')
            out=ROOT/'reproduced_results';out.mkdir(exist_ok=True)
            folder=Path(tempfile.mkdtemp(prefix='posthoc_refit_',dir=out))
            shutil.copytree(temp,folder,dirs_exist_ok=True)
        print('Development-only robustness refit results:',folder,flush=True)
    elif args.stage in ('original','original-full'):
        out=ROOT/'reproduced_results';out.mkdir(exist_ok=True)
        folder=Path(tempfile.mkdtemp(prefix='original_',dir=out))/'original_release'
        shutil.copytree(ROOT/'original_release',folder)
        run(folder/'run_reproduction.py',*(['--bootstrap-primary','--bootstrap-alignment'] if args.stage=='original-full' else []),cwd=folder)
    elif args.stage=='original-reanalysis':
        out=ROOT/'reproduced_results';out.mkdir(exist_ok=True)
        folder=Path(tempfile.mkdtemp(prefix='original_reanalysis_',dir=out))
        run(ROOT/'original_reanalysis/code/audit_csra.py','--repo',ROOT/'original_release','--out',folder/'audit/results','--reps',2000)
        run(ROOT/'original_reanalysis/code/extend_analysis.py','--repo',ROOT/'original_release','--audit',folder/'audit','--out',folder/'revision')
        print('Reanalysis outputs:',folder,flush=True)
    elif args.stage=='factorial':
        out=ROOT/'reproduced_results';out.mkdir(exist_ok=True)
        folder=Path(tempfile.mkdtemp(prefix='factorial_',dir=out))
        run(ROOT/'studies/factorial/run1/frozen_package/code/analyze_pilot.py','--run-dir',ROOT/'studies/factorial/run1','--out',folder)
        print('Factorial outputs:',folder,flush=True)
    elif args.stage=='development':
        folder=working_copy('development',include_analysis=False)
        for algorithm in ('ridge','gradient_boosting'):
            run(folder/'frozen_package/code/analyze_development.py','--run-dir',folder,'--decoder',algorithm)
        print('Newly fitted development outputs:',folder/'analysis',flush=True)
    elif args.stage=='fresh40':
        folder=working_copy('fresh40')
        run(ROOT/'scripts/replay_fresh40.py','--run-dir',folder,'--expected',ROOT/'studies/fresh40/run1')
        print('Replayed Fresh40 outputs:',folder/'analysis',flush=True)
    elif args.stage=='figures':
        out=ROOT/'reproduced_results';out.mkdir(exist_ok=True)
        folder=Path(tempfile.mkdtemp(prefix='figures_',dir=out))/'manuscript_support'
        shutil.copytree(ROOT/'manuscript_support',folder)
        shutil.copytree(ROOT/'original_reanalysis/results',folder/'revision_analysis/results')
        run(ROOT/'original_reanalysis/code/make_manuscript_assets.py','--repo',ROOT/'original_release','--audit',ROOT/'original_reanalysis/audit','--project',folder)
        run(folder/'checks/figure_source/make_figures.py')
        print('Regenerated figures and V1 tables:',folder,flush=True)
if __name__=='__main__':main()
