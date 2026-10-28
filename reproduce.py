#!/usr/bin/env python3
"""Reproduce the Neurocomputing V10 paper. Run without arguments for the CPU paper route."""
from pathlib import Path
import argparse,json,os,platform,subprocess,sys,tempfile,time
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='paper',choices=['paper','verify','posthoc','fresh40','factorial','development','original','original-full','original-reanalysis'])
    args=p.parse_args();env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
    out=ROOT/'reproduced_results/paper'
    if args.stage=='paper':
        out.mkdir(parents=True,exist_ok=True)
        (out/'validation.json').write_text(json.dumps({'status':'RUNNING','command':'python reproduce.py paper'},indent=2)+'\n')
    def run(script,*a):
        try:
            subprocess.run([sys.executable,'-B',str(ROOT/script),*map(str,a)],cwd=ROOT,env=env,check=True)
        except subprocess.CalledProcessError as error:
            if args.stage=='paper':
                (out/'validation.json').write_text(json.dumps({'status':'FAILED','step':str(script),'exit_code':error.returncode},indent=2)+'\n')
            raise
    if args.stage!='paper':
        run('run_reproduction.py','quick' if args.stage=='verify' else args.stage);return
    started=time.perf_counter()
    # The original scripts remain unchanged in their archived environments.
    run('run_reproduction.py','integrity')
    for study in ['factorial','development','fresh40']:run('scripts/check_saved_scores.py',study)
    run('manuscript_support/checks/validate_results.py')
    analysis=out/'revision'
    run('new_analysis/revision_20261027/analyze_saved_scores.py','--repo',ROOT,'--out',analysis)
    run('scripts/build_paper.py','--analysis',analysis,'--out',out)
    summary=json.loads((out/'validation.json').read_text())
    summary.update(python=platform.python_version(),runtime_seconds=round(time.perf_counter()-started,2),raw_jobs_verified=89280,command='python reproduce.py paper')
    (out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('\nPASS: paper tables, figures, score checks and grouped analyses. Results:',out,flush=True)
if __name__=='__main__':main()
