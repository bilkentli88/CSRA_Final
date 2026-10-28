"""CLI called from the five-cell notebook. All model stages run in separate processes."""
import argparse,fcntl,shutil,zipfile,tempfile,contextlib
from core import *
@contextlib.contextmanager
def single_writer(run):
    p=Path(tempfile.gettempdir())/('csra-fresh40-'+sha(str(Path(run).resolve()).encode())+'.lock')
    with p.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another writer is active in this runtime. Use only one Colab runtime per run directory.')
        yield

def initialize(run):
    run=Path(run);run.mkdir(parents=True,exist_ok=True);lock=verify_package();frozen=run/'frozen_package'
    if frozen.exists():
        stored=read_json(frozen/'protocol/package_lock.json')
        if file_sha(frozen/'protocol/package_lock.json')!=lock:raise RuntimeError('Run is bound to a different package; use another run directory.')
        for name,digest in stored['files'].items():
            if file_sha(frozen/name)!=digest:raise RuntimeError('Frozen package file changed: '+name)
    else:shutil.copytree(ROOT,frozen,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    immutable_json(run/'run_identity.json',dict(package_lock_sha256=lock,protocol=cfg()['protocol_id'],review_workbook_sha256=file_sha(ROOT/'inputs/reviewed_facts.xlsx')))
    print('Run initialized:',run,flush=True)
def prepare(run):
    run=Path(run)
    if (run/'pre_outcome_freeze.json').exists():
        verify_freeze(run);print('Existing freeze verified. No re-encoding or refitting.',flush=True);print(read_json(run/'observer_selection.json')['selected']);return
    require_before_outcomes(run)
    from hf_backend import audit
    from encode import encode
    from learning import train
    audit(run,verify_package());encode(run);train(run);freeze_run(run);print('Pre-outcome freeze saved. Ready for scoring.',flush=True)
def targets_checked(run,freeze):
    targets=aggregate_targets(run,freeze);path=Path(run)/'targets.json';immutable_json(path,dict(pre_outcome_freeze_sha256=freeze,cases=targets));return targets

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['validate','init','prepare','answers','targets','reports','analyze','export']);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--model');p.add_argument('--output',type=Path);a=p.parse_args();run=a.run_dir
    with single_writer(run):
        lock=verify_package()
        if a.stage=='validate':print(validate_inputs());return
        if a.stage=='init':initialize(run);return
        if not (run/'run_identity.json').exists() or read_json(run/'run_identity.json')['package_lock_sha256']!=lock:raise RuntimeError('Initialize the bound run first.')
        # Require execution from its preserved package after initialization.
        if ROOT.resolve()!=(run/'frozen_package').resolve():raise RuntimeError('Execute this stage from RUN/frozen_package/code/run_fresh40.py.')
        if a.stage=='prepare':prepare(run);return
        freeze=verify_freeze(run)
        if a.stage in ('answers','reports'):
            from hf_backend import HFBackend
            if a.stage=='answers':jobs=answer_jobs(a.model);path=run/'scores'/f'answer_{a.model}.jsonl'
            else:jobs=report_jobs(a.model,targets_checked(run,freeze));path=run/'scores'/f'report_{a.model}.jsonl'
            existing=recover(path,{j['job_id']:j for j in jobs},freeze)
            if len(existing)==len(jobs):print(a.stage,a.model,'already complete.');return
            backend=HFBackend(a.model,run,lock);execute(jobs,path,freeze,backend);return
        if a.stage=='targets':print('Fresh target cases:',len(targets_checked(run,freeze)));return
        if a.stage=='analyze':
            from analyze import analyze
            analyze(run);return
        if a.stage=='export':
            if not a.output:raise RuntimeError('Provide --output outside the run directory.')
            dest=a.output.resolve()
            if dest.is_relative_to(run.resolve()):raise RuntimeError('Export outside the run directory.')
            # Verify completeness, not just presence of a summary.
            targets=targets_checked(run,freeze);aggregate_reports(run,targets,freeze)
            result=read_json(run/'analysis/qualification_summary.json')
            if result['pre_outcome_freeze_sha256']!=freeze:raise RuntimeError('Analysis belongs to another freeze.')
            dest.parent.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
                for f in sorted(run.rglob('*')):
                    if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc':z.write(f,Path(run.name)/f.relative_to(run))
            print('Saved:',dest,flush=True)
if __name__=='__main__':main()
