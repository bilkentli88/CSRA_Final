"""Development-only execution; Mistral target outcomes and fresh inputs are rejected."""
import argparse,shutil,zipfile,contextlib,tempfile
from common import *

def prepare(run,lock_sha):
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    immutable_json(run/'run_manifest.json',dict(protocol_id=cfg()['protocol_id'],package_lock_sha256=lock_sha,input_sha256=file_sha(ROOT/'inputs/items.csv'),fresh_item_execution_allowed=False,human_re_review_completed=False))
    frozen=run/'frozen_package'
    if not frozen.exists():shutil.copytree(ROOT,frozen,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    elif file_sha(frozen/'protocol/package_lock.json')!=lock_sha:raise RuntimeError('Existing run binds another package.')
    immutable_json(run/'folds.json',dict(item_to_fold=folds(items()),item_to_group=grouping(items())))
    print('Run initialized:',run,flush=True)

def bound(run,lock_sha):
    m=read_json(Path(run)/'run_manifest.json')
    if m['package_lock_sha256']!=lock_sha:raise RuntimeError('Run/package mismatch.')

def targets(run,lock_sha):
    result=aggregate_targets(run,lock_sha);value=dict(package_lock_sha256=lock_sha,answer_file_hashes={m:file_sha(Path(run)/'scores'/f'answer_{m}.jsonl') for m in cfg()['target_models']},targets=result)
    immutable_json(Path(run)/'targets.json',value);print('Complete development target cells:',len(result),flush=True);return value

def get_targets(run,lock_sha):
    saved=read_json(Path(run)/'targets.json');current=dict(package_lock_sha256=lock_sha,answer_file_hashes={m:file_sha(Path(run)/'scores'/f'answer_{m}.jsonl') for m in cfg()['target_models']},targets=aggregate_targets(run,lock_sha))
    if saved!=current:raise RuntimeError('Target manifest differs from committed answer scores.')
    return saved['targets']

@contextlib.contextmanager
def single_writer(run):
    # Lock processes on this runtime using its local filesystem; Drive FUSE need not support flock.
    # Never launch the same Drive run from two runtimes simultaneously.
    import fcntl
    p=Path(tempfile.gettempdir())/('csra-writer-'+sha(str(Path(run).resolve()).encode())+'.lock')
    with p.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError('Another process is writing this run. Wait for it to finish.') from e
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['validate','prepare','audit','answers','targets','reports','assemble','status','export']);ap.add_argument('--run-dir',type=Path);ap.add_argument('--model');ap.add_argument('--max-batches',type=int);ap.add_argument('--output',type=Path);args=ap.parse_args();lock_sha=verify_package()
    if args.stage=='validate':print('Package valid; 100 old items; 14,400 answer + 28,800 report jobs; no Mistral development targets.');return
    if args.run_dir is None:ap.error('--run-dir is required.')
    run=args.run_dir
    if args.stage=='prepare':
        with single_writer(run):prepare(run,lock_sha)
        return
    bound(run,lock_sha)
    if args.stage=='status':
        status={};t=read_json(run/'targets.json')['targets'] if (run/'targets.json').exists() else None
        for model in cfg()['target_models']:
            jobs=answer_jobs(model);status['answer_'+model]=dict(completed=len(recover(run/'scores'/f'answer_{model}.jsonl',{j['job_id']:j for j in jobs},lock_sha,False)),expected=len(jobs))
        if t:
            for model in cfg()['reporter_models']:
                jobs=report_jobs(model,t);status['report_'+model]=dict(completed=len(recover(run/'scores'/f'report_{model}.jsonl',{j['job_id']:j for j in jobs},lock_sha,False)),expected=len(jobs))
        print(json.dumps(status,indent=2));return
    if args.stage=='export':
        if args.output is None:ap.error('--output is required for export.')
        if args.output.resolve().is_relative_to(run.resolve()):raise RuntimeError('Place export archive outside the run directory.')
        with single_writer(run):
            with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED) as z:
                for p in sorted(run.rglob('*')):
                    if p.is_file() and p.name!='writer.lock' and '__pycache__' not in p.parts:z.write(p,run.name+'/'+p.relative_to(run).as_posix())
        print('Exported:',args.output);return
    with single_writer(run):
        if args.stage=='audit':
            from hf_backend import audit
            audit(run,lock_sha)
        elif args.stage=='targets':targets(run,lock_sha)
        elif args.stage in ('answers','reports'):
            if args.stage=='answers':jobs=answer_jobs(args.model)
            else:jobs=report_jobs(args.model,get_targets(run,lock_sha))
            filename=run/'scores'/f'{"answer" if args.stage=="answers" else "report"}_{args.model}.jsonl'
            saved=recover(filename,{j['job_id']:j for j in jobs},lock_sha)
            if len(saved)==len(jobs):print('Already complete:',filename.name);return
            from hf_backend import HFBackend
            backend=HFBackend(args.model,run,lock_sha);execute(jobs,filename,lock_sha,backend,cfg()['initial_batch_size'],args.max_batches)
        elif args.stage=='assemble':
            t=get_targets(run,lock_sha);cases=aggregate_reports(run,t,lock_sha);value=dict(package_lock_sha256=lock_sha,targets_sha256=file_sha(run/'targets.json'),report_file_hashes={m:file_sha(run/'scores'/f'report_{m}.jsonl') for m in cfg()['reporter_models']},cases=cases)
            immutable_json(run/'analysis_inputs.json',value);print('Complete paired analysis cases:',len(cases))
if __name__=='__main__':main()
