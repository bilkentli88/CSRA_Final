"""Static validation, synthetic GPU preflight, internal freeze and explicit run mode."""
import argparse,json
from pathlib import Path
from pilot_common import ROOT,config,jobs,verify_lock,atomic_json,execute,read_json
from execution_context import freeze_run,check_preflight,prepare_header

def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('validate')
    p=sub.add_parser('freeze');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('preflight');p.add_argument('--model',choices=['qwen','granite','phi'],required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--model',choices=['qwen','granite','phi'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['exploratory','registered'],required=True);p.add_argument('--registration',type=Path);p.add_argument('--preflight',type=Path,required=True);p.add_argument('--max-batches',type=int)
    args=ap.parse_args();lock_sha=verify_lock();cfg=config();alljobs=jobs(model=getattr(args,'model',None))
    if args.command=='validate':
        assert len(alljobs)==28800
        print(json.dumps(dict(package_lock_sha256=lock_sha,logical_jobs=len(alljobs),per_model=9600,new_model_calls=0),indent=2));return
    if args.command=='freeze':
        print(json.dumps(freeze_run(args.output),indent=2));return
    if args.command=='run':
        header=prepare_header(args.output,args.model,args.mode,args.registration)
        check_preflight(args.preflight,args.model)
        args.output.mkdir(parents=True,exist_ok=True);headerpath=args.output/f'{args.model}_run_header.json'
        if headerpath.exists() and read_json(headerpath)!=header:raise RuntimeError('Existing run directory is bound to another context/package.')
        if not headerpath.exists():atomic_json(headerpath,header)
        if args.mode=='registered':atomic_json(args.output/f'{args.model}_registration.json',header['registration'])
    from hf_backend import HFBackend
    backend=HFBackend(args.model)
    if args.command=='preflight':
        result=backend.preflight();result['package_lock_sha256']=lock_sha;atomic_json(args.output,result);print(json.dumps(result,indent=2));return
    current=backend.preflight();atomic_json(args.output/f'{args.model}_latest_preflight.json',{**current,'package_lock_sha256':lock_sha})
    result=execute(alljobs,args.output/f'{args.model}_checkpoint.jsonl',lock_sha,backend,backend.runtime,cfg['initial_batch_size'],cfg['fallback_batch_sizes'],args.max_batches)
    print(json.dumps(dict(study_mode=args.mode,committed=len(result),expected=len(alljobs),ok=sum(r['status']=='ok' for r in result.values()),terminal=sum(r['status']=='terminal_error' for r in result.values())),indent=2))

if __name__=='__main__':main()
