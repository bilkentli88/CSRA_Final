"""Explicit study mode and an internal freeze; no independent timestamp claim."""
from pathlib import Path
from datetime import datetime,timezone
import shutil
from pilot_common import ROOT,verify_lock,read_json,atomic_json,file_sha,now,check_registration

def check_preflight(path,model,root=ROOT):
    root=Path(root);value=read_json(path);sha=verify_lock(root)
    spec=read_json(root/'protocol/model_lock.json')['models'][model]
    if value.get('status')!='TECHNICAL_SYNTHETIC_PREFLIGHT_ONLY' or value.get('package_lock_sha256')!=sha:
        raise RuntimeError('A passing preflight for this exact package is required.')
    if value.get('runtime',{}).get('model_revision')!=spec['revision'] or value['runtime'].get('tokenizer_revision')!=spec['tokenizer_revision']:
        raise RuntimeError('Preflight model/tokenizer revision mismatch.')
    return value

def check_freeze(run_dir,root=ROOT):
    run_dir=Path(run_dir);sha=verify_lock(root)
    value=read_json(run_dir/'exploratory_freeze.json')
    if value.get('study_mode')!='exploratory' or value.get('independent_timestamp') is not False or value.get('externally_registered') is not False:
        raise RuntimeError('Expected an explicit internal-only exploratory freeze.')
    if value.get('package_lock_sha256')!=sha:raise RuntimeError('Freeze belongs to another package.')
    stamp=datetime.fromisoformat(value.get('created_utc','').replace('Z','+00:00'))
    if stamp.tzinfo is None or stamp>datetime.now(timezone.utc):raise RuntimeError('Invalid local freeze timestamp.')
    snapshot=run_dir/'frozen_package'
    if file_sha(snapshot/'protocol/package_lock.json')!=sha:raise RuntimeError('Frozen snapshot lock mismatch.')
    verify_lock(snapshot)
    return value

def freeze_run(run_dir,root=ROOT):
    root=Path(root).resolve();run_dir=Path(run_dir).resolve();sha=verify_lock(root)
    if run_dir.is_relative_to(root):raise RuntimeError('Keep the durable run directory outside the source package.')
    if (run_dir/'exploratory_freeze.json').exists():return check_freeze(run_dir,root)
    if list(run_dir.glob('*_checkpoint.jsonl')) or list(run_dir.glob('*_run_header.json')):
        raise RuntimeError('Cannot create a new before-run freeze over existing run records. Preserve that directory and choose a new one.')
    snapshot=run_dir/'frozen_package';snapshot.mkdir(parents=True,exist_ok=True)
    names=list(read_json(root/'protocol/package_lock.json')['files'])+['protocol/package_lock.json']
    for name in names:
        source=root/name;destination=snapshot/name;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            if file_sha(destination)!=file_sha(source):raise RuntimeError('Existing snapshot differs; preserve it and use another run folder.')
        else:shutil.copy2(source,destination)
    value=dict(study_mode='exploratory',package_lock_sha256=sha,created_utc=now(),independent_timestamp=False,externally_registered=False,population='100 previously used items',interpretation='Internal freeze only; not preregistered or an untouched confirmatory study.')
    atomic_json(run_dir/'exploratory_freeze.json',value)
    return check_freeze(run_dir,root)

def prepare_header(run_dir,model,mode,registration=None,root=ROOT):
    run_dir=Path(run_dir);sha=verify_lock(root)
    header=dict(package_lock_sha256=sha,model=model,data_kind='REAL_PILOT_REPORTS',study_mode=mode)
    if mode=='exploratory':
        if registration is not None:raise RuntimeError('Exploratory mode does not accept a registration receipt.')
        header['internal_freeze']=check_freeze(run_dir,root)
    elif mode=='registered':
        if registration is None:raise RuntimeError('Registered mode requires an actual registration receipt.')
        header['registration']=check_registration(registration,sha)
    else:raise RuntimeError('Choose exploratory or registered mode explicitly.')
    for p in run_dir.glob('*_run_header.json'):
        other=read_json(p)
        if other.get('package_lock_sha256')!=sha or other.get('study_mode','registered')!=mode:
            raise RuntimeError('Do not mix packages or study modes within a run directory.')
    return header

def validate_header(run_dir,model,root=ROOT):
    run_dir=Path(run_dir);header=read_json(run_dir/f'{model}_run_header.json')
    if header.get('model')!=model or header.get('package_lock_sha256')!=verify_lock(root) or header.get('data_kind')!='REAL_PILOT_REPORTS':
        raise RuntimeError('Invalid real-run header identity.')
    mode=header.get('study_mode','registered')
    if mode=='exploratory':
        if header.get('internal_freeze')!=check_freeze(run_dir,root):raise RuntimeError('Header and local freeze disagree.')
        if 'registration' in header:raise RuntimeError('Exploratory run must not assert registration.')
    elif mode=='registered':
        receipt=check_registration(run_dir/f'{model}_registration.json',verify_lock(root))
        if header.get('registration')!=receipt:raise RuntimeError('Header and registry receipt disagree.')
    else:raise RuntimeError('Unknown run mode.')
    return mode
