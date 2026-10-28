"""Frozen pilot inputs, jobs, folds, registration binding, and crash recovery."""
from __future__ import annotations
import csv,hashlib,json,math,os,unicodedata
from pathlib import Path
from datetime import datetime,timezone
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1]
ROLES=('document','prior','both','uncertain')
LABELS=('A','B','C','D')
def digest(data): return hashlib.sha256(data).hexdigest()
def file_sha(path): return digest(Path(path).read_bytes())
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def now(): return datetime.now(timezone.utc).isoformat()
def read_json(path):return json.loads(Path(path).read_text())
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as h:h.write(canonical(value)+b'\n');h.flush();os.fsync(h.fileno())
    os.replace(tmp,path)
def config(root=ROOT):return read_json(Path(root)/'protocol/config.json')
def rows(path):
    with Path(path).open(newline='') as h:return list(csv.DictReader(h))
def triple_key(row):
    return tuple(unicodedata.normalize('NFKC',row[k]).strip().casefold() for k in ('parametric_answer','counterfactual_answer','irrelevant_answer'))

def make_folds(items,seed):
    groups={}
    for item in items:groups.setdefault(triple_key(item),[]).append(item)
    total=[0]*5;mapping={}
    for rel in sorted({i['relation'] for i in items}):
        subset=[(key,g) for key,g in groups.items() if g[0]['relation']==rel]
        assert all(len({r['relation'] for r in g})==1 for _,g in subset)
        subset.sort(key=lambda kg:(-len(kg[1]),digest(canonical([seed,rel,kg[0]]))))
        load=[0]*5
        for key,g in subset:
            f=min(range(5),key=lambda k:(load[k],total[k],digest(canonical([seed,rel,key,k]))))
            for item in g:mapping[item['item_id']]=f
            load[f]+=len(g);total[f]+=len(g)
    return mapping

def jobs(root=ROOT,model=None):
    root=Path(root);cfg=config(root);bundle=read_json(root/'protocol/prompts.json')
    items={i['item_id']:i for i in rows(root/'inputs/items.csv')}
    targets=rows(root/'inputs/answer_targets.csv');result=[]
    for target in targets:
        m=target['model_key']
        if model is not None and m!=model:continue
        assert m in cfg['models'] and target['condition'] in cfg['conditions']
        for cell in bundle['cells']:
            for rotation,order in enumerate(bundle['report_orders']):
                options='\n'.join(f'{lab}. {bundle["report_role_text"][role]}' for lab,role in zip(LABELS,order))
                prompt=cell['user_template'].format(document=target['document_text'],question=items[target['item_id']]['question'],answer=target['frozen_answer'],options=options)
                jid=f'{cfg["protocol_id"]}|{m}|{target["item_id"]}|{target["condition"]}|{cell["cell"]}|r{rotation}'
                result.append(dict(job_id=jid,model_key=m,item_id=target['item_id'],condition=target['condition'],cell=cell['cell'],ownership=cell['ownership'],hedging=cell['hedging'],visibility=cell['visibility'],rotation=rotation,label_to_role=dict(zip(LABELS,order)),user_prompt=prompt,prompt_sha256=digest(prompt.encode())))
    assert len({r['job_id'] for r in result})==len(result)
    # Mix cells and strengths across runtime sessions instead of blocking by factor.
    return sorted(result,key=lambda j:digest(canonical([cfg['fold_seed'],'execution',j['job_id']])))

def verify_lock(root=ROOT):
    root=Path(root);lock=read_json(root/'protocol/package_lock.json')
    for name,expected in lock['files'].items():
        p=root/name
        if not p.is_file() or file_sha(p)!=expected:raise RuntimeError(f'Locked file changed or missing: {name}')
    return file_sha(root/'protocol/package_lock.json')

def check_registration(path,lock_sha):
    value=read_json(path)
    if value.get('package_lock_sha256')!=lock_sha:raise RuntimeError('Registration receipt does not bind this package lock.')
    url=urlparse(value.get('registry_url',''))
    if url.scheme!='https' or not url.netloc or url.netloc.endswith('.invalid'):raise RuntimeError('Supply an actual external registration URL before report scoring.')
    stamp=datetime.fromisoformat(value.get('registered_utc','').replace('Z','+00:00'))
    if stamp.tzinfo is None or stamp>datetime.now(timezone.utc):raise RuntimeError('Registration timestamp must be timezone-aware and not in the future.')
    if value.get('author_attests_before_new_reports') is not True:raise RuntimeError('External timestamp attestation is missing.')
    # This checks format and hash binding; it does not verify the registry remotely.
    return value

def recover(path,expected,lock_sha,repair_tail=True):
    """One JSON line is a committed batch; only an unfinished final line is repaired."""
    path=Path(path);saved={}
    if not path.exists():return saved
    data=path.read_bytes();lines=data.splitlines(keepends=True);offset=0
    for ix,line in enumerate(lines):
        if not line.endswith(b'\n'):
            if ix!=len(lines)-1 or not repair_tail:raise RuntimeError('Uncommitted checkpoint tail.')
            tail=path.with_name(path.name+'.partial-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
            tail.write_bytes(line)
            with path.open('r+b') as h:h.truncate(offset);h.flush();os.fsync(h.fileno())
            break
        try:batch=json.loads(line)
        except Exception as exc:raise RuntimeError('Malformed committed checkpoint batch.') from exc
        if batch['package_lock_sha256']!=lock_sha:raise RuntimeError('Checkpoint belongs to another package lock.')
        for row in batch['rows']:
            jid=row['job_id']
            if jid not in expected or row['prompt_sha256']!=expected[jid]['prompt_sha256']:raise RuntimeError('Checkpoint prompt/job identity mismatch.')
            if jid in saved:raise RuntimeError('Duplicate committed job; refusing ambiguous recovery.')
            status=row['status']
            if status not in ('ok','terminal_error'):raise RuntimeError('Unknown checkpoint status.')
            if status=='ok':
                if set(row['logp'])!=set(LABELS) or not all(math.isfinite(float(x)) and float(x)<=1e-5 for x in row['logp'].values()):raise RuntimeError('Invalid saved log probabilities.')
            saved[jid]=row
        offset+=len(line)
    return saved

def append_batch(path,records,lock_sha,runtime):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    batch=dict(package_lock_sha256=lock_sha,committed_utc=now(),runtime=runtime,rows=records)
    with path.open('ab') as h:h.write(canonical(batch)+b'\n');h.flush();os.fsync(h.fileno())

def execute(alljobs,path,lock_sha,backend,runtime,initial_batch=8,fallback=(4,2),max_batches=None):
    """Append-only execution. Successful and terminal rows are never overwritten."""
    expected={j['job_id']:j for j in alljobs};saved=recover(path,expected,lock_sha)
    pending=[j for j in alljobs if j['job_id'] not in saved];cursor=0;batches=0;preferred=initial_batch
    while cursor<len(pending) and (max_batches is None or batches<max_batches):
        batch=pending[cursor:cursor+preferred];attempts=[]
        while True:
            try:
                scores=backend.score(batch)
                if len(scores)!=len(batch):raise RuntimeError('Backend returned the wrong batch length.')
                for score in scores:
                    if set(score)!=set(LABELS) or not all(math.isfinite(float(v)) and float(v)<=1e-5 for v in score.values()):raise FloatingPointError('Invalid backend log probabilities.')
                out=[dict(job_id=j['job_id'],prompt_sha256=j['prompt_sha256'],status='ok',logp=s,batch_size=len(batch),attempts=attempts) for j,s in zip(batch,scores)]
                break
            except MemoryError as exc:
                attempts.append({'batch_size':len(batch),'error':str(exc),'kind':'oom'})
                smaller=[n for n in fallback if n<len(batch)]
                if smaller:
                    preferred=max(smaller);batch=pending[cursor:cursor+preferred];continue
                # Isolate a remaining pair after the 4/2 fallback; this is a
                # technical attribution step, not a new scientific setting.
                if len(batch)>1:batch=batch[:1];preferred=1;continue
                out=[dict(job_id=j['job_id'],prompt_sha256=j['prompt_sha256'],status='terminal_error',error=str(exc),attempts=attempts) for j in batch];break
        append_batch(path,out,lock_sha,runtime);cursor+=len(batch);batches+=1
        print(f'Committed {len(saved)+cursor}/{len(alljobs)} jobs',flush=True)
    return recover(path,expected,lock_sha)
