"""Fresh40 jobs and storage; legacy old-100 files are preserved separately."""
import csv,hashlib,itertools,json,math,os,unicodedata
from pathlib import Path
from datetime import datetime,timezone
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[1]
ROLES=('document','prior','both','uncertain');ANSWER_ROLES=('P','C','I');LABELS='ABCD'
SYSTEM='You answer structured multiple-choice questions. Follow the requested output format exactly and do not explain your answer.'
CUES={'SELF_PRESENTED':'This answer is presented as your earlier answer.','OTHER_PRESENTED':'This answer is presented as an earlier answer by another language model.'}
ROLE_TEXT={'document':'the supplied document','prior':'prior knowledge available without the document','both':'both the document and prior knowledge','uncertain':'neither source clearly, or uncertain'}
FIELDS={'P':'parametric_answer','C':'counterfactual_answer','I':'irrelevant_answer'}
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(v):return hashlib.sha256(v).hexdigest()
def file_sha(p):return sha(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def read_json(p):return json.loads(Path(p).read_text())
@lru_cache(maxsize=1)
def cfg():return read_json(ROOT/'protocol/fresh40_config.json')
@lru_cache(maxsize=1)
def plan():return read_json(ROOT/'protocol/main_study_plan.json')
@lru_cache(maxsize=1)
def items():
    with (ROOT/'inputs/items.csv').open(newline='') as f:return list(csv.DictReader(f))
def atomic_json(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    with t.open('wb') as f:f.write(canonical(v)+b'\n');f.flush();os.fsync(f.fileno())
    t.replace(p)
def immutable_json(p,v):
    p=Path(p)
    if p.exists():
        if read_json(p)!=v:raise RuntimeError('Existing bound artifact differs: '+str(p))
    else:atomic_json(p,v)
def verify_package():
    lock=read_json(ROOT/'protocol/package_lock.json')
    for name,digest in lock['files'].items():
        if file_sha(ROOT/name)!=digest:raise RuntimeError('Changed or missing package file: '+name)
    validate_inputs()
    return file_sha(ROOT/'protocol/package_lock.json')
def normalized(x):return unicodedata.normalize('NFKC',x).strip().casefold()
def grouping(data):
    # Connected components retain both shared subject identities and shared triples.
    parent={x['item_id']:x['item_id'] for x in data}
    def find(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    seen={}
    for x in data:
        keys=[('subject',normalized(x.get('source_subject_uri') or x['subject'])),('triple',*(normalized(x[FIELDS[r]]) for r in ANSWER_ROLES))]
        for key in keys:
            if key in seen:parent[find(x['item_id'])]=find(seen[key])
            else:seen[key]=x['item_id']
    groups={}
    for x in data:groups.setdefault(find(x['item_id']),[]).append(x['item_id'])
    result={}
    for ids in groups.values():
        gid=sha(canonical(sorted(ids)))[:20]
        for k in ids:result[k]=gid
    return result

def folds(data,n=5):
    g=grouping(data);groups={}
    for x in data:groups.setdefault(g[x['item_id']],[]).append(x)
    relations=sorted({x['relation'] for x in data});loads=[{r:0 for r in relations} for _ in range(n)];total=[0]*n;out={}
    for gid,group in sorted(groups.items(),key=lambda kv:(-len(kv[1]),sha(canonical([cfg()['seed'],kv[0]])))):
        counts={r:sum(x['relation']==r for x in group) for r in relations}
        f=min(range(n),key=lambda i:(sum((loads[i][r]+counts[r])**2-loads[i][r]**2 for r in relations),total[i],i))
        for x in group:out[x['item_id']]=f
        for r in relations:loads[f][r]+=counts[r]
        total[f]+=len(group)
    return out

def document(x,condition):
    if condition=='NO_DOC':return ''
    role,strength=condition.split('_',1);a=x[FIELDS[role]];s=x['subject']
    claim={'developer':f'{s} was developed by {a}.','formation_location':f'{s} was formed in {a}.','original_language':f'The original language of {s} was {a}.','original_network':f'{s} originally aired on {a}.'}[x['relation']]
    if strength=='BARE':return 'A reference record states that '+claim[0].lower()+claim[1:] if x['relation']=='original_language' else 'A reference record states that '+claim
    return {'SKEPTICAL':'Source status: unverified and potentially inaccurate. Claim: ','PROVISIONAL':'Source status: provisionally credible, but not independently verified. Claim: '}[strength]+claim

def answer_prompt(x,condition,order):
    doc=document(x,condition);options='\n'.join(f'{lab}. {x[FIELDS[role]]}' for lab,role in zip('ABC',order))
    prefix=f'Document:\n{doc}\n\n' if doc else ''
    return prefix+f'Question:\n{x["question"]}\n\nOptions:\n{options}\n\nReply with A, B, or C only.\nAnswer:'

def report_prompt(doc,question,answer,cue,rotation):
    order=ROLES[rotation:]+ROLES[:rotation];options='\n'.join(f'{lab}. {ROLE_TEXT[role]}' for lab,role in zip(LABELS,order))
    return f'Document:\n{doc}\n\nQuestion:\n{question}\n\nRecorded answer:\n{answer}\n\nDisplayed ownership label:\n{CUES[cue]}\n\nWhich source most influenced the recorded answer?\n\nOptions:\n{options}\n\nReply with A, B, C, or D only.\nAnswer:'
def job(jid,prompt,**kw):return dict(job_id=jid,user_prompt=prompt,prompt_sha256=sha(prompt.encode()),**kw)
def mixed(jobs):return sorted(jobs,key=lambda j:sha(canonical([cfg()['seed'],j['job_id']])))
def answer_jobs(model,data=None):
    if model not in cfg()['target_models']:raise ValueError('Mistral is held out as a target; only the three known development targets are allowed.')
    result=[]
    for x in data or items():
        for condition in cfg()['conditions']:
            for p,order in enumerate(itertools.permutations(ANSWER_ROLES)):
                result.append(job(f'answer|{model}|{x["item_id"]}|{condition}|p{p}',answer_prompt(x,condition,order),model=model,item_id=x['item_id'],condition=condition,label_to_role=dict(zip('ABC',order))))
    return mixed(result)
def possible_report_prompts(data=None):
    # Outcome-blind exhaustive audit of every possible recorded answer, before scoring.
    for x in data or items():
        for strength,role,cue,rotation in itertools.product(cfg()['strengths'],ANSWER_ROLES,cfg()['cues'],range(4)):
            yield report_prompt(document(x,'C_'+strength),x['question'],x[FIELDS[role]],cue,rotation)
def candidate_text(x):
    values=[x[FIELDS[r]] for r in ANSWER_ROLES]
    return '\nUnordered candidate answers:\n'+'\n'.join(sorted(values,key=lambda v:sha(v.encode())))
def baseline_text(x,target,strength,answer):
    return f'Target model: {target}\nQuestion: {x["question"]}\nDocument: {document(x,"C_"+strength)}\nRecorded answer: {answer}'+candidate_text(x)
def report_jobs(reporter,targets):
    if reporter not in cfg()['reporter_models']:raise ValueError('Unknown reporter.')
    result=[]
    for t in targets:
        if t['target'] not in cfg()['target_models']:raise RuntimeError('Held-out target outcome encountered.')
        for cue,rotation in itertools.product(cfg()['cues'],range(4)):
            prompt=report_prompt(t['document'],t['question'],t['answer'],cue,rotation);order=ROLES[rotation:]+ROLES[:rotation]
            result.append(job(f'report|{reporter}|{t["target"]}|{t["item_id"]}|{t["strength"]}|{cue}|r{rotation}',prompt,model=reporter,target=t['target'],item_id=t['item_id'],strength=t['strength'],cue=cue,rotation=rotation,label_to_role=dict(zip(LABELS,order))))
    return mixed(result)

def recover(path,expected,lock_sha,repair_tail=True):
    path=Path(path);saved={}
    if not path.exists():return saved
    lines=path.read_bytes().splitlines(keepends=True);offset=0
    for i,line in enumerate(lines):
        if not line.endswith(b'\n'):
            if i!=len(lines)-1 or not repair_tail:raise RuntimeError('Uncommitted checkpoint tail.')
            tail=path.with_name(path.name+'.partial-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'));tail.write_bytes(line)
            with path.open('r+b') as f:f.truncate(offset);f.flush();os.fsync(f.fileno())
            break
        batch=json.loads(line)
        if batch['pre_outcome_freeze_sha256']!=lock_sha:raise RuntimeError('Checkpoint lock mismatch.')
        for row in batch['rows']:
            jid=row['job_id']
            if jid not in expected or row['prompt_sha256']!=expected[jid]['prompt_sha256'] or jid in saved:raise RuntimeError('Checkpoint job identity mismatch or duplicate.')
            if set(row['logp'])!=set(LABELS) or not all(math.isfinite(v) and v<=1e-5 for v in row['logp'].values()):raise RuntimeError('Invalid saved log probabilities.')
            saved[jid]=row
        offset+=len(line)
    return saved

def execute(jobs,path,lock_sha,backend,initial_batch=8,max_batches=None):
    expected={j['job_id']:j for j in jobs};saved=recover(path,expected,lock_sha);pending=[j for j in jobs if j['job_id'] not in saved];cursor=0;batches=0;b=initial_batch
    while cursor<len(pending) and (max_batches is None or batches<max_batches):
        group=pending[cursor:cursor+b];attempts=[]
        while True:
            try:
                scores=backend.score(group)
                if len(scores)!=len(group):raise RuntimeError('Backend returned wrong batch length.')
                if any(set(s)!=set(LABELS) or not all(math.isfinite(v) and v<=1e-5 for v in s.values()) for s in scores):raise FloatingPointError('Invalid model score; stopped without committing this batch.')
                break
            except MemoryError as e:
                attempts.append(dict(batch_size=len(group),error=str(e),utc=now()))
                sizes=[n for n in cfg()['fallback_batch_sizes'] if n<len(group)]
                if not sizes:raise RuntimeError('Singleton OOM; retain checkpoint and use sufficient hardware.') from e
                b=max(sizes);group=pending[cursor:cursor+b]
        rows=[dict(job_id=j['job_id'],prompt_sha256=j['prompt_sha256'],logp=s,batch_size=len(group),oom_attempts=attempts) for j,s in zip(group,scores)]
        batch=dict(pre_outcome_freeze_sha256=lock_sha,committed_utc=now(),runtime=backend.runtime,rows=rows)
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('ab') as f:f.write(canonical(batch)+b'\n');f.flush();os.fsync(f.fileno())
        cursor+=len(group);batches+=1
        if batches%25==0 or cursor==len(pending):print(f'Committed {len(saved)+cursor}/{len(jobs)} jobs',flush=True)
    return recover(path,expected,lock_sha)

def complete_scores(jobs,path,lock_sha):
    result=recover(path,{j['job_id']:j for j in jobs},lock_sha,repair_tail=False)
    if len(result)!=len(jobs):raise RuntimeError(f'Incomplete {Path(path).name}: {len(result)}/{len(jobs)}. Resume scoring first.')
    return result

def aggregate_targets(run,lock_sha):
    data={x['item_id']:x for x in items()};output=[];g=grouping(items());f=folds(items())
    for model in cfg()['target_models']:
        jobs=answer_jobs(model);scores=complete_scores(jobs,Path(run)/'scores'/f'answer_{model}.jsonl',lock_sha);support={}
        for j in jobs:
            key=(j['item_id'],j['condition']);d=support.setdefault(key,{r:[] for r in ANSWER_ROLES})
            for lab,role in j['label_to_role'].items():d[role].append(scores[j['job_id']]['logp'][lab])
        means={k:{r:sum(v)/len(v) for r,v in d.items()} for k,d in support.items()}
        for item_id,x in data.items():
            for strength in cfg()['strengths']:
                c=means[(item_id,'C_'+strength)];p=means[(item_id,'P_'+strength)];base=means[(item_id,'NO_DOC')];role=max(ANSWER_ROLES,key=lambda r:c[r])
                output.append(dict(target=model,item_id=item_id,relation=x['relation'],subject=x['subject'],group=g[item_id],fold=f[item_id],strength=strength,Y=(c['C']-c['P'])-(p['C']-p['P']),baseline_margin=base['P']-base['C'],answer_role=role,answer=x[FIELDS[role]],question=x['question'],document=document(x,'C_'+strength),text=baseline_text(x,model,strength,x[FIELDS[role]]),answer_support=c,reference_support=p,no_document_support=base))
    return sorted(output,key=lambda x:(x['item_id'],x['target'],x['strength']))

def aggregate_reports(run,targets,lock_sha):
    cases={(t['item_id'],t['target'],t['strength']):dict(t,reports={}) for t in targets}
    for reporter in cfg()['reporter_models']:
        jobs=report_jobs(reporter,targets);scores=complete_scores(jobs,Path(run)/'scores'/f'report_{reporter}.jsonl',lock_sha);parts={}
        for j in jobs:
            k=(j['item_id'],j['target'],j['strength'],j['cue']);d=parts.setdefault(k,{r:[] for r in ROLES})
            for lab,role in j['label_to_role'].items():d[role].append(scores[j['job_id']]['logp'][lab])
        for (item,target,strength,cue),d in parts.items():
            means={r:sum(v)/len(v) for r,v in d.items()};m=max(means.values());p={r:math.exp(v-m) for r,v in means.items()};total=sum(p.values());p={r:v/total for r,v in p.items()}
            cases[(item,target,strength)]['reports'].setdefault(reporter,{})[cue]=dict(log_support=means,q=p,alr=[means[r]-means['prior'] for r in ('document','both','uncertain')])
    return [cases[k] for k in sorted(cases)]

LEGACY=ROOT/'development/frozen_package'
FREEZE_ASSETS=('token_audit.json','embedding_catalog.json','embeddings.npy','encoder_manifest.json','frozen_models.joblib','training_manifest.json','observer_selection.json','old100_oof_errors.json')
def old_items():
    with (LEGACY/'inputs/items.csv').open(newline='') as f:return list(csv.DictReader(f))
def old_cases():
    data={x['item_id']:x for x in old_items()};cases=read_json(ROOT/'development/analysis_inputs.json')['cases']
    for c in cases:c['text']=baseline_text(data[c['item_id']],c['target'],c['strength'],c['answer'])
    return cases
def validate_inputs():
    data=read_json(ROOT/'inputs/items.json');review=read_json(ROOT/'inputs/review_audit.json')
    if len(data)!=40 or len(items())!=40 or len({x['item_id'] for x in data})!=40:raise RuntimeError('Exactly forty unique fresh items required.')
    if file_sha(ROOT/'inputs/reviewed_facts.xlsx')!=review['workbook_sha256']:raise RuntimeError('Review workbook changed.')
    decisions={r['item_id']:r for r in review['records']}
    if len(decisions)!=40 or any(decisions.get(x['item_id'],{}).get('decision')!='APPROVE' for x in data):raise RuntimeError('Forty submitted approvals required.')
    for x,c in zip(data,items()):
        for key in ('item_id','relation','subject','question',*FIELDS.values()):
            if x[key]!=c[key]:raise RuntimeError('JSON/CSV item mismatch.')
        if len({normalized(x[FIELDS[r]]) for r in ANSWER_ROLES})!=3:raise RuntimeError('Repeated alternatives.')
    old=old_items();names={normalized(x['subject']) for x in old};triples={tuple(sorted(normalized(x[FIELDS[r]]) for r in ANSWER_ROLES)) for x in old}
    for x in data:
        aliases={normalized(v) for v in [x['subject']]+x['subject_aliases']};triple=tuple(sorted(normalized(x[FIELDS[r]]) for r in ANSWER_ROLES))
        if names&aliases or triple in triples:raise RuntimeError('Subject/declared alias or full unordered triple overlap.')
        names.update(aliases);triples.add(triple)
    if {r:sum(x['relation']==r for x in data) for r in plan()['relations']}!={r:10 for r in plan()['relations']}:raise RuntimeError('Ten facts per relation required.')
    lock=read_json(LEGACY/'protocol/package_lock.json')
    for name,digest in lock['files'].items():
        if file_sha(LEGACY/name)!=digest:raise RuntimeError('Legacy package was modified: '+name)
    if file_sha(LEGACY/'inputs/items.csv')!=cfg()['input_provenance']['old100_items_sha256']:raise RuntimeError('Historical inputs differ.')
    return dict(approved=40,relations=4,declared_name_alias_and_unordered_triple_overlap=0)
def embedding_key(split,item,target,strength,answer):return '|'.join((split,item,target,strength,sha(answer.encode())))
def embedding_catalog():
    catalog=[]
    for c in old_cases():catalog.append(dict(key=embedding_key('old',c['item_id'],c['target'],c['strength'],c['answer']),text=c['text']))
    for x in items():
        for target,strength,role in itertools.product(cfg()['target_models'],cfg()['strengths'],ANSWER_ROLES):
            answer=x[FIELDS[role]];catalog.append(dict(key=embedding_key('fresh',x['item_id'],target,strength,answer),text=baseline_text(x,target,strength,answer)))
    if len(catalog)!=1980 or len({c['key'] for c in catalog})!=1980:raise RuntimeError('Embedding catalog identity mismatch.')
    return catalog
def has_scores(run):return any(p.stat().st_size for p in (Path(run)/'scores').glob('*.jsonl'))
def require_before_outcomes(run):
    if has_scores(run):raise RuntimeError('Fresh outcomes exist: preparation/refitting is forbidden. Verify and resume the existing freeze.')
def freeze_run(run,technical_fixture=False):
    run=Path(run);require_before_outcomes(run)
    if not technical_fixture:
        from hf_backend import environment
        environment()
        audit=read_json(run/'token_audit.json');encoder=read_json(run/'encoder_manifest.json');training=read_json(run/'training_manifest.json')
        if audit.get('status')!='OUTCOME_BLIND_EXHAUSTIVE_TOKEN_AUDIT_PASSED' or audit.get('package_lock_sha256')!=verify_package() or audit.get('fresh_outcomes')!=0:raise RuntimeError('A passing outcome-blind token audit is required.')
        if set(audit['models'])!=set(cfg()['reporter_models']):raise RuntimeError('All four tokenizer audits are required.')
        if encoder.get('embedding_sha256')!=file_sha(run/'embeddings.npy') or encoder.get('texts_sha256')!=sha(canonical(embedding_catalog())) or encoder.get('encoder')!=plan()['encoder']:raise RuntimeError('Pinned encoder audit differs.')
        if read_json(run/'embedding_catalog.json')!=embedding_catalog() or training.get('fresh_outcomes_used')!=0 or training.get('old100_cases')!=900:raise RuntimeError('Invalid pre-outcome training inputs.')
    assets={name:file_sha(run/name) for name in FREEZE_ASSETS}
    value=dict(status='PRE_OUTCOME_DEVELOPMENT_FREEZE',package_lock_sha256=verify_package(),created_utc=now(),technical_fixture_only=technical_fixture,assets=assets,environment=read_json(run/'training_manifest.json')['environment'],external_registration=False,fresh_outcomes_at_freeze=0)
    immutable_json(run/'pre_outcome_freeze.json',value);return verify_freeze(run,allow_fixture=technical_fixture)
def verify_freeze(run,allow_fixture=False):
    run=Path(run);value=read_json(run/'pre_outcome_freeze.json')
    if value['status']!='PRE_OUTCOME_DEVELOPMENT_FREEZE' or value['package_lock_sha256']!=verify_package():raise RuntimeError('Freeze/package identity differs.')
    if value['technical_fixture_only'] and not allow_fixture:raise RuntimeError('Synthetic technical fixture cannot authorize real model scoring.')
    if set(value['assets'])!=set(FREEZE_ASSETS):raise RuntimeError('Freeze asset inventory differs.')
    for name,digest in value['assets'].items():
        if file_sha(run/name)!=digest:raise RuntimeError('Frozen asset changed: '+name)
    return file_sha(run/'pre_outcome_freeze.json')
def load_embeddings(run,cases,split):
    import numpy as np
    run=Path(run);catalog=read_json(run/'embedding_catalog.json');array=np.load(run/'embeddings.npy',allow_pickle=False)
    if array.shape!=(1980,768) or not np.isfinite(array).all():raise RuntimeError('Invalid frozen embeddings.')
    positions={c['key']:i for i,c in enumerate(catalog)};indices=[]
    for c in cases:
        k=embedding_key(split,c['item_id'],c['target'],c['strength'],c['answer']);i=positions[k]
        if catalog[i]['text']!=c['text']:raise RuntimeError('Visible text differs from its pre-outcome encoding.')
        indices.append(i)
    return array[indices]
