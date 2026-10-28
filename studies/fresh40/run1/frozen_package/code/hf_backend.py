"""Use qualified continuation IDs, FP32 inference and outcome-blind token audits."""
import gc,importlib.metadata,inspect,math,sys,time
from pathlib import Path
from core import *
def environment():
    expected=plan()['scoring_versions'];actual={k:importlib.metadata.version(k) for k in expected}
    if actual!=expected:raise RuntimeError(f'Scoring environment differs from pins: {actual}')
    for name in ('torchvision','torchaudio'):
        try:importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:continue
        raise RuntimeError('Remove unused torchvision and torchaudio as in the notebook installation cell.')
    return actual

def tokenizer_for(model):
    from transformers import AutoTokenizer
    spec=plan()['models'][model];tok=AutoTokenizer.from_pretrained(spec['tokenizer_id'],revision=spec['tokenizer_revision'],use_fast=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tok.padding_side='left';return tok

def render(tok,model,text):
    return tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,**plan()['models'][model].get('chat_template_kwargs',{}))

def audit(run,lock_sha):
    environment()
    existing=Path(run)/'token_audit.json'
    if existing.exists():
        value=read_json(existing)
        if value.get('package_lock_sha256')!=lock_sha or value.get('status')!='OUTCOME_BLIND_EXHAUSTIVE_TOKEN_AUDIT_PASSED':raise RuntimeError('Existing audit differs from this package.')
        print('Outcome-blind token audit already complete.',flush=True);return value
    out={};common_answers=answer_jobs(cfg()['target_models'][0]);alltexts=sorted(set([j['user_prompt'] for j in common_answers]+list(possible_report_prompts())))
    profiles=read_json(ROOT/'protocol/qualified_hardware.json')
    for model in cfg()['reporter_models']:
        tok=tokenizer_for(model);qualification=profiles[model+'_fp32']['label_qualification'];prefix=qualification['label_prefix'];label_ids=qualification['label_ids'];maximum=0
        for lab in LABELS:
            if tok.encode(prefix+lab,add_special_tokens=False)!=[label_ids[lab]]:raise RuntimeError('Qualified label IDs changed: '+model)
        for start in range(0,len(alltexts),64):
            texts=[render(tok,model,t) for t in alltexts[start:start+64]];ids=tok(texts,add_special_tokens=False,truncation=False)['input_ids']
            maximum=max(maximum,max(map(len,ids)))
            if maximum>cfg()['max_prompt_tokens']:raise RuntimeError(f'{model}: actual possible prompt exceeds {cfg()["max_prompt_tokens"]} tokens ({maximum}). Stop before outcomes; revise and re-freeze, never truncate.')
            for lab in LABELS:
                full=tok([t+prefix+lab for t in texts],add_special_tokens=False,truncation=False)['input_ids']
                if any(b!=a+[label_ids[lab]] for a,b in zip(ids,full)):raise RuntimeError(f'{model}: complete-continuation qualification failed; stop before outcomes.')
        out[model]=dict(maximum_prompt_tokens=maximum,unique_possible_prompts=len(alltexts),label_qualification=qualification,model_revision=plan()['models'][model]['revision'])
        print('Token/continuation audit passed:',model,maximum,flush=True)
    from transformers import AutoTokenizer
    enc=plan()['encoder'];tok=AutoTokenizer.from_pretrained(enc['model_id'],revision=enc['revision'],use_fast=True)
    texts=[v['text'] for v in embedding_catalog()]
    lengths=[]
    for start in range(0,len(texts),128):lengths.extend(map(len,tok(texts[start:start+128],add_special_tokens=True,truncation=False)['input_ids']))
    if max(lengths)>enc['max_tokens']:raise RuntimeError(f'Encoder input exceeds {enc["max_tokens"]} tokens; resolve before outcomes.')
    result=dict(status='OUTCOME_BLIND_EXHAUSTIVE_TOKEN_AUDIT_PASSED',package_lock_sha256=lock_sha,models=out,encoder_maximum_tokens=max(lengths),encoder_revision=enc['revision'],fresh_outcomes=0)
    immutable_json(Path(run)/'token_audit.json',result);return result

class HFBackend:
    def __init__(self,model,run,lock_sha):
        freeze_sha=verify_freeze(run,allow_fixture=False)
        import torch
        from transformers import AutoModelForCausalLM
        actual=environment();check=read_json(Path(run)/'token_audit.json')
        if check['package_lock_sha256']!=lock_sha or check['status']!='OUTCOME_BLIND_EXHAUSTIVE_TOKEN_AUDIT_PASSED':raise RuntimeError('Run the outcome-blind token audit first.')
        if model not in cfg()['reporter_models']:raise ValueError('Unknown model.')
        if not torch.cuda.is_available():raise RuntimeError('Select a CUDA GPU runtime.')
        torch.manual_seed(cfg()['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.torch=torch;self.model_key=model;self.tokenizer=tokenizer_for(model);self.qualification=check['models'][model]['label_qualification'];self.ids=self.qualification['label_ids']
        spec=plan()['models'][model];start=time.perf_counter()
        self.model=AutoModelForCausalLM.from_pretrained(spec['model_id'],revision=spec['revision'],device_map={'':0},dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='sdpa');self.model.eval()
        if any(str(v) in ('cpu','disk') for v in getattr(self.model,'hf_device_map',{}).values()):raise RuntimeError('CPU/disk offload is not allowed.')
        if any(p.is_floating_point() and p.dtype!=torch.float32 for p in self.model.parameters()):raise RuntimeError('Unexpected non-FP32 floating parameters.')
        params=inspect.signature(self.model.forward).parameters;self.extra={'logits_to_keep':1} if 'logits_to_keep' in params else ({'num_logits_to_keep':1} if 'num_logits_to_keep' in params else {})
        self.runtime=dict(pre_outcome_freeze_sha256=freeze_sha,started_utc=now(),model=model,model_id=spec['model_id'],model_revision=spec['revision'],tokenizer_revision=spec['tokenizer_revision'],gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,python=sys.version,packages=actual,precision='unquantized_FP32',label_qualification=self.qualification,token_audit_sha256=file_sha(Path(run)/'token_audit.json'),prompt_add_special_tokens=False,tf32=False,attention_implementation='sdpa',forward_optimization=self.extra,model_load_seconds=time.perf_counter()-start)
        # Fictional unequal-length prompts check the new scorer without study outcomes.
        fictional='A fictional archive says the invented city Neral was founded by Tovin.'
        prompts=[report_prompt(fictional,'Who founded the invented city Neral?','Tovin','OTHER_PRESENTED',0),report_prompt(fictional+' '+'An unrelated fictional entry records a blue box. '*18,'Who founded the invented city Neral?','Tovin','SELF_PRESENTED',1)]
        for text in prompts:
            rendered=render(self.tokenizer,model,text);prefix_ids=self.tokenizer.encode(rendered,add_special_tokens=False)
            for lab in LABELS:
                if self.tokenizer.encode(rendered+self.qualification['label_prefix']+lab,add_special_tokens=False)!=prefix_ids+[self.ids[lab]]:raise RuntimeError('Fictional startup continuation qualification failed.')
        batch=[{'user_prompt':p} for p in prompts];a=[self.score([j])[0] for j in batch];b=self.score(batch)
        error=max(abs(x[l]-y[l]) for x,y in zip(a,b) for l in LABELS)
        if error>.05:raise RuntimeError('Single-versus-pair discrepancy exceeds qualification tolerance.')
        self.runtime['startup_single_pair_discrepancy_nats']=error
    def score(self,jobs):
        torch=self.torch
        try:
            started=time.perf_counter()
            texts=[render(self.tokenizer,self.model_key,j['user_prompt']) for j in jobs]
            inputs=self.tokenizer(texts,padding=True,truncation=False,add_special_tokens=False,return_tensors='pt')
            if inputs['input_ids'].shape[1]>cfg()['max_prompt_tokens']:raise RuntimeError('Unexpected overlength prompt; no truncation allowed.')
            inputs={k:v.to(self.model.device) for k,v in inputs.items()}
            with torch.inference_mode():
                out=self.model(**inputs,use_cache=False,**self.extra);logp=torch.log_softmax(out.logits[:,-1,:].float(),-1)[:,[self.ids[l] for l in LABELS]].cpu().tolist()
            torch.cuda.synchronize()
            if hasattr(self,'runtime'):self.runtime['last_forward']=dict(seconds=time.perf_counter()-started,unpadded_input_tokens=int(inputs['attention_mask'].sum().item()),padded_input_tokens=int(inputs['input_ids'].numel()),jobs=len(jobs))
            del inputs,out
            return [dict(zip(LABELS,row)) for row in logp]
        except torch.cuda.OutOfMemoryError as e:
            gc.collect();torch.cuda.empty_cache();raise MemoryError(str(e)) from e
