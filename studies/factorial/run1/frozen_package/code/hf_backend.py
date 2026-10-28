"""Pinned CUDA finite-label scorer; GPU qualification is required in Colab."""
import gc,importlib.metadata,inspect,time,math,sys
from pathlib import Path
from pilot_common import ROOT,read_json,config,now

def qualify_labels(tokenizer,cfg,model_key):
    ids={lab:tokenizer.encode(cfg['expected_label_prefix']+lab,add_special_tokens=False) for lab in 'ABCD'}
    expected=cfg['expected_label_ids'][model_key]
    if any(ids[lab]!=[expected[lab]] for lab in 'ABCD'):raise RuntimeError(f'Frozen single-token label qualification failed: {ids}')
    return expected

class HFBackend:
    def __init__(self,model_key,root=ROOT):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
        self.torch=torch;self.cfg=config(root);self.model_key=model_key
        versions={k:importlib.metadata.version(k) for k in self.cfg['scoring_versions']}
        for k,v in self.cfg['scoring_versions'].items():
            if versions[k]!=v:raise RuntimeError(f'Pinned scoring environment mismatch: {k} {versions[k]} != {v}')
        if not torch.cuda.is_available():raise RuntimeError('The frozen NF4 scorer requires a CUDA GPU.')
        lock=read_json(Path(root)/'protocol/model_lock.json');self.spec=lock['models'][model_key]
        self.prompt_bundle=read_json(Path(root)/'protocol/prompts.json')
        torch.manual_seed(20261016);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.tokenizer=AutoTokenizer.from_pretrained(self.spec['tokenizer_id'],revision=self.spec['tokenizer_revision'],use_fast=True)
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        self.tokenizer.padding_side='left'
        self.label_ids=qualify_labels(self.tokenizer,self.cfg,model_key)
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32)
        self.model=AutoModelForCausalLM.from_pretrained(self.spec['model_id'],revision=self.spec['revision'],device_map='auto',quantization_config=quant,dtype=torch.float32,low_cpu_mem_usage=True)
        self.model.eval()
        placement=getattr(self.model,'hf_device_map',{})
        if any(str(v) in ('cpu','disk') for v in placement.values()):raise RuntimeError('CPU/disk offload is not an accepted scoring configuration.')
        parameters=inspect.signature(self.model.forward).parameters
        self.extra={'logits_to_keep':1} if 'logits_to_keep' in parameters else ({'num_logits_to_keep':1} if 'num_logits_to_keep' in parameters else {})
        self.runtime=dict(started_utc=now(),packages=versions,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,model_revision=self.spec['revision'],tokenizer_revision=self.spec['tokenizer_revision'],label_ids=self.label_ids,precision='NF4_double_quant_FP32_compute',tf32=False,prompt_add_special_tokens=True,forward_optimization=self.extra)
        optional={}
        for name in ('torchvision','torchaudio'):
            try:optional[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:optional[name]=None
        self.runtime.update(python=sys.version,optional_extensions=optional)

    def render(self,text):
        return self.tokenizer.apply_chat_template([{'role':'system','content':self.prompt_bundle['system_text']},{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,**self.spec.get('chat_template_kwargs',{}))

    def score(self,batch):
        torch=self.torch
        try:
            rendered=[self.render(j['user_prompt']) for j in batch]
            # Explicitly retain Study 1's render-then-tokenize(True) convention.
            # Preflight records how it differs from tokenize(False), if at all.
            inputs=self.tokenizer(rendered,return_tensors='pt',padding=True,truncation=False,add_special_tokens=self.cfg['prompt_tokenization_add_special_tokens'])
            inputs={k:v.to(self.model.device) for k,v in inputs.items()}
            with torch.inference_mode():
                out=self.model(**inputs,use_cache=False,**self.extra)
                logits=out.logits[:,-1,:].float()
                logp=torch.log_softmax(logits,dim=-1)[:,[self.label_ids[x] for x in 'ABCD']].cpu().tolist()
            del inputs,out,logits
            return [dict(zip('ABCD',v)) for v in logp]
        except torch.cuda.OutOfMemoryError as exc:
            gc.collect();torch.cuda.empty_cache();raise MemoryError(str(exc)) from exc

    def preflight(self):
        torch=self.torch
        texts=[{'user_prompt':'A fictional record says that the blue key opens the silver box. Which source supports the answer?\nA. the supplied document\nB. prior knowledge\nC. both\nD. uncertain\nAnswer:'},{'user_prompt':'Fictional note: on the invented planet Neral, the first moon is named Tovin. The note is the only information provided about Neral. Which source supports the name?\nA. the supplied document\nB. prior knowledge\nC. both\nD. uncertain\nAnswer:'}]
        singles=[self.score([j])[0] for j in texts]
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        paired=self.score(texts);torch.cuda.synchronize();elapsed=time.perf_counter()-start
        if not all(math.isfinite(float(x[k])) and float(x[k])<=1e-5 for x in singles+paired for k in 'ABCD'):raise RuntimeError('Nonfinite or invalid synthetic qualification log probabilities.')
        discrepancy=max(abs(a[k]-b[k]) for a,b in zip(singles,paired) for k in 'ABCD')
        if discrepancy>.05:raise RuntimeError(f'Padding/batching qualification discrepancy exceeds 0.05 nats: {discrepancy}')
        comparisons=[]
        for j in texts:
            rendered=self.render(j['user_prompt'])
            a=self.tokenizer.encode(rendered,add_special_tokens=True);b=self.tokenizer.encode(rendered,add_special_tokens=False)
            comparisons.append(dict(legacy_token_count=len(a),without_extra_special_token_count=len(b),identical=a==b))
        return dict(runtime=self.runtime,status='TECHNICAL_SYNTHETIC_PREFLIGHT_ONLY',max_single_vs_batch_logp_error=discrepancy,batch_two_seconds=elapsed,seconds_per_logical_job=elapsed/2,peak_allocated_bytes=torch.cuda.max_memory_allocated(),tokenization_comparison=comparisons,qualification_tolerance_nats=.05,benchmark_scope='Two synthetic prompts only; not a reliable full-pilot time or cost estimate.')
