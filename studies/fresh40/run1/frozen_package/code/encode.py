"""CPU MPNet encoding of old900 and all1080 possible fresh visible texts."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
from core import *
def encode(run):
    run=Path(run);require_before_outcomes(run);catalog=embedding_catalog();spec=plan()['encoder']
    identity=dict(package_lock_sha256=verify_package(),encoder=spec,texts_sha256=sha(canonical(catalog)))
    path=run/'embeddings.npy';manifest=run/'encoder_manifest.json'
    if path.exists() and manifest.exists():
        m=read_json(manifest)
        if all(m.get(k)==v for k,v in identity.items()) and m['embedding_sha256']==file_sha(path) and read_json(run/'embedding_catalog.json')==catalog:return
        raise RuntimeError('Existing embedding files differ; use a new run directory.')
    import numpy as np,torch
    from transformers import AutoTokenizer,AutoModel
    from hf_backend import environment
    environment();torch.set_num_threads(2);torch.manual_seed(cfg()['seed'])
    tok=AutoTokenizer.from_pretrained(spec['model_id'],revision=spec['revision'],use_fast=True)
    model=AutoModel.from_pretrained(spec['model_id'],revision=spec['revision'],dtype=torch.float32);model.eval();model.to('cpu');output=[]
    for start in range(0,len(catalog),32):
        batch=tok([c['text'] for c in catalog[start:start+32]],padding=True,truncation=False,return_tensors='pt')
        if batch['input_ids'].shape[1]>spec['max_tokens']:raise RuntimeError('Overlength encoder input; do not truncate.')
        with torch.inference_mode():
            hidden=model(**batch).last_hidden_state;mask=batch['attention_mask'].unsqueeze(-1).float();mean=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1);vec=torch.nn.functional.normalize(mean,p=2,dim=1)
        output.append(vec.cpu().numpy())
        if start%160==0:print(f'CPU encoding {min(start+32,len(catalog))}/{len(catalog)}',flush=True)
    arr=np.concatenate(output).astype('float32')
    if arr.shape!=(1980,768) or not np.isfinite(arr).all():raise RuntimeError('Invalid embeddings.')
    with path.with_suffix('.tmp').open('wb') as f:np.save(f,arr)
    path.with_suffix('.tmp').replace(path)
    atomic_json(run/'embedding_catalog.json',catalog);atomic_json(manifest,dict(identity,embedding_sha256=file_sha(path),device='cpu',normalized=True,supervised_fine_tuning=False))
