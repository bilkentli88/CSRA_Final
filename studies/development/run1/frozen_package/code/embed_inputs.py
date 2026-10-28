"""Pinned frozen encoder, CPU only, no truncation and no supervised fine tuning."""
import argparse,os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
from common import *
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);args=ap.parse_args();run=args.run_dir;lock=verify_package()
    value=read_json(run/'analysis_inputs.json')
    if value['package_lock_sha256']!=lock:raise RuntimeError('Analysis input lock mismatch.')
    spec=plan()['encoder'];texts=[c['text'] for c in value['cases']];identity=dict(package_lock_sha256=lock,analysis_inputs_sha256=file_sha(run/'analysis_inputs.json'),encoder=spec,texts_sha256=sha(canonical(texts)))
    path=run/'analysis/embeddings.npy';manifest=run/'analysis/encoder_manifest.json'
    if path.exists() and manifest.exists():
        old=read_json(manifest)
        if all(old.get(k)==v for k,v in identity.items()) and old['embedding_sha256']==file_sha(path):print('Pinned embeddings already complete.');return
        raise RuntimeError('Existing embeddings belong to different inputs or encoder.')
    from hf_backend import environment
    environment()
    import numpy as np,torch
    from transformers import AutoTokenizer,AutoModel
    torch.set_num_threads(2);torch.manual_seed(cfg()['seed'])
    tok=AutoTokenizer.from_pretrained(spec['model_id'],revision=spec['revision'],use_fast=True);model=AutoModel.from_pretrained(spec['model_id'],revision=spec['revision'],dtype=torch.float32);model.eval();model.to('cpu');output=[]
    for start in range(0,len(texts),32):
        batch=tok(texts[start:start+32],padding=True,truncation=False,return_tensors='pt')
        if batch['input_ids'].shape[1]>spec['max_tokens']:raise RuntimeError('Overlength encoder input; never silently truncate.')
        with torch.inference_mode():
            hidden=model(**batch).last_hidden_state;mask=batch['attention_mask'].unsqueeze(-1).float();mean=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1);embedding=torch.nn.functional.normalize(mean,p=2,dim=1)
        output.append(embedding.cpu().numpy())
        if start%160==0:print(f'CPU embeddings {min(start+32,len(texts))}/{len(texts)}',flush=True)
    arr=np.concatenate(output)
    if not np.isfinite(arr).all() or arr.shape!=(len(texts),768):raise RuntimeError('Invalid frozen embeddings.')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as f:np.save(f,arr)
    path.with_suffix('.tmp').replace(path);atomic_json(manifest,dict(identity,embedding_sha256=file_sha(path),device='cpu',normalized=True,rows=len(texts),dimensions=768,supervised_fine_tuning=False))
if __name__=='__main__':main()
