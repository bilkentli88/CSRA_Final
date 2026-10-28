"""Early import/CUDA check in a fresh process, without loading model weights."""
import importlib.util,importlib.metadata,json,sys
from pilot_common import config,verify_lock

def main():
    verify_lock();cfg=config()
    for name in ('torchvision','torchaudio'):
        if importlib.util.find_spec(name) is not None:
            raise RuntimeError(f'Remove unused {name} using this interpreter before the text-only pilot: {sys.executable} -m pip uninstall -y torchvision torchaudio')
    wanted={**cfg['scoring_versions'],**cfg['analysis_versions']}
    versions={k:importlib.metadata.version(k) for k in wanted}
    for k,v in wanted.items():
        if versions[k]!=v:raise RuntimeError(f'Environment mismatch: {k} {versions[k]} != {v}')
    import torch
    from transformers import Qwen3ForCausalLM
    if not torch.cuda.is_available():raise RuntimeError('Select a CUDA GPU runtime before preflight.')
    print(json.dumps(dict(status='IMPORT_AND_CUDA_CHECK_ONLY',python=sys.version,packages=versions,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,model_weights_loaded=False,new_pilot_reports=0),indent=2))
if __name__=='__main__':main()
