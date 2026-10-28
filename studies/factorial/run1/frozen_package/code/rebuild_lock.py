"""Explicit before-new-reports rebuild only; retain earlier packages and amendments."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def rebuild(root=ROOT):
    root=Path(root)
    files=[]
    for directory in ('code','inputs','protocol','tests'):
        files.extend(p for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('package_lock.json','registration_template.json'))
    files.extend(root/n for n in ('README.md','LICENSE','requirements-analysis.txt','requirements-scoring.txt','CSRA_Study2_Pilot.ipynb','docs/RESEARCH_GAP_CHECK.md','docs/INPUT_PROVENANCE.md'))
    lock=dict(protocol_id='CSRA-STUDY2-PILOT-1',status='EXPLORATORY_NOT_PREREGISTERED',package_revision='EXPLORATORY_V2_20261019',hash_algorithm='sha256',files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))})
    path=root/'protocol/package_lock.json';path.write_text(json.dumps(lock,indent=2,sort_keys=True)+'\n')
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    template=dict(package_lock_sha256=sha,registry_url='',registered_utc='',author_attests_before_new_reports=False)
    (root/'protocol/registration_template.json').write_text(json.dumps(template,indent=2)+'\n')
    print(json.dumps(dict(package_lock_sha256=sha,locked_files=len(lock['files']),study_mode='exploratory',externally_registered=False),indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--before-new-reports',action='store_true',required=True);ap.parse_args();rebuild()
