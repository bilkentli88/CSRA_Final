"""Verify the release inventory and preserved source bytes; standard library only."""
from pathlib import Path
import hashlib,json,sys

ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    inventory=ROOT/'SHA256SUMS.txt'
    if not inventory.is_file():raise RuntimeError('Missing root SHA256SUMS.txt')
    count=0
    for line in inventory.read_text().splitlines():
        expected,name=line.split('  ',1);path=ROOT/name
        if not path.is_file() or digest(path)!=expected:raise RuntimeError('Release checksum mismatch: '+name)
        count+=1
    preserved=json.loads((ROOT/'provenance/PRESERVED_SOURCE_FILES.json').read_text())
    for stage,files in preserved.items():
        for name,expected in files.items():
            if digest(ROOT/name)!=expected:raise RuntimeError('Changed historical source: '+name)
    locks={}
    for study in ('factorial','development','fresh40'):
        base=ROOT/'studies'/study/'run1/frozen_package'
        lock=json.loads((base/'protocol/package_lock.json').read_text())
        for name,expected in lock['files'].items():
            if digest(base/name)!=expected:raise RuntimeError('Frozen package mismatch: '+study+'/'+name)
        locks[study]=len(lock['files'])
    for binding in json.loads((ROOT/'provenance/MANUSCRIPT_BINDINGS.json').read_text()):
        if digest(ROOT/binding['manuscript_copy'])!=digest(ROOT/binding['experimental_source']):
            raise RuntimeError('Manuscript copy differs: '+binding['manuscript_copy'])
    report={'status':'PASSED','release_files_checked':count,'preserved_source_files':{k:len(v) for k,v in preserved.items()},'frozen_package_locked_files':locks,'scope':'Byte integrity, not an external timestamp or proof of experimental independence.'}
    out=ROOT/'reproduced_results/integrity.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
