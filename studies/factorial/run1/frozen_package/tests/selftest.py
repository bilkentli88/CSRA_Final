"""No GPU, downloads, or real model outputs; tests all primary pipeline stages."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,tempfile,unittest,json,math,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from pilot_common import *
from analyze_pilot import load_data,analyze,score_linear
from hf_backend import qualify_labels
from execution_context import freeze_run,check_freeze,prepare_header,validate_header,check_preflight
from unittest.mock import patch
import types,contextlib,io

class SyntheticBackend:
    """Target-derived synthetic logits for code validation, never model evidence."""
    def __init__(self,fail_over=None):
        self.targets={(r['model_key'],r['item_id'],r['condition']):r for r in rows(ROOT/'inputs/answer_targets.csv')}
        self.fail_over=fail_over;self.calls=0
    def logits(self,j):
        t=self.targets[(j['model_key'],j['item_id'],j['condition'])]
        influence=float(t['CIE'])/30
        own=j['ownership']=='OWN';shown=j['visibility']=='shown';hedged=j['hedging']=='hedged'
        return dict(document=.1*influence+(.2*influence if own and shown else 0),prior=0.,both=-.5+.3*own+.4*hedged+.2*shown,uncertain=-1.)
    def score(self,batch):
        self.calls+=1
        if self.fail_over is not None and len(batch)>self.fail_over:raise MemoryError('synthetic OOM')
        result=[]
        for j in batch:
            z=self.logits(j);a=np.array([z[r] for r in ROLES]);norm=np.log(np.exp(a-a.max()).sum())+a.max()
            result.append({lab:float(z[role]-norm+math.log(.8)) for lab,role in j['label_to_role'].items()})
        return result

class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sha=verify_lock();cls.jobs=jobs();cls.backend=SyntheticBackend()
    def test_design_and_visibility(self):
        self.assertEqual(len(self.jobs),28800)
        self.assertEqual(len({j['job_id'] for j in self.jobs}),28800)
        bundle=read_json(ROOT/'protocol/prompts.json')
        for c in bundle['cells']:
            prompt=c['user_template'].format(document='DOC',question='QUESTION',answer='SENTINEL_45cf',options='OPTIONS')
            self.assertEqual(prompt.count('SENTINEL_45cf'),int(c['visibility']=='shown'))
        for model in config()['models']:
            by=[j for j in self.jobs if j['model_key']==model]
            self.assertEqual(len(by),9600)
    def test_folds_separate_triples(self):
        items=rows(ROOT/'inputs/items.csv');folds=read_json(ROOT/'protocol/folds.json');group={}
        self.assertEqual(folds,make_folds(items,config()['fold_seed']))
        for item in items:group.setdefault(triple_key(item),set()).add(folds[item['item_id']])
        self.assertEqual(len(group),57);self.assertTrue(all(len(g)==1 for g in group.values()))
    def test_label_qualification(self):
        class Tokenizer:
            def encode(self,text,add_special_tokens=False):return [ord(text)-33]
        self.assertEqual(qualify_labels(Tokenizer(),config(),'qwen'),{'A':32,'B':33,'C':34,'D':35})
        class WrongTokenizer:
            def encode(self,text,add_special_tokens=False):return [0,1]
        with self.assertRaises(RuntimeError):qualify_labels(WrongTokenizer(),config(),'qwen')
    def test_registered_mode_receipt_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'receipt.json';atomic_json(path,dict(registry_url='',package_lock_sha256=self.sha))
            with self.assertRaises(RuntimeError):check_registration(path,self.sha)
            atomic_json(path,dict(registry_url='https://example.invalid/fake',package_lock_sha256=self.sha,registered_utc='2026-01-01T00:00:00+00:00',author_attests_before_new_reports=True))
            with self.assertRaises(RuntimeError):check_registration(path,self.sha)
    def test_resume_oom_and_partial_tail(self):
        js=self.jobs[:37];backend=SyntheticBackend(fail_over=2)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.jsonl'
            first=execute(js,path,self.sha,backend,{'kind':'SYNTHETIC_TEST_ONLY'},max_batches=2)
            self.assertEqual(len(first),4)
            with path.open('ab') as h:h.write(b'{"unfinished":')
            final=execute(js,path,self.sha,backend,{'kind':'SYNTHETIC_TEST_ONLY'})
            self.assertEqual(len(final),37);self.assertEqual(len(list(Path(folder).glob('*.partial-*'))),1)
            calls=backend.calls;execute(js,path,self.sha,backend,{'kind':'SYNTHETIC_TEST_ONLY'})
            self.assertEqual(backend.calls,calls)
            expected={j['job_id']:dict(j) for j in js};expected[js[0]['job_id']]['prompt_sha256']='wrong'
            with self.assertRaises(RuntimeError):recover(path,expected,self.sha)
            class OneBadJob(SyntheticBackend):
                def score(self,batch):
                    if any(j['job_id']==js[0]['job_id'] for j in batch):raise MemoryError('synthetic terminal OOM')
                    return super().score(batch)
            isolated=execute(js[:3],Path(folder)/'terminal.jsonl',self.sha,OneBadJob(),{'kind':'SYNTHETIC_TEST_ONLY'})
            self.assertEqual(isolated[js[0]['job_id']]['status'],'terminal_error')
            self.assertEqual(sum(r['status']=='ok' for r in isolated.values()),2)
    def test_rotation_aggregation_and_primary_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            for model in config()['models']:
                js=[j for j in self.jobs if j['model_key']==model]
                out=[dict(job_id=j['job_id'],prompt_sha256=j['prompt_sha256'],status='ok',logp=s) for j,s in zip(js,self.backend.score(js))]
                atomic_json(folder/f'{model}_run_header.json',dict(data_kind='SYNTHETIC_TEST_ONLY',package_lock_sha256=self.sha,model=model))
                append_batch(folder/f'{model}_checkpoint.jsonl',out,self.sha,{'kind':'SYNTHETIC_TEST_ONLY'})
            with self.assertRaises(RuntimeError):load_data(folder,synthetic=False)
            agg,selected,bad=load_data(folder,synthetic=True)
            self.assertEqual(len(agg),7200);self.assertEqual(len(bad),0)
            # Compare semantic probabilities with direct eight-cell construction.
            first=self.jobs[0];g=agg[(agg.model_key==first['model_key'])&(agg.item_id==first['item_id'])&(agg.condition==first['condition'])&(agg.cell==first['cell'])].iloc[0]
            z=self.backend.logits(first);expected=np.exp(z['both'])/sum(np.exp(v) for v in z.values())
            self.assertAlmostEqual(g.q_both,expected,places=12)
            result=analyze(agg,selected,reps=39,secondary=False)
            self.assertEqual(len(result['primary']),4);self.assertTrue(np.isfinite(result['primary'].estimate).all())
            self.assertEqual(len(result['interactions']),4)
            self.assertEqual(len(result['per_model_category_contrasts']),9)
            self.assertTrue(np.allclose(result['category_probabilities'][['q_'+r for r in ROLES]].sum(axis=1),1))
            associations=result['answer_association'].dropna(subset=['cramers_v'])
            self.assertTrue(associations.cramers_v.between(0,1+1e-10).all())
            # Independent direct marginal means, rather than reusing analyzer code.
            for factor,high,low in [('ownership','OWN','OTHER'),('hedging','hedged','certain'),('visibility','shown','hidden')]:
                direct=agg.loc[agg[factor]==high,'q_both'].mean()-agg.loc[agg[factor]==low,'q_both'].mean()
                measured=result['primary'].set_index('contrast').loc[factor,'estimate']
                self.assertAlmostEqual(direct,measured,places=12)
            fits=result['predictive_fits'];direct=[]
            for (m,h),group in fits.groupby(['model','hedging']):
                direct.append(group.set_index('visibility').theta_OWN['shown']-group.set_index('visibility').theta_OWN['hidden'])
            self.assertAlmostEqual(np.mean(direct),result['primary'].set_index('contrast').loc['predictive_visibility','estimate'],places=12)
            # Secondary nonlinear smoke check on the same full synthetic fixture.
            secondary=analyze(agg,selected,reps=3,secondary=True,secondary_reps=2)
            self.assertEqual(set(secondary['predictive_fits'].learner),{'Ridge','GradientBoosting'})
            # A terminal failure removes the item jointly, never just its cell.
            model='qwen';path=folder/f'{model}_checkpoint.jsonl';batch=json.loads(path.read_text());batch['rows'][0]={'job_id':batch['rows'][0]['job_id'],'prompt_sha256':batch['rows'][0]['prompt_sha256'],'status':'terminal_error'}
            path.write_bytes(canonical(batch)+b'\n')
            remaining,items,bad=load_data(folder,synthetic=True)
            self.assertEqual(len(items),99);self.assertEqual(len(remaining),99*72);self.assertEqual(len(bad),1)

    def test_internal_freeze_and_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)/'run'
            value=freeze_run(folder)
            self.assertFalse(value['externally_registered']);self.assertFalse(value['independent_timestamp'])
            self.assertEqual(value,freeze_run(folder))
            self.assertEqual(value,check_freeze(folder))
            (folder/'frozen_package/inputs/items.csv').write_text('tampered test snapshot')
            with self.assertRaises(RuntimeError):check_freeze(folder)
            used=Path(td)/'used';used.mkdir();(used/'qwen_checkpoint.jsonl').touch()
            with self.assertRaises(RuntimeError):freeze_run(used)

    def test_explicit_mode_context(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)/'run'
            with self.assertRaises(FileNotFoundError):prepare_header(folder,'qwen','exploratory')
            with self.assertRaises(RuntimeError):prepare_header(folder,'qwen','registered')
            freeze_run(folder)
            header=prepare_header(folder,'qwen','exploratory')
            atomic_json(folder/'qwen_run_header.json',header)
            self.assertEqual(validate_header(folder,'qwen'),'exploratory')
            with self.assertRaises(RuntimeError):prepare_header(folder,'granite','ambiguous')
            with self.assertRaises(RuntimeError):prepare_header(folder,'granite','exploratory',Path(td)/'unwanted_receipt.json')
            atomic_json(folder/'phi_run_header.json',dict(study_mode='registered',package_lock_sha256=self.sha))
            with self.assertRaisesRegex(RuntimeError,'mix'):prepare_header(folder,'granite','exploratory')

    def test_exploratory_real_loader_context(self):
        # Entirely synthetic scores in a temporary directory; test production
        # context checks without any real inference or exported findings.
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);freeze_run(folder)
            for model in config()['models']:
                js=[j for j in self.jobs if j['model_key']==model]
                out=[dict(job_id=j['job_id'],prompt_sha256=j['prompt_sha256'],status='ok',logp=v) for j,v in zip(js,self.backend.score(js))]
                atomic_json(folder/f'{model}_run_header.json',prepare_header(folder,model,'exploratory'))
                append_batch(folder/f'{model}_checkpoint.jsonl',out,self.sha,{'kind':'SYNTHETIC_TEST_ONLY'})
            agg,items,bad=load_data(folder)
            self.assertEqual((len(agg),len(items),len(bad)),(7200,100,0))
            receipt=dict(registry_url='https://example.org/synthetic-unit-test',registered_utc='2026-01-01T00:00:00Z',package_lock_sha256=self.sha,author_attests_before_new_reports=True)
            atomic_json(folder/'granite_registration.json',receipt)
            original=read_json(folder/'granite_run_header.json')
            atomic_json(folder/'granite_run_header.json',dict(data_kind='REAL_PILOT_REPORTS',model='granite',study_mode='registered',package_lock_sha256=self.sha,registration=receipt))
            with self.assertRaisesRegex(RuntimeError,'pooled'):load_data(folder)
            atomic_json(folder/'granite_run_header.json',original)
            with (folder/'phi_checkpoint.jsonl').open('rb+') as h:h.truncate(0)
            with self.assertRaisesRegex(RuntimeError,'incomplete'):load_data(folder)

    def test_exploratory_cli_with_mock_backend(self):
        # Production CLI, but a mocked backend; no GPU or real model calls.
        import run_pilot
        spec=read_json(ROOT/'protocol/model_lock.json')['models']['qwen']
        runtime=dict(kind='SYNTHETIC_TEST_ONLY',model_revision=spec['revision'],tokenizer_revision=spec['tokenizer_revision'])
        class Backend(SyntheticBackend):
            def __init__(self,model):super().__init__();self.runtime=runtime
            def preflight(self):return dict(runtime=self.runtime,status='TECHNICAL_SYNTHETIC_PREFLIGHT_ONLY')
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);freeze_run(folder)
            pre=folder/'qwen_preflight.json';atomic_json(pre,dict(package_lock_sha256=self.sha,runtime=runtime,status='TECHNICAL_SYNTHETIC_PREFLIGHT_ONLY'))
            args=['run_pilot.py','run','--mode','exploratory','--model','qwen','--output',str(folder),'--preflight',str(pre),'--max-batches','1']
            with patch.dict(sys.modules,{'hf_backend':types.SimpleNamespace(HFBackend=Backend)}),patch.object(sys,'argv',args),contextlib.redirect_stdout(io.StringIO()):
                run_pilot.main();run_pilot.main()
            expected={j['job_id']:j for j in self.jobs if j['model_key']=='qwen'}
            saved=recover(folder/'qwen_checkpoint.jsonl',expected,self.sha)
            self.assertEqual(len(saved),16)
            self.assertEqual(read_json(folder/'qwen_run_header.json')['study_mode'],'exploratory')
            self.assertFalse((folder/'qwen_registration.json').exists())
            bad=read_json(pre);bad['package_lock_sha256']='old';atomic_json(pre,bad)
            with self.assertRaises(RuntimeError):check_preflight(pre,'qwen')

if __name__=='__main__':unittest.main(verbosity=2)
