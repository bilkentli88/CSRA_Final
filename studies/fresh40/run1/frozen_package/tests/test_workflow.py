"""Production-path integration with explicitly artificial embeddings and label scores."""
import sys,tempfile,unittest,json,hashlib,ast,shutil,io,contextlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import core as c
import numpy as np
class FixtureBackend:
    def __init__(self,oom_over=None):self.runtime={'technical_fixture_only':True};self.oom_over=oom_over;self.attempts=[]
    def score(self,jobs):
        self.attempts.append(len(jobs))
        if self.oom_over and len(jobs)>self.oom_over:raise MemoryError('artificial OOM')
        result=[]
        for j in jobs:
            seed=int(hashlib.sha256(j['job_id'].encode()).hexdigest()[:16],16);r=np.random.default_rng(seed);v=r.normal(size=4);p=np.exp(v-v.max());p/=p.sum();result.append(dict(zip(c.LABELS,np.log(p).tolist())))
        return result
class Workflow(unittest.TestCase):
    def test_01_inventory_and_source(self):
        self.assertEqual(c.validate_inputs()['approved'],40);self.assertEqual(len(c.embedding_catalog()),1980)
        for target in c.cfg()['target_models']:self.assertEqual(len(c.answer_jobs(target)),1920)
        with self.assertRaises(ValueError):c.answer_jobs('mistral')
        data=c.read_json(c.ROOT/'inputs/items.json');self.assertEqual(len({x['group_id'] for x in data}),40)
        # Candidate text carries strings, not designation order.
        x=dict(c.items()[0]);y=dict(x);y['parametric_answer'],y['irrelevant_answer']=x['irrelevant_answer'],x['parametric_answer']
        self.assertEqual(c.baseline_text(x,'qwen','BARE','anything'),c.baseline_text(y,'qwen','BARE','anything'))
        for path in (c.ROOT/'code').glob('*.py'):ast.parse(path.read_text())
        nb=c.read_json(c.ROOT/'CSRA_Fresh40_Colab.ipynb');codes=[x for x in nb['cells'] if x['cell_type']=='code'];self.assertEqual(len(codes),5)
        for cell in codes:ast.parse(''.join(cell['source']))
    def test_02_checkpoint_and_failure_guards(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scores.jsonl';jobs=c.answer_jobs('qwen')[:9];b=FixtureBackend(oom_over=2)
            with contextlib.redirect_stdout(io.StringIO()):result=c.execute(jobs,path,'fixture',b)
            self.assertEqual(len(result),9);self.assertEqual(b.attempts[:3],[8,4,2])
            before=path.read_bytes()
            with path.open('ab') as f:f.write(b'{"partial"')
            self.assertEqual(len(c.recover(path,{j['job_id']:j for j in jobs},'fixture')),9);self.assertEqual(path.read_bytes(),before)
            with self.assertRaises(RuntimeError):c.recover(path,{j['job_id']:j for j in jobs},'different')
            with path.open('ab') as f:f.write(before.splitlines(keepends=True)[0])
            with self.assertRaises(RuntimeError):c.recover(path,{j['job_id']:j for j in jobs},'fixture')
            bad=FixtureBackend();bad.score=lambda jobs:[{k:float('nan') for k in c.LABELS} for j in jobs]
            with self.assertRaises(FloatingPointError):c.execute(jobs,Path(d)/'bad.jsonl','fixture',bad)
            self.assertFalse((Path(d)/'bad.jsonl').exists())
    def test_03_full_production_training_and_scoring_fixture(self):
        from learning import train,check_probe_invariance
        from analyze import analyze
        from run_fresh40 import initialize,prepare
        with tempfile.TemporaryDirectory(prefix='fresh40-TECHNICAL-ONLY-') as tmp:
            run=Path(tmp);initialize(run);catalog=c.embedding_catalog();c.atomic_json(run/'embedding_catalog.json',catalog)
            rng=np.random.default_rng(12345);E=rng.normal(size=(1980,768)).astype('float32');E/=np.linalg.norm(E,axis=1,keepdims=True)
            np.save(run/'embeddings.npy',E);c.atomic_json(run/'encoder_manifest.json',dict(technical_fixture_only=True,embedding_sha256=c.file_sha(run/'embeddings.npy')));c.atomic_json(run/'token_audit.json',dict(technical_fixture_only=True))
            train(run);freeze=c.freeze_run(run,technical_fixture=True)
            with self.assertRaises(RuntimeError):c.verify_freeze(run)
            self.assertEqual(c.verify_freeze(run,True),freeze)
            # Guard is evaluated before torch import/model loading.
            from hf_backend import HFBackend
            with self.assertRaisesRegex(RuntimeError,'Synthetic technical fixture'):HFBackend('qwen',run,c.verify_package())
            for target in c.cfg()['target_models']:
                jobs=c.answer_jobs(target)
                with contextlib.redirect_stdout(io.StringIO()):c.execute(jobs,run/'scores'/f'answer_{target}.jsonl',freeze,FixtureBackend())
            targets=c.aggregate_targets(run,freeze);self.assertEqual(len(targets),360);check_probe_invariance(targets)
            for reporter in c.cfg()['reporter_models']:
                jobs=c.report_jobs(reporter,targets);self.assertEqual(len(jobs),2880)
                with contextlib.redirect_stdout(io.StringIO()):c.execute(jobs,run/'scores'/f'report_{reporter}.jsonl',freeze,FixtureBackend())
            cases=c.aggregate_reports(run,targets,freeze);self.assertEqual(len(cases),360);self.assertEqual(len({x['group'] for x in cases}),40)
            # Independently verify one permutation-averaged causal contrast and one rotated ALR.
            t=targets[0];scores=c.complete_scores(c.answer_jobs(t['target']),run/'scores'/f'answer_{t["target"]}.jsonl',freeze);means={}
            for condition in ('C_'+t['strength'],'P_'+t['strength']):
                values=[]
                for j in c.answer_jobs(t['target']):
                    if j['item_id']==t['item_id'] and j['condition']==condition:
                        inv={v:k for k,v in j['label_to_role'].items()};v=scores[j['job_id']]['logp'];values.append(v[inv['C']]-v[inv['P']])
                self.assertEqual(len(values),6);means[condition]=np.mean(values)
            self.assertAlmostEqual(t['Y'],means['C_'+t['strength']]-means['P_'+t['strength']],places=10)
            reporter='mistral';cue='SELF_PRESENTED';rjobs=c.report_jobs(reporter,targets);rs=c.complete_scores(rjobs,run/'scores'/f'report_{reporter}.jsonl',freeze);values=[]
            for j in rjobs:
                if j['target']==t['target'] and j['item_id']==t['item_id'] and j['strength']==t['strength'] and j['cue']==cue:
                    inv={v:k for k,v in j['label_to_role'].items()};v=rs[j['job_id']]['logp'];values.append(v[inv['document']]-v[inv['prior']])
            case=next(x for x in cases if all(x[k]==t[k] for k in ('target','item_id','strength')));self.assertEqual(len(values),4);self.assertAlmostEqual(case['reports'][reporter][cue]['alr'][0],np.mean(values),places=10)
            frozen={n:c.file_sha(run/n) for n in c.FREEZE_ASSETS}
            with self.assertRaises(RuntimeError):c.require_before_outcomes(run)
            with self.assertRaises(RuntimeError):train(run)
            with self.assertRaises(RuntimeError):prepare(run) # fixture cannot authorize production resume
            with contextlib.redirect_stdout(io.StringIO()):summary=analyze(run,allow_fixture=True,draws=200)
            self.assertEqual(summary['status'],'SYNTHETIC_TECHNICAL_FIXTURE_ONLY');self.assertEqual(summary['cases'],360);self.assertEqual(set(summary['observer_selection']),set(c.cfg()['target_models']))
            self.assertEqual(frozen,{n:c.file_sha(run/n) for n in c.FREEZE_ASSETS})
            # Idempotent resume produces no new fake forwards.
            b=FixtureBackend();jobs=c.answer_jobs('qwen')
            c.execute(jobs,run/'scores/answer_qwen.jsonl',freeze,b);self.assertEqual(b.attempts,[])
            with (run/'frozen_models.joblib').open('ab') as f:f.write(b'tamper')
            with self.assertRaisesRegex(RuntimeError,'Frozen asset changed'):c.verify_freeze(run,True)
if __name__=='__main__':unittest.main(verbosity=2)
