import unittest,tempfile,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from common import *
class CoreTests(unittest.TestCase):
    def test_inventory_and_holdout(self):
        for m in cfg()['target_models']:self.assertEqual(len(answer_jobs(m)),4800)
        with self.assertRaises(ValueError):answer_jobs('mistral')
        self.assertEqual(len(list(possible_report_prompts())),7200)
        t=[dict(target=m,item_id=x['item_id'],strength=s,document=document(x,'C_'+s),question=x['question'],answer=x['parametric_answer']) for x in items() for m in cfg()['target_models'] for s in cfg()['strengths']]
        for m in cfg()['reporter_models']:self.assertEqual(len(report_jobs(m,t)),7200)
    def test_shared_groups_never_split(self):
        data=items();g=grouping(data);f=folds(data);self.assertEqual(len(set(g.values())),57)
        for group in set(g.values()):self.assertEqual(len({f[k] for k in g if g[k]==group}),1)
        self.assertEqual(set(f.values()),set(range(5)))
        self.assertEqual(len({x['item_id'] for x in data}),100)
    def test_subject_connects_different_triples(self):
        a=dict(items()[0]);b=dict(items()[1]);b.update(item_id='other',source_subject_uri=a['source_subject_uri']);g=grouping([a,b]);self.assertEqual(g[a['item_id']],g['other'])
    def test_prompt_crossing_and_substitution(self):
        x=items()[0];a=report_prompt('DOC','QUESTION','ANSWER','SELF_PRESENTED',0);b=report_prompt('DOC','QUESTION','ANSWER','OTHER_PRESENTED',0)
        self.assertEqual(a.replace(CUES['SELF_PRESENTED'],'CUE'),b.replace(CUES['OTHER_PRESENTED'],'CUE'))
        self.assertNotIn('granite',a);self.assertNotIn('qwen',a)
        for s in cfg()['strengths']:
            self.assertEqual(document(x,'P_'+s).replace(x['parametric_answer'],'ENTITY'),document(x,'C_'+s).replace(x['counterfactual_answer'],'ENTITY'))
    def test_checkpoint_resume_and_partial_tail(self):
        class Fake:
            runtime={'test_only':True}
            calls=0
            def score(self,jobs):self.calls+=1;return [{k:-1.5 for k in LABELS} for _ in jobs]
        jobs=answer_jobs('qwen',items()[:1])[:9]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.jsonl';fake=Fake();execute(jobs,p,'LOCK',fake,8,max_batches=1);self.assertEqual(fake.calls,1)
            with p.open('ab') as f:f.write(b'{unfinished')
            result=execute(jobs,p,'LOCK',fake,8);self.assertEqual(len(result),9);self.assertEqual(fake.calls,2);self.assertEqual(len(list(Path(d).glob('*.partial-*'))),1)
            execute(jobs,p,'LOCK',fake,8);self.assertEqual(fake.calls,2)
            with self.assertRaises(RuntimeError):recover(p,{j['job_id']:j for j in jobs},'WRONG')
    def test_oom_and_nonfinite_not_committed(self):
        class Fake:
            runtime={'test_only':True}
            def score(self,jobs):
                if len(jobs)>2:raise MemoryError('test OOM')
                return [{k:-2.0 for k in LABELS} for _ in jobs]
        jobs=answer_jobs('qwen',items()[:1])[:8]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.jsonl';r=execute(jobs,p,'LOCK',Fake());self.assertTrue(all(v['batch_size']==2 for v in r.values()))
            class Bad(Fake):
                def score(self,jobs):return [{k:float('nan') for k in LABELS} for _ in jobs]
            p=Path(d)/'bad.jsonl'
            with self.assertRaises(FloatingPointError):execute(jobs,p,'LOCK',Bad())
            self.assertFalse(p.exists())
    def test_target_formula_and_permutations(self):
        # Production aggregation over all 100 items, using explicitly fabricated scores.
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'scores').mkdir()
            for model in cfg()['target_models']:
                js=answer_jobs(model);records=[]
                for j in js:
                    values={'P':-2.0,'C':-3.0,'I':-4.0}
                    if j['condition'].startswith('C_'):values={'P':-4.0,'C':-1.0,'I':-5.0}
                    logp={l:-8.0 for l in LABELS}
                    for lab,role in j['label_to_role'].items():logp[lab]=values[role]
                    records.append(dict(job_id=j['job_id'],prompt_sha256=j['prompt_sha256'],logp=logp))
                (run/'scores'/f'answer_{model}.jsonl').write_bytes(canonical(dict(package_lock_sha256='TEST',rows=records))+b'\n')
            targets=aggregate_targets(run,'TEST');self.assertEqual(len(targets),900)
            for t in targets:self.assertEqual(t['Y'],4.0);self.assertEqual(t['baseline_margin'],1.0);self.assertEqual(t['answer_role'],'C')
            for model in cfg()['reporter_models']:
                js=report_jobs(model,targets);records=[]
                for j in js:
                    values={'document':-2.0,'prior':-3.0,'both':-4.0,'uncertain':-5.0}
                    records.append(dict(job_id=j['job_id'],prompt_sha256=j['prompt_sha256'],logp={lab:values[role] for lab,role in j['label_to_role'].items()}))
                (run/'scores'/f'report_{model}.jsonl').write_bytes(canonical(dict(package_lock_sha256='TEST',rows=records))+b'\n')
            cases=aggregate_reports(run,targets,'TEST');self.assertEqual(len(cases),900)
            for c in cases:
                self.assertEqual(set(c['reports']),set(cfg()['reporter_models']))
                for rep in c['reports'].values():
                    for v in rep.values():self.assertEqual(v['alr'],[1.0,-1.0,-2.0]);self.assertAlmostEqual(sum(v['q'].values()),1)
if __name__=='__main__':unittest.main(verbosity=2)
