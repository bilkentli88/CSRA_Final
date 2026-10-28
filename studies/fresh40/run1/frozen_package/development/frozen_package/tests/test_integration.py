"""Entire CLI workflow with artificial scores/embeddings, in a temporary folder."""
import sys,unittest,tempfile,subprocess,os,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from common import *
class IntegrationTests(unittest.TestCase):
    def test_complete_mock_workflow(self):
        import numpy as np
        from run_development import prepare,targets
        lock=verify_package();environment={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        with tempfile.TemporaryDirectory(prefix='csra-artificial-verification-') as d:
            run=Path(d)/'artificial_run';prepare(run,lock);(run/'scores').mkdir()
            def fabricated(j):
                rng=np.random.default_rng(int(sha(j['job_id'].encode())[:12],16));a=rng.normal(0,1,4);a=a-np.max(a);v=a-np.log(np.exp(a).sum());return dict(zip(LABELS,map(float,v)))
            for model in cfg()['target_models']:
                js=answer_jobs(model);rows=[dict(job_id=j['job_id'],prompt_sha256=j['prompt_sha256'],logp=fabricated(j)) for j in js]
                (run/'scores'/f'answer_{model}.jsonl').write_bytes(canonical(dict(package_lock_sha256=lock,artificial_test_only=True,rows=rows))+b'\n')
            ts=targets(run,lock)['targets']
            for model in cfg()['reporter_models']:
                js=report_jobs(model,ts);rows=[dict(job_id=j['job_id'],prompt_sha256=j['prompt_sha256'],logp=fabricated(j)) for j in js]
                (run/'scores'/f'report_{model}.jsonl').write_bytes(canonical(dict(package_lock_sha256=lock,artificial_test_only=True,rows=rows))+b'\n')
            def command(script,*args):
                p=subprocess.run([sys.executable,str(ROOT/'code'/script),*map(str,args)],capture_output=True,text=True,env=environment)
                if p.returncode:self.fail(p.stdout+'\n'+p.stderr)
                return p.stdout
            command('run_development.py','assemble','--run-dir',run);cases=read_json(run/'analysis_inputs.json')['cases'];self.assertEqual(len(cases),900)
            out=run/'analysis';out.mkdir();E=np.random.default_rng(8).normal(size=(900,768)).astype(np.float32);E/=np.linalg.norm(E,axis=1,keepdims=True);np.save(out/'embeddings.npy',E)
            # Explicitly artificial encoder fixture: never saved into the user's package/results.
            atomic_json(out/'encoder_manifest.json',dict(analysis_inputs_sha256=file_sha(run/'analysis_inputs.json'),embedding_sha256=file_sha(out/'embeddings.npy'),encoder=plan()['encoder'],artificial_test_fixture=True,device='ARTIFICIAL_NOT_AN_ENCODER_RUN'))
            command('analyze_development.py','--run-dir',run)
            summary=read_json(out/'ridge/development_summary.json');self.assertEqual(summary['n_items'],100);self.assertFalse(summary['held_out_target_outcomes_used'])
            manifest=read_json(out/'ridge/draft_stack_manifest.json');self.assertFalse(manifest['confirmatory_release_ready']);self.assertNotIn('mistral',manifest['fitted_targets'])
            # A rerun must use saved outer-fold results and the frozen fitted artifact.
            stack_sha=manifest['stack_sha256'];text=command('analyze_development.py','--run-dir',run);self.assertIn('Using completed outer fold',text);self.assertEqual(file_sha(out/'ridge/draft_development_stack.joblib'),stack_sha)
            command('calibrate_size.py','--run-dir',run,'--smoke');cal=read_json(out/'ridge/calibration_smoke/sample_size_scenarios.json');self.assertIsNone(cal['candidate_N_by_precision_only']);self.assertFalse(cal['final_N_frozen']);self.assertEqual(cal['group_counts_by_relation']['original_language'],4)
            export=Path(d)/'artificial_export.zip';command('run_development.py','export','--run-dir',run,'--output',export)
            with zipfile.ZipFile(export) as z:self.assertIsNone(z.testzip());self.assertEqual(sum(n.endswith('.jsonl') for n in z.namelist()),7)
if __name__=='__main__':unittest.main(verbosity=2)
