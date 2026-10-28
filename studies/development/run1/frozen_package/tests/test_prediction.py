"""Artificial outcomes for verification only; never written into study results."""
import unittest,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
from predictive import fit_stack,evaluate_stack,views,base_features,group_splits
from calibrate_size import simulation_intervals,pools_from_rows,wilson
from common import cfg,plan

def artificial_cases(n=24,identical=False):
    rng=np.random.default_rng(91);cases=[]
    for i in range(n):
        for source in cfg()['target_models']:
            Y=float(rng.normal()*3);reports={}
            for reporter in cfg()['reporter_models']:
                v=[Y+.01*rng.normal(),float(rng.normal()),float(rng.normal())] if reporter==source else list(rng.normal(size=3))
                if identical:v=[1.0,2.0,3.0]
                reports[reporter]={cue:{'alr':v} for cue in cfg()['cues']}
            cases.append(dict(item_id=f'artificial_{i}',group=f'g{i}',fold=i%5,relation=plan()['relations'][i%4],target=source,strength='PROVISIONAL',baseline_margin=0.0,answer_role='P',Y=Y,text=f'Question about fictional object token{i}. Document has a common statement. Recorded answer alpha. Target {source}.',reports=reports))
    return cases,rng.normal(size=(len(cases),8))
class PredictiveTests(unittest.TestCase):
    def test_equal_cost_and_slot_symmetry(self):
        cases,E=artificial_cases(4);X,meta=views(cases[0],base_features(cases[0],[0,0]));self.assertEqual(X.shape,(48,22))
        self.assertEqual(sum(m[3]=='native' for m in meta),24);self.assertEqual(sum(m[3]=='external' for m in meta),24)
        for i in range(0,len(X),2):np.testing.assert_allclose(X[i,-6:-3],X[i+1,-3:]);np.testing.assert_allclose(X[i,-3:],X[i+1,-6:-3])
    def test_nested_splits_disjoint_and_source_signal(self):
        cases,E=artificial_cases();train=np.array([i for i,c in enumerate(cases) if c['fold']!=0]);valid=np.array([i for i,c in enumerate(cases) if c['fold']==0])
        for a,b in group_splits(cases,train):self.assertFalse({cases[i]['group'] for i in a}&{cases[i]['group'] for i in b})
        fit=fit_stack(cases,E,train);r=evaluate_stack(fit,cases,E,valid)
        self.assertEqual(set(fit['training_targets']),set(cfg()['target_models']));self.assertFalse(fit['native_identity_feature']);self.assertGreater(np.mean([x['paired_error_improvement'] for x in r]),0)
        with self.assertRaises(RuntimeError):evaluate_stack(fit,cases,E,train)
    def test_identical_reports_give_zero_advantage(self):
        cases,E=artificial_cases(16,identical=True);tr=np.array([i for i,c in enumerate(cases) if c['fold']!=0]);va=np.array([i for i,c in enumerate(cases) if c['fold']==0]);fit=fit_stack(cases,E,tr);r=evaluate_stack(fit,cases,E,va)
        np.testing.assert_allclose([x['paired_error_improvement'] for x in r],0,atol=1e-10)
    def test_calibration_variance_and_sample_size(self):
        pools={r:np.linspace(-.2,.2,12) for r in plan()['relations']}
        a=simulation_intervals(pools,400,30,199,np.random.default_rng(3),.975);b=simulation_intervals(pools,800,30,199,np.random.default_rng(4),.975)
        self.assertLess(np.median(b[:,1]-b[:,0]),np.median(a[:,1]-a[:,0]));self.assertTrue(np.all(a[:,0]<=a[:,1]));self.assertLess(wilson(0,100)[1],.04)
if __name__=='__main__':unittest.main(verbosity=2)
