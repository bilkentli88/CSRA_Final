"""Prospective precision scenarios from old grouped out-of-fold paired errors."""
import argparse,os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
import numpy as np,csv,collections
from common import *

def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n;center=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [center-h,center+h]
def pools_from_rows(rows):
    grouped={}
    for r in rows:
        k=(r['group'],r['relation'],r['target']);grouped.setdefault(k,[]).append(float(r['normalized_improvement']))
    cell={k:float(np.mean(v)) for k,v in grouped.items()};relation_groups={r:sorted({g for g,rr,t in cell if rr==r}) for r in plan()['relations']}
    p1={r:np.array([np.mean([cell[g,r,t] for t in cfg()['target_models']]) for g in gs]) for r,gs in relation_groups.items()}
    p2={t:{r:np.array([cell[g,r,t] for g in gs]) for r,gs in relation_groups.items()} for t in cfg()['target_models']}
    variability={t:sum(float(np.var(v,ddof=1)) for v in rp.values())/16 for t,rp in p2.items()}
    proxy=max(variability,key=lambda t:(variability[t],t));return p1,p2[proxy],proxy,{r:len(g) for r,g in relation_groups.items()}

def simulation_intervals(pools,n,simulations,draws,rng,level):
    relations=plan()['relations'];n_s=n//len(relations);intervals=[];q=[(1-level)/2,1-(1-level)/2]
    # Each artificial future sample draws one grouped residual per hypothetical fresh item.
    # Conditional bootstrap draws multinomial bin counts, avoiding N-long index arrays.
    for simulation in range(simulations):
        point=0.0;boot=np.zeros(draws)
        for relation in relations:
            raw=np.asarray(pools[relation],float);centered=raw-np.mean(raw);k=len(raw)
            counts=rng.multinomial(n_s,np.full(k,1/k));empirical=counts/n_s
            point+=float(empirical@centered)/len(relations)
            bootstrap_counts=rng.multinomial(n_s,empirical,size=draws)
            boot+=(bootstrap_counts@centered/n_s)/len(relations)
        lo,hi=np.quantile(boot,q);intervals.append((float(lo),float(hi),point))
    return np.asarray(intervals)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--smoke',action='store_true',help='Testing only: 20 simulations, 199 draws; cannot recommend N.');args=ap.parse_args();run=args.run_dir;lock=verify_package();c=cfg()['calibration'];folder=run/'analysis/ridge'
    summary=read_json(folder/'development_summary.json')
    if summary['identity']['package_lock_sha256']!=lock:raise RuntimeError('Development summary lock differs.')
    if summary['heldout_errors_sha256']!=file_sha(folder/'heldout_errors.csv'):raise RuntimeError('Held-out diagnostic errors differ from their analysis manifest.')
    with (folder/'heldout_errors.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=900 or {r['target'] for r in rows}!=set(cfg()['target_models']):raise RuntimeError('Complete known-target old-100 diagnostic errors are required.')
    p1,p2,proxy,counts=pools_from_rows(rows)
    if min(counts.values())<2:raise RuntimeError('Insufficient relation-specific independent groups for calibration.')
    S,B=(20,199) if args.smoke else (c['simulations'],c['bootstrap_draws']);out=folder/('calibration_smoke' if args.smoke else 'calibration');out.mkdir(exist_ok=True)
    identity=dict(package_lock_sha256=lock,heldout_errors_sha256=file_sha(folder/'heldout_errors.csv'),simulations=S,bootstrap_draws=B,seed=cfg()['seed'])
    immutable_json(out/'identity.json',identity);result=[];zlow=(1-c['interval_level'])/2
    for metric,pools in (('P1_known_target_mean',p1),('P2_known_target_variance_proxy',p2)):
        for n in c['candidate_sizes']:
            cache=out/f'{metric}_N{n}.json'
            if cache.exists():
                cached=read_json(cache)
                if cached['identity']!=identity:raise RuntimeError('Cached calibration identity differs.')
                result.extend(cached['rows']);continue
            seed=int(sha(canonical([cfg()['seed'],metric,n]))[:16],16);intervals=simulation_intervals(pools,n,S,B,np.random.default_rng(seed),c['interval_level']);local=[]
            for variance in ([1.0] if metric.startswith('P1') else c['variance_multipliers']):
                scaled=intervals*math.sqrt(variance);width=(scaled[:,1]-scaled[:,0])/2;coverage=(scaled[:,0]<=0)&(scaled[:,1]>=0)
                scenarios=[]
                for shift in c['shift_scenarios']:
                    positive=scaled[:,0]+shift>0;k=int(positive.sum());scenarios.append(dict(assumed_shift=shift,positive_interval_fraction=k/S,monte_carlo_95_interval=wilson(k,S)))
                local.append(dict(metric=metric,N=n,variance_multiplier=variance,median_halfwidth=float(np.median(width)),p90_halfwidth=float(np.quantile(width,c['width_quantile_for_selection'])),halfwidth_p90_mc95_approx=[float(np.quantile(width,max(0,.9-1.96*math.sqrt(.9*.1/S)))),float(np.quantile(width,min(1,.9+1.96*math.sqrt(.9*.1/S))))],coverage_under_empirical_location_model=float(np.mean(coverage)),coverage_monte_carlo_95_interval=wilson(int(coverage.sum()),S),positive_interval_scenarios=scenarios))
            atomic_json(cache,dict(identity=identity,rows=local));result.extend(local);print('Completed precision scenarios:',metric,n,flush=True)
    passing=[n for n in c['candidate_sizes'] if all(r['p90_halfwidth']<=c['halfwidth_goal'] for r in result if r['N']==n)]
    decision=dict(status='SMOKE_TEST_NO_SAMPLE_SIZE_DECISION' if args.smoke else 'DEVELOPMENT_PRECISION_SCENARIOS_NOT_A_POWER_GUARANTEE',identity=identity,group_counts_by_relation=counts,unseen_target_proxy=proxy,unseen_target_outcomes_used=False,variance_multiplier_means='Multiplier on paired-error residual variance; centered residuals scale by its square root.',group_resampling='Equal mass to each distinct development subject/triple group within each relation; old groups are proxy support for hypothetical independent fresh items, not literal repeated fresh facts.',interval_level=c['interval_level'],engineering_halfwidth_goal=c['halfwidth_goal'],precision_selection_rule='Smallest N with 90th-percentile halfwidth <= goal for P1 and every P2 variance scenario. Sign/significance is not used.',candidate_N_by_precision_only=None if args.smoke or not passing else min(passing),fresh_run_authorized=False,final_N_frozen=False,human_re_review_completed=False,limitations=['Only four old answer-triple groups in the original-language relation; empirical support is sparse.','Unseen-target variance scenarios are assumptions, not an observed bound on Mistral.','Conditioning on development fits does not capture all training-set or population-shift uncertainty.','The .01 halfwidth target is an engineering resolution, not a validated utility/equivalence margin.','Final N also requires actual review yield, staffing and compute feasibility before fresh outcomes.'],results=result)
    atomic_json(out/'sample_size_scenarios.json',decision)
    if not args.smoke:
        with (out/'sample_size_table.csv').open('w',newline='') as f:
            keys=['metric','N','variance_multiplier','median_halfwidth','p90_halfwidth','coverage_under_empirical_location_model'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in result)
    print(json.dumps({k:v for k,v in decision.items() if k!='results'},indent=2))
if __name__=='__main__':main()
