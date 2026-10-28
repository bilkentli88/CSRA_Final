"""Frozen prediction, paired group bootstrap, and descriptive qualification screen."""
from core import *
from learning import *
import csv
POLICIES=('visible','self','observer','probe','probe_self','probe_observer','reference_ceiling')
CONTRASTS={'self_over_visible':('visible','self'),'self_over_selected_observer':('observer','self'),'self_added_to_probe':('probe','probe_self'),'self_over_probe_cost_comparison':('probe','self')}
def cost_summary(run):
    costs={}
    for path in sorted((Path(run)/'scores').glob('*.jsonl')):
        batches=[json.loads(line) for line in path.read_text().splitlines()];loads={};seconds=tokens=padded=jobs=oom=0
        for b in batches:
            rt=b['runtime'];loads[rt.get('started_utc','fixture')]=rt.get('model_load_seconds',0);f=rt.get('last_forward',{})
            seconds+=f.get('seconds',0);tokens+=f.get('unpadded_input_tokens',0);padded+=f.get('padded_input_tokens',0);jobs+=len(b['rows']);oom+=len(b['rows'][0]['oom_attempts']) if b['rows'] else 0
        costs[path.stem]=dict(jobs=jobs,committed_forward_seconds=seconds,unpadded_input_tokens=tokens,padded_input_tokens=padded,unique_model_load_seconds=sum(loads.values()),model_loads=len(loads),oom_retries=oom)
    return dict(stages=costs,meter_limits='Successful committed forwards and unique model loads only; interrupted uncommitted work, failed OOM calls, download, startup qualification, CPU preparation and Colab idle time require whole-session billing. Tokens are model-specific; batch wall time cannot be uniquely allocated to an individual policy.',policy_calls_per_target_item={'visible':0,'self':24,'observer':24,'probe':6,'probe_self':30,'probe_observer':30},common_input_costs_excluded='Given question/document/recorded answer and causal audit Y collection; CPU encoder and decoder costs must be added or amortized.',planning_A100_forward_hours_reference=1.158308984633345)
def analyze(run,allow_fixture=False,draws=None):
    run=Path(run);freeze=verify_freeze(run,allow_fixture);check_environment();targets=aggregate_targets(run,freeze);cases=aggregate_reports(run,targets,freeze)
    if len(cases)!=360 or len({c['group'] for c in cases})!=40:raise RuntimeError('Fresh40 must have 360 cases and forty groups.')
    model=joblib.load(run/'frozen_models.joblib');old=old_cases()
    if set(model['training_groups'])&{c['group'] for c in cases}:raise RuntimeError('Training/fresh group overlap.')
    E=load_embeddings(run,cases,'fresh');idx=np.arange(len(cases));tp=text_predictions(model['text_models'],cases,E,idx);check_probe_invariance(cases);rows=[]
    for algo in ALGORITHMS:
        pred=predict_decoders(model['decoders'][algo],cases,idx,tp)
        for i,c in enumerate(cases):
            reporter=model['selected'][c['target']];specs={'visible':('visible',None),'self':('visible',c['target']),'observer':('visible',reporter),'probe':('probe',None),'probe_self':('probe',c['target']),'probe_observer':('probe',reporter),'reference_ceiling':('reference_ceiling',None)}
            row=dict(algorithm=algo,item_id=c['item_id'],group=c['group'],target=c['target'],strength=c['strength'],relation=c['relation'],Y=c['Y'],selected_observer=reporter,old100_Y_variance=model['old100_Y_variance'][c['target']])
            for policy,(tier,r) in specs.items():
                value=pred[(i,tier,r)];row[policy+'_prediction']=value;row[policy+'_squared_error']=(c['Y']-value)**2
            rows.append(row)
    groups=sorted({c['group'] for c in cases});draws=draws or cfg()['bootstrap_draws'];rng=np.random.default_rng(cfg()['seed']);samples=rng.integers(0,len(groups),size=(draws,len(groups)));summaries={}
    for algo in ALGORITHMS:
        data=[r for r in rows if r['algorithm']==algo];contrasts={}
        for name,(baseline,new) in CONTRASTS.items():
            values=np.array([np.mean([(r[baseline+'_squared_error']-r[new+'_squared_error'])/r['old100_Y_variance'] for r in data if r['group']==g]) for g in groups]);ci=np.quantile(values[samples].mean(1),[.025,.975]);bytarget={t:float(np.mean([(r[baseline+'_squared_error']-r[new+'_squared_error'])/r['old100_Y_variance'] for r in data if r['target']==t])) for t in cfg()['target_models']}
            estimate=float(values.mean());positive=sum(v>0 for v in bytarget.values())
            contrasts[name]=dict(normalized_MSE_reduction=estimate,descriptive_95pct_cluster_bootstrap_interval=ci.tolist(),by_target=bytarget,positive_targets=positive,practical_development_screen=bool(estimate>=cfg()['practical_screen_threshold'] and positive>=2))
        metrics={}
        for target in cfg()['target_models']:
            rr=[r for r in data if r['target']==target];variance=float(np.var([r['Y'] for r in rr]));metrics[target]={p:dict(MSE=float(np.mean([r[p+'_squared_error'] for r in rr])),R2_fresh=(1-float(np.mean([r[p+'_squared_error'] for r in rr]))/variance) if variance>1e-12 else None) for p in POLICIES}
        summaries[algo]=dict(contrasts=contrasts,target_metrics=metrics)
    result=dict(status='SYNTHETIC_TECHNICAL_FIXTURE_ONLY' if allow_fixture else 'FRESH40_DEVELOPMENT_QUALIFICATION',package_lock_sha256=verify_package(),pre_outcome_freeze_sha256=freeze,fresh_items=40,cases=360,groups=40,bootstrap_draws=draws,primary_algorithm='gradient_boosting',sensitivity_algorithm='ridge',observer_selection=model['selected'],results=summaries,threshold_note='0.01 normalized MSE is a proposed engineering screen, not a validated scientific minimum. At least two targets must have positive improvements. Intervals are descriptive; no multiplicity-adjusted confirmatory decision or publication guarantee.',sample_note='Purposive relation-balanced convenience sample. After inspection these forty items are development data and must not enter later confirmatory tests.',ownership_note='SELF_PRESENTED and OTHER_PRESENTED are prompt labels. Self means same weights/configuration, not access to retained answer-generation activations.')
    out=run/'analysis';out.mkdir(exist_ok=True);atomic_json(out/'qualification_summary.json',result);atomic_json(out/'actual_costs.json',cost_summary(run));atomic_json(out/'fresh_analysis_inputs.json',dict(pre_outcome_freeze_sha256=freeze,cases=cases))
    with (out/'per_case_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=['# Fresh40 development qualification','',f'Status: {result["status"]}. Forty fresh items, 360 cases, forty clusters.','', 'Positive values below favor the policy named first. Scaling uses the target model’s old100 outcome variance.','', '| Contrast | Normalized MSE reduction | Descriptive 95% interval | Positive targets | Practical screen |','|---|---:|---|---:|---|']
    for name,c in summaries['gradient_boosting']['contrasts'].items():
        lo,hi=c['descriptive_95pct_cluster_bootstrap_interval'];text.append(f'| {name} | {c["normalized_MSE_reduction"]:.4f} | [{lo:.4f}, {hi:.4f}] | {c["positive_targets"]}/3 | {c["practical_development_screen"]} |')
    text+=['',result['threshold_note'],'',result['sample_note'],'',result['ownership_note'],'','Training-selected observers: '+json.dumps(model['selected']), '', 'See qualification_summary.json for each target and Ridge sensitivity, per_case_predictions.csv for all errors, and actual_costs.json for measured successful forward costs. No predictor was fitted to Fresh40 outcomes.']
    (out/'READ_RESULTS.md').write_text('\n'.join(text)+'\n');print('\n'.join(text),flush=True);return result
