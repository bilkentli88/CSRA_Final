from __future__ import annotations
from pathlib import Path
import sys,json,hashlib
import pandas as pd
import numpy as np

EXPECTED={
'report_audit_sha':'89015fbd5f073e0c6e653d66016e510cee61913f71a194a4192c05e3ad3cc790',
'tr3_repair_sha':'7946c6fc1ef7bb123d784826bd9db3d6cb4c83824fc1bbd2a0ee95bb5f409b3b',
'analysis_sha':'7b582ff2d39aa1c49c6bd2dab282ba581f938944c882d14c3621fc1038c0a3b7',
'common_sha':'97a7933c2679687010eab2a87233ff280498755353f14f488b52fe5a9446783d',
'analysis_spec_sha':'529b0a2dec87136d8f4c533a3cd69c3e7560ddc002f94030f14a135542e940f0',
'answer_audit_sha':'debd40b4938633342f986f0b866481984e398b8c6695119878335bb4abf29377',
'postreview_sha':'92e7c26fd9f7a0782b309c419d9ad2b69c8fa5e84d1677a32cee77b54edb9bd2',
'final_items_sha':'19c57a4a96db4d8bbe2cb529ed94a74e156eb9f1f2c407660cf2530132c90b98',
'complete_ids_sha':'005890a778ae1ecc02054c3abd52a6cbbebaf70e1eb704615b84d1138206b021',
'answer_constants_sha':'494d9c2c46cadeeddd6bb033212c2639a2c2f48d27755cd062119838677b8c8b',
'analysis_bundle_sha':'9edc3257f4567c0da8d4ce16292aa8cef89168adb73f5bac19adddacc183524f',
'env_sha':'5aad9de33fc5ad45f2369bec8d48c4f667c0cd16967d557ce0469920204f3473',
'component_manifest_sha':'32dd87e0d9f84ca55bcd35d979a72dc1914e007bf26bee505f9dc84b645b6d0f',
'model_manifests':{'qwen': '0ca8fba38c8396b6d3f2b4a09963d60c835f0e18f7a558cc4e8cf4c9d6d0be19', 'granite': '561c64dddd9ab2937fcfc38d19d4a0981164fbc3d1c14534dbc743968878cc8f', 'phi': 'fa36e2aeeab5e1c78e303a7d90db083f331af3cbaba859489a0fc2e7853acab2'},
}
APPROVAL_PHRASE='I_APPROVE_V08_ANALYSIS_TECHNICAL_REPAIR_TR4_AND_UNLOCK_ANALYSIS'

def file_sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def req(p,h,label):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(f'Missing {label}: {p}')
    got=file_sha(p)
    if got!=h: raise RuntimeError(f'{label} SHA mismatch: expected {h}, got {got}')
    return got

def _verify_bound_files(run_root,manifest,label):
    files=manifest.get('files',{})
    if not isinstance(files,dict): raise RuntimeError(f'{label} files binding missing')
    for rel,h in files.items(): req(Path(run_root)/rel,h,f'{label} bound file {rel}')
    return len(files)

def _load_base(run_root,package_root):
    run_root=Path(run_root); package_root=Path(package_root); src=run_root/'step2'/'src'
    if str(src) not in sys.path: sys.path.insert(0,str(src))
    from csra_v08_common import load_json,PRIMARY_MODELS
    from csra_v08_activation import verify_environment
    req(run_root/'step2'/'src'/'csra_v08_analysis.py',EXPECTED['analysis_sha'],'frozen analysis module')
    req(run_root/'step2'/'src'/'csra_v08_common.py',EXPECTED['common_sha'],'frozen common module')
    req(run_root/'step2'/'config'/'csra_v08_analysis_spec.json',EXPECTED['analysis_spec_sha'],'analysis spec')
    req(run_root/'step2'/'CSRA_v08_ANALYSIS_NOTEBOOK_FROZEN.zip',EXPECTED['analysis_bundle_sha'],'frozen analysis bundle')
    req(run_root/'step2'/'config'/'csra_v08_environment_lock.json',EXPECTED['env_sha'],'environment lock')
    req(run_root/'step2'/'manifests'/'CSRA_v08_STEP2_COMPONENT_MANIFEST.json',EXPECTED['component_manifest_sha'],'component manifest')
    req(run_root/'CSRA_v08_ANSWER_PHASE_AUDIT_R4_A2.json',EXPECTED['answer_audit_sha'],'answer audit')
    req(run_root/'locks'/'complete_item_ids.json',EXPECTED['complete_ids_sha'],'complete IDs')
    req(run_root/'locks'/'answer_derived_constants.json',EXPECTED['answer_constants_sha'],'answer constants')
    req(run_root/'locks'/'postreview'/'postreview_analysis_lock.json',EXPECTED['postreview_sha'],'postreview analysis lock')
    req(run_root/'locks'/'postreview'/'final_common_items.csv',EXPECTED['final_items_sha'],'final common items')
    report_path=run_root/'CSRA_v08_REPORT_PHASE_AUDIT_R4_A2.json'; req(report_path,EXPECTED['report_audit_sha'],'TR3 report-phase audit')
    report=load_json(report_path)
    if report.get('status')!='REPORT_PHASE_COMPLETE_ANALYSIS_NOT_RUN_AFTER_TR3_TECHNICAL_REPAIR': raise RuntimeError('Unexpected report audit status')
    if report.get('tr3_repair_record_sha256')!=EXPECTED['tr3_repair_sha']: raise RuntimeError('TR3 lineage mismatch in report audit')
    n_report_bound=_verify_bound_files(run_root,report,'report audit')
    for m,h in EXPECTED['model_manifests'].items(): req(run_root/'results'/f'{m}_report_phase_manifest_R4_A2.json',h,f'{m} report manifest')
    answer=load_json(run_root/'CSRA_v08_ANSWER_PHASE_AUDIT_R4_A2.json')
    n_answer_bound=_verify_bound_files(run_root,answer,'answer audit') if isinstance(answer.get('files'),dict) else 0
    env=verify_environment(load_json(run_root/'step2'/'config'/'csra_v08_environment_lock.json'))
    complete=load_json(run_root/'locks'/'complete_item_ids.json')
    if len(complete)!=100 or len(set(complete))!=100: raise RuntimeError('Complete item set is not exactly 100 unique items')
    selected=pd.read_csv(run_root/'locks'/'postreview'/'final_common_items.csv')
    selected=selected[selected['item_id'].isin(set(complete))].copy()
    counts=selected.groupby('relation')['item_id'].nunique().to_dict()
    if int(counts.get('country_of_origin',0))!=0: raise RuntimeError('TR4 defect premise no longer holds: country_of_origin is unexpectedly present')
    analysis_spec=load_json(run_root/'step2'/'config'/'csra_v08_analysis_spec.json')
    if analysis_spec.get('status')!='FROZEN_PRE_ACTIVATION' or int(analysis_spec['bootstrap_reps'])!=2000 or int(analysis_spec['alignment']['bootstrap_reps'])!=2000: raise RuntimeError('Frozen analysis specification changed')
    return dict(run_root=run_root,package_root=package_root,src=src,results=run_root/'results',checkpoints=run_root/'checkpoints',runtime=run_root/'runtime',locks=run_root/'locks',postreview=run_root/'locks'/'postreview',complete_ids=complete,selected=selected,analysis_spec=analysis_spec,env=env,relation_counts=counts,n_report_bound=n_report_bound,n_answer_bound=n_answer_bound)

def _tr4_design_vector(fit,model_key,cie,self_value):
    import patsy
    frame=fit.model.data.frame
    def observed(col):
        vals=[x for x in frame[col].dropna().tolist()]
        if not vals: raise RuntimeError(f'No observed level for {col}')
        return sorted(set(map(str,vals)))[0]
    row=pd.DataFrame([{'VSA_z':0.0,'CIE_z':float(cie),'SELF':int(self_value),'model_key':model_key,'frozen_answer_role':observed('frozen_answer_role'),'baseline_margin_z':0.0,'strength':observed('strength'),'analysis_relation':observed('analysis_relation')}])
    mat=patsy.build_design_matrices([fit.model.data.design_info],row,return_type='dataframe')[0]
    return mat.loc[:,fit.fe_params.index].to_numpy(float)[0]

def prepare_analysis_runtime_tr4(run_root,package_root):
    from csra_v08_common import atomic_write_json
    ctx=_load_base(run_root,package_root)
    repair_spec=Path(package_root)/'CSRA_v08_ANALYSIS_TECHNICAL_REPAIR_TR4_SPEC.json'
    repair_sha=file_sha(repair_spec)
    pre={'protocol_id':'CSRA-0.8','study':'CSRA v0.8 R4/A2','phase':'analysis_runtime_preflight','status':'ANALYSIS_LOCKED_PENDING_TR4_APPROVAL','scientific_change':False,'technical_change':True,'technical_repair':'TR4','report_phase_audit_sha256':EXPECTED['report_audit_sha'],'answer_audit_sha256':EXPECTED['answer_audit_sha'],'frozen_analysis_module_sha256':EXPECTED['analysis_sha'],'analysis_spec_sha256':EXPECTED['analysis_spec_sha'],'tr4_repair_spec_sha256':repair_sha,'relation_counts':{k:int(v) for k,v in ctx['relation_counts'].items()},'country_of_origin_selected_count':0,'report_bound_files_verified':int(ctx['n_report_bound']),'answer_bound_files_verified':int(ctx['n_answer_bound']),'analysis_outputs_present_before_unlock':sorted([p.name for p in ctx['results'].glob('primary_*')]+[p.name for p in ctx['results'].glob('alignment_*')]),'boundary':'No predictive or inferential analysis executed.'}
    if pre['analysis_outputs_present_before_unlock']: raise RuntimeError(f'Protective stop: analysis-like outputs already present before unlock: {pre["analysis_outputs_present_before_unlock"]}')
    p=ctx['runtime']/'CSRA_v08_ANALYSIS_PREFLIGHT_R4_A2_TR4.json'
    if p.exists():
        old=json.loads(p.read_text())
        if old!=pre: raise RuntimeError('Existing analysis preflight differs')
    else: atomic_write_json(pre,p)
    ctx['preflight_path']=p; ctx['repair_spec_path']=repair_spec
    print('ANALYSIS RUNTIME PREFLIGHT: PASS')
    print('Report-phase audit:',EXPECTED['report_audit_sha'])
    print('Frozen analysis module:',EXPECTED['analysis_sha'])
    print('TR4 compatibility condition verified: country_of_origin selected count = 0')
    print('Analysis preflight SHA-256:',file_sha(p))
    print('ANALYSIS REMAINS LOCKED: run explicit TR4 approval next.')
    return ctx

def approve_tr4_and_unlock_analysis(ctx):
    from csra_v08_common import atomic_write_json
    req(ctx['run_root']/'CSRA_v08_REPORT_PHASE_AUDIT_R4_A2.json',EXPECTED['report_audit_sha'],'report audit before analysis unlock')
    print('Report-phase audit:',EXPECTED['report_audit_sha'])
    print('Frozen analysis module:',EXPECTED['analysis_sha'])
    phrase=input('TR4 analysis approval phrase: ').strip()
    if phrase!=APPROVAL_PHRASE: raise RuntimeError('Exact TR4 analysis approval phrase not supplied; analysis remains locked')
    rec={'protocol_id':'CSRA-0.8','study':'CSRA v0.8 R4/A2','status':'ANALYSIS_UNLOCKED_AFTER_TR4_TECHNICAL_REPAIR_BEFORE_FIRST_ANALYSIS_RESULT','scientific_change':False,'technical_change':True,'technical_repair':'TR4','approval_phrase':APPROVAL_PHRASE,'report_phase_audit_sha256':EXPECTED['report_audit_sha'],'analysis_preflight_sha256':file_sha(ctx['preflight_path']),'tr4_repair_spec_sha256':file_sha(ctx['repair_spec_path']),'frozen_analysis_module_sha256':EXPECTED['analysis_sha']}
    p=ctx['runtime']/'CSRA_v08_ANALYSIS_UNLOCK_R4_A2_TR4.json'
    if p.exists():
        old=json.loads(p.read_text())
        if old!=rec: raise RuntimeError('Existing analysis unlock differs')
    else: atomic_write_json(rec,p)
    ctx['unlock_path']=p
    print('TR4 analysis repair approved; analysis unlocked.')
    print('Analysis unlock record SHA-256:',file_sha(p))
    return p

def _require_unlock(ctx):
    p=ctx.get('unlock_path',ctx['runtime']/'CSRA_v08_ANALYSIS_UNLOCK_R4_A2_TR4.json')
    if not Path(p).exists(): raise RuntimeError('Analysis is not unlocked')
    rec=json.loads(Path(p).read_text())
    if rec.get('status')!='ANALYSIS_UNLOCKED_AFTER_TR4_TECHNICAL_REPAIR_BEFORE_FIRST_ANALYSIS_RESULT': raise RuntimeError('Invalid analysis unlock status')
    if rec.get('report_phase_audit_sha256')!=EXPECTED['report_audit_sha']: raise RuntimeError('Unlock lineage mismatch')
    return Path(p)

def _load_analysis_inputs(ctx):
    from csra_v08_common import PRIMARY_MODELS,load_json
    answer={}; main={}; alt={}; fa={}; fr={}
    ids=set(ctx['complete_ids'])
    for m in PRIMARY_MODELS:
        answer[m]=pd.read_csv(ctx['results']/f'{m}_answer_condition_summary.csv')
        main[m]=pd.read_csv(ctx['results']/f'{m}_main_report_results.csv')
        alt[m]=pd.read_csv(ctx['results']/f'{m}_alternate_report_results.csv')
        fa[m]=pd.read_csv(ctx['results']/f'{m}_fiction_answer_summary.csv')
        fr[m]=pd.read_csv(ctx['results']/f'{m}_fiction_report_results.csv')
    variances=load_json(ctx['locks']/'answer_derived_constants.json')['cie_population_variance_ddof0']
    return answer,main,alt,fa,fr,variances

def run_primary_analysis_tr4(ctx):
    _require_unlock(ctx)
    from csra_v08_common import atomic_write_json,atomic_write_csv
    from csra_v08_analysis import assemble_primary_units,primary_point_estimate,bootstrap_primary,bootstrap_summary
    answer,main,alt,fa,fr,variances=_load_analysis_inputs(ctx)
    units=assemble_primary_units(answer,main,ctx['selected'],ctx['postreview'])
    point,item_effects=primary_point_estimate(units,variances,1.0)
    atomic_write_json(point,ctx['results']/'primary_point_estimate.json'); atomic_write_csv(item_effects,ctx['results']/'primary_item_effects.csv')
    boot=bootstrap_primary(units,ctx['selected'],variances,reps=ctx['analysis_spec']['bootstrap_reps'],seed=ctx['analysis_spec']['bootstrap_seed'],checkpoint_path=ctx['checkpoints']/'primary_full_refit_bootstrap.csv')
    summary=bootstrap_summary(point,boot,ctx['analysis_spec']['minimum_valid_bootstrap'],ctx['analysis_spec']['practical_delta'])
    atomic_write_json(summary,ctx['results']/'primary_validation_result.json')
    print(json.dumps(summary,indent=2))
    return summary

def run_predictive_sensitivity_tr4(ctx):
    _require_unlock(ctx)
    from csra_v08_common import atomic_write_csv
    from csra_v08_analysis import assemble_primary_units,fixed_alpha_sensitivity,historical_nested_grid
    answer,main,alt,fa,fr,variances=_load_analysis_inputs(ctx)
    units=assemble_primary_units(answer,main,ctx['selected'],ctx['postreview'])
    sens=fixed_alpha_sensitivity(units,variances,ctx['analysis_spec'],ctx['postreview']); atomic_write_csv(sens,ctx['results']/'primary_sensitivity_summary.csv')
    hist=historical_nested_grid(answer,main,ctx['complete_ids']); atomic_write_csv(hist,ctx['results']/'historical_predictive_validation.csv')
    print('Predictive sensitivity/historical analyses: COMPLETE')
    return sens,hist

def run_alignment_analysis_tr4(ctx):
    _require_unlock(ctx)
    from csra_v08_common import atomic_write_csv,atomic_write_json
    import csra_v08_analysis as ana
    answer,main,alt,fa,fr,variances=_load_analysis_inputs(ctx)
    # TR4 in-memory compatibility repair. Frozen module bytes are unchanged.
    ana._design_vector=_tr4_design_vector
    data=ana.assemble_alignment_data(answer,main,ctx['selected'],ctx['postreview']); atomic_write_csv(data,ctx['results']/'alignment_analysis_dataset.csv')
    point,delta=ana.alignment_point(data)
    boot=ana.alignment_bootstrap(data,ctx['selected'],reps=ctx['analysis_spec']['alignment']['bootstrap_reps'],seed=ctx['analysis_spec']['alignment']['bootstrap_seed'],checkpoint_path=ctx['checkpoints']/'alignment_common_item_bootstrap.csv')
    summary,delta=ana.alignment_bootstrap_summary(point,delta,boot,ctx['analysis_spec']['alignment']['minimum_valid_bootstrap'])
    atomic_write_json(summary,ctx['results']/'alignment_results.json'); atomic_write_csv(delta,ctx['results']/'model_specific_alignment_deltas.csv')
    print(json.dumps(summary,indent=2))
    return summary,delta

def run_robustness_controls_tr4(ctx):
    _require_unlock(ctx)
    from csra_v08_common import atomic_write_csv
    from csra_v08_analysis import wording_robustness,fictional_summary
    answer,main,alt,fa,fr,variances=_load_analysis_inputs(ctx)
    word=wording_robustness(main,alt); atomic_write_csv(word,ctx['results']/'wording_robustness_summary.csv')
    fic=fictional_summary(fa,fr); atomic_write_csv(fic,ctx['results']/'fictional_control_summary.csv')
    print('Wording robustness and fictional controls: COMPLETE')
    return word,fic

def finalize_analysis_tr4(ctx):
    from csra_v08_common import atomic_write_json
    unlock=_require_unlock(ctx)
    expected=['primary_point_estimate.json','primary_item_effects.csv','primary_validation_result.json','primary_sensitivity_summary.csv','historical_predictive_validation.csv','alignment_analysis_dataset.csv','alignment_results.json','model_specific_alignment_deltas.csv','wording_robustness_summary.csv','fictional_control_summary.csv']
    files={}
    for name in expected:
        p=ctx['results']/name
        if not p.exists(): raise RuntimeError(f'Missing frozen analysis output: {name}')
        files[str(p.relative_to(ctx['run_root']))]=file_sha(p)
    for name in ['primary_full_refit_bootstrap.csv','alignment_common_item_bootstrap.csv']:
        p=ctx['checkpoints']/name
        if not p.exists(): raise RuntimeError(f'Missing bootstrap checkpoint: {name}')
        d=pd.read_csv(p)
        if len(d)!=2000 or d['replicate'].nunique()!=2000: raise RuntimeError(f'{name} does not contain exactly 2000 unique replicates')
        files[str(p.relative_to(ctx['run_root']))]=file_sha(p)
    audit={'protocol_id':'CSRA-0.8','study':'CSRA v0.8 R4/A2','artifact_type':'analysis-phase audit manifest','status':'ANALYSIS_COMPLETE_AFTER_TR4_TECHNICAL_REPAIR','scientific_change':False,'technical_change':True,'technical_repair':'TR4','report_phase_audit_sha256':EXPECTED['report_audit_sha'],'analysis_unlock_sha256':file_sha(unlock),'frozen_analysis_module_sha256':EXPECTED['analysis_sha'],'analysis_spec_sha256':EXPECTED['analysis_spec_sha'],'tr4_repair_spec_sha256':file_sha(ctx['repair_spec_path']),'files':files}
    p=ctx['run_root']/'CSRA_v08_ANALYSIS_PHASE_AUDIT_R4_A2.json'
    if p.exists():
        old=json.loads(p.read_text())
        if old!=audit: raise RuntimeError('Existing analysis audit differs; protective stop')
    else: atomic_write_json(audit,p)
    print('ANALYSIS PHASE FINALIZATION: PASS UNDER DOCUMENTED TR4 REPAIR')
    print('Analysis-phase audit SHA-256:',file_sha(p))
    return audit
