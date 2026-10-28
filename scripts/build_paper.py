"""Rebuild V10 manuscript tables/figures from numerical sources.

The reference LaTeX supplies captions and layout only. Numerical table bodies
are calculated or rendered from identified data sources and checked against V10.
"""
from pathlib import Path
import argparse,csv,hashlib,json,re,shutil,subprocess,sys,tempfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
TARGETS=['granite','phi','qwen']
NAMES={k:k.title() for k in TARGETS}
POLICIES=['visible','source','observer','probe','probe_source','probe_observer']


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def body(text):
    return text.split(r'\midrule',1)[1].rsplit(r'\bottomrule',1)[0].strip()
def numbers(text):
    text=re.sub(r'\\\[[^]]*\]','',text)
    text=re.sub(r'\\multicolumn\{\d+\}\{[^}]*\}', '',text)
    return re.findall(r'(?<![A-Za-z\d])[-+]?\d+(?:\.\d+)?(?![A-Za-z\d])',text)
def interval(point,lo,hi,digits=4,scale=1):
    return r'\shortstack{$'+f'{scale*point:.{digits}f}'+r'$\\$['+rf'{scale*lo:.{digits}f},\,'+f'{scale*hi:.{digits}f}'+r']$}'
def row(cells):return ' & '.join(map(str,cells))+r' \\'
def one(frame,**kwargs):
    mask=np.ones(len(frame),bool)
    for k,v in kwargs.items():mask &= frame[k].eq(v).to_numpy()
    result=frame.loc[mask]
    if len(result)!=1:raise ValueError((kwargs,len(result)))
    return result.iloc[0]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis',type=Path,required=True,help='Regenerated October 27 results')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    tables=out/'tables';figures=out/'figures';tables.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    refs=ROOT/'paper/reference/tables';coverage=[];cells=[]
    binding=read(ROOT/'paper/manuscript_binding.json')
    order={x['label'].split(':')[1]:x['number'] for x in binding['tables']}
    display={x['label'].split(':')[1]:x['display_label'] for x in binding['tables']}
    def emit(key,rows,sources,mode='derived from released numerical data'):
        reference=(refs/(key+'.tex')).read_text()
        rows=list(rows);actual='\n'.join(rows)
        expected_numbers=numbers(body(reference));actual_numbers=numbers(actual)
        # Compare rounded numeric values, so harmless formatting and + signs do
        # not conceal a changed estimate. Preserve multiplicity and row order.
        if len(expected_numbers)!=len(actual_numbers) or not np.allclose(np.array(expected_numbers,float),np.array(actual_numbers,float),rtol=0,atol=1e-12):
            raise AssertionError(f'Article table mismatch: {key}\nexpected {expected_numbers}\nactual {actual_numbers}')
        prefix=reference.split(r'\midrule',1)[0]+r'\midrule'+'\n'
        suffix=r'\bottomrule'+reference.rsplit(r'\bottomrule',1)[1]
        (tables/(key+'.tex')).write_text(prefix+actual+'\n'+suffix)
        with (tables/(key+'.csv')).open('w',newline='') as f:
            writer=csv.writer(f)
            data=[r.rstrip(' \\').split(' & ') for r in rows if ' & ' in r]
            width=max(map(len,data))
            writer.writerow(['row']+[f'column_{i+1}' for i in range(width)])
            for i,values in enumerate(data):writer.writerow([i+1,*values])
        for i,(e,a) in enumerate(zip(expected_numbers,actual_numbers),1):cells.append({'table':key,'numeric_position':i,'article_display':e,'reproduced_display':a,'absolute_difference':abs(float(e)-float(a))})
        coverage.append({'number':order[key],'display_label':display[key],'label':'tab:'+key,'table':key,'numeric_entries':len(actual_numbers),'status':'PASSED','mode':mode,'sources':sources})
    with tempfile.TemporaryDirectory(prefix='csra_paper_assets_') as tmp:
        project=Path(tmp);shutil.copytree(ROOT/'original_reanalysis/results',project/'revision_analysis/results')
        subprocess.run([sys.executable,'-B',str(ROOT/'original_reanalysis/code/make_manuscript_assets.py'),'--repo',str(ROOT/'original_release'),'--audit',str(ROOT/'original_reanalysis/audit'),'--project',str(project)],check=True)
        for p in sorted((project/'generated').glob('*.tex')):
            label=re.search(r'\\label\{tab:([^}]+)\}',p.read_text()).group(1)
            if (refs/(label+'.tex')).exists():
                sources={
                    'scalar_prediction':['original_reanalysis/audit/results/predictive_intervals.csv'],
                    'full_role_prediction':['original_reanalysis/results/full_role_prediction.csv'],
                    'alignment':['original_reanalysis/results/alignment_restandardized_bootstrap.csv'],
                    'descriptives':['original_reanalysis/audit/results/reconstructed_primary_units.csv','original_release/data/supporting/*_answer_condition_summary.csv'],
                    'all_roles':['original_reanalysis/audit/results/report_role_distributions.csv'],
                    'controls':['original_reanalysis/audit/results/fictional_controls_by_frame.csv'],
                    'correlations':['original_reanalysis/audit/results/alignment_scale_comparison.csv'],
                    'historical_alignment':['original_reanalysis/audit/results/alignment_intervals.csv'],
                    'alternative_targets':['original_reanalysis/results/alternative_targets.csv'],
                    'original_sensitivity':['original_release/expected_results/primary_sensitivity_summary.csv']}
                emit(label,body(p.read_text()).splitlines(),sources[label], 'rendered from released original-study summaries and archived statistical refits')
        for name in ['descriptive_diagnostics','predictive_baselines']:shutil.copy2(project/'figures'/(name+'.pdf'),figures/(name+'.pdf'))
        shutil.copy2(project/'csra_framework.pdf',figures/'csra_framework.pdf')
    # Review totals use the recovered audit and actual final selection, rather
    # than typing the manuscript's counts back into the output.
    review=pd.read_csv(ROOT/'original_reanalysis/audit/recovered_provenance/factual_review/csra_v08_blinded_common_review_AUDITED.csv')
    selected=pd.read_csv(ROOT/'original_release/data/final_common_items.csv')
    rows=[];totals=np.zeros(4,int)
    for rel,name in [('country_of_origin','Country of origin'),('developer','Developer'),('formation_location','Formation location'),('original_language','Original language'),('original_network','Original network')]:
        part=review[review.relation.eq(rel)];approved=part.review_status.eq('APPROVE').sum()
        values=np.array([len(part),approved,len(part)-approved,selected.relation.eq(rel).sum()]);totals+=values
        rows.append(row([name,*values]))
    rows += [r'\midrule',row(['Total',*totals])]
    emit('retention_stages',rows,['original_reanalysis/audit/recovered_provenance/factual_review/csra_v08_blinded_common_review_AUDITED.csv','original_release/data/final_common_items.csv'])
    primary=pd.read_csv(ROOT/'studies/factorial/run1/primary_analysis/primary.csv')
    rows=[]
    for key,name in [('ownership','OWN minus OTHER'),('hedging','Hedged minus certain'),('visibility','Answer shown minus hidden'),('predictive_visibility','Shown minus hidden OWN increment')]:
        r=one(primary,contrast=key);scale=1 if key=='predictive_visibility' else 100;digits=6 if scale==1 else 3
        ci=r'\shortstack{$['+f'{scale*r.family_ci_low:.{digits}f}'+r',\,$\\$'+f'{scale*r.family_ci_high:.{digits}f}'+r']$}'
        rows.append(row([name,f'{scale*r.estimate:.{digits}f}',ci,'fit units' if scale==1 else 'pp']))
    emit('factorial_primary',rows,['studies/factorial/run1/primary_analysis/primary.csv','studies/factorial/run1/primary_analysis/primary_draws.csv'],'primary point estimates and archived 10,000-refit bootstrap; raw category estimates checked before rendering')
    factors=pd.read_csv(args.analysis/'factorial_contrasts.csv');association=pd.read_csv(args.analysis/'factorial_answer_association.csv')
    rows=[]
    for t in TARGETS:
        a=one(factors,target=t,outcome='q_both',contrast='visibility');b=one(factors,target=t,outcome='document_minus_prior',contrast='visibility');c=one(association,target=t,visibility='shown_minus_hidden')
        rows.append(row([NAMES[t],interval(a.estimate,a.ci_low,a.ci_high,4 if t=='qwen' else 2,100),interval(b.estimate,b.ci_low,b.ci_high,2,100),interval(c.estimate,c.ci_low,c.ci_high,3)]))
    emit('factorial_expanded',rows,['regenerated revision/factorial_contrasts.csv','regenerated revision/factorial_answer_association.csv'],'newly recomputed 57-group bootstrap')
    costs=read(ROOT/'studies/fresh40/run1/analysis/actual_costs.json')
    # Counts are specified by the frozen rotations/permutations and strengths.
    config=read(ROOT/'studies/fresh40/run1/frozen_package/protocol/main_study_plan.json')
    report_calls=config['report_strengths']*len(config['ownership_cues'])*config['report_rotations'];probe_calls=config['answer_permutations']
    assert costs['policy_calls_per_target_item']=={'visible':0,'self':report_calls,'observer':report_calls,'probe':probe_calls,'probe_self':probe_calls+report_calls,'probe_observer':probe_calls+report_calls}
    emit('policy_access',[
        row(['Visible inputs','Two text forecasts, relation and strength; no direct target scores',0]),
        row(['Source report',r'Target-corresponding reporter\textquotesingle s three log ratios',report_calls]),
        row(['Selected observer',r'Development-selected different reporter\textquotesingle s three log ratios',report_calls]),
        row(['Direct probe','Symmetric no-document support contrasts and entropy',probe_calls]),
        row(['Probe + report','Direct probe plus source or selected-observer report',report_calls+probe_calls])],['studies/fresh40/run1/frozen_package/protocol/main_study_plan.json','studies/fresh40/run1/frozen_package/code/learning.py'],'design/access definitions and configured forward counts')
    contrasts=pd.read_csv(args.analysis/'fresh_pooled.csv');targets=pd.read_csv(args.analysis/'fresh_by_target.csv')
    predictions=pd.read_csv(ROOT/'new_analysis/posthoc_20261026/results/per_case_predictions.csv')
    absolute=[]
    for keys,df in predictions.groupby(['outcome','variant','algorithm','target']):
        for pol in POLICIES:
            mse=np.mean((df.Y-df[pol+'_prediction'])**2)
            absolute.append(dict(zip(['outcome','variant','algorithm','target'],keys),policy=pol,R2=1-mse/np.var(df.Y),MSE=mse))
    absolute=pd.DataFrame(absolute);absolute.to_csv(out/'absolute_performance.csv',index=False)
    source=ROOT/'studies/fresh40/run1/analysis/qualification_summary.json';summary=read(source);observer=summary['observer_selection']
    pairs=[('source_over_observer','Source versus observer'),('source_over_visible','Source versus visible'),('source_added_to_probe','Source added to probe'),('source_alone_over_probe','Source alone versus probe alone')]
    for algorithm,label in [('gradient_boosting','fresh_primary'),('ridge','fresh_ridge')]:
        rows=[]
        current=pairs if algorithm=='gradient_boosting' else [pairs[1],pairs[0],pairs[2],pairs[3]]
        for key,name in current:
            r=one(contrasts,outcome='log_support',variant='original',algorithm=algorithm,contrast=key)
            ci=interval(r.estimate,r.ci_low,r.ci_high)
            if algorithm=='ridge' and key=='source_over_observer':
                ci=r'\shortstack{$'+f'{r.estimate:.4f}'+r'$\\$['+rf'{r.ci_low:.4f},\,{r.ci_high:.5f}'+r']$}'
            rows.append(row([name,ci]))
        emit(label,rows,['regenerated revision/fresh_pooled.csv'],'newly recomputed paired 40-group intervals from fixed predictions')
    rows=[]
    for t in TARGETS:
        values=[one(absolute,outcome='log_support',variant='original',algorithm='gradient_boosting',target=t,policy=pol).R2 for pol in POLICIES]
        rows.append(row([NAMES[t],NAMES[observer[t]],*[f'{x:.3f}' for x in values]]))
    emit('fresh_by_target',rows,['new_analysis/posthoc_20261026/results/per_case_predictions.csv',str(source.relative_to(ROOT))],'fresh R2 recomputed from per-case predictions')
    rows=[]
    for outcome,variant,name in [('log_support','original','Log support; base features'),('log_support','agreement','Log support; agreement added'),('bounded_probability','original','Bounded probability; base features'),('bounded_probability','agreement','Bounded probability; agreement added')]:
        entries=[name]
        for alg in ['gradient_boosting','ridge']:
            r=one(contrasts,outcome=outcome,variant=variant,algorithm=alg,contrast='source_added_to_probe');entries.append(interval(r.estimate,r.ci_low,r.ci_high))
        rows.append(row(entries))
    emit('csra_posthoc_robustness',rows,['regenerated revision/fresh_pooled.csv'])
    rows=[]
    for outcome,name in [('log_support','Log support'),('bounded_probability','Bounded')]:
        for alg,title in [('gradient_boosting','Boosting'),('ridge','Ridge')]:
            entries=[name,title]
            for t in TARGETS:
                r=one(targets,outcome=outcome,variant='agreement',algorithm=alg,contrast='source_added_to_probe',target=t);entries.append(interval(r.estimate,r.ci_low,r.ci_high))
            rows.append(row(entries))
    emit('csra_posthoc_targets',rows,['regenerated revision/fresh_by_target.csv'],'newly recomputed target-specific paired intervals')
    rows=[]
    for outcome,name in [('log_support','Log-support outcome'),('bounded_probability','Bounded probability outcome')]:
        if rows:rows.append(r'\midrule')
        rows.append(r'\multicolumn{7}{@{}l}{\textit{'+name+r'}} \\')
        for t in TARGETS:
            values=[one(absolute,outcome=outcome,variant='agreement',algorithm='gradient_boosting',target=t,policy=pol).R2 for pol in POLICIES]
            rows.append(row([NAMES[t],*[f'${x:.3f}$' for x in values]]))
    emit('csra_posthoc_absolute',rows,['new_analysis/posthoc_20261026/results/per_case_predictions.csv'],'fresh R2 recomputed from per-case predictions')
    dev={}
    for alg,folder in [('ridge','studies/development/run1/analysis/ridge'),('gradient_boosting','studies/development/secondary_refit/gradient_boosting')]:dev[alg]=read(ROOT/folder/'development_summary.json')
    rows=[]
    for t in TARGETS:
        entries=[NAMES[t]]
        for alg in ['ridge','gradient_boosting']:
            d=next(x for x in dev[alg]['known_target_diagnostics'] if x['target']==t)
            entries.append(r'\shortstack{'+f"{d['R2_native']:.6f} / {d['R2_second_observer']:.6f}"+r'\\'+f"{d['delta_R2']:+.6f}"+'}')
        rows.append(row(entries))
    rows.append(row(['Mean difference',*[f"{dev[a]['equal_target_mean_delta_R2']:+.6f}" for a in ['ridge','gradient_boosting']]]))
    emit('development_results',rows,['studies/development/run1/analysis/ridge/development_summary.json','studies/development/secondary_refit/gradient_boosting/development_summary.json'],'development summaries independently checked against held-out errors')
    rows=[]
    for t in ['qwen','granite','phi']:
        m=config['models'][t];rows+=[row([NAMES[t],r'\path{'+m['model_id']+'}']),row(['',r'\texttt{'+m['revision']+'}'])]
    emit('frozen_models',rows,['studies/fresh40/run1/frozen_package/protocol/main_study_plan.json'],'immutable model identities, not a numerical estimate')
    # Render later figures from numerical files, replacing historical rounded
    # constants with the underlying estimates and explicitly exporting plot data.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':10.5,'axes.labelsize':11,'savefig.facecolor':'white'})
    colors=['#1f4e79','#b25b24','#33785b'];y=np.arange(3)
    fig,ax=plt.subplots(1,2,figsize=(7,2.6833333333),gridspec_kw={'width_ratios':[1.1,1]},layout='constrained')
    cat=primary[primary.contrast.isin(['ownership','hedging','visibility'])].set_index('contrast').loc[['ownership','hedging','visibility']]
    means=100*cat.estimate.to_numpy();lo=100*cat.family_ci_low.to_numpy();hi=100*cat.family_ci_high.to_numpy()
    ax[0].errorbar(means,y,xerr=[means-lo,hi-means],fmt='o',color=colors[0],capsize=3);ax[0].axvline(0,color='0.5',lw=.7);ax[0].set_yticks(y,['Ownership','Hedging','Answer visibility']);ax[0].invert_yaxis();ax[0].set_xlabel('Change in both-source\nsupport (pp)');ax[0].set_title('Primary contrasts\n(98.75% intervals)')
    per=pd.read_csv(ROOT/'studies/factorial/run1/primary_analysis/per_model_category_contrasts.csv')
    for j,t in enumerate(TARGETS):
        values=[100*one(per,model=t,contrast=c).estimate for c in ['ownership','hedging','visibility']]
        ax[1].plot(values,y+(j-1)*.15,'o',label=NAMES[t],color=colors[j],markersize=4)
    ax[1].axvline(0,color='0.5',lw=.7);ax[1].set_yticks(y,[]);ax[1].invert_yaxis();ax[1].set_xlabel('Change in both-source\nsupport (pp)');ax[1].set_title('System point estimates');ax[1].legend(frameon=False,loc='upper right',fontsize=10)
    for ext in ['pdf','png']:fig.savefig(figures/('factorial.'+ext),dpi=240)
    plt.close(fig);cat.to_csv(out/'factorial_plot_data.csv')
    fig,ax=plt.subplots(1,2,figsize=(7,2.8777777778),gridspec_kw={'width_ratios':[1.1,1]},layout='constrained')
    keys=['self_over_visible','self_over_selected_observer','self_added_to_probe'];cc=summary['results']['gradient_boosting']['contrasts']
    mean=np.array([cc[k]['normalized_MSE_reduction'] for k in keys]);ints=np.array([cc[k]['descriptive_95pct_cluster_bootstrap_interval'] for k in keys])
    ax[0].errorbar(mean,y,xerr=[mean-ints[:,0],ints[:,1]-mean],fmt='o',color=colors[0],capsize=3);ax[0].axvline(0,color='0.5',lw=.7);ax[0].set_yticks(y,['Source vs visible','Source vs observer','Source added to probe']);ax[0].invert_yaxis();ax[0].set_xlabel('Normalized MSE\nreduction');ax[0].set_title('Primary contrasts\n(descriptive 95% intervals)')
    for j,t in enumerate(TARGETS):ax[1].plot([cc[k]['by_target'][t] for k in keys],y+(j-1)*.15,'o',label=NAMES[t],color=colors[j],markersize=4)
    ax[1].axvline(0,color='0.5',lw=.7);ax[1].set_yticks(y,[]);ax[1].invert_yaxis();ax[1].set_xlabel('Normalized MSE\nreduction');ax[1].set_title('Target point estimates');ax[1].legend(frameon=False,loc='lower right',fontsize=10)
    for ext in ['pdf','png']:fig.savefig(figures/('fresh40.'+ext),dpi=240)
    plt.close(fig)
    fresh=pd.read_csv(ROOT/'studies/fresh40/run1/analysis/per_case_predictions.csv');fresh=fresh[fresh.algorithm.eq('gradient_boosting')]
    fig,axes=plt.subplots(1,3,figsize=(7,2.6833333333),sharey=True,layout='constrained');rng=np.random.default_rng(20261021)
    for j,(ax,t) in enumerate(zip(axes,TARGETS)):
        values=[fresh[fresh.target.eq(t)&fresh.strength.eq(st)].Y.to_list() for st in ['SKEPTICAL','PROVISIONAL','BARE']]
        ax.boxplot(values,patch_artist=True,showfliers=False,boxprops={'facecolor':'#e4e9ef','linewidth':.8},medianprops={'color':colors[j]},whiskerprops={'linewidth':.7},capprops={'linewidth':.7})
        for k,v in enumerate(values):ax.scatter(rng.normal(k+1,.045,len(v)),v,s=7,alpha=.45,c=colors[j],edgecolors='none')
        ax.axhline(0,color='0.7',lw=.6);ax.set_xticks([1,2,3],['Skeptical','Provisional','Bare'],rotation=20);ax.set_title(NAMES[t])
    axes[0].set_ylabel('Substitution effect (nats)')
    for ext in ['pdf','png']:fig.savefig(figures/('outcomes.'+ext),dpi=240)
    plt.close(fig)
    assert len(coverage)==22,len(coverage)
    expected=set(p.stem for p in refs.glob('*.tex'));assert {x['table'] for x in coverage}==expected
    pd.DataFrame(cells).to_csv(out/'table_numeric_comparison.csv',index=False)
    coverage.sort(key=lambda x:x['number'])
    (out/'table_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    report={'status':'PASSED','manuscript_revision':'V10 Neurocomputing','tables_rebuilt':len(coverage),'main_tables_rebuilt':10,'supplementary_tables_rebuilt':12,'numeric_entries_checked':len(cells),'maximum_displayed_numeric_difference':max(x['absolute_difference'] for x in cells),'figures_rebuilt':6,'new_model_calls':0,'scope':'All table bodies regenerated from identified numerical/design records; historical statistical refit results are inputs where stated. Later grouped intervals recomputed. Figure numerical precision may exceed the original rounded plotting constants; PDF metadata/font bytes need not match.'}
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
