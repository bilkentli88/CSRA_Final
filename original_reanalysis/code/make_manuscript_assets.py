"""Generate manuscript tables and vector figures from verified released scores.

No model calls. Run after audit_csra.py and extend_analysis.py; see README.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

MODELS = ['qwen', 'granite', 'phi']
NAMES = {'qwen':'Qwen', 'granite':'Granite', 'phi':'Phi', 'equal_model_mean':'Equal-model mean', 'H':r'\(H\)'}
STRENGTHS = ['SKEPTICAL', 'PROVISIONAL', 'BARE']
ROLES = ['document', 'prior', 'both', 'uncertain']
COLORS = ['#246b94', '#c48625', '#8b689f', '#b7c1c7']


def interval_cell(row, digits=4):
    a,b,c = (float(row[k]) for k in ['estimate','ci_low','ci_high'])
    return r'\shortstack{$'+f'{a:.{digits}f}'+r'$\\$['+f'{b:.{digits}f},'+r'\,'+f'{c:.{digits}f}'+r']$}'


def table(path, caption, label, header, rows, spec):
    text = '\\begin{table}[tbp]\n\\centering\\small\n'
    text += '\\caption{'+caption+'}\n\\label{'+label+'}\n'
    text += '\\renewcommand{\\arraystretch}{1.25}\n\\begin{tabular}{'+spec+'}\n\\toprule\n'
    text += header+r'\\'+'\n\\midrule\n'
    text += '\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    out = args.project/'generated'; figdir = args.project/'figures'
    out.mkdir(exist_ok=True);figdir.mkdir(exist_ok=True)
    new = args.project/'revision_analysis/results'
    old = args.audit/'results'
    scalar = pd.read_csv(old/'predictive_intervals.csv')
    full = pd.read_csv(new/'full_role_prediction.csv')
    for filename, source, caption, label in [
        ('scalar_prediction.tex', scalar[scalar.effect.eq('SELF')],
         r'SELF increment after PEER with scalar VSA. Values are changes in out-of-fold $R^2$ with 95\% paired item-bootstrap intervals. The answer-omitted analysis changes the predictors only; every report prompt still displays the recorded answer.', 'tab:scalar_prediction'),
        ('full_role_prediction.tex', full,
         r'Exploratory SELF increments with all three relative four-role log-score contrasts for each frame. The mean with the recorded-answer predictor remains below 0.01, but individual-model intervals and the answer-omitted analysis are less decisive.', 'tab:full_role_prediction')]:
        rows=[]
        for m in MODELS+['equal_model_mean']:
            cells=[NAMES[m]]
            for case in ['with_recorded_answer','without_recorded_answer']:
                cells.append(interval_cell(source[source.model_key.eq(m)&source.case.eq(case)].iloc[0],5))
            rows.append(' & '.join(cells)+r'\\[5pt]')
        table(out/filename,caption,label,r'Model & With answer predictor & Answer predictor omitted',rows,'@{}lcc@{}')
    alignment=pd.read_csv(new/'alignment_restandardized_bootstrap.csv')
    rows=[]
    for m in MODELS+['equal_model_mean','H']:
        cells=[NAMES[m]]
        for est in ['correlation_difference','conditional_frame_contrast']:
            cells.append(interval_cell(alignment[alignment.model_key.eq(m)&alignment.estimand.eq(est)].iloc[0]))
        rows.append(' & '.join(cells)+r'\\[5pt]')
    table(out/'alignment.tex',r'Exploratory alignment contrasts with standardization re-estimated within each of 2,000 paired item-bootstrap draws. Intervals are marginal 95\% percentile intervals. $H$ compares Qwen with the average of Granite and Phi.', 'tab:alignment',r'Model & $\Delta^{\mathrm{corr}}$ & $\Delta^{\mathrm{cond}}$',rows,'@{}lcc@{}')

    unit=pd.read_csv(old/'reconstructed_primary_units.csv')
    desc=[]
    for m in MODELS:
        answer=pd.read_csv(args.repo/f'data/supporting/{m}_answer_condition_summary.csv')
        baseline=answer[answer.condition.eq('NO_DOC')][['item_id','frozen_answer_role']].rename(columns={'frozen_answer_role':'baseline_answer'})
        frame=unit[unit.model_key.eq(m)].merge(baseline,on='item_id',validate='many_to_one')
        assert frame.baseline_answer.eq('P').all()
        for s in STRENGTHS:
            g=frame[frame.strength.eq(s)]
            desc.append({'model_key':m,'strength':s,'n':len(g),'CIE_mean':g.CIE.mean(),'CIE_sd':g.CIE.std(ddof=0),'CIE_median':g.CIE.median(),'CIE_q25':g.CIE.quantile(.25),'CIE_q75':g.CIE.quantile(.75),'flips':int(g.frozen_answer_role.ne(g.baseline_answer).sum()),'to_C':int(g.frozen_answer_role.eq('C').sum()),'to_I':int(g.frozen_answer_role.eq('I').sum())})
    pd.DataFrame(desc).to_csv(new/'descriptive_statistics_with_flips.csv',index=False)
    rows=[]
    for r in desc:
        rows.append(f"{NAMES[r['model_key']]} & {r['strength'].title()} & {r['CIE_mean']:.2f} ({r['CIE_sd']:.2f}) & {r['CIE_median']:.2f} [{r['CIE_q25']:.2f}, {r['CIE_q75']:.2f}] & {r['flips']} ({r['to_C']}/{r['to_I']})"+r'\\')
    table(out/'descriptives.tex',r'CIE distributions in natural-log units and modal answer changes, with 100 items per row. Every no-document recorded answer is $P$. The last column gives total changes from that answer, followed by counts to $C$ and $I$. SD uses the population convention.', 'tab:descriptives',r'Model & Strength & Mean (SD) & Median [IQR] & Flips ($C/I$)',rows,'@{}llrrr@{}')

    reports=pd.read_csv(old/'report_role_distributions.csv')
    rows=[]
    for m in MODELS:
        for f in ['PEER','SELF']:
            g=reports[reports.model_key.eq(m)&reports.frame.eq(f)&reports.condition_group.eq('PRIMARY_C')].set_index('report_role')
            rows.append(' & '.join([NAMES[m],f]+[str(int(g.loc[r,'count'])) for r in ROLES])+r'\\')
    table(out/'primary_roles.tex',r'Modal report roles across the three $C$-asserting strengths (300 units per model and frame). Modal categories use the maximum permutation-averaged report log support.', 'tab:primary_roles','Model & Frame & Document & Prior & Both & Uncertain',rows,'@{}llrrrr@{}')

    rows=[]
    conditions=['C_SKEPTICAL','C_PROVISIONAL','C_BARE','P_PROVISIONAL','I_PROVISIONAL']
    abbrev={'C_SKEPTICAL':r'$C$: skeptical','C_PROVISIONAL':r'$C$: provisional','C_BARE':r'$C$: bare','P_PROVISIONAL':r'$P$: provisional','I_PROVISIONAL':r'$I$: provisional'}
    for m in MODELS:
        for cond in conditions:
            cells=[NAMES[m],abbrev[cond]]
            for f in ['PEER','SELF']:
                g=reports[reports.model_key.eq(m)&reports.frame.eq(f)&reports.condition_group.eq(cond)].set_index('report_role')
                cells.append('/'.join(str(int(g.loc[r,'count'])) for r in ROLES))
            rows.append(' & '.join(cells)+r'\\')
        if m!='phi':rows.append(r'\midrule')
    table(out/'all_roles.tex',r'Complete modal report distributions for all five principal conditions. Each cell lists document/prior/both/uncertain counts and sums to 100. These are role selections, not calibrated attribution probabilities.', 'tab:all_roles','Model & Document condition & PEER ($D/P/B/U$) & SELF ($D/P/B/U$)',rows,'@{}llrr@{}')

    controls=pd.read_csv(old/'fictional_controls_by_frame.csv')
    rows=[]
    for m in MODELS:
        for f in ['PEER','SELF']:
            r=controls[controls.model_key.eq(m)&controls.frame.eq(f)].iloc[0]
            rows.append(f"{NAMES[m]} & {f} & {int(r.document_report_count)}/10 & {r.mean_VSA:.2f}"+r'\\')
    table(out/'controls.tex',r'Fictional-control reports by frame. All three models select the contextual answer in 10/10 controls. Granite\textquotesingle s PEER reports show why answer sensitivity alone does not validate report faithfulness.', 'tab:controls','Model & Frame & Document reports & Mean VSA',rows,'@{}llrr@{}')

    scale=pd.read_csv(old/'alignment_scale_comparison.csv')
    rows=[]
    for r in scale.itertuples():
        rows.append(f'{NAMES[r.model_key]} & {r.r_CIE_PEER:.3f} & {r.r_CIE_SELF:.3f} & {r.SD_SELF_div_PEER:.3f} & {r.spearman_CIE_PEER:.3f} & {r.spearman_CIE_SELF:.3f}'+r'\\')
    table(out/'correlations.tex',r'Unadjusted CIE--VSA association and relative report-score dispersion on the 300 primary units per model. Rank correlations are descriptive point estimates.', 'tab:correlations',r'Model & $r_{\rm PEER}$ & $r_{\rm SELF}$ & $s_{\rm SELF}/s_{\rm PEER}$ & $\rho_{\rm PEER}$ & $\rho_{\rm SELF}$',rows,'@{}lrrrrr@{}')
    historical=pd.read_csv(old/'alignment_intervals.csv')
    rows=[]
    for m,label in [('qwen','Delta_qwen'),('granite','Delta_granite'),('phi','Delta_phi'),('equal_model_mean','common'),('H','H')]:
        cells=[NAMES[m]]
        for s in ['pooled','within_frame']:
            cells.append(interval_cell(historical[historical.scale.eq(s)&historical.estimand.eq(label)].iloc[0]))
        rows.append(' & '.join(cells)+r'\\[5pt]')
    table(out/'historical_alignment.tex',r'Historical alignment checks using normalization fixed at original-sample values. The pooled-scale column reproduces the original five bootstrap intervals. The within-frame column is the earlier fixed-normalization diagnostic; revised restandardized intervals are in Table~\ref{tab:alignment}. These are not interchangeable inferential procedures.', 'tab:historical_alignment',r'Model & Pooled-scale slope difference & Within-frame difference',rows,'@{}lcc@{}')
    alt=pd.read_csv(new/'alternative_targets.csv')
    rows=[]
    for r in alt.itertuples():
        name=r'$C$ vs. no document' if r.target=='C_vs_no_document' else 'Probability contrast'
        rows.append(f'{NAMES[r.model_key]} & {name} & {r.correlation_with_original:.3f} & {r.theta_SELF:.5f}'+r'\\')
    table(out/'alternative_targets.tex',r'Alternative-target point checks using scalar VSA and the original answer-containing baseline. No bootstrap interval or practical-threshold decision is assigned to these checks.', 'tab:alternative_targets',r'Model & Target & $r$ with original CIE & $\theta_{\rm SELF}$',rows,'@{}llrr@{}')
    sensitivity=pd.read_csv(args.repo/'expected_results/primary_sensitivity_summary.csv')
    rows=[]
    for r in sensitivity.itertuples():
        name=(r'$\alpha='+r.setting+'$') if r.analysis=='alpha' else ('Fold '+r.setting.split('_')[-1])
        rows.append(name+f' & {r.theta_SELF:.6f} & {r.theta_PEER:.6f}'+r'\\')
    table(out/'original_sensitivity.tex',r'Originally specified penalty and fold point checks for scalar VSA with the recorded-answer predictor. They assess neither the answer-omitted baseline nor a broader class of prediction functions.', 'tab:original_sensitivity',r'Setting & $\theta_{\rm SELF}$ & $\theta_{\rm PEER}$',rows,'@{}lrr@{}')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(8.0,6.5),gridspec_kw={'height_ratios':[1.25,1]},layout='constrained')
    for j,m in enumerate(MODELS):
        ax=axes[0,j]; frame=unit[unit.model_key.eq(m)]
        data=[frame[frame.strength.eq(s)].CIE for s in STRENGTHS]
        bp=ax.boxplot(data,positions=range(3),widths=.5,patch_artist=True,showfliers=True,flierprops={'markersize':2,'alpha':.35})
        for patch in bp['boxes']:patch.set(facecolor=COLORS[0],alpha=.55)
        ax.axhline(0,c='#777777',lw=.7,ls='--');ax.set_title(NAMES[m],fontweight='bold')
        ax.set_xticks(range(3),['Skeptical','Provisional','Bare'],rotation=25,ha='right',fontsize=9)
        ax.set_ylabel('CIE (nats)' if j==0 else '')
        ax=axes[1,j];bottom=np.zeros(2)
        for role,c in zip(ROLES,COLORS):
            values=[]
            for f in ['PEER','SELF']:
                x=reports[reports.model_key.eq(m)&reports.frame.eq(f)&reports.condition_group.eq('PRIMARY_C')&reports.report_role.eq(role)]
                values.append(float(x['share'].iloc[0]))
            ax.bar([0,1],values,bottom=bottom,color=c,width=.55,label=role.title())
            bottom+=values
        ax.set_xticks([0,1],['PEER','SELF']);ax.set_ylim(0,1.03)
        ax.set_ylabel('Modal report share' if j==0 else '')
        ax.set_yticks([0,.5,1],['0%','50%','100%'])
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False,fontsize=9)
    fig.savefig(figdir/'descriptive_diagnostics.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(8,4.2),layout='constrained')
    for j,case in enumerate(['with_recorded_answer','without_recorded_answer']):
        ax=axes[j]
        for k,(source,offset,color,label) in enumerate([(scalar[scalar.effect.eq('SELF')],.10,COLORS[0],'Scalar VSA'),(full,-.10,COLORS[2],'Four-role profile')]):
            for y,m in enumerate(MODELS+['equal_model_mean']):
                r=source[source.case.eq(case)&source.model_key.eq(m)].iloc[0]
                ax.errorbar(r.estimate,y+offset,xerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],fmt='o',ms=4,c=color,capsize=2,label=label if y==0 else None)
        ax.axvline(0,color='#888888',ls=':',lw=.8)
        ax.axvline(.01,color='#b96035',ls='--',lw=1)
        ax.set_yticks(range(4),[NAMES[m] for m in MODELS+['equal_model_mean']]);ax.invert_yaxis()
        ax.set_title('With answer predictor' if j==0 else 'Answer predictor omitted',fontsize=11)
        ax.set_xlabel(r'SELF increment ($\Delta R^2$)')
        ax.set_xlim(-.035,.08 if j==1 else .035)
        ax.grid(axis='x',alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,fontsize=10)
    fig.savefig(figdir/'predictive_baselines.pdf');plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.7));ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis('off')
    def box(x,y,w,h,title,body,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.07,rounding_size=.10',facecolor=color,edgecolor='#66727a',linewidth=.8))
        ax.text(x+w/2,y+h-.22,title,ha='center',va='top',fontsize=11,fontweight='bold')
        ax.text(x+w/2,y+h/2-.05,body,ha='center',va='center',fontsize=10,linespacing=1.5)
    def arrow(start,end):ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=14,color='#59656d',lw=1))
    box(.15,4.05,3.3,1.55,'Answer measurement','Question + matched documents\n6 label permutations\nNo document / P / C / I','#edf3f6')
    box(5.65,4.05,4.1,1.55,'Intervention target','CIE: C-vs-P support shift\nC-asserting minus matched\nP-asserting document','#e5eff4')
    arrow((3.55,4.85),(5.5,4.85))
    box(.15,1.2,3.3,1.9,'Visible report inputs','Question + document\n+ aggregate recorded answer\nNew stateless prompts','#f1edf5')
    arrow((1.8,3.95),(1.8,3.2))
    box(4.0,1.5,2.1,1.3,'PEER frame','Other-instance wording\nSame model','#f1edf5')
    box(7.25,1.5,2.5,1.3,'SELF frame','Own-answer wording\nSame model','#f1edf5')
    arrow((3.55,2.45),(3.9,2.45));arrow((3.55,1.4),(7.15,1.4))
    ax.text(6.85,.77,'4 report roles; 4 cyclic orderings',ha='center',fontsize=10)
    ax.text(5,.12,'Analysis: baseline prediction, frame scale and complete role profiles',ha='center',fontsize=11,fontweight='bold')
    ax.text(6.8,3.3,'Frame wording varies; computational access is held fixed.',ha='center',fontsize=9,color='#4d5260')
    fig.tight_layout(pad=.2);fig.savefig(args.project/'csra_framework.pdf');plt.close(fig)
    print('Generated manuscript tables, descriptive statistics, and three vector figures.')


if __name__=='__main__':main()
