from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path(__file__).resolve().parent;out=base.parents[1]/'figures';out.mkdir(exist_ok=True)
s=json.loads((base/'plot_data.json').read_text());cases=s['cases']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':10.5,'axes.labelsize':11,'savefig.facecolor':'white'})
colors=['#1f4e79','#b25b24','#33785b'];targets=['granite','phi','qwen'];names=['Granite','Phi','Qwen']
fig,ax=plt.subplots(1,2,figsize=(7,2.6833333333),gridspec_kw={'width_ratios':[1.1,1]},layout='constrained')
means=np.array([-2.157,2.931,11.726]);lo=np.array([-3.572,2.392,10.090]);hi=np.array([-.773,3.384,13.492]);y=np.arange(3)
ax[0].errorbar(means,y,xerr=[means-lo,hi-means],fmt='o',color=colors[0],capsize=3);ax[0].axvline(0,color='0.5',lw=.7);ax[0].set_yticks(y,['Ownership','Hedging','Answer visibility']);ax[0].invert_yaxis();ax[0].set_xlabel('Change in both-source\nsupport (pp)');ax[0].set_title('Primary contrasts\n(98.75% intervals)')
per=np.array([[-5.8035,8.2717,27.2598],[-.6683,.5202,7.9149],[.0014,.0013,.0022]])
for j in range(3):ax[1].plot(per[j],y+(j-1)*.15,'o',label=names[j],color=colors[j],markersize=4)
ax[1].axvline(0,color='0.5',lw=.7);ax[1].set_yticks(y,[]);ax[1].invert_yaxis();ax[1].set_xlabel('Change in both-source\nsupport (pp)');ax[1].set_title('System point estimates');ax[1].legend(frameon=False,loc='upper right',fontsize=10)
fig.savefig(out/'factorial.pdf');fig.savefig(out/'factorial.png',dpi=240);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(7,2.8777777778),gridspec_kw={'width_ratios':[1.1,1]},layout='constrained')
keys=['self_over_visible','self_over_selected_observer','self_added_to_probe'];labels=['Source vs visible','Source vs observer','Source added to probe'];cc=s['results']['gradient_boosting']['contrasts'];mean=np.array([cc[k]['normalized_MSE_reduction'] for k in keys]);ints=np.array([cc[k]['descriptive_95pct_cluster_bootstrap_interval'] for k in keys])
ax[0].errorbar(mean,y,xerr=[mean-ints[:,0],ints[:,1]-mean],fmt='o',color=colors[0],capsize=3);ax[0].axvline(0,color='0.5',lw=.7);ax[0].set_yticks(y,labels);ax[0].invert_yaxis();ax[0].set_xlabel('Normalized MSE\nreduction');ax[0].set_title('Primary contrasts\n(descriptive 95% intervals)')
for j,t in enumerate(targets):ax[1].plot([cc[k]['by_target'][t] for k in keys],y+(j-1)*.15,'o',label=names[j],color=colors[j],markersize=4)
ax[1].axvline(0,color='0.5',lw=.7);ax[1].set_yticks(y,[]);ax[1].invert_yaxis();ax[1].set_xlabel('Normalized MSE\nreduction');ax[1].set_title('Target point estimates');ax[1].legend(frameon=False,loc='lower right',fontsize=10)
fig.savefig(out/'fresh40.pdf');fig.savefig(out/'fresh40.png',dpi=240);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(7,2.6833333333),sharey=True,layout='constrained');rng=np.random.default_rng(20261021)
for j,(ax,t) in enumerate(zip(axes,targets)):
 values=[[c['Y'] for c in cases if c['target']==t and c['strength']==st] for st in ['SKEPTICAL','PROVISIONAL','BARE']]
 ax.boxplot(values,patch_artist=True,showfliers=False,boxprops={'facecolor':'#e4e9ef','linewidth':.8},medianprops={'color':colors[j]},whiskerprops={'linewidth':.7},capprops={'linewidth':.7})
 for k,v in enumerate(values):ax.scatter(rng.normal(k+1,.045,len(v)),v,s=7,alpha=.45,c=colors[j],edgecolors='none')
 ax.axhline(0,color='0.7',lw=.6);ax.set_xticks([1,2,3],['Skeptical','Provisional','Bare'],rotation=20);ax.set_title(names[j])
axes[0].set_ylabel('Substitution effect (nats)');fig.savefig(out/'outcomes.pdf');fig.savefig(out/'outcomes.png',dpi=240);plt.close(fig)
print('Created3 figures from verified values and all360 fresh outcomes.')
