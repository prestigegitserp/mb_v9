"""Publication-sized figures from measured exploratory validation CSVs."""
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 from matplotlib.ticker import FormatStrFormatter
 from v4.figures import _style,_save
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();e=a.evidence
 trials=pd.read_csv(e/'validation_trials.csv');seven=pd.read_csv(e/'seven_model_comparison.csv');selected=pd.read_csv(e/'selected_validation_models.csv');blend=json.loads((e/'ENSEMBLE_NULL.json').read_text())
 def value(study,trial):
  hit=trials[(trials.study==study)&trials.trial.str.endswith('/'+trial)]
  if len(hit)!=1:raise ValueError('Missing/ambiguous measured arm '+trial)
  return float(hit.validation_RMSE.iloc[0])
 def draw(ax,labels,values,colors,title):
  y=np.arange(len(labels));ax.scatter(values,y,c=colors,s=38,zorder=3,edgecolors='white',linewidths=.5)
  for j,v in enumerate(values):ax.annotate(f'{v:.4f}',(v,j),xytext=(6,0),textcoords='offset points',va='center',fontsize=8,color='#33404B')
  ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_title(title,loc='left',fontsize=10,pad=9,fontweight='semibold');ax.grid(axis='x',alpha=.16);ax.set_xlabel('Validation RMSE  (lower is better)',fontsize=9);ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'));ax.margins(x=.18,y=.13)
 with _style() as plt:
  with plt.rc_context({'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8}):
   fig,axes=plt.subplots(2,2,figsize=(9.3,7.1),constrained_layout=True)
   ax=axes[0,0];line=seven.sort_values('method');y=np.arange(len(line))
   for j,row in enumerate(line.itertuples()):
    ax.plot([row.prior_validation_RMSE,row.validation_RMSE],[j,j],color='#CCD2D8',lw=1.5)
   ax.scatter(line.prior_validation_RMSE,y,c='#AEB8C1',s=33,label='Previous selected V4.1',zorder=2)
   ax.scatter(line.validation_RMSE,y,c='#0072B2',s=28,label='Selected V4.2',zorder=3,edgecolors='white',linewidths=.4)
   for j,row in enumerate(line.itertuples()):ax.annotate(f'{row.validation_RMSE:.4f}',(row.validation_RMSE,j),xytext=(5,-6),textcoords='offset points',fontsize=8,color='#173C55')
   ax.set_yticks(y,[m.replace('Method','M') for m in line.method]);ax.invert_yaxis();ax.set_title('a  Proposed models: paired configuration comparison',loc='left',fontsize=10,fontweight='semibold',pad=9)
   ax.set_xlabel('Validation RMSE');ax.set_xlim(.885,.967);ax.grid(axis='x');ax.legend(loc='upper center',ncol=2,fontsize=7.5,columnspacing=.9,handletextpad=.4);ax.set_ylim(len(line)-.5,-1.2);ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
   order=['Blend[AutoRec+Method4]','AutoRec','Method6','Method4','MBMF-N','MF','MBMF-F','BiasMF','AEMC-NE','MBMF-C'];vals=[];labels=[];colors=[]
   for name in order:
    if name.startswith('Blend'):v=blend['validation_RMSE'];label='AutoRec + M4 (ensemble control)';color='#E69F00'
    else:
     hit=selected[selected.method==name]
     if hit.empty:continue
     v=float(hit.validation_RMSE.iloc[0]);label=name.replace('Method','M');color='#0072B2' if name.startswith('Method') else '#606D78'
    vals.append(v);labels.append(label);colors.append(color)
   draw(axes[0,1],labels,vals,colors,'b  Strong competitors remain in the comparison')
   specs=[('V4.1 selected control',value('v4_1_selected_reference','Method4__control')),
          ('Readout change only',value('v4_2_runs','Method4__linear_full')),
          ('Strong L2 only',value('v4_2_factorial','Method4__regularization_only')),
          ('Readout + strong L2',value('v4_2_regularized','Method4__l2_1000.0'))]
   for trial,label in [('Method4__regularized_decoder_null','Strong decoder control')]:
    hit=trials[(trials.study=='v4_2_factorial')&trials.trial.str.endswith('/'+trial)]
    if len(hit):specs.append((label,float(hit.validation_RMSE.iloc[0])))
   draw(axes[1,0],[x[0] for x in specs],[x[1] for x in specs],['#AEB8C1']*3+['#0072B2']+(['#E69F00'] if len(specs)>4 else []),'c  M4: a non-additive exploratory interaction')
   specs=[('V4.1 selected control',float(seven[seven.method=='Method6'].prior_validation_RMSE.iloc[0])),
          ('Shared readout only',value('v4_2_runs','Method6__coupled')),
          ('Strong L2 only',value('v4_2_m6_factorial','Method6__regularization_only')),
          ('Shared readout + strong L2',value('v4_2_regularized','Method6__l2_1000.0'))]
   for trial,label in [('Method6__regularized_decoder_null','Strong decoder control'),('Method6__pure_decoder_null','Decoder without geometry auxiliary')]:
    hit=trials[(trials.study=='v4_2_m6_factorial')&trials.trial.str.endswith('/'+trial)]
    if len(hit):specs.append((label,float(hit.validation_RMSE.iloc[0])))
   draw(axes[1,1],[x[0] for x in specs],[x[1] for x in specs],['#AEB8C1']*3+['#0072B2']+['#E69F00']*(len(specs)-4),'d  M6: component removals and strong nulls')
   fig.suptitle('MovieLens 100K | Exploratory validation',fontsize=14,fontweight='semibold',color='#243746')
   fig.supxlabel('One split (2026), one seed (1). No fresh TEST metrics or independent-split confidence intervals.',fontsize=8,color='#596773')
   _save(fig,e,'v4_2_validation_overview',False)
   fig,axes=plt.subplots(1,2,figsize=(9.3,3.7),constrained_layout=True)
   ax=axes[0];names=['MF','MBMF-F','MBMF-N','MBMF-C','AutoRec','AEMC-NE']
   table=selected.set_index('method').loc[names];y=np.arange(len(names))
   raw=table.validation_RMSE.to_numpy();clipped=table.validation_RMSE_clipped_1_5.to_numpy()
   for j,(left,right) in enumerate(zip(raw,clipped)):ax.plot([left,right],[j,j],color='#BFC7CE',lw=1.5)
   ax.scatter(raw,y,s=38,c='#AEB8C1',label='Raw prediction',zorder=2)
   ax.scatter(clipped,y,s=23,c='#0072B2',label='Uniform clip to [1,5]',zorder=3)
   for j,v in enumerate(clipped):ax.annotate(f'{v:.4f}',(v,j),xytext=(5,-7),textcoords='offset points',fontsize=8,color='#173C55')
   ax.set_yticks(y,names);ax.set_ylim(len(names)-.5,-1.3);ax.set_xlim(.88,1.014)
   ax.set_title('a  The output rule can change competitor ordering',loc='left',fontsize=10,fontweight='semibold',pad=9)
   ax.set_xlabel('Validation RMSE');ax.grid(axis='x');ax.legend(loc='upper center',ncol=2,fontsize=7.5,handletextpad=.4,columnspacing=.8)
   ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
   ax=axes[1]
   for method,color,marker in [('Method4','#0072B2','o'),('Method6','#D55E00','s')]:
    rows=trials[(trials.study=='v4_2_regularized')&(trials.method==method)&(trials.arm=='stronger_regularization')].sort_values('collaborative_l2')
    if len(rows)!=3:raise ValueError('Expected three fixed-configuration L2 controls')
    ax.plot(rows.collaborative_l2,rows.validation_RMSE,marker=marker,color=color,label=method.replace('Method','M'),ms=4,lw=1.3)
   ax.set_xscale('log');ax.set_xticks([10,100,1000],['10','100','1000']);ax.set_xlabel('Explicit collaborative L2 coefficient');ax.set_ylabel('Validation RMSE')
   ax.set_title('b  Regularization sensitivity',loc='left',fontsize=10,fontweight='semibold',pad=9);ax.grid(alpha=.17);ax.legend(fontsize=8)
   fig.suptitle('MovieLens 100K | Fairness and regularization controls',fontsize=13,fontweight='semibold',color='#243746')
   fig.supxlabel('Exploratory: one split, one seed. Each L2 curve fixes the other settings. AEMC-NE is a local reimplementation.',fontsize=8,color='#596773')
   _save(fig,e,'v4_2_accuracy_controls',False)
if __name__=='__main__':main()
