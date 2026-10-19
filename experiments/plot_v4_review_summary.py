"""Static review figures from auditable CSVs; no single-split confidence bands."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def export(fig,root,name):
    for ext in ('svg','png'):fig.savefig(root/(name+'.'+ext),dpi=350,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    args=parser.parse_args();root=args.evidence
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
    data=pd.read_csv(root/'proposed_control_comparison.csv')
    winners=pd.read_csv(root/'validation_winners.csv')
    ar=float(winners.loc[winners.method.eq('AutoRec'),'val_RMSE'].iloc[0])
    fig,ax=plt.subplots(figsize=(10,4.7))
    x=np.arange(len(data))
    for j,row in data.iterrows():
        ax.plot([j,j],[row.control_val_RMSE,row.selected_val_RMSE],color='#B6C0CA',lw=3,zorder=1)
    ax.scatter(x-.035,data.control_val_RMSE,color='#A96A22',s=40,label='Fresh V4 control',zorder=3)
    ax.scatter(x+.035,data.selected_val_RMSE,color='#087F7A',s=45,marker='D',label='Validation-selected V4.1',zorder=3)
    ax.axhline(ar,color='#505C69',lw=1.3,ls='--',label='Tuned AutoRec (raw RMSE)')
    ax.set_xticks(x,data.method);ax.set_ylabel('Selection validation RMSE (lower is better)')
    ax.set_title('One fixed MovieLens-100K split and model seed; K = 20')
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.legend(frameon=False,loc='upper center',ncol=3,fontsize=8)
    ax.set_ylim(min(ar,float(data.selected_val_RMSE.min()))-.008,max(float(data.control_val_RMSE.max()),float(data.selected_val_RMSE.max()))+.013)
    fig.text(.12,-.01,'Same parameter count within each proposed method. Unequal stopping/targets and tuning counts. No across-split inference.',fontsize=8,color='#5A6670')
    export(fig,root,'fresh_validation_comparison')
    candidates=pd.read_csv(root/'validation_candidates.csv').set_index('job')
    keys=[('Euclidean','Method4__control','Method4__density'),('Weighted metric','Method4__metric','Method4__metric__density'),
          ('Mean priority','Method4__priority','Method4__priority__density')]
    fig,ax=plt.subplots(figsize=(7.8,4.6))
    for column,label,color in [(1,'Fixed masking probability','#A96A22'),(2,'Variable masking probability','#087F7A')]:
        y=[float(candidates.loc[k[column],'val_RMSE']) for k in keys]
        ax.plot(range(3),y,'o-',color=color,label=label,lw=1.6)
        other=[float(candidates.loc[k[2 if column==1 else 1],'val_RMSE']) for k in keys]
        for j,v in enumerate(y):
            ax.annotate(f'{v:.6f}',(j,v),xytext=(0,11 if v>=other[j] else -18),textcoords='offset points',ha='center',fontsize=8,color=color)
    ax.set_xticks(range(3),[k[0] for k in keys]);ax.set_ylabel('Selection validation RMSE')
    ax.set_title('M4: projection × context-density control')
    ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,fontsize=8)
    ax.margins(x=.15,y=.25)
    fig.text(.12,-.01,'Same architecture width, K, parameter count and seed; early stopping and realized targets differ.',fontsize=8,color='#5A6670')
    export(fig,root,'m4_projection_factorial')


if __name__=='__main__':main()
