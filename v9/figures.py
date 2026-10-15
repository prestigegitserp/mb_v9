"""V9 readable sensitivity/convergence charts. Never uses test to choose a config."""
from __future__ import annotations
import json,pathlib
import numpy as np
import pandas as pd


def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt

def plot_k_rmse(base_folder, output_folder, methods=None):
    """Fixed-K locked experiments only: separates validation selection from test sensitivity."""
    base=pathlib.Path(base_folder);out=pathlib.Path(output_folder);out.mkdir(parents=True,exist_ok=True)
    entries=[]
    for f in sorted(base.glob('K_*/locked_test_runs.csv')):
        d=pd.read_csv(f)
        expected=int(f.parent.name.split('_',1)[1])
        d=d[d.K.notna()].copy();d['K']=d.K.astype(int)
        if (d.K!=expected).any():raise ValueError(f'Cross-K selection in fixed-K study: {f}')
        entries.append(d)
    if not entries:return []
    allrows=pd.concat(entries,ignore_index=True)
    allrows=allrows[~allrows.method.isin(['AutoRec','AEMC-NE'])]
    if methods:allrows=allrows[allrows.method.isin(methods)]
    summary=allrows.groupby(['dataset','method','K'],as_index=False)[['train_RMSE','val_RMSE','test_RMSE']].agg(['mean','std'])
    summary.to_csv(out/'k_rmse_summary.csv')
    generated=[];plt=_mpl()
    for ds,group in allrows.groupby('dataset'):
        meth=sorted(group.method.unique());nc=3;nr=int(np.ceil(len(meth)/nc))
        fig,axes=plt.subplots(nr,nc,figsize=(16,3.8*nr),squeeze=False)
        for idx,m in enumerate(meth):
            ax=axes[idx//nc,idx%nc];part=group[group.method==m]
            av=part.groupby('K')[['train_RMSE','val_RMSE','test_RMSE']].agg(['mean','std']).sort_index()
            for split in ('train','val','test'):
                x=av.index.to_numpy(dtype=float);mean=av[(f'{split}_RMSE','mean')].to_numpy(dtype=float)
                std=av[(f'{split}_RMSE','std')].fillna(0).to_numpy(dtype=float)
                ax.plot(x,mean,marker='o',label=split)
                ax.fill_between(x,mean-std,mean+std,alpha=.08)
            ax.set(title=m,xlabel='Latent dimension K',ylabel='RMSE');ax.legend(fontsize=8)
        for idx in range(len(meth),nr*nc):axes[idx//nc,idx%nc].axis('off')
        fig.suptitle(f'{ds}: train / validation / descriptive locked-test sensitivity to K\nBands: SD across split and model seeds; Test is never used to select K',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.96));name=f'{ds}_K_RMSE_train_val_test'
        for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig);generated.append(name)
    return generated

def plot_search_results(validation_csv,output_folder):
    csv=pathlib.Path(validation_csv);out=pathlib.Path(output_folder);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(csv)
    if any(x.startswith('test_') for x in d.columns):raise ValueError('Test leakage detected')
    d=d[d.K.notna()];d=d.groupby(['method','K'],as_index=False).val_RMSE.agg(['mean','std'])
    # Usually run a fixed hyperparameter scan separately to prevent confounding.
    d.to_csv(out/'validation_k_scatter_summary.csv')

def plot_convergence(epoch_csv,output_folder,max_curves=25):
    f=pathlib.Path(epoch_csv)
    if not f.exists():return []
    d=pd.read_csv(f);d=d[(d.epoch.notna()) & (d.val_RMSE.notna())]
    if d.empty:return []
    out=pathlib.Path(output_folder);out.mkdir(parents=True,exist_ok=True)
    plt=_mpl();done=[]
    for method,group in d.groupby('method'):
        fig,ax=plt.subplots(figsize=(9,5.5))
        for name,g in list(group.groupby(['split_seed','K','model_seed','init'],dropna=False))[:max_curves]:
            g=g.sort_values('epoch');ax.plot(g.epoch,g.val_RMSE,alpha=.35,label=str(name))
        ax.set(xlabel='Epoch (evaluation ticks)',ylabel='Validation RMSE',title=f'{method} validation convergence')
        if group.groupby(['split_seed','K','model_seed','init']).ngroups<8:ax.legend(fontsize=7)
        fig.tight_layout();name=f'convergence_{method}'
        fig.savefig(out/f'{name}.png',dpi=180);fig.savefig(out/f'{name}.pdf');plt.close(fig);done.append(name)
    return done

def plot_validation_landscape(ranked_csv,output_folder,top_n=16):
    """Validation-only per-model search landscape; pooled scores are NOT the selector."""
    p=pathlib.Path(ranked_csv)
    if not p.exists():return []
    d=pd.read_csv(p)
    if any(x.startswith('test_') for x in d.columns):raise ValueError('Test leakage in tuning plot input')
    out=pathlib.Path(output_folder);out.mkdir(parents=True,exist_ok=True)
    plt=_mpl();produced=[]
    for m,g in d.groupby('method'):
        # Means here are used for VISUALIZATION only; per-split selection was already done.
        a=g.groupby(['scan_job','config_key'],as_index=False).val_mean.mean().sort_values('val_mean').head(top_n)
        if a.empty:continue
        names=a.scan_job.astype(str).tolist()
        fig,ax=plt.subplots(figsize=(11,max(5,.4*len(a))))
        ax.barh(range(len(a)),a.val_mean)
        ax.set_yticks(range(len(a)),names,fontsize=8)
        ax.invert_yaxis();ax.set_xlabel('Validation RMSE (mean of per-split means; descriptive only)')
        ax.set_title(f'{m}: top {len(a)} scanned configurations — not test scores')
        fig.tight_layout();name=f'hyper_validation_{m}'
        fig.savefig(out/f'{name}.png',dpi=170,bbox_inches='tight')
        fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig);produced.append(name)
    return produced

def plot_train_keep_sensitivity(study_root,output_folder):
    root=pathlib.Path(study_root);out=pathlib.Path(output_folder);out.mkdir(parents=True,exist_ok=True)
    frames=[]
    for f in root.glob('keep_*/locked_test_runs.csv'):
        x=pd.read_csv(f);x['train_keep_requested']=float(f.parent.name[5:]);frames.append(x)
    if not frames:return []
    d=pd.concat(frames,ignore_index=True);plt=_mpl();produced=[]
    d.groupby(['dataset','method','train_keep_requested'],as_index=False)[['train_RMSE','val_RMSE','test_RMSE']].mean().to_csv(out/'train_keep_metric_summary.csv',index=False)
    for ds,z in d.groupby('dataset'):
        fig,ax=plt.subplots(figsize=(11,6))
        for method,g in z.groupby('method'):
            av=g.groupby('train_keep_requested').test_RMSE.mean().sort_index()
            ax.plot(av.index,av.values,marker='o',label=method)
        ax.set(xlabel='Requested keep-rate of original TRAIN only',ylabel='Locked test RMSE',title=f'{ds} training sparsity, SAME held-out val/test')
        ax.legend(fontsize=7,bbox_to_anchor=(1.01,1),loc='upper left')
        fig.tight_layout();name=f'{ds}_train_keep_test_rmse';fig.savefig(out/f'{name}.png',dpi=170,bbox_inches='tight');fig.savefig(out/f'{name}.pdf');plt.close(fig);produced.append(name)
    return produced
