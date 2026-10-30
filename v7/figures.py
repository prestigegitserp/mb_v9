"""Publication sensitivity figures; outer splits are the statistical units."""
from contextlib import contextmanager
import hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from .reporting import COLORS,BASELINE_MARKERS


def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _smoke(path):
    for parent in (Path(path),*Path(path).parents):
        f=parent/'V9_JOB_STATUS.json'
        if f.is_file():
            if json.loads(f.read_text()).get('command_arguments',{}).get('quick',False):return True
        f=parent/'V7_STUDY_MODE.json'
        if f.is_file() and json.loads(f.read_text()).get('quick',False):return True
    return False


@contextmanager
def _style():
    plt=_mpl()
    from .publication import STYLE
    with plt.rc_context(STYLE):yield plt


def _save(fig,out,name,smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if smoke:fig.text(.5,.5,'SMOKE · NOT FOR PUBLICATION',ha='center',va='center',fontsize=22,color='#A33D32',alpha=.2,rotation=20,zorder=100)
    try:
        for ext in ('png','pdf','svg'):
            target=out/(name+'.'+ext);temp=out/('.'+name+'.partial.'+ext)
            try:fig.savefig(temp,format=ext,dpi=400,bbox_inches='tight');temp.replace(target)
            finally:temp.unlink(missing_ok=True)
    finally:_mpl().close(fig)


def _outer_summary(data,x,metrics):
    keys=['dataset','method',x]
    outer=data.groupby(keys+['split_seed'],dropna=False)[metrics].mean().reset_index()
    summary=outer.groupby(keys,dropna=False)[metrics].agg(['mean','std','count'])
    summary.columns=[metric+'__'+stat for metric,stat in summary.columns]
    return outer,summary.reset_index()


def _write_provenance(out,sources,smoke):
    out=Path(out)
    (out/'FIGURE_PROVENANCE.json').write_text(json.dumps(dict(
        statistical_unit='outer split; model seeds averaged within split; bands are SD, not CI',
        smoke=smoke,renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources=[dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in sources]),indent=2))


def plot_k_rmse(base_folder,output_folder,methods=None):
    base=Path(base_folder);out=Path(output_folder);out.mkdir(parents=True,exist_ok=True)
    frames=[];sources=sorted(base.glob('K_*/locked_test_runs.csv'))
    for file in sources:
        d=pd.read_csv(file);expected=int(file.parent.name[2:])
        d=d[~d.method.isin(['AutoRec','AEMC-NE'])].copy()
        if d.K.isna().any() or not d.K.eq(expected).all():raise ValueError('Cross-K selection in fixed-K study: '+str(file))
        frames.append(d)
    if not frames:return []
    data=pd.concat(frames,ignore_index=True)
    if methods:data=data[data.method.isin(methods)]
    metrics=['train_RMSE','val_RMSE','test_RMSE']
    outer,summary=_outer_summary(data,'K',metrics)
    data.to_csv(out/'k_rmse_source.csv',index=False);outer.to_csv(out/'k_outer_split_means.csv',index=False);summary.to_csv(out/'k_rmse_summary.csv',index=False)
    smoke=any(_smoke(p.parent) for p in sources);done=[]
    with _style() as plt:
        for dataset,g in summary.groupby('dataset'):
            methods=list(g.method.unique());cols=min(4,len(methods));rows=math.ceil(len(methods)/cols)
            fig,axes=plt.subplots(rows,cols,figsize=(3.1*cols,2.65*rows),squeeze=False,constrained_layout=True)
            for ax,method in zip(axes.flat,methods):
                line=g[g.method==method].sort_values('K');x=line.K.to_numpy(float)
                for metric,col,label in zip(metrics,['#777777','#E69F00','#0072B2'],['Train (in-sample)','Validation','Locked test (descriptive)']):
                    mean=line[metric+'__mean'].to_numpy(float);sd=line[metric+'__std'].to_numpy(float)
                    ax.plot(x,mean,'o-',ms=3,lw=1,color=col,label=label)
                    if np.isfinite(sd).any():ax.fill_between(x,mean-sd,mean+sd,color=col,alpha=.12)
                ax.set(title=method,xlabel='Latent dimension K',ylabel='RMSE');ax.grid()
            for ax in list(axes.flat)[len(methods):]:ax.set_visible(False)
            handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=7)
            fig.suptitle(f'{dataset} · fixed-K sensitivity · bands: outer-split SD when available',fontsize=10)
            name=f'{dataset}_K_RMSE_train_val_test';_save(fig,out,name,smoke);done.append(name)
    _write_provenance(out,sources,smoke);return done


def plot_convergence(epoch_csv,output_folder,max_curves=25):
    file=Path(epoch_csv)
    if not file.is_file():return []
    data=pd.read_csv(file)
    if data.empty or 'val_RMSE' not in data:return []
    data=data[data.epoch.notna()&data.val_RMSE.notna()]
    out=Path(output_folder);out.mkdir(parents=True,exist_ok=True);data.to_csv(out/'convergence_source.csv',index=False)
    done=[];smoke=_smoke(file.parent)
    with _style() as plt:
        for method,g in data.groupby('method'):
            fig,ax=plt.subplots(figsize=(6.3,3.8),constrained_layout=True)
            for name,line in list(g.groupby(['split_seed','K','model_seed','init'],dropna=False))[:max_curves]:
                line=line.sort_values('epoch');ax.plot(line.epoch,line.val_RMSE,color=COLORS.get(method,'#777777'),alpha=.5,lw=1,label=str(name))
            ax.set(xlabel='Actual training epoch',ylabel='Validation RMSE',title=method+' · convergence');ax.grid()
            if g.groupby(['split_seed','K','model_seed','init'],dropna=False).ngroups<6:ax.legend(fontsize=6)
            name='convergence_'+method;_save(fig,out,name,smoke);done.append(name)
    _write_provenance(out,[file],smoke);return done


def plot_validation_landscape(ranked_csv,output_folder,top_n=16):
    file=Path(ranked_csv)
    if not file.is_file():return []
    data=pd.read_csv(file)
    if any(str(c).startswith('test_') for c in data):raise ValueError('Test leakage in tuning plot input')
    out=Path(output_folder);out.mkdir(parents=True,exist_ok=True);data.to_csv(out/'validation_landscape_source.csv',index=False)
    smoke=_smoke(file.parent);done=[]
    with _style() as plt:
        for method,g in data.groupby('method'):
            a=g.groupby(['scan_job','config_key'],dropna=False,as_index=False).val_mean.mean().sort_values('val_mean').head(top_n)
            if a.empty:continue
            a.to_csv(out/('validation_landscape_'+method+'.csv'),index=False)
            fig,ax=plt.subplots(figsize=(7.5,max(2.8,.29*len(a))),constrained_layout=True)
            ax.scatter(a.val_mean,np.arange(len(a)),color=COLORS.get(method,'#555555'),s=32)
            ax.set_yticks(np.arange(len(a)),a.scan_job.astype(str),fontsize=7);ax.invert_yaxis();ax.grid(axis='x')
            ax.set(xlabel='Validation RMSE (mean of per-split means; descriptive)',title=method+' · scanned configurations')
            name='hyper_validation_'+method;_save(fig,out,name,smoke);done.append(name)
    _write_provenance(out,[file],smoke);return done


def plot_train_keep_sensitivity(study_root,output_folder):
    sources=sorted(Path(study_root).glob('keep_*/locked_test_runs.csv'));frames=[]
    for file in sources:
        d=pd.read_csv(file);d['train_keep_requested']=float(file.parent.name[5:]);frames.append(d)
    if not frames:return []
    data=pd.concat(frames,ignore_index=True);out=Path(output_folder);out.mkdir(parents=True,exist_ok=True)
    outer,summary=_outer_summary(data,'train_keep_requested',['train_RMSE','val_RMSE','test_RMSE'])
    actual='train_keep_realized' in data and data.train_keep_realized.notna().all()
    if actual:
        summary=summary.merge(data.groupby(['dataset','method','train_keep_requested'],as_index=False).train_keep_realized.mean(),on=['dataset','method','train_keep_requested'],validate='one_to_one')
    data.to_csv(out/'train_keep_source.csv',index=False);outer.to_csv(out/'train_keep_outer_split_means.csv',index=False);summary.to_csv(out/'train_keep_metric_summary.csv',index=False)
    done=[];smoke=any(_smoke(p.parent) for p in sources)
    with _style() as plt:
        for dataset,g in summary.groupby('dataset'):
            fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
            for method,line in g.groupby('method'):
                line=line.sort_values('train_keep_requested');x=line['train_keep_realized' if actual else 'train_keep_requested'].to_numpy(float)
                mean=line.test_RMSE__mean.to_numpy(float);sd=line.test_RMSE__std.to_numpy(float);color=COLORS.get(method,'#777777')
                ax.plot(x,mean,marker=BASELINE_MARKERS.get(method,'o'),ms=3,lw=1,color=color,label=method)
                if np.isfinite(sd).any():ax.fill_between(x,mean-sd,mean+sd,alpha=.1,color=color)
            ax.set(xlabel='Realized fraction retained from TRAIN' if actual else 'Requested TRAIN fraction (subject to support constraints)',ylabel='Locked test RMSE',title=dataset+' · fixed held-out validation/test');ax.grid()
            ax.legend(fontsize=7,ncol=3);name=dataset+'_train_keep_test_rmse';_save(fig,out,name,smoke);done.append(name)
    _write_provenance(out,sources,smoke);return done


def plot_ablation_effects(data,output_folder):
    """Paired descriptive effects; never manufacture a full-model reference."""
    required={'dataset','method','split_seed','model_seed','K','ablation','test_RMSE'}
    if not required<=set(data):raise ValueError('Missing ablation columns')
    out=Path(output_folder);out.mkdir(parents=True,exist_ok=True);source=out/'ablation_source.csv'
    data.to_csv(source,index=False);smoke=_smoke(out.parent);done=[];records=[]
    with _style() as plt:
        for (dataset,method),group in data.groupby(['dataset','method']):
            full=group[group.ablation==method+'__full']
            if full.empty:raise ValueError('Missing full ablation reference: '+method)
            keys=['dataset','method','split_seed','model_seed']
            if full.duplicated(keys).any():raise ValueError('Ambiguous full-model reference')
            pairs=group[group.ablation!=method+'__full'].merge(full[keys+['K','test_RMSE']],on=keys,suffixes=('','_full'),validate='many_to_one')
            if len(pairs)!=len(group)-len(full):raise ValueError('Missing paired full-model runs')
            pairs['delta_RMSE']=pairs.test_RMSE-pairs.test_RMSE_full;pairs['matched_K']=pairs.K.eq(pairs.K_full)
            records.append(pairs)
            outer=pairs.groupby(['ablation','split_seed']).delta_RMSE.mean().reset_index()
            stats=outer.groupby('ablation').delta_RMSE.agg(['mean','std','count']).sort_values('mean')
            fig,ax=plt.subplots(figsize=(7,max(2.8,.32*len(stats))),constrained_layout=True)
            ax.scatter(stats['mean'],np.arange(len(stats)),c=['#B64B45' if x>0 else '#16836C' for x in stats['mean']],s=34,zorder=3)
            for j,(_,row) in enumerate(stats.iterrows()):
                if np.isfinite(row['std']):ax.errorbar(row['mean'],j,xerr=row['std'],fmt='none',ecolor='#777777',capsize=2,lw=.8)
            ax.axvline(0,color='#777777',lw=.8);ax.set_yticks(np.arange(len(stats)),[x.split('__',1)[-1] for x in stats.index],fontsize=8)
            ax.set(xlabel='Δ locked test RMSE (ablated − full); outer-split SD when available',
                title=f'{dataset} · {method} · '+('same K' if pairs.matched_K.all() else 'selected K differs; descriptive only'))
            ax.grid(axis='x');name=dataset+'_ablation_'+method;_save(fig,out,name,smoke);done.append(name)
    if records:pd.concat(records,ignore_index=True).to_csv(out/'ablation_paired_effects.csv',index=False)
    _write_provenance(out,[source],smoke);return done
