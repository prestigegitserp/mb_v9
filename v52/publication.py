"""Journal-sized, rebuildable figures plus reusable user/item/tail tables."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

STYLE={'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
       'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#A5ADB4',
       'axes.linewidth':.65,'axes.labelcolor':'#243541','xtick.color':'#48545D','ytick.color':'#48545D',
       'grid.color':'#D9E0E5','grid.linewidth':.55,'grid.alpha':.65,'legend.frameon':False,
       'figure.facecolor':'white','axes.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42,
       'svg.fonttype':'none','savefig.facecolor':'white','lines.linewidth':1.55,'lines.markersize':3.5}


def enrich_tables(report,primary):
    tables=Path(report)/'tables';entities=[];tails=[]
    for path in sorted((tables/'per_pair').glob('*.csv.gz')):
        if not path.name.endswith('__'+primary+'.csv.gz'):continue
        frame=pd.read_csv(path);method=path.name.split('__',1)[0]
        for dimension in ('userId','movieId'):
            if dimension not in frame:continue
            group=frame.groupby(dimension).agg(n=('squared_error','size'),MSE=('squared_error','mean'),MAE=('absolute_error','mean'),bias=('residual','mean')).reset_index()
            group['RMSE']=np.sqrt(group.MSE);group.insert(0,'method',method);group.insert(1,'partition',primary)
            group['dimension']=dimension;group['source']=path.name
            group.rename(columns={dimension:'entity_id'},inplace=True);entities.append(group)
        tails.append(dict(method=method,partition=primary,source=path.name,n=len(frame),
            error_le_0_5=np.mean(frame.absolute_error<=.5),error_le_1=np.mean(frame.absolute_error<=1),
            error_le_2=np.mean(frame.absolute_error<=2),P50AE=frame.absolute_error.quantile(.5),
            P90AE=frame.absolute_error.quantile(.9),P95AE=frame.absolute_error.quantile(.95),P99AE=frame.absolute_error.quantile(.99)))
    pd.concat(entities,ignore_index=True).to_csv(tables/'entity_errors.csv',index=False) if entities else pd.DataFrame(columns=['method','partition','dimension','entity_id','n','RMSE','MAE','bias']).to_csv(tables/'entity_errors.csv',index=False)
    pd.DataFrame(tails,columns=['method','partition','source','n','error_le_0_5','error_le_1','error_le_2','P50AE','P90AE','P95AE','P99AE']).to_csv(tables/'error_tail_summary.csv',index=False)


def render_clean(report,runs,primary,smoke=False):
    from .reporting import COLORS,BASELINE_MARKERS
    from .figures import _save
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=Path(report);out=report/'figures';out.mkdir(parents=True,exist_ok=True)
    with plt.rc_context(STYLE):
        for metric in ('RMSE','MAE','P95AE','Probability_NLL'):
            key=primary+'_'+metric
            if key not in runs or not runs[key].notna().any():continue
            outer=runs.groupby(['method','split_seed'])[key].mean().reset_index()
            values=outer.groupby('method')[key].agg(['mean','std','count']).dropna(subset=['mean']).sort_values('mean')
            values.to_csv(report/'tables'/('plot_'+key+'.csv'))
            fig,ax=plt.subplots(figsize=(6.6,max(3,.30*len(values)+.8)),layout='constrained')
            ax.set_axisbelow(True);ax.grid(axis='x');ax.invert_yaxis()
            for y,(method,row) in enumerate(values.iterrows()):
                ax.scatter(row['mean'],y,s=38,color=COLORS.get(method,'#66727B'),marker=BASELINE_MARKERS.get(method,'o'),zorder=3)
                if row['count']>1:ax.errorbar(row['mean'],y,xerr=row['std'],fmt='none',ecolor=COLORS.get(method,'#66727B'),capsize=2,lw=1)
                ax.text(1.01,y,f"{row['mean']:.4f}",transform=ax.get_yaxis_transform(),va='center',fontsize=8,color='#263742')
            ax.set_yticks(range(len(values)),values.index)
            ax.set(xlabel=primary.capitalize()+' '+metric,ylabel='',title='Rating prediction · '+('validation' if primary=='val' else 'locked evaluation'))
            ax.margins(x=.13,y=.07)
            ax.set_ylim(len(values)-.5,-.5)
            ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=8)
            _save(fig,out,'rank_'+key,smoke)
        ecdf_file=report/'tables/absolute_error_ecdf.csv'
        if ecdf_file.is_file():
            data=pd.read_csv(ecdf_file);data=data[data.partition==primary]
            if len(data):
                fig,axes=plt.subplots(1,2,figsize=(7.2,3.7),sharex=True,sharey=True,layout='constrained')
                for ax,proposed,title in zip(axes,[True,False],['Proposed models','Comparison methods']):
                    subset=data[data.method.str.startswith('Method')==proposed]
                    for method,line in subset.groupby('method'):
                        line=line.groupby('fraction',as_index=False).absolute_error.mean()
                        ax.plot(line.absolute_error,line.fraction,color=COLORS.get(method,'#66727B'),label=method,ls='-' if proposed else '--')
                    ax.set(title=title,xlabel='Absolute rating error',ylim=(0,1));ax.grid();ax.legend(loc='lower right',fontsize=7,ncol=2)
                axes[0].set_ylabel('Cumulative fraction of interactions')
                _save(fig,out,'absolute_error_ecdf',smoke)
        epoch_file=report/'tables/epoch_history.csv'
        if epoch_file.is_file():
            history=pd.read_csv(epoch_file)
            if len(history) and 'val_RMSE' in history:
                methods=sorted(history.method.unique());cols=3;rows=(len(methods)+cols-1)//cols
                fig,axes=plt.subplots(rows,cols,figsize=(7.2,2.05*rows),squeeze=False,layout='constrained')
                for ax,method in zip(axes.flat,methods):
                    group=history[history.method==method]
                    for _,line in group.groupby([c for c in ('history_id','source') if c in group]):
                        ax.plot(line.epoch,line.val_RMSE,color=COLORS.get(method,'#66727B'),lw=1.15)
                        if 'train_RMSE' in line:ax.plot(line.epoch,line.train_RMSE,color='#89949C',lw=.9,ls='--')
                    ax.set(title=method,xlabel='Epoch',ylabel='RMSE');ax.grid()
                for ax in list(axes.flat)[len(methods):]:ax.set_visible(False)
                _save(fig,out,'training_generalization_facets',smoke)
        if 'train_wall_sec' in runs and primary+'_RMSE' in runs:
            fig,ax=plt.subplots(figsize=(6.8,4),layout='constrained')
            grouped=runs.groupby('method')[['train_wall_sec',primary+'_RMSE']].mean()
            # Labels live in a separate legend; points are not overprinted.
            for method,row in grouped.iterrows():
                ax.scatter(max(row.train_wall_sec,1e-9),row[primary+'_RMSE'],color=COLORS.get(method,'#66727B'),marker=BASELINE_MARKERS.get(method,'o'),s=37,label=method)
            ax.set(xscale='log',xlabel='Training time (seconds)',ylabel=primary.capitalize()+' RMSE');ax.grid()
            ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=7,ncol=1)
            _save(fig,out,'accuracy_efficiency',smoke)


def finish_report(root,report,runs,primary,*,plots,smoke):
    from .export import export_workbook
    root,report=Path(root),Path(report)
    enrich_tables(report,primary)
    fidelity=root/'BASELINE_FIDELITY.csv'
    if fidelity.is_file():shutil.copy2(fidelity,report/'tables/baseline_fidelity.csv')
    devices=[]
    for file in sorted(root.rglob('DEVICE_AND_STORAGE.json')):
        event=json.loads(file.read_text());place=event.get('placement',{})
        devices.append(dict(candidate=file.parent.name,source=str(file.relative_to(root)),requested_device=place.get('requested_device'),
            actual_device=place.get('actual_device'),initialization_fallback=place.get('initialization_fallback'),
            training_fallback=json.dumps(place.get('fallback')),parameter_bytes=place.get('parameter_bytes'),
            peak_allocated_bytes=event.get('peak_allocated_bytes'),peak_reserved_bytes=event.get('peak_reserved_bytes'),
            storage_attempts=json.dumps(event.get('storage_attempts')),wall_seconds=place.get('total_candidate_seconds')))
    pd.DataFrame(devices,columns=['candidate','source','requested_device','actual_device','initialization_fallback','training_fallback','parameter_bytes','peak_allocated_bytes','peak_reserved_bytes','storage_attempts','wall_seconds']).to_csv(report/'tables/candidate_resources.csv',index=False)
    if plots:render_clean(report,runs,primary,smoke)
    workbook=export_workbook(report)
    destination=root/'figures';destination.mkdir(parents=True,exist_ok=True)
    images=[]
    if plots:
        for path in sorted((report/'figures').glob('*')):
            if path.suffix in ('.png','.pdf','.svg') and not path.name.startswith('.'):
                target=destination/path.name;temporary=destination/('.'+path.name+'.copying')
                try:shutil.copy2(path,temporary);temporary.replace(target)
                finally:temporary.unlink(missing_ok=True)
                images.append('figures/'+path.name)
    (root/'PAPER_OUTPUTS.md').write_text('# V52 research outputs\n\n'
        '- `publication/research_tables.xlsx`: indexed, formatted Excel workbook.\n'
        '- `publication/tables/`: full CSV tables, including compressed pair data.\n'
        '- `figures/`: publication figures in PNG (400 DPI), vector PDF and editable SVG.\n'
        '- `predictions/`: numerical prediction and probability archives.\n'
        '- `mechanisms/*/DEVICE_AND_STORAGE.json`: actual device, fallback and offload events.\n\n'
        'Validation reports intentionally contain no test results. Train curves are in-sample full-context errors; masked query loss is separate. One observed split does not supply independent confirmation.\n')
    (root/'V52_PAPER_OUTPUTS.json').write_text(json.dumps(dict(workbook=str((report/workbook['workbook']).relative_to(root)),figures=images,primary_partition=primary,smoke=smoke),indent=2))
