"""Journal-sized, rebuildable figures plus reusable user/item/tail tables."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

STYLE={'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9,'axes.labelsize':10,'axes.titlesize':11,
       'axes.titlepad':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
       'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#A5ADB4',
       'axes.linewidth':.65,'axes.labelcolor':'#243541','xtick.color':'#48545D','ytick.color':'#48545D',
       'grid.color':'#D9E0E5','grid.linewidth':.55,'grid.alpha':.65,'legend.frameon':False,
       'figure.facecolor':'white','axes.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42,
       'svg.fonttype':'none','savefig.facecolor':'white','lines.linewidth':1.55,'lines.markersize':3.5}


def selected_rmse_panel(report,runs,primary,smoke=False,title=None):
    """A manuscript comparison with an explicit numeric column and real SDs."""
    metric=primary+"_RMSE"
    if metric not in runs or runs.dataset.nunique()!=1:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .reporting import COLORS
    from .figures import _save
    report=Path(report);out=report/"figures";out.mkdir(parents=True,exist_ok=True)
    outer=runs.groupby(["method","split_seed"],as_index=False)[metric].mean()
    table=outer.groupby("method")[metric].agg(["mean","std","count"]).reset_index().sort_values("mean")
    table.to_csv(report/"tables"/"selected_rmse_panel.csv",index=False)
    competitors=table[~table.method.str.startswith("Method")]
    reference=competitors.loc[competitors["mean"].idxmin()] if len(competitors) else None
    with plt.rc_context(STYLE):
        fig,(ax,values)=plt.subplots(1,2,sharey=True,figsize=(7.2,max(3.1,.34*len(table)+1.5)),
            gridspec_kw={"width_ratios":[5.,1.05]},constrained_layout=True)
        for y,(_,row) in enumerate(table.iterrows()):
            if y%2==0:
                ax.axhspan(y-.45,y+.45,color="#F3F5F7",zorder=0)
                values.axhspan(y-.45,y+.45,color="#F3F5F7",zorder=0)
            color=COLORS.get(row.method,"#344054") if row.method.startswith("Method") else "#667085"
            sd=float(row["std"]) if row["count"]>1 else None
            ax.errorbar(row["mean"],y,xerr=sd,fmt="o",ms=5.2,color=color,
                        ecolor=color,elinewidth=1.2,capsize=2.4,zorder=3)
            values.text(.5,y,f'{row["mean"]:.4f}',ha="center",va="center",
                        transform=values.get_yaxis_transform(),fontsize=9,color=color)
        ax.set_yticks(np.arange(len(table)),table.method)
        ax.invert_yaxis();ax.tick_params(axis="y",length=0,pad=7)
        ax.grid(axis="x",linestyle=":",alpha=.65);ax.set_axisbelow(True)
        if reference is not None:
            ax.axvline(reference["mean"],color="#98A2B3",ls="--",lw=.9,
                       label=str(reference.method)+" reference",zorder=1)
            ax.legend(loc="lower right",fontsize=8)
        ax.set_xlabel(primary.capitalize()+" RMSE · lower is better")
        values.set_title("RMSE",fontsize=10);values.set_xlim(0,1);values.axis("off")
        n=int(table["count"].min())
        detail=("Single outer split · descriptive estimates" if n==1 else
                f"{n} outer splits · whiskers show outer-split SD, not confidence intervals")
        fig.suptitle((title or "Validation-selected configurations")+"\n"+detail,fontsize=11)
        _save(fig,out,"selected_"+primary+"_rmse",smoke)


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
    if plots:
        render_clean(report,runs,primary,smoke)
        selected_rmse_panel(report,runs,primary,smoke)
    objective_tables(report,runs,plots=plots,smoke=smoke)
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
    (root/'PAPER_OUTPUTS.md').write_text('# V7 research outputs\n\n'
        '- `publication/research_tables.xlsx`: indexed, formatted Excel workbook.\n'
        '- `publication/tables/`: full CSV tables, including compressed pair data.\n'
        '- `figures/`: publication figures in PNG (400 DPI), vector PDF and editable SVG.\n'
        '- `predictions/`: numerical prediction and probability archives.\n'
        '- `mechanisms/*/DEVICE_AND_STORAGE.json`: actual device, fallback and offload events.\n\n'
        'Validation reports intentionally contain no test results. Train curves are in-sample full-context errors; masked query loss is separate. One observed split does not supply independent confirmation.\n')
    (root/'V7_PAPER_OUTPUTS.json').write_text(json.dumps(dict(workbook=str((report/workbook['workbook']).relative_to(root)),figures=images,primary_partition=primary,smoke=smoke),indent=2))


def objective_tables(report,runs,*,plots,smoke):
    """Export operative weights and per-epoch objective components, then plot."""
    from .config import ModelConfig
    from .losses import coefficients
    tables=Path(report)/'tables';records=[]
    for _,row in runs.iterrows():
        config=json.loads(row.config_key)['config']
        if row.method in ('Method'+str(i) for i in range(1,8)):
            cfg=ModelConfig(**config)
            weights={key:getattr(cfg,key) for key in coefficients(cfg)}
        elif row.method=='Method8':weights={'original_fused_frobenius_coefficient':1.}
        else:weights={key:config[key] for key in ('lambda_l2','lambda_main','lambda_element','weight_decay') if key in config}
        for key,value in weights.items():
            records.append(dict(method=row.method,split_seed=row.split_seed,model_seed=row.model_seed,
                                coefficient=key,value=value,config_key=row.config_key))
    pd.DataFrame(records,columns=['method','split_seed','model_seed','coefficient','value','config_key']).to_csv(tables/'objective_coefficients.csv',index=False)
    history_file=tables/'epoch_history.csv'
    if not history_file.is_file():return
    history=pd.read_csv(history_file)
    loss_columns=[name for name in history if name.startswith('L_')]
    if not loss_columns:return
    ids=[name for name in ('method','history_id','source','epoch') if name in history]
    long=history.melt(id_vars=ids,value_vars=loss_columns,var_name='term',value_name='value').dropna(subset=['value'])
    long.to_csv(tables/'objective_components_long.csv',index=False)
    if not plots or long.empty:return
    import matplotlib.pyplot as plt
    from .figures import _save
    methods=sorted(long.method.unique());columns=2 if len(methods) in (2,4) else min(3,len(methods));rows=(len(methods)+columns-1)//columns
    palette=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(rows,columns,figsize=(7.2,2.25*rows),squeeze=False,layout='constrained')
        from matplotlib.ticker import MaxNLocator,FuncFormatter
        for ax,method in zip(axes.flat,methods):
            subset=long[long.method==method]
            seen_curves=[]
            for index,(term,group) in enumerate(subset.groupby('term')):
                # Split and seed curves are descriptive means; no fabricated intervals.
                curve=group.groupby('epoch').value.mean()
                if any(np.array_equal(curve.values,prior) for prior in seen_curves):continue
                seen_curves.append(curve.values.copy())
                ax.plot(curve.index,curve.values,label=term.removeprefix('L_').replace('_',' '),color=palette[index%len(palette)])
            ax.set(title=method,xlabel='Epoch',ylabel='Loss component');ax.grid(axis='y');ax.legend(fontsize=6.5)
            ax.yaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.2g}'))
            ax.xaxis.set_major_locator(MaxNLocator(4,integer=True))
            if subset.value.min()>=-1e-8:ax.set_ylim(bottom=0)
        for ax in list(axes.flat)[len(methods):]:ax.set_visible(False)
        _save(fig,Path(report)/'figures','objective_component_facets',smoke)
