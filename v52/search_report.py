"""Rebuildable, validation-only scan dossier with explicit selection provenance."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .reporting import _write,generate_report,COLORS


def generate_search_report(root,table,winners,ranked,*,plots=True):
    root=Path(root)
    if any(str(c).startswith('test_') for c in table):raise ValueError('Test leakage in scan dossier')
    join=['dataset','method','split_seed','scan_job','config_key']
    selected=table.merge(winners[join],on=join,validate='many_to_one')
    manifest=generate_report(root,selected,validation_only=True,plots=plots)
    tables=root/'publication'/'tables';figures=root/'publication'/'figures'
    _write(table,tables/'search_all_candidates.csv')
    _write(winners,tables/'search_validation_winners.csv')
    _write(ranked,tables/'search_ranked_configurations.csv')
    audit=ranked.merge(winners[join].assign(selected=True),on=join,how='left',validate='one_to_one')
    audit['selected']=audit.selected.eq(True)
    audit['selection_rank']=audit.groupby(['dataset','method','split_seed']).cumcount()+1
    _write(audit,tables/'selection_trace.csv')
    params=[]
    for _,row in table.drop_duplicates(['method','scan_job','config_key']).iterrows():
        config=json.loads(row.config_key)['config']
        for key,value in config.items():
            if isinstance(value,(dict,list)):continue
            params.append(dict(method=row.method,scan_job=row.scan_job,config_key=row.config_key,
                parameter=key,value=json.dumps(value),numeric_value=float(value) if isinstance(value,(int,float)) and not isinstance(value,bool) else np.nan))
    parameters=pd.DataFrame(params)
    _write(parameters,tables/'hyperparameters_long.csv')
    budget=[]
    for method,group in table.groupby('method'):
        budget.append(dict(method=method,scan_jobs=group.scan_job.nunique(),unique_configurations=group.config_key.nunique(),
            completed_trainings=len(group),split_count=group.split_seed.nunique(),model_seed_count=group.model_seed.nunique(),
            total_training_wall_seconds=group.train_wall_sec.sum(),
            budget_comparability='actual counts; family-specific grids are not equal-compute experiments'))
    _write(pd.DataFrame(budget),tables/'search_budget.csv')
    summary=ranked.groupby(['method','scan_job','config_key'],as_index=False).agg(
        val_mean=('val_mean','mean'),val_sd_outer=('val_mean','std'),outer_splits=('split_seed','nunique'))
    sensitivity=summary.merge(parameters,on=['method','scan_job','config_key'],validate='one_to_many')
    _write(sensitivity,tables/'hyperparameter_sensitivity.csv')
    if plots and len(summary):
        from .figures import _style,_save,_smoke
        with _style() as plt:
            methods=sorted(table.method.unique())
            cols=min(3,len(methods));rows=int(np.ceil(len(methods)/cols))
            fig,axes=plt.subplots(rows,cols,figsize=(4.4*cols,2.8*rows),squeeze=False,constrained_layout=True)
            for ax,method in zip(axes.flat,methods):
                frame=ranked[ranked.method==method]
                for split,line in frame.groupby('split_seed'):
                    # Job order is declared search order, not accuracy sorting.
                    order={tag:n for n,tag in enumerate(table[table.method==method].scan_job.unique())}
                    line=line.assign(trial=line.scan_job.map(order)).sort_values('trial')
                    best=line.val_mean.cummin()
                    ax.plot(line.trial+1,best,lw=1.4,label=str(split),alpha=.8)
                ax.set(title=method,xlabel='Declared scan job',ylabel='Best validation RMSE so far');ax.grid()
            for ax in list(axes.flat)[len(methods):]:ax.set_visible(False)
            fig.suptitle('Validation search progress · each split selects independently',fontsize=11)
            _save(fig,figures,'search_progress',_smoke(root))
            varied=sensitivity.groupby(['method','parameter']).numeric_value.nunique()
            for (method,parameter),count in varied.items():
                if count<2 or parameter in ('max_epochs','eval_every','latent_dim'):continue
                values=sensitivity[(sensitivity.method==method)&(sensitivity.parameter==parameter)].dropna(subset=['numeric_value'])
                # Each point is a complete configuration; no marginal minimum is
                # passed off as an isolated causal hyperparameter effect.
                fig,ax=plt.subplots(figsize=(5.4,3.5),constrained_layout=True)
                ax.scatter(values.numeric_value,values.val_mean,color=COLORS.get(method,'#525A63'),s=26,alpha=.7,edgecolors='white',linewidth=.4)
                if (values.numeric_value>0).all() and values.numeric_value.max()/values.numeric_value.min()>=10:ax.set_xscale('log')
                ax.set(title=method+' · complete candidate configurations',xlabel=parameter,ylabel='Mean validation RMSE (descriptive across splits)');ax.grid()
                _save(fig,figures,method+'_sensitivity_'+parameter,_smoke(root))
    from .publication import finish_report
    from .figures import _smoke
    finish_report(root,root/'publication',selected,'val',plots=plots,smoke=_smoke(root))
    manifest.update(scope='frozen validation winners plus all scan diagnostics',
        selection_rule='each outer split: mean validation RMSE across model seeds; no global cross-split tuning',
        tables=[dict(path=str(p.relative_to(root/'publication')),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(tables.rglob('*.csv*'))],
        figures=[str(p.relative_to(root/'publication')) for p in sorted(figures.glob('*')) if p.suffix in ('.pdf','.svg','.png')],
        test_evaluated=False)
    (root/'publication'/'REPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2,default=str))
    return manifest
