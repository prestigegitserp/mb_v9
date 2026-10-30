"""Validation-only OFAT sensitivity from frozen winners; no TEST evaluation."""
import copy
import json
from pathlib import Path
import shutil
from dataclasses import replace
import pandas as pd
from .losses import coefficients, trial_values, registry_rows,coefficient_overrides,coefficient_kind


def run(pub,core,base,root,args):
    from .studies import frozen_configurations,anchor_protocol
    from . import engine
    anchors=frozen_configurations(pub,base,root,args)
    folder=Path(root)/'lossstudy';folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for tag,anchor in anchors:
        method=anchor.methods[0]
        if method not in anchor.v7.models: continue
        cfg=anchor.v7.models[method]
        for parameter in coefficients(cfg):
            for value in trial_values(cfg,parameter):
                trial=copy.deepcopy(anchor)
                changes=coefficient_overrides(cfg,{parameter:value})
                trial.v7.models[method]=replace(cfg,**changes).validate()
                trial.validation_only=True
                trial.save_predictions=False;trial.save_selected_checkpoints=False
                trial.save_candidate_checkpoints=False
                name=engine.basename(tag+'__'+parameter+'='+str(value))
                table=engine.run_job(pub,core,trial,folder/'jobs'/name,args=args,validation_only=True,
                                     extra=dict(type='loss-sensitivity',anchor=tag,parameter=parameter,value=value,changed_settings=changes))
                for _,row in table.iterrows():
                    rows.append(dict(method=method,split_seed=int(row.split_seed),model_seed=int(row.model_seed),
                                     K=row.K,parameter=parameter,value=value,anchor_value=getattr(cfg,parameter),
                                     train_RMSE=row.get('train_RMSE'),parameter_kind=coefficient_kind(parameter),
                                     changed_settings_json=json.dumps(changes,sort_keys=True),
                                     **{key:row[key] for key in table.columns if key.startswith('val_')},
                                     source_lineage=cfg.lineage,config_key=row.config_key,
                                     artifact_root=str((folder/'jobs'/name).resolve())))
                # Aggregate values/histories remain reproducible; transient model files are unnecessary here.
                trial_folder=folder/'jobs'/name
                for child in ('_candidate_checkpoints','selected_models'):
                    p=trial_folder/child
                    if p.is_dir() and not p.is_symlink():shutil.rmtree(p)
                engine.json_write(trial_folder/'V7_AGGREGATE_ONLY.json',dict(reason='loss sensitivity; all metrics and histories retained'))
                pd.DataFrame(rows).to_csv(folder/'loss_coefficient_validation.csv',index=False)
    engine.json_write(folder/'LOSS_STUDY_PROTOCOL.json',anchor_protocol(anchors,kind='validation-only loss sensitivity',
                        test_evaluated=False,ofat_after_joint_search=True,coefficients_frozen_for_final=True,
                        method8='immutable original objective; no loss ablations'))
    pd.DataFrame(registry_rows(base.v7)).to_csv(folder/'objective_registry.csv',index=False)
    rebuild(folder,plots=engine.REPORT_PLOTS)


def rebuild(folder,plots=True):
    folder=Path(folder);path=folder/'loss_coefficient_validation.csv'
    if not path.is_file(): return
    table=pd.read_csv(path)
    outer=table.groupby(['method','parameter','value','split_seed'],as_index=False).val_RMSE.mean()
    summary=outer.groupby(['method','parameter','value']).val_RMSE.agg(['mean','std','count']).reset_index()
    summary.to_csv(folder/'loss_coefficient_summary.csv',index=False)
    probability=None
    if 'val_Probability_NLL' in table and table.val_Probability_NLL.notna().any():
        per_split=table.dropna(subset=['val_Probability_NLL']).groupby(
            ['method','parameter','value','split_seed'],as_index=False).val_Probability_NLL.mean()
        probability=per_split.groupby(['method','parameter','value']).val_Probability_NLL.agg(['mean','std','count']).reset_index()
        probability.to_csv(folder/'loss_coefficient_probability_summary.csv',index=False)
    with pd.ExcelWriter(folder/'loss_sensitivity.xlsx',engine='xlsxwriter') as writer:
        table.to_excel(writer,sheet_name='all_validation_runs',index=False)
        summary.to_excel(writer,sheet_name='outer_split_summary',index=False)
        if probability is not None:probability.to_excel(writer,sheet_name='probability_summary',index=False)
        for sheet in writer.sheets.values():sheet.freeze_panes(1,0);sheet.autofilter(0,0,sheet.dim_rowmax,sheet.dim_colmax);sheet.set_column(0,5,22)
    if not plots:return
    from .publication import STYLE
    from .reporting import COLORS
    from .figures import _save
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metric_frames=[('Validation RMSE','',summary)]
    if probability is not None:metric_frames.append(('Validation NLL','probability_',probability))
    for ylabel,prefix,frame in metric_frames:
      for parameter,group in frame.groupby('parameter'):
        with plt.rc_context(STYLE):
            fig,ax=plt.subplots(figsize=(5.2,3.4),layout='constrained')
            for method,line in group.groupby('method'):
                line=line.sort_values('value');color=COLORS.get(method,'#667085')
                ax.plot(line.value,line['mean'],'o-',label=method,color=color)
                if (line['count']>1).any():ax.fill_between(line.value,line['mean']-line['std'].fillna(0),line['mean']+line['std'].fillna(0),color=color,alpha=.12)
            ax.set(xlabel=parameter.replace('_',' '),ylabel=ylabel,title='Frozen-winner objective sensitivity')
            ax.set_xscale('symlog',linthresh=max(float(group.value.max())*.01,1e-6));ax.grid(axis='y');ax.legend(fontsize=7,ncol=2)
            _save(fig,folder/'figures',prefix+'coefficient_'+parameter,False)
