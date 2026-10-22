"""Rebuildable research tables and publication figures, strictly post-training."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd

COLORS=dict(zip([f"Method{i}" for i in range(1,8)],
    ["#0072B2","#E69F00","#009E73","#CC79A7","#D55E00","#56B4E9","#6C4A97"]))
COLORS.update({"MF":"#262626","BiasMF":"#535353","AutoRec":"#707070","AEMC-NE":"#909090",
               "MBMF-F":"#3D5366","MBMF-C":"#6B7B89","MBMF-N":"#93A0AA"})
COLORS.update(dict(zip(["MF","BiasMF","AutoRec","AEMC-NE","MBMF-F","MBMF-C","MBMF-N"],
    ["#332288","#88CCEE","#44AA99","#117733","#999933","#AA4499","#CC6677"])))
BASELINE_MARKERS=dict(zip(["MF","BiasMF","AutoRec","AEMC-NE","MBMF-F","MBMF-C","MBMF-N"],["s","D","^","v","P","X","*"]))


def _write(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not len(frame.columns):
        schemas={"epoch_history.csv":["method","history_id","epoch","val_RMSE"],
            "phase_and_execution_diagnostics.csv":["method","history_id","phase","details_json"],
            "device_utilization_samples.csv":["method","history_id","seconds","device_kernel_active_percent","device_memory_bus_active_percent"],
            "conditional_errors.csv":["method","partition","dimension","group","n","RMSE","MAE","bias"],
            "absolute_error_ecdf.csv":["method","partition","fraction","absolute_error","residual_quantile"],
            "residual_histogram.csv":["method","partition","left","right","n"],
            "probability_coverage.csv":["method","partition","nominal","coverage","mean_width"],
            "paired_comparison_vs_MF.csv":["method","reference","partition","n","delta_RMSE","delta_MSE"],
            "recomputed_prediction_metrics.csv":["method","partition","n","RMSE","MAE","mean_error"],
            "auxiliary_phase_history.csv":["method","history_id","phase","epoch","objective"]}
        frame=pd.DataFrame(columns=schemas.get(path.name,["status","reason"]))
    kwargs={"compression":{"method":"gzip","mtime":0}} if path.suffix==".gz" else {}
    temporary=path.with_name('.'+path.name+'.partial')
    try:
        frame.to_csv(temporary,index=False,float_format="%.17g",**kwargs)
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


def _archive(root,row,part):
    given=row.get(part+"_prediction_file")
    if isinstance(given,str):
        original=Path(given)
        if original.is_file():return original
        local=root/"predictions"/Path(given.replace("\\","/")).name
        if local.is_file():return local
    k="*" if pd.isna(row.K) else str(int(row.K))
    pattern=f"{row.dataset}__{row.method}__split{int(row.split_seed)}__seed{int(row.model_seed)}__K{k}__{row.init_mode}__{part}.npz"
    paths=sorted((root/"predictions").rglob(pattern))
    return paths[0] if paths else None


def _source_root(root,row,combined=False):
    if combined:return root/"final"/str(row.method)/f"split_{int(row.split_seed)}"
    given=row.get("artifact_root")
    if isinstance(given,str) and Path(given).is_dir():return Path(given)
    if "scan_job" in row:
        from .engine import basename
        candidate=root/"jobs"/basename(str(row.scan_job))
        if candidate.is_dir():return candidate
    return root


def _histories(root,combined,runs=None):
    paths=sorted((root/"final").rglob("training_histories.json")) if combined else (
        sorted({_source_root(root,row)/"training_histories.json" for _,row in runs.iterrows()}) if runs is not None else [root/"training_histories.json"])
    rows,phases,auxiliary=[],[],[]
    for path in paths:
        if not path.is_file():continue
        for identity,history in json.loads(path.read_text()).items():
            method,*parts=identity.split("|")
            meta=dict(method=method,history_id=identity,source=str(path))
            meta.update(dict(p.split("=",1) for p in parts))
            if runs is not None and "scan_job" in runs:
                matched=runs[(runs.method==method)&(runs.split_seed==int(meta['split']))&(runs.model_seed==int(meta['seed']))]
                if matched.empty:continue
                matched=matched[matched.apply(lambda row:_source_root(root,row)==path.parent,axis=1)]
                if method not in ('AutoRec','AEMC-NE'):matched=matched[matched.K==int(meta['K'])]
                if matched.empty:continue
            for record in history:
                plain={k:v for k,v in record.items() if not isinstance(v,(dict,list))}
                if "val_RMSE" not in plain and "val_RMSE_log_only" in plain:plain["val_RMSE"]=plain["val_RMSE_log_only"]
                rows.append({**meta,**plain})
                for key,value in record.items():
                    if not isinstance(value,dict):continue
                    phases.append({**meta,"phase":key,"details_json":json.dumps(value,sort_keys=True)})
                    for trace in ("router_history","shape_history"):
                        for event in value.get(trace,[]):auxiliary.append({**meta,"phase":trace,**event})
    return pd.DataFrame(rows),pd.DataFrame(phases),pd.DataFrame(auxiliary)


def summarise_locked_runs(runs):
    """One row per method: model-seed means first, then outer-split dispersion."""
    numeric=[c for c in runs.select_dtypes(include=np.number) if c not in ("split_seed","model_seed","K")]
    outer=runs.groupby(["dataset","method","split_seed"],dropna=False)[numeric].mean().reset_index()
    grouped=outer.groupby(["dataset","method"],dropna=False)[numeric]
    result=pd.concat([grouped.mean().add_suffix("_mean"),grouped.std().add_suffix("_std"),
                      grouped.count().add_suffix("_count")],axis=1).reset_index()
    choices=[]
    for (dataset,method),frame in runs.groupby(["dataset","method"]):
        choices.append(dict(dataset=dataset,method=method,n_outer_splits=frame.split_seed.nunique(),
            n_model_seeds=frame.model_seed.nunique(),
            selected_Ks=",".join(str(int(k)) for k in sorted(frame.K.dropna().unique())) or "NA",
            selected_initializations=",".join(sorted(frame.init_mode.fillna("NA").astype(str).unique()))))
    return result.merge(pd.DataFrame(choices),on=["dataset","method"],validate="one_to_one")


def generate_report(root,runs,*,validation_only=False,plots=True,combined=False):
    root=Path(root);runs=runs.copy()
    if validation_only and any(str(c).startswith("test_") for c in runs):raise ValueError("Test columns forbidden in validation reports")
    if "init_mode" in runs:runs["init_mode"]=runs.init_mode.fillna("NA")
    report=root/"publication";tables=report/"tables";figures=report/"figures"
    old_manifest=report/'REPORT_MANIFEST.json'
    if plots and old_manifest.is_file():
        for item in json.loads(old_manifest.read_text()).get('figures',[]):
            previous=report/item
            if previous.parent==figures and previous.suffix in ('.pdf','.svg','.png'):previous.unlink(missing_ok=True)
    tables.mkdir(parents=True,exist_ok=True);primary="val" if validation_only else "test"
    keys=[k for k in ("dataset","method","split_seed","model_seed","K","init_mode") if k in runs]
    metrics=[c for c in runs.select_dtypes(include=np.number) if c.startswith(("train_","val_","test_")) and not c.endswith("sec")]
    _write(runs,tables/"all_run_metrics.csv")
    _write(runs.melt(id_vars=keys,value_vars=metrics,var_name="metric",value_name="value"),tables/"metrics_long.csv")
    outer=runs.groupby(["dataset","method","split_seed"])[metrics].mean().reset_index()
    _write(outer,tables/"outer_split_means.csv")
    variation=runs.groupby(["dataset","method","split_seed"])[metrics].agg(["mean","std","count"])
    variation.columns=[m+"__"+stat for m,stat in variation.columns]
    _write(variation.reset_index(),tables/"within_split_seed_variation.csv")
    rows=[]
    for (dataset,method),frame in outer.groupby(["dataset","method"]):
        for metric in metrics:
            values=frame[metric].dropna()
            if len(values):rows.append(dict(dataset=dataset,method=method,metric=metric,mean=values.mean(),
                std_outer_splits=values.std(ddof=1) if len(values)>1 else np.nan,n_outer_splits=len(values),
                inference="descriptive_one_split" if len(values)==1 else "outer_split_summary"))
    _write(pd.DataFrame(rows),tables/"metric_summary.csv")
    _write(runs[keys+(["config_key"] if "config_key" in runs else [])],tables/"model_configurations.csv")
    history,phases,auxiliary=_histories(root,combined,runs)
    _write(history,tables/"epoch_history.csv");_write(phases,tables/"phase_and_execution_diagnostics.csv")
    _write(auxiliary,tables/"auxiliary_phase_history.csv")
    responses=[]
    if len(phases):
        for _,event in phases[phases.phase=="v6_response"].iterrows():
            responses.append(dict(method=event.method,history_id=event.history_id,
                                  **json.loads(event.details_json)))
    _write(pd.DataFrame(responses),tables/"response_corrections.csv")
    budgets=[]
    if len(phases):
        for _,event in phases[phases.phase=='v6_training_budget'].iterrows():
            budgets.append(dict(method=event.method,history_id=event.history_id,**json.loads(event.details_json)))
    _write(pd.DataFrame(budgets),tables/'training_budget.csv')
    samples=[]
    if len(phases):
        for _,event in phases[phases.phase=="v6_execution"].iterrows():
            for sample in json.loads(event.details_json).get("device_sampling",{}).get("samples",[]):
                samples.append(dict(method=event.method,history_id=event.history_id,**sample))
    _write(pd.DataFrame(samples),tables/"device_utilization_samples.csv")
    reference={}
    for _,row in runs[runs.method=="MF"].iterrows():
        reference_root=_source_root(root,row,combined)
        path=_archive(reference_root,row,primary)
        if path:
            with np.load(path,allow_pickle=False) as archive:
                order=np.argsort(archive["pair_id"])
                reference[(row.dataset,int(row.split_seed),int(row.model_seed))]={k:archive[k][order] for k in ("pair_id","truth","prediction")}
    groups,ecdf,histograms,coverage,recomputed,paired,sources,calibration=[],[],[],[],[],[],[],[]
    for _,row in runs.iterrows():
        meta={k:row[k] for k in keys}
        source_root=_source_root(root,row,combined)
        assignment=source_root/f"split_assignments_{int(row.split_seed)}.csv"
        full=pd.read_csv(assignment) if assignment.is_file() else None
        train=full[full.split=="train"] if full is not None else None
        ni=int(full.movieId.max())+1 if full is not None else None
        dataset_manifest=source_root/'dataset_manifest.json'
        rating_range=json.loads(dataset_manifest.read_text()).get('documented_range') if dataset_manifest.is_file() else None
        for part in (("train","val") if validation_only else ("train","val","test")):
            path=_archive(source_root,row,part)
            if path is None:continue
            if validation_only and "__test.npz" in path.name:raise ValueError("Test archive in validation report")
            with np.load(path,allow_pickle=False) as archive:
                pair,truth,prediction=[archive[k] for k in ("pair_id","truth","prediction")]
                if not np.isfinite(prediction).all() or not np.isfinite(truth).all():raise FloatingPointError(str(path))
                error=prediction.astype(np.float64)-truth;absolute=abs(error)
                rmse=float(np.sqrt(np.mean(error**2)));reported=row.get(part+"_RMSE",np.nan)
                if np.isfinite(reported) and abs(rmse-float(reported))>2e-5:raise ValueError("Prediction/metric mismatch: "+str(path))
                record={**meta,"partition":part,"n":len(error),"RMSE":rmse,"MAE":absolute.mean(),"mean_error":error.mean(),"prediction_std":prediction.std()}
                if rating_range is not None:
                    clipped=np.clip(prediction.astype(np.float64),*rating_range);ce=clipped-truth
                    record.update(RMSE_clipped=np.sqrt(np.mean(ce**2)),MAE_clipped=np.mean(abs(ce)),clipping_rate=np.mean(clipped!=prediction),
                                  NRMSE_range=rmse/(rating_range[1]-rating_range[0]),
                                  NMAE_range=absolute.mean()/(rating_range[1]-rating_range[0]))
                recomputed.append(record)
                sources.append(dict(path=str(path),partition=part,n=len(error),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                frame=pd.DataFrame(dict(pair_id=pair,truth=truth,prediction=prediction,residual=error,absolute_error=absolute,squared_error=error**2))
                for name in archive.files:
                    if name=="levels":continue
                    value=archive[name]
                    if name not in frame and value.ndim==1 and len(value)==len(frame):frame[name]=value
                    if name in ("probabilities","member_means") and value.ndim==2 and len(value)==len(frame):
                        for j in range(value.shape[1]):frame[f"{name}_{j}"]=value[:,j]
                if ni is not None:
                    frame["userId"],frame["movieId"]=pair//ni,pair%ni
                    frame["user_train_count"]=frame.userId.map(train.groupby("userId").size()).fillna(0).astype(int)
                    frame["item_train_count"]=frame.movieId.map(train.groupby("movieId").size()).fillna(0).astype(int)
                token=f"{row.method}__split{int(row.split_seed)}__seed{int(row.model_seed)}__K{row.K}__{row.init_mode}__{part}"
                _write(frame,tables/"per_pair"/(token+".csv.gz"))
                # Quantile bins are descriptive within each fitted predictor;
                # they do not alter training, calibration or model selection.
                bins=pd.qcut(frame.prediction,q=10,duplicates='drop')
                for label,subset in frame.groupby(bins,observed=True):
                    calibration.append({**meta,'partition':part,'prediction_bin':str(label),'n':len(subset),
                        'predicted_mean':subset.prediction.mean(),'observed_mean':subset.truth.mean(),'RMSE':np.sqrt(subset.squared_error.mean())})
                for dimension in ("truth","user_train_count","item_train_count"):
                    if dimension not in frame:continue
                    bucket=frame.truth if dimension=="truth" else pd.cut(frame[dimension],[-1,5,20,50,100,np.inf],labels=["0–5","6–20","21–50","51–100",">100"])
                    for label,subset in frame.groupby(bucket,observed=True,sort=True):
                        groups.append({**meta,"partition":part,"dimension":dimension,"group":str(label),"n":len(subset),"RMSE":np.sqrt(subset.squared_error.mean()),"MAE":subset.absolute_error.mean(),"bias":subset.residual.mean()})
                for fraction,a,r in zip(np.linspace(0,1,201),np.quantile(absolute,np.linspace(0,1,201)),np.quantile(error,np.linspace(0,1,201))):
                    ecdf.append({**meta,"partition":part,"fraction":fraction,"absolute_error":a,"residual_quantile":r})
                counts,edges=np.histogram(error,bins=60)
                for n,left,right in zip(counts,edges[:-1],edges[1:]):histograms.append({**meta,"partition":part,"left":left,"right":right,"n":n})
                if part==primary and row.method!="MF":
                    ref=reference.get((row.dataset,int(row.split_seed),int(row.model_seed)))
                    if ref is not None:
                        order=np.argsort(pair)
                        if not np.array_equal(pair[order],ref["pair_id"]) or not np.array_equal(truth[order],ref["truth"]):raise ValueError("Unpaired evaluation cohorts")
                        ref_error=ref["prediction"].astype(np.float64)-ref["truth"]
                        paired.append({**meta,"reference":"MF selected on validation","partition":part,"n":len(error),"delta_RMSE":rmse-np.sqrt(np.mean(ref_error**2)),"delta_MSE":np.mean(error**2)-np.mean(ref_error**2),"inference":"descriptive fixed models; no training-seed uncertainty"})
                if part==primary and "probabilities" in archive and "levels" in archive:
                    prob=archive["probabilities"].astype(np.float64);levels=archive["levels"];cdf=prob.cumsum(-1)
                    reliability=[]
                    for j,level in enumerate(levels[:-1]):
                        bins=np.minimum((cdf[:,j]*10).astype(int),9)
                        for cell in np.unique(bins):
                            mask=bins==cell
                            reliability.append(dict(threshold=float(level),bin=int(cell),n=int(mask.sum()),predicted_cdf=cdf[mask,j].mean(),observed_cdf=(truth[mask]<=level).mean()))
                    _write(pd.DataFrame(reliability),tables/"probability"/(token+"__cdf_reliability.csv"))
                    index=np.abs(truth[:,None]-levels).argmin(-1);previous=np.where(index>0,cdf[np.arange(len(index)),np.maximum(index-1,0)],0.)
                    pit=np.clip(previous+np.random.default_rng(2026).random(len(index))*prob[np.arange(len(index)),index],0,1)
                    counts,edges=np.histogram(pit,bins=np.linspace(0,1,21))
                    _write(pd.DataFrame(dict(left=edges[:-1],right=edges[1:],n=counts)),tables/"probability"/(token+"__randomized_pit.csv"))
                    for nominal in (.5,.8,.9,.95):
                        tail=(1-nominal)/2
                        low=levels[(cdf<tail).sum(-1).clip(max=len(levels)-1)]
                        high=levels[(cdf<1-tail).sum(-1).clip(max=len(levels)-1)]
                        coverage.append({**meta,"partition":part,"nominal":nominal,"coverage":((truth>=low)&(truth<=high)).mean(),"mean_width":(high-low).mean()})
    datasets={"conditional_errors":groups,"absolute_error_ecdf":ecdf,"residual_histogram":histograms,"probability_coverage":coverage,
              "recomputed_prediction_metrics":recomputed,"paired_comparison_vs_MF":paired}
    for name,rows in datasets.items():_write(pd.DataFrame(rows),tables/(name+".csv"))
    _write(pd.DataFrame(calibration),tables/'point_prediction_calibration.csv')
    statuses=sorted((root/"final").rglob("V9_JOB_STATUS.json")) if combined else [root/"V9_JOB_STATUS.json"]
    smoke=any(json.loads(p.read_text()).get("command_arguments",{}).get("quick",False) for p in statuses if p.is_file())
    if not smoke:
        from .figures import _smoke
        smoke=_smoke(root)
    if plots:_render(figures,runs,primary,history,pd.DataFrame(groups),pd.DataFrame(ecdf),pd.DataFrame(coverage),auxiliary,smoke=smoke)
    if plots and any(row['partition']==primary for row in calibration):
        from .figures import _style,_save
        with _style() as plt:
            points=pd.DataFrame(calibration);points=points[points.partition==primary]
            methods=sorted(points.method.unique());cols=min(4,len(methods));rows=math.ceil(len(methods)/cols)
            fig,axes=plt.subplots(rows,cols,figsize=(3.1*cols,2.6*rows),squeeze=False,constrained_layout=True)
            for ax,method in zip(axes.flat,methods):
                line=points[points.method==method]
                ax.scatter(line.predicted_mean,line.observed_mean,s=15,color=COLORS.get(method,'#64717B'),alpha=.7)
                lo=min(line.predicted_mean.min(),line.observed_mean.min());hi=max(line.predicted_mean.max(),line.observed_mean.max())
                ax.plot([lo,hi],[lo,hi],ls='--',color='#A2AAB2',lw=.8)
                ax.set(title=method,xlabel='Mean predicted rating',ylabel='Mean observed rating');ax.grid()
            for ax in list(axes.flat)[len(methods):]:ax.set_visible(False)
            fig.suptitle(primary.capitalize()+' response calibration · quantile-bin means for each fitted model',fontsize=10)
            _save(fig,figures,'point_prediction_calibration',smoke)
    from .publication import finish_report
    finish_report(root,report,runs,primary,plots=plots,smoke=smoke)
    manifest=dict(format="MBMF-V6-report",stage="validation_only" if validation_only else "locked_test",primary_metric=primary+"_RMSE",
        test_used_for_selection=False,plots_generated=plots,smoke_only=smoke,
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        single_split_caution="No independent-split CI when n=1; multi-split bars are SD",
        training_error="In-sample full-context error; masked target risk is logged separately",
        structural_bounds="Support of point predictor, not confidence/credible intervals; full M4/M5 use global support",
        ranking="NDCG_testset ranks held-out rated items, not full-catalog retrieval",
        probability="Discrete equal-tail coverage can exceed nominal even under calibration",sources=sources,
        tables=[dict(path=str(p.relative_to(report)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(tables.rglob("*.csv*")) if p.is_file()],
        figures=[str(p.relative_to(report)) for p in sorted(figures.glob("*")) if plots and p.suffix in (".pdf",".svg",".png") and not p.name.startswith(".")],
        mechanism_files=[str(p.relative_to(root)) for p in sorted((root/"final" if combined else root).rglob("*")) if p.is_file() and "mechanisms" in p.relative_to(root).parts],
        calibration_archives=[str(p.relative_to(root)) for p in sorted((root/"final" if combined else root).rglob("*.npz")) if "calibration" in p.relative_to(root).parts])
    (report/"REPORT_MANIFEST.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str))
    return manifest


def _render(folder,runs,primary,history,groups,ecdf,coverage,auxiliary,*,smoke=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True,exist_ok=True)
    color=lambda method:COLORS.get(method,"#777777")
    def save(fig,name):
        if smoke:fig.text(.5,.5,"SMOKE · NOT FOR PUBLICATION",ha="center",va="center",fontsize=22,color="#A33D32",alpha=.2,rotation=20,zorder=100)
        try:
            for extension in ("pdf","svg","png"):
                target=folder/(name+"."+extension);temporary=folder/("."+name+".partial."+extension)
                try:fig.savefig(temporary,format=extension,dpi=400,bbox_inches="tight");temporary.replace(target)
                finally:temporary.unlink(missing_ok=True)
        finally:plt.close(fig)
    with plt.rc_context({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
        "axes.labelcolor":"#27313A","axes.titleweight":"semibold","grid.alpha":.18,"pdf.fonttype":42,
        "ps.fonttype":42,"svg.fonttype":"none","figure.facecolor":"white","axes.facecolor":"white","legend.frameon":False}):
        for metric in ("RMSE","MAE","P95AE","P99AE","NMAE_range","NRMSE_range","Global_OOB_Rate","Mean_OOB_Violation",
                       "IndividualBoundViolationRate","MeanIndividualBoundViolation","MeanBoundWidth","Probability_NLL","RankedProbabilityScore"):
            key=primary+"_"+metric
            if key not in runs or not runs[key].notna().any():continue
            outer=runs.groupby(["method","split_seed"])[key].mean().reset_index()
            values=outer.groupby("method")[key].mean().dropna().sort_values(ascending=False)
            sd=outer.groupby("method")[key].std()
            fig,ax=plt.subplots(figsize=(6.2,max(2.8,.28*len(values))),constrained_layout=True)
            ax.scatter(values,np.arange(len(values)),c=[color(m) for m in values.index],s=38,zorder=3)
            for y,(method,value) in enumerate(values.items()):
                if np.isfinite(sd.get(method,np.nan)):ax.errorbar(value,y,xerr=sd[method],fmt="none",ecolor="#888888",lw=.8,capsize=2)
                ax.annotate(f"{value:.4f}",(value,y),xytext=(6,0),textcoords="offset points",va="center",fontsize=8)
            ax.set_yticks(np.arange(len(values)),values.index);ax.margins(x=.2);ax.grid(axis="x")
            ax.set_xlabel(primary.capitalize()+" "+metric+(" (raw rating units)" if metric in ("RMSE","MAE","P95AE","P99AE") else ""))
            ax.set_title(("Validation selection" if primary=="val" else "Locked test evaluation")+(" · mean ± outer-split SD" if sd.notna().any() else ""),loc="left")
            save(fig,"rank_"+key)
        if len(history) and "val_RMSE" in history:
            methods=history.method.unique();columns=min(4,len(methods));rows=math.ceil(len(methods)/columns)
            fig,axes=plt.subplots(rows,columns,figsize=(3.1*columns,2.35*rows),squeeze=False,constrained_layout=True)
            for ax,method in zip(axes.flat,methods):
                for _,line in history[history.method==method].groupby(["history_id","source"]):
                    ax.plot(line.epoch,line.val_RMSE,color=color(method),lw=1.1,alpha=.85)
                    match=runs[(runs.method==method)&(runs.model_seed.astype(str)==str(line.seed.iloc[0]))&(runs.split_seed.astype(str)==str(line.split.iloc[0]))]
                    if "init_mode" in match:match=match[match.init_mode.astype(str)==str(line.init.iloc[0])]
                    if method not in ("AutoRec","AEMC-NE"):match=match[match.K==int(line.K.iloc[0])]
                    if "best_epoch" in match:
                        chosen=line[line.epoch.isin(match.best_epoch.dropna().unique())]
                        ax.scatter(chosen.epoch,chosen.val_RMSE,color=color(method),s=18,zorder=3)
                ax.set(title=method,xlabel="Epoch (method-specific budget)",ylabel="Validation RMSE");ax.grid()
            for ax in list(axes.flat)[len(methods):]:ax.set_visible(False)
            fig.suptitle("Mean training · separate panel scales · dots mark retained epochs",fontsize=10)
            save(fig,"training_validation_facets")
            if "train_RMSE" in history and history.train_RMSE.notna().any():
                fig,axes=plt.subplots(rows,columns,figsize=(3.1*columns,2.35*rows),squeeze=False,constrained_layout=True)
                for ax,method in zip(axes.flat,methods):
                    for _,line in history[history.method==method].groupby(["history_id","source"]):
                        ax.plot(line.epoch,line.val_RMSE,color=color(method),lw=1.1)
                        ax.plot(line.epoch,line.train_RMSE,color=color(method),linestyle="--",lw=1)
                    ax.set(title=method,xlabel="Epoch",ylabel="RMSE");ax.grid()
                for ax in list(axes.flat)[len(methods):]:ax.set_visible(False)
                fig.suptitle("Solid: validation · dashed: in-sample full-context train",fontsize=10)
                save(fig,"training_generalization_facets")
            if "optimizer_steps" in history:
                fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
                for method,group in history.dropna(subset=["optimizer_steps"]).groupby("method"):
                    for _,line in group.groupby(["history_id","source"]):ax.plot(line.optimizer_steps,line.val_RMSE,lw=1,color=color(method),label=method)
                ax.set(xlabel="Mean optimizer steps (OOF/shape recorded separately)",ylabel="Validation RMSE");ax.grid();ax.legend(fontsize=7,ncol=3)
                save(fig,"validation_vs_optimizer_steps")
        if len(auxiliary):
            phases=auxiliary.phase.unique();fig,axes=plt.subplots(1,len(phases),figsize=(5*len(phases),3.5),squeeze=False,constrained_layout=True)
            for ax,phase in zip(axes.flat,phases):
                for (method,identity),line in auxiliary[auxiliary.phase==phase].groupby(["method","history_id"]):
                    ax.plot(line.epoch,line.objective,color=color(method),lw=1,label=method)
                ax.set(title=phase,xlabel="Phase epoch",ylabel="Held-out phase objective (see configuration)");ax.grid();ax.legend(fontsize=7)
            save(fig,"auxiliary_phase_convergence")
        if len(ecdf):
            selected=ecdf[ecdf.partition==primary]
            fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
            for method,values in selected.groupby("method"):
                line=values.groupby("fraction",as_index=False).absolute_error.mean()
                ax.plot(line.absolute_error,line.fraction,label=method,color=color(method),lw=1.3,linestyle="--" if method in BASELINE_MARKERS else "-")
            ax.set(xlabel="Absolute prediction error",ylabel="Fraction of interactions",ylim=(0,1));ax.grid();ax.legend(fontsize=7,ncol=3,loc="lower right")
            save(fig,"absolute_error_ecdf")
            stats=[]
            for method,values in selected.groupby("method"):
                q=values.groupby("fraction").residual_quantile.mean()
                at=lambda level:float(q.iloc[np.argmin(abs(q.index.to_numpy()-level))])
                stats.append(dict(label=method,med=at(.5),q1=at(.25),q3=at(.75),whislo=at(.05),whishi=at(.95),fliers=[]))
            fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True);ax.bxp(stats,showfliers=False,vert=False)
            ax.axvline(0,color="#999999",lw=.8);ax.set_xlabel("Residual; whiskers: mean per-run 5th/95th percentiles");ax.grid(axis="x")
            save(fig,"residual_boxplot")
        if "train_RMSE" in runs and primary+"_RMSE" in runs:
            line=runs.assign(gap=runs[primary+"_RMSE"]-runs.train_RMSE).groupby("method").gap.mean().sort_values()
            fig,ax=plt.subplots(figsize=(6.5,4),constrained_layout=True);ax.scatter(line,np.arange(len(line)),c=[color(m) for m in line.index],s=34)
            ax.set_yticks(np.arange(len(line)),line.index);ax.grid(axis="x");ax.set_xlabel(primary.capitalize()+" RMSE − in-sample train RMSE");save(fig,"generalization_gap")
        for key in ("train_wall_sec","PeakGPUAllocated_MB","PeakGPUReserved_MB","PeakRSS_Delta_MB"):
            if key not in runs or not runs[key].notna().any():continue
            line=runs.groupby("method")[key].mean().dropna().sort_values()
            fig,ax=plt.subplots(figsize=(6.5,4),constrained_layout=True);ax.scatter(line,np.arange(len(line)),c=[color(m) for m in line.index],s=34)
            ax.set_yticks(np.arange(len(line)),line.index);ax.set_xlabel(key);ax.grid(axis="x");save(fig,"resource_"+key)
        ranking=sorted([c for c in runs if c.startswith(primary+"_NDCG_testset@")],key=lambda x:int(x.split("@")[-1]))
        if ranking:
            fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
            for method,frame in runs.groupby("method"):
                ax.plot([int(c.split("@")[-1]) for c in ranking],frame[ranking].mean(),ms=3,marker=BASELINE_MARKERS.get(method,"o"),
                    linestyle="--" if method in BASELINE_MARKERS else "-",color=color(method),label=method,lw=1)
            ax.set(xlabel="k (rank among held-out rated items)",ylabel="NDCG_testset@k");ax.grid();ax.legend(fontsize=7,ncol=3);save(fig,"ndcg_testset")
        if "train_wall_sec" in runs and primary+"_RMSE" in runs:
            values=runs.groupby("method",as_index=False)[["train_wall_sec",primary+"_RMSE"]].mean()
            fig,ax=plt.subplots(figsize=(7,4.3),constrained_layout=True)
            for _,row in values.iterrows():
                ax.scatter(max(row.train_wall_sec,1e-9),row[primary+"_RMSE"],color=color(row.method),s=40)
                ax.annotate(row.method,(max(row.train_wall_sec,1e-9),row[primary+"_RMSE"]),xytext=(4,4),textcoords="offset points",fontsize=7)
            ax.set_xscale("log");ax.set(xlabel="Training wall time (seconds, logarithmic)",ylabel=primary.capitalize()+" RMSE");ax.grid();save(fig,"accuracy_efficiency")
        if len(groups):
            for dimension in ("truth","user_train_count","item_train_count"):
                values=groups[(groups.partition==primary)&(groups.dimension==dimension)]
                if values.empty:continue
                fig,axes=plt.subplots(1,2,figsize=(9,3.3),constrained_layout=True)
                for method,frame in values.groupby("method"):
                    line=frame.groupby("group",sort=False,as_index=False)[["RMSE","bias"]].mean()
                    for ax,key in zip(axes,("RMSE","bias")):
                        ax.plot(line.group,line[key],marker=BASELINE_MARKERS.get(method,"o"),ms=3,lw=1,
                            linestyle="--" if method in BASELINE_MARKERS else "-",label=method,color=color(method))
                        ax.set(xlabel=dimension,ylabel=key);ax.grid();ax.tick_params(axis="x",rotation=45)
                axes[1].axhline(0,color="#777777",lw=.6)
                handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="outside lower center",ncol=7,fontsize=7)
                save(fig,"error_by_"+dimension)
        if len(coverage):
            fig,axes=plt.subplots(1,2,figsize=(8,3.3),constrained_layout=True)
            for method,frame in coverage[coverage.partition==primary].groupby("method"):
                line=frame.groupby("nominal",as_index=False)[["coverage","mean_width"]].mean()
                axes[0].plot(line.nominal,line.coverage,"o-",color=color(method),label=method)
                axes[1].plot(line.nominal,line.mean_width,"o-",color=color(method),label=method)
            axes[0].plot([.5,1],[.5,1],"--",color="#999999",lw=.8)
            for ax in axes:ax.set_xlabel("Nominal probability");ax.grid();ax.legend()
            axes[0].set_ylabel("Observed coverage (ordinal equal-tail)");axes[1].set_ylabel("Mean interval width")
            save(fig,"probability_coverage_and_width")
            probability_root=folder.parent/"tables"/"probability"
            files=sorted(probability_root.glob("*__cdf_reliability.csv"))
            if files:
                frame=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
                fig,ax=plt.subplots(figsize=(5.2,4),constrained_layout=True)
                for threshold,values in frame.groupby("threshold"):
                    # Recombine bins by their actual interaction counts.
                    rows=[]
                    for cell,part in values.groupby("bin"):
                        rows.append((np.average(part.predicted_cdf,weights=part.n),np.average(part.observed_cdf,weights=part.n)))
                    line=np.asarray(rows);ax.plot(line[:,0],line[:,1],"o-",ms=3,label=f"P(Y ≤ {threshold:g})",lw=1)
                ax.plot([0,1],[0,1],"--",color="#999999",lw=.8)
                ax.set(xlabel="Predicted cumulative probability",ylabel="Observed cumulative frequency",xlim=(0,1),ylim=(0,1));ax.grid();ax.legend(fontsize=8)
                save(fig,"ordinal_cdf_reliability")
            files=sorted(probability_root.glob("*__randomized_pit.csv"))
            if files:
                frame=pd.concat([pd.read_csv(p) for p in files],ignore_index=True).groupby(["left","right"],as_index=False).n.sum()
                fig,ax=plt.subplots(figsize=(5.2,3.5),constrained_layout=True)
                ax.bar(frame.left,frame.n/frame.n.sum()/(frame.right-frame.left),width=frame.right-frame.left,align="edge",color="#6C4A97",alpha=.75,edgecolor="white",linewidth=.4)
                ax.axhline(1,color="#555555",linestyle="--",lw=.8);ax.set(xlabel="Randomized ordinal PIT (fixed diagnostic seed)",ylabel="Density",xlim=(0,1))
                save(fig,"randomized_ordinal_pit")
