"""Variant-aware aggregate workbooks; cuts are never pooled as seed repeats."""
import json
from pathlib import Path
import pandas as pd
from .export import export_workbook
from .reporting import _write


def export_study(root, frame, variants=(), primary="test"):
    root=Path(root)
    if frame.empty:return
    report=root/"publication";tables=report/"tables";tables.mkdir(parents=True,exist_ok=True)
    if primary=="val" and any(str(c).startswith("test_") for c in frame):
        raise ValueError("Test metrics reached validation-only secondary reporting")
    metrics=[c for c in frame.select_dtypes(include="number") if c.startswith(("train_","val_","test_")) and c not in variants]
    groups=list(dict.fromkeys(["dataset","method",*variants,"split_seed"]))
    outer=frame.groupby(groups,dropna=False)[metrics].mean().reset_index()
    summary=outer.groupby(groups[:-1],dropna=False)[metrics].agg(["mean","std","count"])
    summary.columns=["__".join(x) for x in summary.columns]
    ids=list(dict.fromkeys([c for c in [*groups,"model_seed","K","init_mode"] if c in frame]))
    _write(frame,tables/"study_all_runs.csv")
    _write(outer,tables/"study_outer_split_means.csv")
    _write(summary.copy().reset_index(),tables/"study_summary.csv")
    _write(frame.melt(id_vars=ids,value_vars=metrics,var_name="metric",value_name="value"),
           tables/"study_metrics_long.csv")
    if "config_key" in frame:
        _write(frame[[*ids,"config_key"]],tables/"study_configurations.csv")
    export_workbook(report)
    manifest=dict(variants=list(variants),primary=primary,rows=len(frame),
        n_outer_splits=int(frame.split_seed.nunique()),ablation_variants_pooled=False,
        confirmatory_significance=False,child_pair_data_preserved=True,
        note="Variant-specific model-seed means precede outer-split summaries; no invented one-split CI.")
    (report/"STUDY_REPORT_MANIFEST.json").write_text(json.dumps(manifest,indent=2))
    (root/"PUBLICATION_SCOPE.md").write_text(
        "# Secondary study report\n\n"
        "publication/research_tables.xlsx contains variant-aware aggregate tables.\n"
        "Complete prediction archives, pair CSVs, histories and individual reports remain in each child job; "
        "they are not duplicated here. Follow prediction-file columns in study_all_runs.csv.\n")


def rebuild_secondary_reports(dsroot,plots=True):
    dsroot=Path(dsroot)
    for folder in dsroot.glob("ablations*"):
        path=folder/"ablation_locked_test_runs.csv"
        if path.is_file():export_study(folder,pd.read_csv(path),("ablation",))
    kstudy=dsroot/"kstudy";path=kstudy/"all_fixed_k_locked_runs.csv"
    if path.is_file():export_study(kstudy,pd.read_csv(path),("K",))
    sparsity=dsroot/"sparsity";pieces=[]
    for path in sorted(sparsity.glob("keep_*/locked_test_runs.csv")):
        frame=pd.read_csv(path);frame["train_keep_requested"]=float(path.parent.name[5:]);pieces.append(frame)
    if pieces:
        frame=pd.concat(pieces,ignore_index=True,sort=False)
        _write(frame,sparsity/"all_fixed_holdout_runs.csv")
        export_study(sparsity,frame,("train_keep_requested",))
    calibration=dsroot/"calibration";path=calibration/"calibration_locked_test_runs.csv"
    if path.is_file():export_study(calibration,pd.read_csv(path))
    convergence=dsroot/"convergence"
    pieces=[pd.read_csv(p) for p in convergence.rglob("candidate_validation_runs.csv")]
    if pieces:export_study(convergence,pd.concat(pieces,ignore_index=True,sort=False),primary="val")
    density=dsroot/"synthetic_density";pieces=[]
    for path in sorted(density.glob("density_*/locked_test_runs.csv")):
        frame=pd.read_csv(path);frame["density"]=float(path.parent.name[8:]);pieces.append(frame)
    if pieces:
        frame=pd.concat(pieces,ignore_index=True,sort=False)
        _write(frame,density/"all_density_runs.csv")
        export_study(density,frame,("density",))
        if plots:
            from .figures import _style,_save,_smoke
            from .reporting import COLORS
            outer=frame.groupby(["method","density","split_seed"],as_index=False).test_RMSE.mean()
            summary=outer.groupby(["method","density"]).test_RMSE.agg(["mean","std","count"]).reset_index()
            with _style() as plt:
                fig,axes=plt.subplots(2,1,figsize=(6.6,7.2),sharex=True,constrained_layout=True)
                for ax,proposed in zip(axes,(False,True)):
                    for method,line in summary.groupby("method"):
                        if method.startswith("Method")!=proposed:continue
                        line=line.sort_values("density")
                        yerr=line["std"] if (line["count"]>1).all() else None
                        ax.errorbar(line.density,line["mean"],yerr=yerr,marker="o",
                                    color=COLORS.get(method),label=method,lw=1.2,capsize=2)
                    ax.set(ylabel="Test RMSE",title="Proposed methods" if proposed else "Competitors")
                    ax.grid();ax.legend(fontsize=7,ncol=3)
                axes[-1].set_xlabel("Observed-data density")
                fig.suptitle("Generated-data sensitivity · descriptive cohorts",fontsize=10)
                _save(fig,density/"figures","density_rmse",_smoke(density))
