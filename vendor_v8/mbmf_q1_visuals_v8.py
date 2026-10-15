"""Publication-oriented *presentation only* of existing locked V8 results.

No training, splits, models, metrics, or hyperparameters are changed here.
Every chart is descriptive; confidence intervals are omitted for single replicates.
Figures are generated from locked_test_runs and saved prediction archives only.
"""
from __future__ import annotations
import json
import math
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# 14 distinct qualitative colors, stable across every figure and dataset.
# We also encode method identity via sorted labels/markers, not color alone.
_METHOD_ORDER = ["MF", "BiasMF", "MBMF-F", "MBMF-C", "MBMF-N",
                 "AutoRec", "AEMC-NE"] + [f"Method{i}" for i in range(1, 8)]
_PALETTE = ["#3659B5", "#D35400", "#008F87", "#9C4B95", "#AC8000",
            "#367DAB", "#B13453", "#4F7730", "#BB6D3A", "#6E61B5",
            "#117B65", "#91592C", "#B3428D", "#565E68"]
_MARKERS = ["o", "s", "^", "D", "P", "v", "X", "<", ">", "h", "p", "*", "8", "d"]


def _style():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.0, "axes.titlesize": 11,
        "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
        "legend.fontsize": 8, "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "axes.grid": False,
    })
    return plt


def _name_order(methods):
    methods = [str(m) for m in methods]
    return sorted(methods, key=lambda m: (_METHOD_ORDER.index(m) if m in _METHOD_ORDER else 999, m))


def method_style(method):
    m = str(method)
    if m in _METHOD_ORDER:
        j = _METHOD_ORDER.index(m)
    else:
        # Stable hash avoids Python's randomized hash seed.
        import hashlib
        j = int(hashlib.sha256(m.encode()).hexdigest()[:8], 16) % len(_PALETTE)
    return _PALETTE[j], _MARKERS[j]


def _save(fig, folder, filename, notes, registry, plt):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{filename}.png", dpi=320, bbox_inches="tight", pad_inches=.16)
    fig.savefig(folder / f"{filename}.pdf", bbox_inches="tight", pad_inches=.16)
    plt.close(fig)
    registry.append({"file": filename, "source": notes, "formats": ["PNG_320dpi", "PDF_vector"]})


def _clean(runs):
    required = {"dataset", "method"}
    if not required.issubset(runs.columns):
        raise ValueError(f"Missing locked-result columns: {required - set(runs.columns)}")
    d = runs.copy()
    d["method"] = d["method"].astype(str)
    return d


def _desc(runs, metric):
    if metric not in runs.columns:
        return pd.DataFrame()
    d = runs[["method", metric]].copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return pd.DataFrame()
    return d.groupby("method", observed=True)[metric].agg(["mean", "std", "count"])


def _ranked_metric(runs, metric, folder, registry, plt):
    s = _desc(runs, metric)
    if s.empty:
        return
    lower_better = any(x in metric for x in ["RMSE", "MAE", "OOB", "Violation", "P95AE", "NLL"])
    s = s.sort_values("mean", ascending=lower_better)
    colors = [method_style(m)[0] for m in s.index]
    height = max(4.1, .41 * len(s) + 1.1)
    fig, ax = plt.subplots(figsize=(8.8, height))
    y = np.arange(len(s))
    ax.hlines(y, 0, s["mean"].to_numpy(float), color=colors, alpha=.45, lw=1.5)
    ax.scatter(s["mean"], y, s=54, color=colors, edgecolor="white", linewidth=.8, zorder=3)
    # Only draw uncertainty if there are independent replicates; never show fake
    # error bars on a one-seed pilot or describe std as a 95% CI.
    for j, (_, v) in enumerate(s.iterrows()):
        if v["count"] > 1 and np.isfinite(v["std"]) and v["std"] > 0:
            ax.errorbar(v["mean"], j, xerr=v["std"], fmt="none", ecolor=colors[j], capsize=2, lw=1.1)
    ax.set_yticks(y, s.index)
    ax.invert_yaxis()
    ax.set_xlabel(metric.replace("test_", "Test ").replace("_", " "))
    ax.set_title(metric.replace("test_", "Test ").replace("_", " ") + " | method comparison", loc="left", fontweight="semibold")
    ax.grid(axis="x", color="#DDE3EA", linewidth=.7)
    ax.set_axisbelow(True)
    mn = float(s["mean"].min());mx = float(s["mean"].max());span = max(1e-6, mx-mn)
    # Zero-centered when possible: no misleading truncated bar baseline.
    ax.set_xlim(min(0.0, mn-.10*span), mx+.18*max(span,abs(mx)*.06,1e-3))
    for yy, val in enumerate(s["mean"]):
        ax.annotate(f"{val:.4g}", (float(val), yy), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7.7)
    note = ("Descriptive only — one replicate; no uncertainty interval" if (s["count"]<2).all()
            else "Markers: mean; whiskers: ±1 SD (where n≥2)")
    fig.text(.13, .013, note, fontsize=7.5, color="#555E68")
    fig.tight_layout(rect=(0,.035,1,1))
    _save(fig, folder, f"q1_rank_{metric}", f"locked_test_runs.csv:{metric}; {note}", registry, plt)


def _tradeoff(runs, folder, registry, plt):
    cols={"method", "test_RMSE", "train_wall_sec"}
    if not cols.issubset(runs.columns):return
    d=runs.groupby("method")[["test_RMSE","train_wall_sec"]].mean().replace([np.inf,-np.inf], np.nan).dropna()
    d=d[d.train_wall_sec>0]
    if d.empty:return
    fig, ax=plt.subplots(figsize=(9,5.8))
    for m in _name_order(d.index):
        r=d.loc[m];color,marker=method_style(m)
        ax.scatter(r.train_wall_sec,r.test_RMSE,s=85,c=color,marker=marker,edgecolors="white",lw=.5,zorder=3)
        ax.annotate(m,(r.train_wall_sec,r.test_RMSE),xytext=(5,5),textcoords="offset points",fontsize=7.5)
    if d.train_wall_sec.max()/max(d.train_wall_sec.min(),1e-9)>15:ax.set_xscale("log")
    ax.set_xlabel("Training time per run (seconds)");ax.set_ylabel("Test RMSE (lower is better)")
    ax.set_title("Accuracy–efficiency trade-off",loc="left",fontweight="semibold")
    ax.grid(alpha=.22);fig.tight_layout()
    _save(fig,folder,"q1_accuracy_efficiency","locked_test_runs.csv:test_RMSE,train_wall_sec; descriptive",registry,plt)


def _split_gap(runs, folder, registry, plt):
    cs=[c for c in ("train_RMSE","val_RMSE","test_RMSE") if c in runs]
    if len(cs)<2:return
    a=runs.groupby("method")[cs].mean().dropna(how="all")
    a=a.reindex([m for m in _name_order(a.index)])
    fig, ax=plt.subplots(figsize=(max(9, len(a)*.62),5.5));x=np.arange(len(a))
    markerstyle=[("train_RMSE","s","#64748B"),( "val_RMSE","^","#2F7E90"),("test_RMSE","o","#C64D3C")]
    for col,marker,color in markerstyle:
        if col in a:ax.plot(x,a[col],marker=marker,lw=1.5,ms=5,label=col.replace("_RMSE","" ).title(),color=color)
    ax.set_xticks(x,a.index,rotation=40,ha="right");ax.set_ylabel("RMSE");ax.set_title("Train / validation / locked test",loc="left")
    ax.grid(axis="y",alpha=.25);ax.legend(frameon=False,ncol=3);fig.tight_layout()
    _save(fig,folder,"q1_generalization_gap","locked_test_runs.csv:train/val/test RMSE",registry,plt)


def _ndcg(runs,folder,registry,plt):
    cs=[(int(c.split("@")[-1]),c) for c in runs.columns if c.startswith("test_NDCG_testset@")]
    if not cs:return
    cs=sorted(cs);fig,ax=plt.subplots(figsize=(9,5.8))
    for m in _name_order(runs.method.unique()):
        d=runs[runs.method==m]
        vals=[pd.to_numeric(d[c],errors="coerce").mean() for _,c in cs]
        color,marker=method_style(m)
        ax.plot([k for k,_ in cs],vals,color=color,marker=marker,ms=4.4,lw=1.5,label=m)
    ax.set_xlabel("k");ax.set_ylabel("NDCG@k");ax.set_title("Held-out-rated-item ranking (NOT full-catalog)",loc="left")
    ax.grid(alpha=.2);ax.legend(bbox_to_anchor=(1.01,1),loc="upper left",frameon=False,ncol=1)
    fig.tight_layout();_save(fig,folder,"q1_ndcg_testset","locked_test_runs.csv:test_NDCG_testset@k; testset ranking only",registry,plt)


def _prediction_figures(runs,folder,registry,plt):
    if "test_prediction_file" not in runs:return
    arrays=[]
    for m in _name_order(runs.method.unique()):
        paths=runs.loc[runs.method==m,"test_prediction_file"].dropna()
        if len(paths)==0:continue
        path=Path(str(paths.iloc[0]))
        if not path.exists():continue
        try:
            with np.load(path) as z:
                truth=np.asarray(z["truth"],float); pred=np.asarray(z["prediction"],float)
            good=np.isfinite(truth)&np.isfinite(pred)
            if good.sum()==0:continue
            arrays.append((m,truth[good],pred[good]))
        except (OSError,KeyError,ValueError) as exc:
            warnings.warn(f"Prediction archive for {m} skipped: {exc}")
    if not arrays:return
    # 1: ECDF of |error|, more diagnostic than many near-duplicate bars.
    fig,ax=plt.subplots(figsize=(8,5.7))
    for m,t,p in arrays:
        e=np.sort(np.abs(p-t));
        if len(e)>5000:e=e[np.linspace(0,len(e)-1,5000,dtype=int)]
        color,_=method_style(m);ax.step(e,np.linspace(1/len(e),1,len(e)),color=color,lw=1.5,label=m,where="post")
    ax.set_xlabel("Absolute rating error");ax.set_ylabel("Cumulative proportion")
    ax.set_title("Absolute-error distribution on locked test",loc="left");ax.set_ylim(0,1.01)
    ax.grid(alpha=.2);ax.legend(bbox_to_anchor=(1.01,1),loc="upper left",frameon=False)
    fig.tight_layout();_save(fig,folder,"q1_absolute_error_ecdf","selected-model saved test predictions + ground truth",registry,plt)
    # 2: Mean absolute error conditional on true ordinal score (no invented bins).
    fig,ax=plt.subplots(figsize=(8.7,5.6))
    for m,t,p in arrays:
        scores=np.unique(t)
        if len(scores)>25:continue # only a meaningful categorical scale
        counts=np.array([(t==v).sum() for v in scores]);mae=np.array([np.abs(p[t==v]-t[t==v]).mean() for v in scores])
        color,marker=method_style(m);ax.plot(scores,mae,color=color,marker=marker,lw=1.4,ms=4,label=m)
    if ax.lines:
        ax.set_xlabel("Ground-truth rating");ax.set_ylabel("MAE")
        ax.set_title("Error stratified by true rating",loc="left");ax.grid(alpha=.2)
        ax.legend(bbox_to_anchor=(1.01,1),loc="upper left",frameon=False)
        fig.tight_layout();_save(fig,folder,"q1_error_by_rating","selected-model saved test predictions + ground truth; no smoothing",registry,plt)
    else:plt.close(fig)
    # 3: residual distributions via boxplot (deterministic capped sample per method).
    fig,ax=plt.subplots(figsize=(max(8,len(arrays)*.57),5.4));box_data=[];labels=[]
    for m,t,p in arrays:
        resid=p-t
        if len(resid)>4000:resid=resid[np.linspace(0,len(resid)-1,4000,dtype=int)]
        box_data.append(resid);labels.append(m)
    art=ax.boxplot(box_data,showfliers=False,patch_artist=True,medianprops={"color":"white","linewidth":1.8})
    for b,m in zip(art["boxes"],labels):b.set_facecolor(method_style(m)[0]);b.set_alpha(.86)
    ax.axhline(0,color="#334155",ls="--",lw=1);ax.set_xticks(np.arange(1,len(labels)+1),labels,rotation=42,ha="right")
    ax.set_ylabel("Predicted − true rating");ax.set_title("Signed-error distribution (outliers hidden)",loc="left")
    ax.grid(axis="y",alpha=.2);fig.tight_layout()
    _save(fig,folder,"q1_residual_boxplot","selected-model saved test predictions; deterministic max 4000 samples/method",registry,plt)


def _training_curves(path,folder,registry,plt):
    path=Path(path)
    if not path.exists():return
    with path.open(encoding="utf-8") as f:histories=json.load(f)
    fig,ax=plt.subplots(figsize=(9,5.5));n=0
    for key,hist in histories.items():
        if not isinstance(hist,list) or not hist or "val_RMSE" not in hist[0]:continue
        m=key.split("|")[0];color,_=method_style(m)
        ax.plot([v["epoch"] for v in hist],[v["val_RMSE"] for v in hist],color=color,lw=1.4,label=m,alpha=.88)
        n+=1
        if n>=25:break
    if n==0:plt.close(fig);return
    if n>1:ax.legend(bbox_to_anchor=(1.01,1),loc="upper left",frameon=False)
    ax.set_xlabel("Epoch");ax.set_ylabel("Validation RMSE");ax.set_title("Optimization traces (descriptive)",loc="left")
    ax.grid(alpha=.22);fig.tight_layout()
    _save(fig,folder,"q1_training_validation_rmse","training_histories.json (validation only)",registry,plt)


def generate_publication_figures_v8(runs:pd.DataFrame,output_dir,history_json=None):
    """Generate publication-style plots, plus machine-readable figure provenance.

    Called only AFTER locked evaluation. Missing input causes documented omissions,
    not invented curves, fabricated significance, or silent misleading error bars.
    """
    runs=_clean(runs);root=Path(output_dir);folder=root/"figures";folder.mkdir(parents=True,exist_ok=True)
    plt=_style();registry=[]
    metrics=["test_RMSE","test_MAE","test_NMAE_range","test_P95AE",
             "test_Global_OOB_Rate","test_Mean_OOB_Violation",
             "test_IndividualBoundViolationRate","test_MeanBoundWidth",
             "test_Gaussian_NLL","test_PICP_90","train_wall_sec","PeakRSS_Delta_MB","PeakGPUAllocated_MB"]
    for metric in metrics:
        _ranked_metric(runs,metric,folder,registry,plt)
    _split_gap(runs,folder,registry,plt)
    _tradeoff(runs,folder,registry,plt)
    _ndcg(runs,folder,registry,plt)
    _prediction_figures(runs,folder,registry,plt)
    if history_json is not None:_training_curves(history_json,folder,registry,plt)
    # Single-seed design cannot support paired Wilcoxon/Friedman: explicit warning.
    nsplit=int(runs["split_seed"].nunique()) if "split_seed" in runs else 0
    provenance={
       "schema":"q1-viz-v1", "source":"locked_test_runs.csv + selected prediction archives",
       "figures":registry, "n_split_seeds":nsplit,
       "statistical_note":("One split seed: paired significance tests and confidence intervals are NOT inferentially valid."
                           if nsplit<2 else "Replicated splits present; distinguish SD from CI."),
       "legend":"NDCG_testset@k is on held-out rated items, not catalog-wide ranking.",
       "consistent_model_palette":dict(zip(_METHOD_ORDER,_PALETTE)),
       "exclusions":"Ablation/sparsity/K-sensitivity figures require corresponding experimental variation; not fabricated for one K.",
    }
    (root/"figure_manifest.json").write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding="utf-8")
    return provenance


def generate_ablation_figures_v8(ablations: pd.DataFrame, output_dir):
    """Per-family ablation effect plots; every original ablation remains present."""
    if not {"ablation", "method", "test_RMSE"}.issubset(ablations):
        raise ValueError("Ablation figure input must contain ablation, method and test_RMSE")
    plt = _style();folder=Path(output_dir)/"figures";registry=[]
    for method in _name_order(ablations.method.unique()):
        d=ablations[ablations.method == method]
        stats=d.groupby("ablation",observed=True)["test_RMSE"].agg(["mean","count"])
        full=f"m{method[6:]}_full" if method.startswith("Method") else None
        if full not in stats.index:
            warnings.warn(f"Missing full ablation baseline for {method}; skipping differential plot")
            continue
        baseline=float(stats.loc[full,"mean"])
        d=stats.drop(index=full).sort_values("mean")
        if d.empty:continue
        delta=d["mean"].to_numpy(float)-baseline
        colors=["#BF483F" if v>0 else "#178169" for v in delta]
        fig,ax=plt.subplots(figsize=(9,max(3,.56*len(d)+1.4)))
        y=np.arange(len(d));ax.barh(y,delta,color=colors,alpha=.88,height=.67)
        ax.axvline(0,color="#334155",lw=1.1)
        ax.set_yticks(y,d.index);ax.invert_yaxis()
        ax.set_xlabel("Δ test RMSE (ablated − full; positive means worse)")
        ax.set_title(f"{method} | mechanism ablations",loc="left",fontweight="semibold")
        ax.grid(axis="x",color="#DDE3EA",alpha=.7);ax.set_axisbelow(True)
        fig.tight_layout()
        _save(fig,folder,f"q1_ablation_delta_{method}",
              f"ablation_locked_test_runs.csv; comparison against {full} within same dataset/seeds/K",registry,plt)
    (Path(output_dir)/"ablation_figure_manifest.json").write_text(json.dumps({"figures":registry,
        "note":"Each named ablation remains unchanged; plot shows descriptive means and Δ to its full model."},indent=2),encoding="utf-8")
    return registry


def generate_multi_dataset_figures_v8(runs:pd.DataFrame,output_dir):
    """Cross-dataset RMSE heatmap with numerical cells; no invalid pooling of scales."""
    runs=_clean(runs)
    if runs.dataset.nunique()<2 or "test_RMSE" not in runs:return []
    folder=Path(output_dir)/"figures";plt=_style();registry=[]
    table=runs.groupby(["dataset","method"])["test_RMSE"].mean().unstack("method")
    table=table.reindex(columns=_name_order(table.columns))
    rows,cols=table.shape
    fig,ax=plt.subplots(figsize=(max(9,cols*.7+2),max(4.5,rows*.6+1.5)))
    # Per-dataset ranks avoid comparing incompatible rating scales.
    rank=table.rank(axis=1,method="average",ascending=True)
    im=ax.imshow(rank.to_numpy(float),aspect="auto",cmap="viridis_r",vmin=1,vmax=max(cols,2))
    ax.set_yticks(np.arange(rows),table.index)
    ax.set_xticks(np.arange(cols),table.columns,rotation=40,ha="right")
    for i in range(rows):
        for j in range(cols):
            v=table.iat[i,j]
            if np.isfinite(v):ax.text(j,i,f"{v:.3f}",ha="center",va="center",fontsize=7,
                                      color="black" if rank.iat[i,j]<cols*.48 else "white")
    cb=fig.colorbar(im,ax=ax,shrink=.75);cb.set_label("Within-dataset rank (lower is better)")
    ax.set_title("RMSE across datasets (cells: absolute RMSE; color: rank)",loc="left")
    fig.tight_layout();_save(fig,folder,"q1_cross_dataset_rmse_heatmap",
      "all_datasets_locked_runs.csv; per-dataset rank colors, raw RMSE cell text",registry,plt)
    return registry
