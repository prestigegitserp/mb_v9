"""Dataset-specific, validation-locked V7 experiments without changing releases."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DATASETS = ("synthetic","ml-100k","ml-1m","ml-10m","jester","bookcrossing")
STAGES = ("plan","pilot","develop","scan","final","lossstudy","ablation","ablation-fixed","kstudy",
          "sparsity","density","convergence","calibration","temporal","report","audit","all")


def build_commands(a):
    datasets = DATASETS if a.datasets == "all" else tuple(a.datasets.split(","))
    if not datasets or set(datasets)-set(DATASETS):
        raise ValueError("Unknown datasets")
    profile = a.profile
    splits = a.split_seeds or ("2026" if profile in ("development","common") else ",".join(map(str,range(4101,4121))))
    seeds = a.model_seeds or ("1" if profile in ("development","common") else "11,22,33")
    ks = a.k or ("20" if profile in ("development","common") else "10,20,50")
    commands = []
    for dataset in datasets:
        config = ROOT/"examples/v7"/profile/(dataset+".json")
        common = ["--datasets",dataset,"--data-root",str(a.data_root.resolve()),
                  "--output",str(a.output.resolve()),"--split-seeds",splits,
                  "--model-seeds",seeds,"--k",ks,"--scan-mode","hybrid",
                  "--analysis-source","winners","--v7-config",str(config),
                  "--v7-threads",str(a.threads),"--v7-retain-candidates","metrics",
                  "--v7-device",a.device,"--v7-storage",a.storage,
                  "--v7-host-budget-mb",str(a.host_budget_mb),
                  "--v7-ssd-budget-gb",str(a.ssd_budget_gb),
                  "--v7-cache-dir",str(a.cache_dir.resolve())]
        if getattr(a,"manual_seed",None) is not None:
            index=common.index("--model-seeds");common[index:index+2]=["--manual-seed",str(a.manual_seed)]
        if getattr(a,"deterministic",False):common += ["--v7-deterministic"]
        if a.methods:
            common += ["--methods",a.methods]
        if a.blend_methods:
            common += ["--include-blend","--blend-methods",a.blend_methods]
        if a.search_jobs_per_method is not None:
            common += ["--v7-search-jobs-per-method",str(a.search_jobs_per_method)]
        if a.quick:
            common += ["--quick","--v7-search-jobs-per-method","1"]
        for name in ("max_epochs","baseline_max_epochs","steps_per_epoch"):
            value=getattr(a,name)
            if value is not None:
                common += ["--v7-"+name.replace("_","-"),str(value)]
        if a.no_plots:
            common += ["--v7-no-plots"]
        common += ['--epoch-train-rmse' if getattr(a,'epoch_train_rmse',False) else '--no-epoch-train-rmse']
        stages = (["scan","final","lossstudy","ablation","ablation-fixed","kstudy","sparsity",
                   "convergence","calibration","report","audit"] if a.stage=="all" else [a.stage])
        if a.stage=="all" and dataset=="synthetic":
            stages.insert(-2,"density")
        if a.stage=="all" and a.include_temporal and dataset.startswith("ml-"):
            stages.insert(-2,"temporal")
        for stage in stages:
            command="main" if stage=="pilot" else "scan" if stage=="develop" else stage
            extra = []
            if stage=="scan":
                extra += ["--scan-only"]
            if command not in ("plan","report","audit"):
                extra += ["--resume"]
            commands.append([sys.executable,str(ROOT/"run_v7.py"),command,*common,*extra])
    return commands


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=STAGES,default="plan")
    p.add_argument("--profile",choices=("common","development","paper"),default="development")
    p.add_argument("--datasets",default="ml-100k")
    p.add_argument("--data-root",type=Path,default=Path("./datasets"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",choices=("auto","cpu","cuda","cuda:0"),default="auto")
    p.add_argument("--split-seeds");p.add_argument("--model-seeds");p.add_argument("--k")
    p.add_argument("--manual-seed",type=int)
    p.add_argument("--deterministic",action="store_true")
    p.add_argument("--methods");p.add_argument("--search-jobs-per-method",type=int)
    p.add_argument("--blend-methods",help="Optional predeclared two-model ensemble control; reported separately")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--storage",choices=("auto","resident","host","ssd"),default="auto")
    p.add_argument("--host-budget-mb",type=int,default=512)
    p.add_argument("--ssd-budget-gb",type=float,default=8.)
    p.add_argument("--cache-dir",type=Path,default=Path("./v7_ssd_cache"))
    p.add_argument("--max-epochs",type=int);p.add_argument("--baseline-max-epochs",type=int)
    p.add_argument("--steps-per-epoch",type=int)
    p.add_argument("--include-temporal",action="store_true")
    p.add_argument("--no-plots",action="store_true")
    p.add_argument('--epoch-train-rmse',action=argparse.BooleanOptionalAction,default=False,
                   help='Extra full TRAIN evaluation at every tick; convergence enables its own curves')
    p.add_argument("--quick",action="store_true",help="Smoke only, never manuscript accuracy")
    p.add_argument("--dry-run",action="store_true",help="Print commands without running them")
    return p


def main(argv=None):
    p=parser();a=p.parse_args(argv)
    if a.manual_seed is not None and (a.manual_seed<0 or a.model_seeds is not None):p.error("Use a nonnegative --manual-seed OR --model-seeds")
    if a.quick and a.profile=="paper":
        p.error("A quick smoke cannot use the paper profile")
    if a.threads<1 or a.host_budget_mb<0 or a.ssd_budget_gb<=0:
        p.error("Invalid resource budget")
    if a.search_jobs_per_method is not None and a.search_jobs_per_method<0:
        p.error("Search limit must be nonnegative")
    if any(v is not None and v<1 for v in (a.max_epochs,a.baseline_max_epochs,a.steps_per_epoch)):
        p.error("Training budgets must be positive")
    try:
        commands=build_commands(a)
    except ValueError as exc:
        p.error(str(exc))
    # Different protocols/configurations must have different study roots.
    if not a.dry_run:
        a.output.mkdir(parents=True,exist_ok=True)
    for command in commands:
        print(subprocess.list2cmdline(command) if sys.platform=="win32" else shlex.join(command),flush=True)
        if a.dry_run:
            continue
        start=time.time();result=subprocess.run(command,cwd=ROOT)
        event=dict(stage=a.stage,profile=a.profile,arguments=command,
                   start_unix=start,end_unix=time.time(),exit_code=result.returncode)
        with (a.output/"V7_RECIPE_RUNS.jsonl").open("a",encoding="utf8") as handle:
            handle.write(json.dumps(event,ensure_ascii=False)+"\n")
        if result.returncode:
            return result.returncode
    if not a.dry_run:
        datasets=DATASETS if a.datasets=="all" else tuple(a.datasets.split(","))
        index={ds:{name:str(a.output/ds/name) for name in
                   ("search/best_configurations.json","final_locked_test_summary.csv",
                    "publication/research_tables.xlsx","figures","final_selection_provenance.json")}
               for ds in datasets}
        (a.output/"V7_SUITE_INDEX.json").write_text(json.dumps(index,indent=2),encoding="utf8")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
