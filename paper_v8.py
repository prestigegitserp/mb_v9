"""Explicit TRAIN/VAL search -> frozen TEST -> manuscript analyses for V8.

The initial recipe is saved. Later stages need only --stage and --output;
omitted settings are restored, and explicit scientific changes are rejected.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
DATASETS=('synthetic','ml-100k','ml-1m','ml-10m','jester','bookcrossing')
STAGES=('plan','smoke','pilot','develop','scan','final','test','lossstudy','ablation',
        'ablation-fixed','kstudy','sparsity','density','convergence','calibration',
        'temporal','report','curves','audit','status','all')
DEFAULTS=dict(profile='common',datasets='ml-100k',data_root='./datasets',device='auto',
    threads=4,storage='auto',host_budget_mb=512,ssd_budget_gb=8.,cache_dir='./v8_ssd_cache',
    quick=False,no_plots=False,epoch_train_rmse=False,deterministic=False,
    include_temporal=False,budget='standard')
FLEXIBLE={'stage','dry_run','resume','loss_methods','loss_parameters','no_plots',
          'threads','storage','host_budget_mb','ssd_budget_gb','cache_dir','device',
          'epoch_train_rmse','output','include_temporal'}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=STAGES,default='plan')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=('common','development','paper'))
    p.add_argument('--datasets');p.add_argument('--data-root')
    p.add_argument('--device',choices=('auto','cpu','cuda','cuda:0'))
    p.add_argument('--split-seeds');p.add_argument('--model-seeds');p.add_argument('--k')
    p.add_argument('--manual-seed',type=int);p.add_argument('--methods');p.add_argument('--blend-methods')
    p.add_argument('--search-jobs-per-method',type=int)
    p.add_argument('--max-epochs',type=int);p.add_argument('--baseline-max-epochs',type=int)
    p.add_argument('--steps-per-epoch',type=int)
    p.add_argument('--budget',choices=('light','standard','full'))
    p.add_argument('--threads',type=int);p.add_argument('--host-budget-mb',type=int)
    p.add_argument('--ssd-budget-gb',type=float);p.add_argument('--cache-dir')
    p.add_argument('--storage',choices=('auto','resident','host','ssd'))
    for name in ('quick','deterministic','no-plots','include-temporal'):
        p.add_argument('--'+name,action='store_true',default=None)
    p.add_argument('--plots',dest='no_plots',action='store_false',default=None,
                   help='Render figures, including after a --no-plots training run')
    p.add_argument('--epoch-train-rmse',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--loss-methods',help='Optional comma-separated subset for lossstudy only')
    p.add_argument('--loss-parameters',help='Optional comma-separated active coefficients for lossstudy only')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true',help='Accepted explicitly; training stages always resume safely')
    return p


def command_text(command):
    """Print a literal PowerShell command on Windows, a shell command elsewhere."""
    if sys.platform=='win32':
        # The call operator is required for a quoted executable path. Single
        # quotes also keep $, backticks and apostrophes in paths literal.
        return '& '+' '.join("'"+str(arg).replace("'","''")+"'" for arg in command)
    return shlex.join(command)


def resolve_recipe(a):
    given=vars(a).copy();root=a.output.resolve()
    if given.get('data_root') is not None:
        given['data_root']=str(Path(given['data_root']).resolve())
    path=root/'V8_RECIPE.json'
    saved=json.loads(path.read_text())['settings'] if path.is_file() else {}
    for key,value in given.items():
        if key not in FLEXIBLE and value is not None and key in saved and value!=saved[key]:
            raise ValueError(f'Existing study has {key}={saved[key]!r}; choose a fresh --output for {value!r}')
    values={**saved,**{k:v for k,v in given.items() if v is not None}}
    for key,value in given.items():values.setdefault(key,value)
    for key,value in DEFAULTS.items():
        if values.get(key) is None:values[key]=value
    if a.stage=='smoke':values['quick']=True
    if saved and bool(values['quick'])!=bool(saved.get('quick',False)):
        raise ValueError('Smoke and research runs require a fresh --output; the saved study cannot change quick mode')
    profile=values['profile']
    values['split_seeds']=values.get('split_seeds') or (','.join(map(str,range(4101,4121))) if profile=='paper' else '2026')
    if values.get('manual_seed') is not None:
        if given.get('model_seeds') is not None:raise ValueError('Choose --manual-seed OR --model-seeds')
        values['model_seeds']=str(values['manual_seed'])
    else:values['model_seeds']=values.get('model_seeds') or ('11,22,33' if profile=='paper' else '1')
    values['k']=values.get('k') or ('10,20,50' if profile=='paper' else '20')
    if values['budget']=='light':
        for key,value in dict(max_epochs=120,steps_per_epoch=2,baseline_max_epochs=200,search_jobs_per_method=6).items():
            if values.get(key) is None:values[key]=value
    if values['budget']=='full' and values.get('search_jobs_per_method') is None:
        values['search_jobs_per_method']=0
    if values['threads']<1 or values['host_budget_mb']<0 or values['ssd_budget_gb']<=0:
        raise ValueError('Invalid resource budget')
    for key in ('max_epochs','baseline_max_epochs','steps_per_epoch'):
        if values.get(key) is not None and values[key]<1:raise ValueError(key+' must be positive')
    if values.get('search_jobs_per_method') is not None and values['search_jobs_per_method']<0:
        raise ValueError('Search budget must be nonnegative')
    if values['quick'] and profile=='paper':raise ValueError('Smoke cannot use the paper profile')
    if values.get('manual_seed') is not None and values['manual_seed']<0:raise ValueError('Seed must be nonnegative')
    values['data_root']=str(Path(values['data_root']).resolve())
    values['cache_dir']=str(Path(values['cache_dir']).resolve())
    values['output']=root
    resolved=argparse.Namespace(**values)
    return resolved,path


def build_commands(a):
    datasets=DATASETS if a.datasets=='all' else tuple(a.datasets.split(','))
    if not datasets or set(datasets)-set(DATASETS):raise ValueError('Unknown datasets')
    commands=[]
    stages=(['scan','final','lossstudy','ablation','ablation-fixed','kstudy','sparsity',
             'convergence','calibration','report','audit'] if a.stage=='all' else [a.stage])
    for dataset in datasets:
        common=['--datasets',dataset,'--data-root',a.data_root,'--output',str(a.output),
            '--split-seeds',a.split_seeds,'--k',a.k,'--scan-mode','hybrid',
            '--analysis-source','winners','--v8-config',str(ROOT/'examples/v8'/a.profile/(dataset+'.json')),
            '--v8-device',a.device,'--v8-threads',str(a.threads),'--v8-storage',a.storage,
            '--v8-host-budget-mb',str(a.host_budget_mb),'--v8-ssd-budget-gb',str(a.ssd_budget_gb),
            '--v8-cache-dir',a.cache_dir,'--v8-retain-candidates','metrics']
        common+=['--manual-seed',str(a.manual_seed)] if a.manual_seed is not None else ['--model-seeds',a.model_seeds]
        if a.methods:common+=['--methods',a.methods]
        if a.blend_methods:common+=['--include-blend','--blend-methods',a.blend_methods]
        if a.deterministic:common+=['--v8-deterministic']
        if a.no_plots:common+=['--v8-no-plots']
        common+=['--epoch-train-rmse' if a.epoch_train_rmse else '--no-epoch-train-rmse']
        if a.quick:common+=['--quick','--v8-search-jobs-per-method','1']
        elif a.search_jobs_per_method is not None:common+=['--v8-search-jobs-per-method',str(a.search_jobs_per_method)]
        for key in ('max_epochs','baseline_max_epochs','steps_per_epoch'):
            if getattr(a,key) is not None:common+=['--v8-'+key.replace('_','-'),str(getattr(a,key))]
        if a.loss_methods:common+=['--loss-methods',a.loss_methods]
        if a.loss_parameters:common+=['--loss-parameters',a.loss_parameters]
        requested=list(stages)
        if a.stage=='all' and dataset=='synthetic':requested.insert(-2,'density')
        if a.stage=='all' and a.include_temporal and dataset.startswith('ml-'):requested.insert(-2,'temporal')
        for stage in requested:
            command={'pilot':'main','smoke':'main','develop':'scan','test':'final'}.get(stage,stage)
            extra=['--scan-only'] if stage=='scan' else []
            if command not in ('plan','report','curves','audit'):extra+=['--resume']
            commands.append([sys.executable,str(ROOT/'run_v8.py'),command,*common,*extra])
    return commands


def main(argv=None):
    p=parser();given=p.parse_args(argv)
    try:
        a,recipe_path=resolve_recipe(given)
        if a.stage=='status':
            from v8.readiness import build_checklist
            datasets=DATASETS if a.datasets=='all' else a.datasets.split(',')
            for ds in datasets:
                status=build_checklist(a.output/ds)
                print(ds,json.dumps(status,ensure_ascii=False,indent=2))
            return 0
        commands=build_commands(a)
    except (ValueError,KeyError) as exc:p.error(str(exc))
    if not a.dry_run:
        a.output.mkdir(parents=True,exist_ok=True)
        if not recipe_path.exists():
            settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in FLEXIBLE}
            # Keep placement settings as defaults, while allowing explicit changes.
            settings.update({k:getattr(a,k) for k in ('data_root','cache_dir','device','threads','storage','host_budget_mb','ssd_budget_gb','no_plots','epoch_train_rmse')})
            recipe_path.write_text(json.dumps(dict(version=8,settings=settings),indent=2)+'\n')
    for command in commands:
        print(command_text(command),flush=True)
        if a.dry_run:continue
        started=time.time();result=subprocess.run(command,cwd=ROOT)
        with (a.output/'V8_RECIPE_RUNS.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(dict(stage=a.stage,command=command,start=started,end=time.time(),exit_code=result.returncode))+'\n')
        if result.returncode:
            print('[INCOMPLETE] Stage failed; existing completed jobs remain resumable.',flush=True)
            return result.returncode
    if not a.dry_run:
        from v8.readiness import build_checklist
        datasets=DATASETS if a.datasets=='all' else a.datasets.split(',')
        for dataset in datasets:build_checklist(a.output/dataset)
        if a.stage=='scan':
            print('[VAL ONLY] TEST has not run. Next: '+command_text([sys.executable,'paper_v8.py','--stage','test','--output',str(a.output)]))
        elif a.stage in ('develop','final','test','pilot','smoke','all'):
            print('[TEST OUTPUTS] See <output>/<dataset>/MANUSCRIPT_CHECKLIST.md and publication/figures/selected_test_rmse.png')
    return 0


if __name__=='__main__':raise SystemExit(main())
