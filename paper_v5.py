"""Small cross-platform launcher for the complete, validation-locked V5 workflow.

Examples:
  python paper_v5.py --phase plan --dataset ml-100k --output ./paper_v5
  python paper_v5.py --phase scan --dataset ml-100k --device cuda:0 --output ./paper_v5
  python paper_v5.py --phase all --dataset ml-100k --device cuda:0 --output ./paper_v5
"""
import argparse,json,shlex,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--phase',choices=['plan','smoke','scan','final','all','ablation','ablation-fixed','kstudy','sparsity','convergence','calibration','report','audit'],default='plan')
 p.add_argument('--dataset',choices=['synthetic','ml-100k','ml-1m','ml-10m','jester','bookcrossing'],default='ml-100k')
 p.add_argument('--data-root',type=Path,default=Path('./datasets'));p.add_argument('--output',type=Path,required=True)
 p.add_argument('--device',choices=['auto','cpu','cuda:0'],default='auto');p.add_argument('--config',type=Path)
 p.add_argument('--split-seeds',default='3031,3032,3033');p.add_argument('--model-seeds',default='11,22,33');p.add_argument('--k',default='10,20,50')
 p.add_argument('--scan-mode',choices=['one-factor','hybrid','joint'],default='hybrid');p.add_argument('--threads',type=int,default=4)
 p.add_argument('--methods');p.add_argument('--protocol',choices=['common','aemc-paper','autorec-paper'],default='common')
 p.add_argument('--search-jobs-per-method',type=int,default=0)
 p.add_argument('--host-budget-mb',type=int,default=1024);p.add_argument('--cache-dir',type=Path,default=Path('./v5_ssd_cache'))
 p.add_argument('--no-plots',action='store_true');p.add_argument('--dry-run',action='store_true')
 a=p.parse_args()
 if a.threads<1 or a.host_budget_mb<0 or a.search_jobs_per_method<0:p.error('Invalid thread, search or host memory budget')
 output=a.output.resolve()/('smoke' if a.phase=='smoke' else '')
 config=a.config.resolve() if a.config else (ROOT/'examples/v5/recommended_v5_ml100k.json' if a.dataset=='ml-100k' else None)
 if config is not None and not config.is_file():p.error('Configuration file not found')
 command='scan' if a.phase in ('scan','smoke') else a.phase
 args=[sys.executable,str(ROOT/'run_v5.py'),command,'--datasets',a.dataset,'--data-root',str(a.data_root.resolve()),'--output',str(output),
       '--split-seeds',a.split_seeds,'--model-seeds',a.model_seeds,'--k',a.k,'--scan-mode',a.scan_mode,'--analysis-source','winners',
       '--v5-threads',str(a.threads),'--v5-retain-candidates','metrics']
 args+=['--v5-protocol',a.protocol]
 if a.methods:args+=['--methods',a.methods]
 if a.search_jobs_per_method:args+=['--v5-search-jobs-per-method',str(a.search_jobs_per_method)]
 if a.phase not in ('plan','report','audit'):
  args+=['--v5-device',a.device,'--v5-storage','auto','--v5-host-budget-mb',str(a.host_budget_mb),'--v5-cache-dir',str(a.cache_dir.resolve())]
 if config is not None:args+=['--v5-config',str(config)]
 declared=set(a.methods.split(',')) if a.methods else None
 if a.dataset.startswith('ml-') and a.protocol=='common' and (declared is None or {'AutoRec','Method4'}<=declared):args+=['--include-blend']
 if a.phase=='all':args+=['--include-convergence']
 if a.phase=='scan':args+=['--scan-only','--resume']
 elif a.phase=='smoke':args+=['--quick','--v5-search-jobs-per-method','1']
 elif a.phase not in ('plan','report','audit'):args+=['--resume']
 if a.no_plots:args+=['--v5-no-plots']
 print(subprocess.list2cmdline(args) if sys.platform=='win32' else shlex.join(args),flush=True)
 if a.dry_run:return
 output.mkdir(parents=True,exist_ok=True)
 start=time.time();completed=subprocess.run(args,cwd=ROOT)
 event=dict(phase=a.phase,arguments=args,start_unix=start,end_unix=time.time(),exit_code=completed.returncode)
 with (output/'V5_RECIPE_RUNS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
 raise SystemExit(completed.returncode)
if __name__=='__main__':main()
