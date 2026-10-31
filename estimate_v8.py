"""Count actual V8 plans and give explicitly uncertain wall-time budgets."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from v8.config import StudyConfig
from v8.execution import ExecutionConfig
from v8.runtime import installed
from v8.worker import hydrate
from v8 import engine,search
from v8.losses import coefficients,trial_values
from v9.search import candidate_count


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('synthetic','ml-100k','ml-1m','ml-10m','jester','bookcrossing'),default='ml-100k')
    parser.add_argument('--profile',choices=['common','development','paper'],default='common')
    parser.add_argument('--data-root',default='./datasets')
    parser.add_argument('--split-seeds');parser.add_argument('--model-seeds');parser.add_argument('--k')
    parser.add_argument('--calibration',type=Path,help='candidate_validation_runs.csv measured on your own hardware')
    parser.add_argument('--search-jobs-per-method',type=int)
    parser.add_argument('--max-epochs',type=int)
    parser.add_argument('--baseline-max-epochs',type=int)
    parser.add_argument('--steps-per-epoch',type=int)
    parser.add_argument('--methods')
    parser.add_argument('--output',type=Path)
    parser.add_argument("--include-temporal",action="store_true")
    a=parser.parse_args(argv)
    if any(value is not None and value<1 for value in (a.max_epochs,a.baseline_max_epochs,a.steps_per_epoch)):
        parser.error('Training caps must be positive')
    if a.search_jobs_per_method is not None and a.search_jobs_per_method<0:
        parser.error('Search limit must be nonnegative')
    if a.dataset!="ml-100k" and not a.calibration:parser.error("Other datasets require their own measured --calibration CSV")
    root=Path(__file__).resolve().parent
    payload=json.loads((root/'examples/v8'/a.profile/(a.dataset+'.json')).read_text())
    template=asdict(StudyConfig())
    for key,value in payload.items():
        if key=='models':
            for method,overrides in value.items():template['models'][method].update(overrides)
        else:template[key]=value
    study=hydrate(StudyConfig(),template).validate()
    if a.search_jobs_per_method is not None:study.search_jobs_per_method=a.search_jobs_per_method
    paper=a.profile=='paper'
    args=SimpleNamespace(data_root=a.data_root,backend='auto',split_seeds=a.split_seeds or (','.join(map(str,range(4101,4121))) if paper else '2026'),
         model_seeds=a.model_seeds or ('11,22,33' if paper else '1'),k=a.k or ('10,20,50' if paper else '20'),
         methods=a.methods,eval_every=None,quick=False,resume=True)
    options=SimpleNamespace(study=study,execution=ExecutionConfig(),max_epochs=a.max_epochs,baseline_max_epochs=a.baseline_max_epochs,
         steps_per_epoch=a.steps_per_epoch,no_plots=True,retain_candidates='metrics',v8_recovery_interval=5,v8_isolate_jobs=False)
    # The audited timing reference belongs to the frozen V7 evidence tree.
    calibration=a.calibration or root/'docs/v7_evidence/V6_PILOT_CANDIDATES.csv'
    timings=pd.read_csv(calibration).groupby('method').train_wall_sec.median().to_dict()
    timings['Method8']=timings.get('Method8',40.)
    def seconds(config):
        total=0.
        for method in config.methods:
            single=copy.deepcopy(config);single.methods=(method,)
            total+=candidate_count(single)*timings.get(method,40.)
        return total
    with installed(engine,options) as (_,pub):
        base=engine.make_base(pub,args,a.dataset)
        jobs=search.build_jobs(pub,base,engine.numbers(args.k))
        secondary=copy.deepcopy(base);secondary.core_config.latent_dims=base.core_config.latent_dims[:1]
        variants=search.ablations(secondary)
        ns=len(base.core_config.split_seeds);nm=len(base.core_config.model_seeds)
        # Final consumes ONE selected K/init per method and outer split.
        final_count=len(base.methods)*ns*nm
        final_time=sum(timings.get(method,40.) for method in base.methods)*ns*nm
        loss_count=0;loss_time=0.
        for method,cfg in base.v8.models.items():
            if method not in base.methods:continue
            count=sum(len(trial_values(cfg,name)) for name in coefficients(cfg))*ns*nm
            loss_count+=count;loss_time+=count*timings.get(method,40.)
        levels=[('main',candidate_count(base),seconds(base)),
                ('hyperparameter_scan',sum(candidate_count(c) for c in jobs.values()),sum(seconds(c) for c in jobs.values())),
                ('final',final_count,final_time),
                ('ablation_defaults_plan',sum(candidate_count(c) for _,c in variants),sum(seconds(c) for _,c in variants)),
                ('loss_sensitivity_defaults_plan',loss_count,loss_time)]
        k_methods=[m for m in base.methods if m not in ('AutoRec','AEMC-NE')]
        k_count=len(k_methods)*ns*nm*len(base.core_config.latent_dims)
        k_time=sum(timings.get(m,40.) for m in k_methods)*ns*nm*len(base.core_config.latent_dims)
        ablation=levels[3]
        calibration_count=ns*nm if 'Method7' in base.methods and base.v8.models['Method7'].probability_fit else 0
        levels += [('ablation_fixed_defaults_plan',ablation[1],ablation[2]),
                   ('kstudy',k_count,k_time),('sparsity_four_rates',final_count*4,final_time*4),
                   ('convergence',final_count,final_time),
                   ('calibration',calibration_count,calibration_count*timings.get('Method7',40.))]
        if a.include_temporal:
            levels += [('temporal_scan_and_final',levels[1][1]+final_count,levels[1][2]+final_time)]
        suite=levels[1:]
        levels += [('all_requested_stages_estimate',sum(row[1] for row in suite),sum(row[2] for row in suite))]
    rows=[dict(stage=name,candidate_trainings=count,reference_hours=value/3600,
               low_hours=value*.5/3600,high_hours=value*4/3600) for name,count,value in levels]
    report=dict(target='User hardware; supply a same-device pilot CSV for a relevant reference',dataset=a.dataset,profile=a.profile,
         calibration_file=str(calibration),user_supplied_calibration=bool(a.calibration),
         measured_on_target_hardware=None if a.calibration else False,
         reference_hardware='not verified from CSV alone' if a.calibration else 'NVIDIA GeForce RTX 5090; supplied V6 pilot device logs',
         requested_training_caps=dict(max_epochs=a.max_epochs,baseline_max_epochs=a.baseline_max_epochs,steps_per_epoch=a.steps_per_epoch),
         timing_scope='Measured median per-method time times actual outer training count; caps/architecture/early stopping are NOT silently extrapolated linearly',
         limitations=['Planning heuristic, not a measured benchmark of the target hardware',
         '0.5x to 4x is an assumption range, not a statistical confidence interval',
         'Different epochs, early stopping, SSD and CPU fallback can exceed the range',
         'Method8 uses a 40-second placeholder unless measured in calibration CSV',
         'Default timing reference is the supplied RTX 5090 pilot; a supplied CSV alone does not verify hardware provenance',
         'Winner-based ablation/loss counts change with selected architecture; defaults shown',
         'All estimate includes scan, final, both ablations, loss, K, four sparsity rates, convergence and eligible calibration; temporal only if requested'],
         stages=rows)
    print(pd.DataFrame(rows).to_string(index=False))
    print('\n'+json.dumps({k:v for k,v in report.items() if k!='stages'},indent=2))
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':main()
