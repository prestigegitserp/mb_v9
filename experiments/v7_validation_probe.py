"""Bounded, reproducible ML-100K validation probes; never evaluates TEST.

Compare within-method variants at the same epoch/step/width caps. Baselines
have their own recorded optimizer and epoch budgets, so this is not an equal
compute leaderboard. The full paper recipe is paper_v7.py, not this probe.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict,fields,replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--steps',type=int,default=2)
    parser.add_argument('--hidden',type=int,default=64)
    parser.add_argument('--k',type=int,default=20)
    parser.add_argument('--split-seed',type=int,default=2026)
    parser.add_argument('--model-seed',type=int,default=1)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--families',default='reference,prior,gated')
    parser.add_argument('--methods',default=','.join('Method'+str(i) for i in range(1,8)))
    parser.add_argument('--baselines',action='store_true')
    parser.add_argument('--method8',action='store_true')
    parser.add_argument('--baseline-epochs',type=int,default=100)
    parser.add_argument('--resume',action='store_true')
    a=parser.parse_args(argv)
    for key in ('epochs','steps','hidden','k','threads','baseline_epochs'):
        if getattr(a,key)<1:parser.error(key+' must be positive')
    families=a.families.split(',');methods=a.methods.split(',')
    if set(families)-{'reference','prior','gated'}:parser.error('Unknown family')
    import pandas as pd
    import torch
    from v7 import engine
    from v7.config import ModelConfig,StudyConfig,implementation_signature
    from v7.execution import ExecutionConfig
    from v7.runtime import installed
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    a.data_root=a.data_root.resolve()
    torch.set_num_threads(a.threads)
    opts=SimpleNamespace(study=StudyConfig(profile='controlled-validation-probe'),
        execution=ExecutionConfig(device=a.device).validate(),max_epochs=a.epochs,
        baseline_max_epochs=a.baseline_epochs,steps_per_epoch=a.steps,no_plots=True,
        retain_candidates='metrics',v7_recovery_interval=5,v7_isolate_jobs=True,
        threads=a.threads,v7_deterministic=False)
    args=SimpleNamespace(data_root=str(a.data_root),backend='sparse',
        split_seeds=str(a.split_seed),model_seeds=str(a.model_seed),k=str(a.k),
        methods=None,eval_every=5,quick=False,resume=a.resume,epoch_train_rmse=False,output=str(root))
    prior=json.loads((ROOT/'examples/v7/validated_candidates.json').read_text())
    allowed={f.name for f in fields(ModelConfig)}
    rows=[];configs={};started=time.perf_counter()
    declaration=dict(purpose='development validation only; not TEST or paper reproduction',
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='resume'},
        source_sha256=implementation_signature(),test_evaluated=False,
        budgets='same caps within Methods1-7; architecture and early stopping can change actual compute',
        variants='gated starts from the same prior config and changes history_signal only')
    declaration_file=root/'PROBE_PROTOCOL.json'
    if declaration_file.is_file() and json.loads(declaration_file.read_text())!=declaration:
        raise ValueError('Changed source, config or cohort; choose a fresh --output')
    engine.json_write(declaration_file,declaration)
    with installed(engine,opts) as (core,pub):
        base=engine.make_base(pub,args,'ml-100k')
        jobs=[]
        for family in families:
            for method in methods:
                cfg=copy.deepcopy(base);cfg.methods=(method,)
                model=cfg.v7.models[method]
                if family!='reference':
                    model=ModelConfig(**{k:v for k,v in prior[method]['config'].items() if k in allowed})
                model=replace(model,max_epochs=a.epochs,steps_per_epoch=a.steps,hidden_dim=a.hidden,
                    eval_every=5,router_expert_epochs=a.epochs,router_epochs=a.epochs,
                    shape_epochs=a.epochs,probability_fit=base.v7.models[method].probability_fit)
                if family=='gated':model=replace(model,history_signal='gated')
                cfg.v7.models[method]=model.validate();jobs.append((family,method,cfg))
        if a.baselines:
            for method in ('AutoRec','AEMC-NE'):
                cfg=copy.deepcopy(base);cfg.methods=(method,);jobs.append(('baseline_control',method,cfg))
        if a.method8:
            cfg=copy.deepcopy(base);cfg.methods=('Method8',);jobs.append(('original_Method10','Method8',cfg))
        for family,method,cfg in jobs:
            tag=family+'__'+method;folder=root/'ml-100k/validation_probes/jobs'/tag
            configs[tag]=asdict(cfg)
            table=engine.run_job(pub,core,cfg,folder,args=args,validation_only=True,
                extra=dict(type='hyper',probe_family=family,probe_tag=tag))
            if any(key.startswith('test_') for key in table):raise AssertionError('TEST evaluation in probe')
            for _,row in table.iterrows():
                values=row.to_dict();values.update(probe_family=family,probe_tag=tag)
                rows.append(values)
            pd.DataFrame(rows).to_csv(root/'VALIDATION_PROBES.csv',index=False)
            engine.json_write(root/'PROBE_CONFIGURATIONS.json',configs)
            print('[PROBE]',tag,'val_RMSE',table.val_RMSE.min(),flush=True)
    engine.json_write(root/'PROBE_COMPLETION.json',dict(completed=True,elapsed_seconds=time.perf_counter()-started,
        candidate_trainings=len(rows),test_evaluated=False,source_sha256=declaration['source_sha256']))


if __name__=='__main__':main()
