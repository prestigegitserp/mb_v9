"""Matched-capacity V7/V8 development probes; TEST labels are poisoned.

This bounded point-prediction check is not a manuscript benchmark. It records
all seven results, including regressions, without selecting a model on TEST.
"""
from pathlib import Path
from dataclasses import asdict,replace
from types import SimpleNamespace
import argparse
import copy
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--k',type=int,default=20)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--split-seed',type=int,default=2026)
    p.add_argument('--model-seed',type=int,default=1)
    p.add_argument('--methods',default=','.join(f'Method{i}' for i in range(1,8)))
    p.add_argument('--variants',default='v7_reference,v8_fixed_control,v8_updated')
    a=p.parse_args(argv)
    import numpy as np
    import pandas as pd
    import torch
    from v8.engine import load_v8
    from v7.config import default_models as defaults7,StudyConfig as Study7
    from v8.config import default_models as defaults8,StudyConfig as Study8,implementation_signature
    from v7.protocol import validation_roles as roles7,data_signature as signature7
    from v8.protocol import validation_roles as roles8,data_signature as signature8
    from v7.models import instantiate as instantiate7
    from v8.models import instantiate as instantiate8
    from v7.training import train_v7
    from v8.training import train_v8
    if min(a.epochs,a.steps,a.hidden,a.k,a.threads)<1:p.error('Budgets must be positive')
    torch.set_num_threads(a.threads)
    core,pub=load_v8()
    dc=pub.PublicationDataConfig(dataset='ml-100k',data_root=str(a.data_root.resolve()),split_seed=a.split_seed,
                                 force_backend='sparse')
    raw,spec=pub.load_publication_dataset(dc)
    split=pub.make_core_split(raw,spec,dc)
    initial,_=pub.build_publication_matrix_data_v8(split,spec,dc,torch.device('cpu'))
    d7=roles7(initial,Study7(),a.split_seed)
    d8=roles8(initial,Study8(),a.split_seed)
    assert signature7(d7)==signature8(d8)
    # Fail loudly if any training/post-fit code attempts to use TEST labels.
    d7.test_ratings_raw=torch.full_like(d7.test_ratings_raw,float('nan'))
    d8.test_ratings_raw=torch.full_like(d8.test_ratings_raw,float('nan'))
    def evaluate(model,data,part,**kw):
        if part=='test':raise AssertionError('TEST must never be evaluated in this probe')
        with torch.no_grad():
            model.eval()
            pred=model.predict_raw_pairs(data,getattr(data,part+'_users'),getattr(data,part+'_items'))
            return dict(RMSE=float((pred-getattr(data,part+'_ratings_raw')).square().mean().sqrt()))
    training_pub=SimpleNamespace(MetricConfig=lambda **kw:SimpleNamespace(**kw),evaluate_publication_model=evaluate)
    a.output.mkdir(parents=True,exist_ok=True)
    declaration=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        source_sha256=implementation_signature(),test_evaluated=False,test_labels_poisoned=True,
        data_signature=signature8(d8),purpose='bounded development validation; point-only Method7',
        comparison='V7 default versus V8 default and V8 fixed-coupling control at matching caps; not a single-factor cross-version ablation',
        source_file_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                            for f in sorted((ROOT/'v8').glob('*.py'))})
    (a.output/'PROBE_PROTOCOL.json').write_text(json.dumps(declaration,indent=2)+'\n')
    all_configs={};rows=[]
    for method in a.methods.split(','):
        for variant in a.variants.split(','):
            old=variant=='v7_reference'
            cfg=(defaults7() if old else defaults8())[method]
            cfg=replace(cfg,latent_dim=a.k,hidden_dim=a.hidden,max_epochs=a.epochs,
                steps_per_epoch=a.steps,eval_every=5,val_patience=16,
                router_expert_epochs=a.epochs,router_epochs=a.epochs,
                posterior_probes=2,posterior_cg_steps=16,graph_probes=4,cg_steps=16,
                probability_fit=False)
            if variant=='v8_fixed_control':cfg=replace(cfg,adaptive_readout=False,calibrate_geometry=False,continuous_response=False)
            elif variant not in ('v7_reference','v8_updated'):raise ValueError('Unknown variant '+variant)
            data=copy.copy(d7 if old else d8)
            model=(instantiate7 if old else instantiate8)(data,cfg,a.model_seed)
            started=time.perf_counter()
            result=(train_v7 if old else train_v8)(training_pub,model,data,cfg,None)
            val=evaluate(model,data,'val')['RMSE']
            response=result.get('v7_response',{}) if old else result.get('v8_response',{})
            rows.append(dict(method=method,variant=variant,val_RMSE=val,
                best_epoch=result['best_epoch'],best_raw_val_RMSE=result['best_val_RMSE'],
                source_lineage=cfg.lineage,hidden_dim=a.hidden,max_epochs=a.epochs,steps_per_epoch=a.steps,
                selected_response_scale=response.get('selected_scale',1.),
                selected_geometry_scale=response.get('selected_geometry_scale',1.),
                seconds=time.perf_counter()-started,test_evaluated=False))
            all_configs[method+'__'+variant]=asdict(cfg)
            pd.DataFrame(rows).to_csv(a.output/'VALIDATION_COMPARISON.csv',index=False)
            (a.output/'PROBE_CONFIGURATIONS.json').write_text(json.dumps(all_configs,indent=2)+'\n')
            print(method,variant,'VAL',val,flush=True)
            del model
    (a.output/'PROBE_COMPLETION.json').write_text(json.dumps(dict(completed=True,trainings=len(rows),test_evaluated=False),indent=2)+'\n')


if __name__=='__main__':main()
