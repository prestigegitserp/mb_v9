"""Native validation-only V3/V4 diagnostic; intentionally bypasses test reporting.

Example: python experiments/v4_controlled_pilot.py --assignment results/ml-100k/main/split_assignments_2026.csv --version 4 --output pilot_v4
A completed result cannot be overwritten. Test labels are poisoned before fitting.
"""
import argparse,hashlib,importlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assignment',type=Path,required=True)
    p.add_argument('--version',choices=['3','4'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--methods',default=','.join('Method'+str(i) for i in range(1,8)))
    p.add_argument('--k',type=int,default=20);p.add_argument('--model-seed',type=int,default=1)
    p.add_argument('--split-seed',type=int,default=2026);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    torch.set_num_threads(a.threads)
    from v9.engine import load_v8
    _,pub=load_v8();torch.set_num_threads(a.threads)
    version='v'+a.version
    conf,models,protocol,training=[importlib.import_module(version+'.'+part) for part in ('config','models','protocol','training')]
    frame=pd.read_csv(a.assignment)
    data,_=pub.build_publication_matrix_data_v8(frame,pub.DATASET_SPECS['ml-100k'],pub.PublicationDataConfig(dataset='ml-100k',force_backend='dense'),torch.device('cpu'))
    if 'timestamp' in frame:data.train_timestamps=torch.tensor(frame.loc[frame.split=='train','timestamp'].to_numpy(),dtype=torch.float64)
    data.test_ratings_raw[:]=torch.nan
    study=conf.StudyConfig();data=protocol.validation_roles(data,study,a.split_seed)
    names=a.methods.split(',')
    if set(names)-set(study.models):p.error('Unknown method')
    hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(models.__file__).parent.glob('*.py')}
    a.output.mkdir(parents=True,exist_ok=True)
    for method in names:
        target=a.output/(version+'_'+method+'.json')
        if target.exists():raise ValueError('Refusing to overwrite '+str(target))
        from dataclasses import replace
        cfg=replace(study.models[method],latent_dim=a.k);model=models.instantiate(data,cfg,seed=a.model_seed)
        def evaluate(current,data,partition,**kwargs):
            if partition!='val':raise AssertionError('Validation-only pilot')
            with torch.no_grad():
                current.eval();pred=current.predict_raw_pairs(data,data.val_users,data.val_items)
                return {'RMSE':float((pred-data.val_ratings_raw).square().mean().sqrt())}
        evaluator=SimpleNamespace(MetricConfig=lambda **kw:SimpleNamespace(**kw),evaluate_publication_model=evaluate)
        start=time.perf_counter();info=getattr(training,'train_'+version)(evaluator,model,data,cfg,None)
        with torch.no_grad():pred=model.predict_raw_pairs(data,data.val_users,data.val_items)
        if hashes!={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(models.__file__).parent.glob('*.py')}:raise RuntimeError('Source changed during pilot')
        payload=dict(version=version,method=method,config=vars(cfg),info=info,validation_RMSE=float((pred-data.val_ratings_raw).square().mean().sqrt()),parameter_count=sum(v.numel() for v in model.parameters()),wall_seconds=time.perf_counter()-start,test_labels_poisoned=True,source_file_sha256=hashes,data_assignment_sha256=hashlib.sha256(a.assignment.read_bytes()).hexdigest())
        target.write_text(json.dumps(payload,indent=2));torch.save(model.state_dict(),target.with_suffix('.pt'))
        np.savez_compressed(a.output/(version+'_'+method+'__val.npz'),pair_id=(data.val_users*data.num_items+data.val_items).numpy(),truth=data.val_ratings_raw.numpy(),prediction=pred.numpy())
        print(method,payload['validation_RMSE'],flush=True)


if __name__=='__main__':main()
