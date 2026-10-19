"""Isolated M4 geometry candidate; no production source is modified.

Projection uses the same mean/dispersion coordinates as the training loss.
The prototype explicitly patches one operator only while fitting M4 and saves
a prototype checkpoint, not a checkpoint advertised as an unchanged V4 model.
"""
from dataclasses import asdict, replace
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch


def metric_projection(point, base, lo, hi, step, dispersion_weight):
    """Argmin over the exact polygon of (delta P-delta N)^2+gamma(delta P+delta N)^2."""
    if not 0 < dispersion_weight <= 1:
        raise ValueError('dispersion_weight must be in (0,1]')
    zero=torch.zeros_like(base)
    if step is None:
        lower=upper=base
    else:
        scaled=(base-lo)/float(step)
        lower=(lo+scaled.floor()*step).clamp(lo,hi)
        upper=(lo+scaled.ceil()*step).clamp(lo,hi)
    vertices=torch.stack([torch.stack([zero,base-lo],-1),torch.stack([hi-base,zero],-1),
                          torch.stack([(upper-base).clamp_min(0),zero],-1),
                          torch.stack([zero,(base-lower).clamp_min(0)],-1)],-2)
    edge=vertices.roll(-1,dims=-2)-vertices
    rel=point.unsqueeze(-2)-vertices
    cross=edge[...,0]*rel[...,1]-edge[...,1]*rel[...,0]
    inside=(cross<=1e-6).all(-1)

    def inner(a,b):
        return (a[...,0]-a[...,1])*(b[...,0]-b[...,1])+dispersion_weight*(a[...,0]+a[...,1])*(b[...,0]+b[...,1])

    position=(inner(rel,edge)/inner(edge,edge).clamp_min(1e-12)).clamp(0,1)
    candidates=vertices+position.unsqueeze(-1)*edge
    diff=candidates-point.unsqueeze(-2)
    index=inner(diff,diff).argmin(-1)
    nearest=candidates.gather(-2,index[...,None,None].expand(*index.shape,1,2)).squeeze(-2)
    return torch.where(inside[...,None],point,nearest)


def priority_projection(point,base,lo,hi,step):
    """Lexicographic mean-first projection, then nearest feasible dispersion.

    This is a strong simple null for the weighted geometry, not an SPD metric
    at gamma=0. It keeps the exact same continuous/ordinal feasible polygon.
    """
    mean=(point[...,0]-point[...,1]).clamp(min=lo-base,max=hi-base)
    expected=base+mean
    minimum=mean.abs()
    if step is not None:
        lower=(lo+((base-lo)/step).floor()*step).clamp(lo,hi)
        upper=(lo+((base-lo)/step).ceil()*step).clamp(lo,hi)
        gap=(upper-lower).clamp_min(1e-12)
        chord=((upper-expected)*(base-lower)+(expected-lower)*(upper-base))/gap
        minimum=torch.where((expected>=lower)&(expected<=upper)&(upper>lower),chord,minimum)
    maximum=(2*(hi-base)*(base-lo)+(hi+lo-2*base)*mean)/(hi-lo)
    minimum=torch.minimum(minimum.clamp_min(0),maximum)
    dispersion=torch.maximum(minimum,torch.minimum(maximum,point.sum(-1)))
    return torch.stack([.5*(dispersion+mean),.5*(dispersion-mean)],-1)


def main():
    import numpy as np
    import pandas as pd
    from v9.engine import load_v8
    from v4.config import StudyConfig,implementation_signature
    from v4.protocol import validation_roles
    from v4.training import train_v4
    import v4.models as models
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gamma',type=float,required=True)
    parser.add_argument('--mask-jitter',type=float,default=0.)
    parser.add_argument('--projection',choices=['metric','mean-priority'],default='metric')
    args=parser.parse_args()
    if not 0<args.gamma<=1:parser.error('gamma must be inside (0,1]')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'PLAN.json').exists():raise ValueError('Refusing to overwrite an existing candidate')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source=implementation_signature()
    plan=dict(implementation_sha256=source,prototype_sha256=sha(__file__),assignment_sha256=sha(args.assignment),
              gamma=args.gamma,projection=args.projection,mask_fraction_jitter=args.mask_jitter,seed=1,source_note='M4 projection prototype; exact experiment source required',
              test_evaluation=False)
    (args.output/'PLAN.json').write_text(json.dumps(plan,indent=2))
    core,pub=load_v8();torch.set_num_threads(1)
    frame=pd.read_csv(args.assignment)
    data,_=pub.build_publication_matrix_data_v8(frame,pub.DATASET_SPECS['ml-100k'],pub.PublicationDataConfig(dataset='ml-100k',force_backend='dense'),torch.device('cpu'))
    data.train_timestamps=torch.tensor(frame.loc[frame.split=='train','timestamp'].to_numpy(),dtype=torch.float64)
    data.test_ratings_raw[:]=torch.nan
    study=StudyConfig();data=validation_roles(data,study)
    overrides=dict(mask_fraction_jitter=args.mask_jitter)
    if hasattr(study.models['Method4'],'moment_projection_mode'):overrides['moment_projection_mode']='metric'
    cfg=replace(study.models['Method4'],**overrides).validate()
    model=models.instantiate(data,cfg,1);model._v4_artifact_dir=args.output/'trace'

    def evaluate(model,data,partition,**kwargs):
        assert partition=='val'
        with torch.no_grad():
            model.eval();prediction=model.predict_raw_pairs(data,data.val_users,data.val_items)
            return {'RMSE':float((prediction-data.val_ratings_raw).square().mean().sqrt())}

    original=models.signed_moment_projection
    models.signed_moment_projection=(lambda point,base,lo,hi,step,*unused:metric_projection(point,base,lo,hi,step,args.gamma)
                                    if args.projection=='metric' else priority_projection(point,base,lo,hi,step))
    start=time.perf_counter()
    try:
        info=train_v4(SimpleNamespace(MetricConfig=lambda **kw:None,evaluate_publication_model=evaluate),model,data,cfg,None)
        with torch.no_grad():prediction=model.predict_raw_pairs(data,data.val_users,data.val_items)
        np.savez_compressed(args.output/'validation.npz',pair_id=(data.val_users*data.num_items+data.val_items).numpy(),truth=data.val_ratings_raw.numpy(),prediction=prediction.numpy())
        torch.save(dict(format='V4-MomentMetric-Prototype',plan=plan,config=asdict(cfg),state_dict=model.state_dict()),args.output/'prototype.pt')
    finally:
        models.signed_moment_projection=original
    if source!=implementation_signature() or plan['prototype_sha256']!=sha(__file__):raise ValueError('Source changed')
    result=dict(plan=plan,config=asdict(cfg),info=info,validation_RMSE=float((prediction-data.val_ratings_raw).square().mean().sqrt()),
                wall_seconds=time.perf_counter()-start,test_labels_poisoned=True,test_evaluated=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(result['validation_RMSE'],flush=True)


if __name__=='__main__':main()
