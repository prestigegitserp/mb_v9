"""Rebind review weights to the release only after full validation-prediction parity.

This is an explicit provenance migration, not a claim that old weights were
trained with a new source hash. No optimizer step or test evaluation is run.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    import pandas as pd
    import torch
    from v9.engine import load_v8
    from v4.config import ModelConfig,StudyConfig,implementation_signature
    from v4.models import instantiate,Method7
    from v4.protocol import validation_roles
    from v4 import checkpoints
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--assignment',type=Path,required=True)
    parser.add_argument('--expected-source-sha',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Refusing to overwrite a release checkpoint')
    result=json.loads(args.result.read_text())
    if result['test_evaluated'] or not result['test_labels_poisoned']:
        raise ValueError('Source result does not declare blinded test labels')
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    fmt=payload.get('format')
    if fmt=='MBMF-V4':
        source=payload['implementation_sha256'];seed=payload['model_seed']
        cfg=dict(payload['config']);cfg['moment_projection_weight']=1.
        cfg['moment_projection_mode']='metric'
    elif fmt=='V4-MomentMetric-Prototype':
        if payload['plan']!=result['plan']:raise ValueError('Prototype plan/result mismatch')
        source=payload['plan']['implementation_sha256'];seed=payload['plan']['seed']
        cfg=dict(payload['config']);cfg['moment_projection_weight']=payload['plan']['gamma']
        cfg['moment_projection_mode']=payload['plan'].get('projection','metric')
    else:raise ValueError('Unsupported source checkpoint')
    if source!=args.expected_source_sha:raise ValueError('Source implementation hash mismatch')
    if fmt=='MBMF-V4' and sha(args.checkpoint)!=result['checkpoint_sha256']:
        raise ValueError('Source checkpoint differs from the frozen result')
    _,pub=load_v8();torch.set_num_threads(1)
    frame=pd.read_csv(args.assignment)
    data,_=pub.build_publication_matrix_data_v8(frame,pub.DATASET_SPECS['ml-100k'],pub.PublicationDataConfig(dataset='ml-100k',force_backend='dense'),torch.device('cpu'))
    data.train_timestamps=torch.tensor(frame.loc[frame.split=='train','timestamp'].to_numpy(),dtype=torch.float64)
    data.test_ratings_raw[:]=torch.nan
    data=validation_roles(data,StudyConfig())
    cfg=ModelConfig(**cfg).validate();model=instantiate(data,cfg,seed)
    model.load_state_dict(payload['state_dict'],strict=True)
    if cfg.method=='Method4':model.bias.requires_grad_(False)
    model.eval()
    expected_path=args.result.parent/'validation.npz'
    with np.load(expected_path,allow_pickle=False) as archive:
        old_pair,old_prediction,old_truth=[archive[k] for k in ('pair_id','prediction','truth')]
    pair=(data.val_users*data.num_items+data.val_items).numpy()
    np.testing.assert_array_equal(pair,old_pair)
    np.testing.assert_array_equal(data.val_ratings_raw.numpy(),old_truth)
    with torch.no_grad():prediction=model.predict_raw_pairs(data,data.val_users,data.val_items).numpy()
    np.testing.assert_allclose(prediction,old_prediction,atol=2e-6,rtol=1e-6)
    provenance=dict(action='EXPLICIT_RELEASE_REBIND_WITHOUT_TRAINING',source_format=fmt,
                    source_implementation_sha256=source,release_implementation_sha256=implementation_signature(),
                    source_checkpoint_sha256=sha(args.checkpoint),source_result_sha256=sha(args.result),
                    source_validation_archive_sha256=sha(expected_path),assignment_sha256=sha(args.assignment),
                    validation_pairs_verified=len(pair),max_abs_prediction_difference=float(np.max(np.abs(prediction-old_prediction))),
                    optimizer_steps_run=0,new_test_evaluated=False)
    info=dict(result['info']);info['release_rebind']=provenance
    checkpoints.save(args.output,model,config=cfg,data=data,train_info=info,extra={'release_rebind':provenance})
    restored,_=checkpoints.load(args.output,data,torch.device('cpu'))
    with torch.no_grad():roundtrip=restored.predict_raw_pairs(data,data.val_users,data.val_items).numpy()
    np.testing.assert_allclose(roundtrip,prediction,atol=0,rtol=0)
    extras={}
    if isinstance(restored,Method7):
        with torch.no_grad():dist=restored.distribution(data,data.val_users,data.val_items)
        for name in ('probabilities','member_means','member_probabilities','variance','within_variance','between_variance','levels'):
            extras[name]=dist[name].cpu().numpy()
        np.testing.assert_allclose(extras['probabilities']@extras['levels'],prediction,atol=2e-5,rtol=1e-6)
    np.savez_compressed(args.output.with_suffix('.validation.npz'),pair_id=pair,truth=old_truth,prediction=prediction,**extras)
    diagnostics=restored.diagnostics(data,data.val_users,data.val_items)
    if extras:
        probabilities=extras['probabilities'];levels=extras['levels'];index=np.abs(old_truth[:,None]-levels).argmin(-1)
        cdf=probabilities.cumsum(-1)
        diagnostics['validation_probability_NLL']=float(-np.log(probabilities[np.arange(len(pair)),index].clip(1e-12)).mean())
        diagnostics['validation_probability_RPS']=float(((cdf[:,:-1]-(levels[None,:-1]>=old_truth[:,None]))**2).sum(-1).mean())
        lower=levels[(cdf>=.05).argmax(-1)];upper=levels[(cdf>=.95).argmax(-1)]
        diagnostics['validation_PICP_90']=float(((old_truth>=lower)&(old_truth<=upper)).mean())
        diagnostics['validation_mean_interval_width_90']=float((upper-lower).mean())
    args.output.with_suffix('.diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    args.output.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance,indent=2))


if __name__=='__main__':main()
