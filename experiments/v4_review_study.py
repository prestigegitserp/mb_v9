"""Frozen, validation-only review study on a supplied fixed split.

Prepare once, then execute jobs in separate processes. TEST labels are poisoned
before model construction. No test prediction or metric is produced. Completed
jobs are immutable and can be resumed; source/data mismatches fail explicitly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(assignment, output):
    from v4.config import default_models, implementation_signature
    output.mkdir(parents=True, exist_ok=True)
    path = output/'PLAN.json'
    if path.exists():
        raise ValueError('Refusing to replace an existing frozen plan')
    jobs = []
    for method, cfg in default_models().items():
        for arm, changes in [('control', {}), ('density', {'mask_fraction_jitter': .15})]:
            c = replace(cfg, **changes).validate()
            jobs.append(dict(id=method+'__'+arm, method=method, arm=arm,
                             config=asdict(c), seed=1, init_mode='NA'))
    for method, arm, changes in [
        ('Method1', 'decoder_gradient', dict(gradient_surgery=False)),
        ('Method5', 'plugin', dict(posterior_mean_steps=0)),
        ('Method6', 'isotropic', dict(angular_reference_weight=0.)),
        ('Method6', 'decoder_context', dict(context_strength=2., angular_reference_weight=0.)),
    ]:
        jobs.append(dict(id=method+'__'+arm, method=method, arm=arm,
                         config=asdict(replace(default_models()[method], **changes).validate()),
                         seed=1, init_mode='NA'))
    for method in ('MF', 'BiasMF'):
        jobs.append(dict(id=method+'__control', method=method, arm='control', config={}, seed=1, init_mode='NA'))
    for method in ('MBMF-F', 'MBMF-C', 'MBMF-N'):
        for init in ('legacy_uniform_2pi', 'canonical_uniform', 'isotropic_gaussian'):
            jobs.append(dict(id=method+'__'+init, method=method, arm='validation_checkpoint',
                             config={}, seed=1, init_mode=init))
    for lam in (.001, 100.):
        jobs.append(dict(id='AutoRec__lambda'+str(lam), method='AutoRec', arm='regularization',
                         config=dict(lambda_l2=lam, optimizer='rprop', lr=.01), seed=1, init_mode='NA'))
    payload = dict(assignment=str(assignment.resolve()), assignment_sha256=digest(assignment),
                   implementation_sha256=implementation_signature(),
                   runner_sha256=digest(__file__), dataset='ml-100k', split_seed=2026,
                   test_policy='NaN labels during construction/training; no test evaluation',
                   evidence_status='exploratory validation; historically observed split',
                   jobs=jobs)
    path.write_text(json.dumps(payload, indent=2), encoding='utf8')
    return path


def worker(plan_path, job_id):
    import numpy as np
    import pandas as pd
    import torch
    from v4.config import ModelConfig, StudyConfig, implementation_signature
    from v4.protocol import validation_roles
    from v4.models import instantiate
    from v4.training import train_v4
    from v4.baselines import train_mbmf, train_autorec, paper_config
    from v4 import checkpoints
    from v9.engine import load_v8

    plan = json.loads(plan_path.read_text())
    if plan['implementation_sha256'] != implementation_signature() or plan['runner_sha256'] != digest(__file__):
        raise ValueError('Frozen study source mismatch')
    if plan['assignment_sha256'] != digest(plan['assignment']):
        raise ValueError('Frozen study data mismatch')
    job = next(x for x in plan['jobs'] if x['id'] == job_id)
    out = plan_path.parent/job_id
    out.mkdir(exist_ok=True)
    result_path = out/'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['plan_sha256'] != digest(plan_path) or result['job'] != job:
            raise ValueError('Existing result differs from frozen job')
        print('RESUME', job_id, result['validation_RMSE'], flush=True)
        return
    core, pub = load_v8()
    torch.set_num_threads(1)
    frame = pd.read_csv(plan['assignment'])
    data, _ = pub.build_publication_matrix_data_v8(
        frame, pub.DATASET_SPECS['ml-100k'],
        pub.PublicationDataConfig(dataset='ml-100k', force_backend='dense'), torch.device('cpu'))
    data.train_timestamps = torch.tensor(frame.loc[frame.split == 'train', 'timestamp'].to_numpy(), dtype=torch.float64)
    data.test_ratings_raw[:] = torch.nan
    data = validation_roles(data, StudyConfig(), plan['split_seed'])

    def evaluate(model, current_data, partition, **kwargs):
        if partition != 'val':
            raise AssertionError('Review training can evaluate selection validation only')
        with torch.no_grad():
            model.eval()
            pred = (model.predict_raw_pairs(current_data.val_users, current_data.val_items)
                    if isinstance(model, core.PaperMBMF) else
                    model.predict_raw_pairs(current_data, current_data.val_users, current_data.val_items))
            return {'RMSE': float((pred-current_data.val_ratings_raw).square().mean().sqrt())}

    evaluator = SimpleNamespace(MetricConfig=lambda **kwargs: SimpleNamespace(**kwargs),
                                evaluate_publication_model=evaluate)
    start = time.perf_counter()
    method = job['method']
    if method.startswith('Method'):
        cfg = ModelConfig(**job['config']).validate()
        model = instantiate(data, cfg, job['seed'])
        model._v4_artifact_dir = out/'trace'
        info = train_v4(evaluator, model, data, cfg, None)
        checkpoints.save(out/'checkpoint.pt', model, config=cfg, data=data, train_info=info)
    else:
        study = pub.PublicationExperimentConfigV8()
        study.data.dataset = 'ml-100k'
        for name, value in job['config'].items():
            setattr(study.autorec, name, value)
        model, cfg = pub.instantiate_model_v8(method, data, study, 20, job['seed'], torch.device('cpu'),
                                            None if job['init_mode'] == 'NA' else job['init_mode'])
        if method.startswith('MBMF-'):
            cfg = paper_config(cfg)
            info = train_mbmf(evaluator, model, data, cfg)
        elif method == 'AutoRec':
            info = train_autorec(evaluator, model, data, cfg)
        else:
            original = pub.evaluate_publication_model
            pub.evaluate_publication_model = evaluate
            try:
                info = pub.train_biasmf(model, data, cfg)
            finally:
                pub.evaluate_publication_model = original
        pub.save_model_checkpoint_v8(out/'checkpoint.pt', model, config=cfg, data=data, train_info=info,
                                     extra=dict(init_mode=job['init_mode'], method=method))
    with torch.no_grad():
        model.eval()
        pred = (model.predict_raw_pairs(data.val_users, data.val_items) if method.startswith('MBMF-') else
                model.predict_raw_pairs(data, data.val_users, data.val_items))
    if plan['implementation_sha256'] != implementation_signature():
        raise ValueError('Source changed during training')
    np.savez_compressed(out/'validation.npz', pair_id=(data.val_users*data.num_items+data.val_items).numpy(),
                        truth=data.val_ratings_raw.numpy(), prediction=pred.numpy())
    payload = dict(status='COMPLETED', plan_sha256=digest(plan_path), job=job, actual_config=asdict(cfg),
                   validation_RMSE=float((pred-data.val_ratings_raw).square().mean().sqrt()),
                   parameter_count=sum(p.numel() for p in model.parameters()),
                   wall_seconds=time.perf_counter()-start, info=info,
                   checkpoint_sha256=digest(out/'checkpoint.pt'), test_labels_poisoned=True,
                   test_evaluated=False, protocol=data.v4_protocol,
                   packages=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__))
    tmp = result_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2), encoding='utf8')
    tmp.replace(result_path)
    print('COMPLETED', job_id, payload['validation_RMSE'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--job')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.assignment:
        if args.output is None:
            parser.error('--output is required for preparation')
        print(prepare(args.assignment, args.output))
        return
    if args.plan is None:
        parser.error('Provide --assignment/--output or --plan')
    if args.job:
        worker(args.plan, args.job)
        return
    if args.workers < 1:
        parser.error('--workers must be positive')
    plan = json.loads(args.plan.read_text())

    def execute(job):
        log = args.plan.parent/(job['id']+'.log')
        with log.open('a') as handle:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--plan', str(args.plan.resolve()),
                                     '--job', job['id']], stdout=handle, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
        print(job['id'], 'PASS' if result.returncode == 0 else 'FAILED; see '+str(log), flush=True)
        return result.returncode

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        failed = [future.result() for future in as_completed([pool.submit(execute, job) for job in plan['jobs']])]
    if any(failed):
        raise SystemExit('Review study has failed jobs; no aggregate success is claimed')


if __name__ == '__main__':
    main()
