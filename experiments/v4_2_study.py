"""Frozen validation-only test of a stronger readout and its reductive null.

Reuses the V4.1 runner's NaN TEST boundary, atomic result/checkpoint recording
and isolated workers. Preparation is explicit and never consults TEST metrics.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import v4_review_study as runner


def prepare(assignment,output,methods=None):
    from v4.config import default_models,implementation_signature
    output.mkdir(parents=True,exist_ok=True)
    path=output/'PLAN.json'
    if path.exists():raise ValueError('Existing frozen plan')
    reviewed=json.loads((ROOT/'examples/v4/reviewed_ml100k.json').read_text())['models']
    models={name:replace(cfg,**reviewed.get(name,{})).validate() for name,cfg in default_models().items()}
    jobs=[]
    for method,cfg in models.items():
        for arm,changes in [('coupled',dict(readout_mode='coupled')),('decoder_null',dict(readout_mode='decoder'))]:
            c=replace(cfg,**changes).validate()
            jobs.append(dict(id=method+'__'+arm,method=method,arm=arm,config=asdict(c),seed=1,init_mode='NA'))
    for arm,changes in [('control',{}),('linear_quarter',dict(moment_refinement_link='linear')),
                        ('linear_full',dict(moment_refinement_link='linear',moment_refinement_scale=1.)),
                        ('tanh_full',dict(moment_refinement_scale=1.))]:
        c=replace(models['Method4'],**changes).validate()
        jobs.append(dict(id='Method4__'+arm,method='Method4',arm=arm,config=asdict(c),seed=1,init_mode='NA'))
    if methods is not None:
        if set(methods)-set(models):raise ValueError('Unknown methods')
        jobs=[job for job in jobs if job['method'] in methods]
    payload=dict(assignment=str(assignment.resolve()),assignment_sha256=runner.digest(assignment),
        implementation_sha256=implementation_signature(),runner_sha256=runner.digest(runner.__file__),
        design_sha256=runner.digest(__file__),dataset='ml-100k',split_seed=2026,
        test_policy='NaN labels before model construction; validation selection ONLY',
        evidence_status='exploratory one split/seed; historical TEST already observed in prior versions',jobs=jobs)
    path.write_text(json.dumps(payload,indent=2));return path


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assignment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--methods')
    a=p.parse_args();print(prepare(a.assignment,a.output,None if a.methods is None else a.methods.split(',')))
