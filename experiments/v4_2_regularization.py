"""Predeclare a stronger regularization test motivated by the AutoRec control."""
import argparse,json
from dataclasses import asdict,replace
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import v4_review_study as runner


def prepare(assignment,output):
    from v4.config import default_models,implementation_signature
    output.mkdir(parents=True,exist_ok=True);path=output/'PLAN.json'
    if path.exists():raise ValueError('Existing frozen plan')
    reviewed=json.loads((ROOT/'examples/v4/reviewed_ml100k.json').read_text())['models'];models=default_models()
    jobs=[]
    for method in ('Method4','Method6'):
        cfg=replace(models[method],**reviewed[method])
        if method=='Method4':cfg=replace(cfg,moment_refinement_link='linear',moment_refinement_scale=1.)
        else:cfg=replace(cfg,readout_mode='coupled')
        for penalty in (10.,100.,1000.):
            c=replace(cfg,collaborative_l2=penalty).validate()
            jobs.append(dict(id=method+'__l2_'+str(penalty),method=method,arm='stronger_regularization',config=asdict(c),seed=1,init_mode='NA'))
        c=replace(cfg,readout_mode='decoder',collaborative_l2=100.).validate()
        jobs.append(dict(id=method+'__regularized_decoder_null',method=method,arm='regularized_null',config=asdict(c),seed=1,init_mode='NA'))
    path.write_text(json.dumps(dict(assignment=str(assignment.resolve()),assignment_sha256=runner.digest(assignment),
        implementation_sha256=implementation_signature(),runner_sha256=runner.digest(runner.__file__),design_sha256=runner.digest(__file__),
        dataset='ml-100k',split_seed=2026,test_policy='NaN TEST before construction; no TEST metrics',
        evidence_status='adaptive exploratory validation; motivated by separate AutoRec regularization evidence; no test tuning',jobs=jobs),indent=2));return path
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assignment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(prepare(a.assignment,a.output))
