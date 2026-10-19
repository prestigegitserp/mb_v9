"""Matched strong-regularization geometry and decoder controls for M6."""
import argparse,json,sys
from dataclasses import asdict,replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import v4_review_study as runner

def prepare(assignment,output):
 from v4.config import default_models,implementation_signature
 output.mkdir(parents=True,exist_ok=True);path=output/'PLAN.json'
 if path.exists():raise ValueError('Existing frozen plan')
 settings=json.loads((ROOT/'examples/v4/reviewed_ml100k.json').read_text())['models']['Method6']
 base=replace(default_models()['Method6'],**settings)
 arms=[('regularization_only',dict(collaborative_l2=1000.)),
       ('regularized_decoder_null',dict(collaborative_l2=1000.,readout_mode='decoder')),
       ('pure_decoder_null',dict(collaborative_l2=1000.,readout_mode='decoder',transport_moment_weight=0.,reconstruction_weight=0.))]
 jobs=[dict(id='Method6__'+arm,method='Method6',arm=arm,config=asdict(replace(base,**changes).validate()),seed=1,init_mode='NA') for arm,changes in arms]
 path.write_text(json.dumps(dict(assignment=str(assignment.resolve()),assignment_sha256=runner.digest(assignment),
     implementation_sha256=implementation_signature(),runner_sha256=runner.digest(runner.__file__),design_sha256=runner.digest(__file__),
     dataset='ml-100k',split_seed=2026,test_policy='NaN TEST before construction; no TEST metrics',
     evidence_status='adaptive exploratory validation; strong nulls; not confirmatory',jobs=jobs),indent=2));return path
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assignment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(prepare(a.assignment,a.output))
