"""AutoRec controls close to the allocated parameter counts of M4 and M6."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import v4_2_baseline_audit as runner

def prepare(assignment,output):
 from v4.config import implementation_signature
 output.mkdir(parents=True,exist_ok=True);path=output/'PLAN.json'
 if path.exists():raise ValueError('Existing frozen plan')
 jobs=[]
 for hidden in (146,265):
  for penalty in (10.,100.):
   jobs.append(dict(id=f'AutoRec__H{hidden}__l2_{penalty}',method='AutoRec',config=dict(hidden_dim=hidden,lambda_l2=penalty,lr=.01),seed=1,init='NA'))
 path.write_text(json.dumps(dict(assignment=str(assignment.resolve()),assignment_sha256=runner.sha(assignment),
   implementation_sha256=implementation_signature(),runner_sha256=runner.sha(runner.__file__),design_sha256=runner.sha(__file__),jobs=jobs,
   protocol='same common V4 selection; TEST NaN before construction; no TEST metrics',
   limitation='one split/seed; allocated parameter matching only, not equal training information/optimization cost'),indent=2));return path
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assignment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(prepare(a.assignment,a.output))
