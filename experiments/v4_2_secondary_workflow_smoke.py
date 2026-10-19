"""Small end-to-end QA of winner-anchored K/sparsity/convergence workflows.

Uses only synthetic data; its TEST outputs are software checks, not evidence.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
 import numpy as np,pandas as pd
 from v4 import engine
 from v4.entry import run
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Choose a new QA output directory')
 a.output.mkdir(parents=True)
 original=engine.make_base
 def small(pub,args,dataset):
  cfg=original(pub,args,dataset);cfg.data.synthetic_users=24;cfg.data.synthetic_items=28
  cfg.data.synthetic_rank=3;cfg.data.synthetic_density=.6
  return cfg
 engine.make_base=small
 common=['--datasets','synthetic','--quick','--methods','Method4','--k','4,8','--split-seeds','2026','--model-seeds','1',
   '--v4-threads','1','--v4-no-plots','--v4-search-jobs-per-method','1','--v4-retain-candidates','metrics','--output',str(a.output)]
 try:
  run(['scan','--scan-only',*common]);root=a.output/'synthetic'
  assert not list(root.rglob('*__test.npz'))
  winner=pd.read_csv(root/'search/validation_winners.csv').iloc[0]
  for phase in ('kstudy','sparsity','convergence'):
   run([phase,*common,*(['--keep-rates','0.7,1.0'] if phase=='sparsity' else [])])
  ks=pd.read_csv(root/'kstudy/all_fixed_k_locked_runs.csv');assert set(ks.K)=={4,8}
  heldouts=[];retained=[]
  for rate in (.7,1.):
   directory=root/'sparsity'/f'keep_{rate:g}';r=pd.read_csv(directory/'locked_test_runs.csv')
   assert r.train_keep_realized.between(rate,1.).all();retained.append(float(r.train_keep_realized.iloc[0]))
   assignment=pd.read_csv(next(directory.rglob('split_assignments_2026.csv')))
   heldouts.append(assignment[assignment.split!='train'].sort_values(['userId','movieId']).reset_index(drop=True))
  pd.testing.assert_frame_equal(*heldouts)
  protocol=json.loads((root/'convergence/CONVERGENCE_PROTOCOL.json').read_text())
  assert protocol['anchors'][0]['frozen']['scan_job']==winner.scan_job
  assert not list((root/'convergence').rglob('*__test.npz'))
  train=np.load(next((root/'convergence').rglob('*__train.npz')))
  assert np.isfinite(train['prediction']).all()
  run(['report','--datasets','synthetic','--v4-threads','1','--v4-no-plots','--output',str(a.output)])
  proof=dict(status='PASS',synthetic_only=True,real_test_evaluated=False,frozen_winner=str(winner.scan_job),
   K_checked=[4,8],train_keep_requested=[.7,1.],train_keep_realized=retained,heldout_assignments_identical=True,
   convergence_test_archives=0,convergence_checkpoint_schedule=protocol['checkpoint_schedule'],aggregate_report_rebuilt=True)
  (a.output/'SECONDARY_WORKFLOW_QA.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
 finally:engine.make_base=original

if __name__=='__main__':main()
