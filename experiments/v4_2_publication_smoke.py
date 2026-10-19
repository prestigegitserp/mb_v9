"""Render a full synthetic scan/final/report, including real probability charts.

Software QA only. Quick synthetic TEST values are not research measurements.
"""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
 import pandas as pd
 from PIL import Image
 from v4 import engine
 from v4.entry import run
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--verify-existing',action='store_true',help='Check completed artifacts without training or rendering')
 a=p.parse_args()
 if a.output.exists() and not a.verify_existing:raise ValueError('Choose a new QA output directory')
 a.output.mkdir(parents=True,exist_ok=True)
 original=engine.make_base
 def small(pub,args,dataset):
  cfg=original(pub,args,dataset);cfg.data.synthetic_users=24;cfg.data.synthetic_items=28
  cfg.data.synthetic_rank=3;cfg.data.synthetic_density=.6
  return cfg
 engine.make_base=small
 common=['--datasets','synthetic','--quick','--methods','MF,Method4,Method7','--include-blend','--blend-methods','MF,Method4',
  '--k','4','--split-seeds','2026','--model-seeds','1','--v4-threads','1','--v4-search-jobs-per-method','1',
  '--v4-retain-candidates','metrics','--output',str(a.output)]
 try:
  root=a.output/'synthetic';hashes_unchanged=None
  if not a.verify_existing:
   run(['scan','--scan-only',*common]);assert not list(root.rglob('*__test.npz'))
   run(['final',*common])
   before={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in root.rglob('*.pt')}
   run(['report','--datasets','synthetic','--v4-threads','1','--output',str(a.output)])
   after={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in root.rglob('*.pt')};assert before==after
   hashes_unchanged=True
  assert not list((root/'search').rglob('*__test.npz'))
  assert list((root/'search/publication/figures').glob('*.pdf'))
  assert len(pd.read_csv(root/'final_locked_test_all.csv'))==3
  assert (root/'ensemble/ENSEMBLE_PROTOCOL.json').is_file()
  files=list(root.rglob('*.png'))
  assert files
  legacy=0
  for image in files:
   assert image.with_suffix('.pdf').is_file(),str(image)
   if image.stem.startswith('q1_'):legacy+=1 # frozen V8 extra figures export PNG/PDF
   else:assert image.with_suffix('.svg').is_file(),str(image)
   with Image.open(image) as im:im.verify()
  for name in ('training_budget.csv','epoch_history.csv','conditional_errors.csv','recomputed_prediction_metrics.csv','point_prediction_calibration.csv'):
   assert (root/'publication/tables'/name).is_file(),name
  assert list(root.rglob('*__train.csv.gz')) and list(root.rglob('*__val.csv.gz')) and list(root.rglob('*__test.csv.gz'))
  proof=dict(status='PASS',synthetic_only=True,real_test_evaluated=False,search_had_no_test_archives=True,
   families=['MF','Method4','Method7'],complete_V4_PNG_PDF_SVG_triplets=len(files)-legacy,legacy_extra_PNG_PDF_pairs=legacy,PNG_files_decodable=True,
   final_and_ensemble_generated=True,raw_train_val_test_per_pair_tables=True,report_checkpoint_hashes_unchanged=hashes_unchanged,
   verification_mode='existing artifacts' if a.verify_existing else 'fresh end-to-end run',
   output_marker='SMOKE; not suitable for publication results')
  (a.output/'PUBLICATION_QA.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
 finally:engine.make_base=original

if __name__=='__main__':main()
