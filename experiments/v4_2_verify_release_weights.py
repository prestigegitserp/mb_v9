"""Explicit checkpoint migration with full validation parity, never training or TEST."""
import argparse,hashlib,json,shutil,sys
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 import numpy as np,pandas as pd,torch
 from v4.config import ModelConfig,StudyConfig,implementation_signature
 from v4.protocol import validation_roles
 from v4.models import instantiate,Method7
 from v4 import checkpoints
 from v4.baselines import restore_mbmf
 from v9.engine import load_v8
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--assignment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists() and any(a.output.iterdir()):raise ValueError('Nonempty output directory')
 a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
 core,pub=load_v8();frame=pd.read_csv(a.assignment)
 data,_=pub.build_publication_matrix_data_v8(frame,pub.DATASET_SPECS['ml-100k'],pub.PublicationDataConfig(dataset='ml-100k',force_backend='dense'),torch.device('cpu'))
 data.train_timestamps=torch.tensor(frame.loc[frame.split=='train','timestamp'].to_numpy(),dtype=torch.float64)
 data.test_ratings_raw[:]=torch.nan;data=validation_roles(data,StudyConfig(),2026)
 pairs=(data.val_users*data.num_items+data.val_items).numpy();records=[]
 for entry in json.loads(a.manifest.read_text())['entries']:
  result=json.loads(Path(entry['result']).read_text());source=Path(entry['checkpoint']);expected=Path(entry['validation'])
  if result.get('test_evaluated') is not False or result.get('test_labels_poisoned') is not True:raise ValueError('Unblinded source result')
  if sha(source)!=result['checkpoint_sha256']:raise ValueError('Source checkpoint hash differs from frozen result')
  payload=torch.load(source,map_location='cpu',weights_only=False);method=entry['method'];destination=a.output/(method+'.pt')
  provenance=dict(action='explicit release verification; no training',source_checkpoint_sha256=sha(source),source_result_sha256=sha(entry['result']),
    source_validation_sha256=sha(expected),assignment_sha256=sha(a.assignment),release_implementation_sha256=implementation_signature(),optimizer_steps_run=0,test_evaluated=False)
  if payload.get('format')=='MBMF-V4':
   provenance.update(source_architecture=payload['architecture_version'],source_implementation_sha256=payload['implementation_sha256'])
   cfg=ModelConfig(**payload['config']).validate();model=instantiate(data,cfg,payload['model_seed']);model.load_state_dict(payload['state_dict'],strict=True)
   if method=='Method4':model.bias.requires_grad_(False)
  else:
   if payload.get('family')=='PaperMBMF':model,_=restore_mbmf(pub,core,payload,data,torch.device('cpu'))
   else:model,_=pub.load_model_checkpoint_v8(source,data,torch.device('cpu'))
   provenance.update(source_format=payload.get('format'),source_frozen_scientific_core_sha256=payload.get('frozen_v7_core_sha256'))
  model.eval()
  with np.load(expected,allow_pickle=False) as z:
   np.testing.assert_array_equal(pairs,z['pair_id']);np.testing.assert_array_equal(data.val_ratings_raw.numpy(),z['truth']);reference=z['prediction'].copy()
  def predict(m):
   with torch.no_grad():
    m.eval();return (m.predict_raw_pairs(data.val_users,data.val_items) if isinstance(m,core.PaperMBMF) else m.predict_raw_pairs(data,data.val_users,data.val_items)).cpu().numpy()
  prediction=predict(model);np.testing.assert_allclose(prediction,reference,atol=2e-6,rtol=1e-6)
  provenance.update(validation_pairs_verified=len(pairs),max_abs_prediction_difference=float(np.max(abs(prediction-reference))))
  if payload.get('format')=='MBMF-V4':
   checkpoints.save(destination,model,config=cfg,data=data,train_info={**result['info'],'release_rebind':provenance},extra={'release_rebind':provenance})
   restored,_=checkpoints.load(destination,data,torch.device('cpu'))
  else:
   shutil.copy2(source,destination)
   if payload.get('family')=='PaperMBMF':restored,_=restore_mbmf(pub,core,payload,data,torch.device('cpu'))
   else:restored,_=pub.load_model_checkpoint_v8(destination,data,torch.device('cpu'))
  restored_prediction=predict(restored);np.testing.assert_allclose(restored_prediction,prediction,atol=0,rtol=0)
  extras={}
  if isinstance(restored,Method7):
   with torch.no_grad():law=restored.distribution(data,data.val_users,data.val_items)
   for key in ('probabilities','member_means','variance','within_variance','between_variance','levels'):extras[key]=law[key].cpu().numpy()
   np.testing.assert_allclose(extras['probabilities']@extras['levels'],prediction,atol=2e-5,rtol=1e-6)
  np.savez_compressed(destination.with_suffix('.validation.npz'),pair_id=pairs,truth=data.val_ratings_raw.numpy(),prediction=prediction,**extras)
  provenance['delivered_checkpoint_sha256']=sha(destination)
  destination.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2))
  records.append(dict(method=method,validation_RMSE=float(np.sqrt(np.mean((prediction.astype(float)-data.val_ratings_raw.numpy())**2))),**provenance))
  print(method,'verified',provenance['max_abs_prediction_difference'],flush=True)
 (a.output/'VERIFIED_WEIGHTS.json').write_text(json.dumps(dict(records=records,all_test_labels_poisoned=True,test_evaluated=False),indent=2))
if __name__=='__main__':main()
