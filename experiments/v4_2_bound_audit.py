"""TRAIN-derived support audit on selection validation; never evaluate TEST."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
 import numpy as np,pandas as pd,torch
 from v9.engine import load_v8
 from v4.protocol import validation_roles
 from v4.config import StudyConfig
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assignment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 core,pub=load_v8();torch.set_num_threads(1);frame=pd.read_csv(a.assignment)
 data,_=pub.build_publication_matrix_data_v8(frame,pub.DATASET_SPECS['ml-100k'],pub.PublicationDataConfig(dataset='ml-100k',force_backend='dense'),torch.device('cpu'))
 data.train_timestamps=torch.tensor(frame.loc[frame.split=='train','timestamp'].to_numpy(),dtype=torch.float64);data.test_ratings_raw[:]=torch.nan
 data=validation_roles(data,StudyConfig(),2026);rows=[];details=[]
 for variant in ('MBMF-F','MBMF-C','MBMF-N'):
  cfg=core.PaperBaselineConfig(latent_dim=20);model=core.PaperMBMF(data,cfg,variant,'isotropic_gaussian',torch.device('cpu'))
  for split in ('train','val'):
   u,i,y=[getattr(data,split+'_'+part) for part in ('users','items','ratings_raw')]
   bound=model.rW[u,0]*model.rH[i,0]
   if variant=='MBMF-F':lo=torch.full_like(y,data.rating_min);hi=torch.full_like(y,data.rating_max)
   elif variant=='MBMF-C':lo=torch.full_like(y,model.data_meta[2]);hi=lo+2*bound
   else:lo=torch.zeros_like(y);hi=bound
   residual=torch.relu(lo-y)+torch.relu(y-hi)
   rows.append(dict(method=variant,partition=split,n=len(y),outside_support_fraction=float((residual>1e-5).float().mean()),
     RMSE_support_lower_bound=float(residual.square().mean().sqrt()),mean_support_width=float((hi-lo).mean()),interpretation='entrywise lower bound only; joint low-rank constraints can cause larger unavoidable error'))
   for level in torch.unique(y):
    hit=y==level
    details.append(dict(method=variant,partition=split,rating=float(level),n=int(hit.sum()),outside_support_fraction=float((residual[hit]>1e-5).float().mean()),RMSE_support_lower_bound=float(residual[hit].square().mean().sqrt())))
 pd.DataFrame(rows).to_csv(a.output/'mbmf_support_feasibility.csv',index=False)
 pd.DataFrame(details).to_csv(a.output/'mbmf_support_by_rating.csv',index=False)
 (a.output/'BOUND_AUDIT_PROTOCOL.json').write_text(json.dumps(dict(test_evaluated=False,test_labels_poisoned=True,protocol=data.v4_protocol,config='TRAIN-only original radii, rho=.05; no fitted directions needed for support bound'),indent=2))
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
