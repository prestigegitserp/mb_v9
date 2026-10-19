"""Blinded validation pilot for a baseline whose best epoch hit its old cap."""
from dataclasses import asdict
import argparse,hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import numpy as np,pandas as pd,torch
    from v9.engine import load_v8
    from v4.config import StudyConfig,implementation_signature
    from v4.protocol import validation_roles
    from v4.baselines import paper_config,train_mbmf
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--weight-decay',type=float,default=.0001)
    parser.add_argument('--method',choices=['MF','MBMF-F','MBMF-C','MBMF-N'],default='MF')
    parser.add_argument('--init-mode',default='isotropic_gaussian')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'PLAN.json').exists():raise ValueError('Refusing to overwrite a frozen baseline pilot')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source=implementation_signature()
    core,pub=load_v8();torch.set_num_threads(1)
    cfg=(pub.BiasMFConfig(latent_dim=20,use_bias=False,max_epochs=1000,weight_decay=args.weight_decay) if args.method=='MF'
         else paper_config(core.PaperBaselineConfig(latent_dim=20,max_epochs=2000)))
    plan=dict(method=args.method,init_mode=args.init_mode,config=asdict(cfg),seed=1,implementation_sha256=source,
              experiment_sha256=sha(__file__),assignment_sha256=sha(args.assignment),test_evaluated=False)
    (args.output/'PLAN.json').write_text(json.dumps(plan,indent=2))
    frame=pd.read_csv(args.assignment)
    data,_=pub.build_publication_matrix_data_v8(frame,pub.DATASET_SPECS['ml-100k'],pub.PublicationDataConfig(dataset='ml-100k',force_backend='dense'),torch.device('cpu'))
    data.train_timestamps=torch.tensor(frame.loc[frame.split=='train','timestamp'].to_numpy(),dtype=torch.float64)
    data.test_ratings_raw[:]=torch.nan;data=validation_roles(data,StudyConfig())
    core.set_global_seed(1)
    model=(pub.BiasMF(data.num_users,data.num_items,cfg,torch.device('cpu')) if args.method=='MF'
           else core.PaperMBMF(data,cfg,args.method,args.init_mode,torch.device('cpu')))
    def evaluate(model,data,partition,**kwargs):
        assert partition=='val'
        with torch.no_grad():
            model.eval();pred=(model.predict_raw_pairs(data,data.val_users,data.val_items) if args.method=='MF'
                               else model.predict_raw_pairs(data.val_users,data.val_items))
            return {'RMSE':float((pred-data.val_ratings_raw).square().mean().sqrt())}
    old=pub.evaluate_publication_model;pub.evaluate_publication_model=evaluate;start=time.perf_counter()
    try:
        info=(pub.train_biasmf(model,data,cfg) if args.method=='MF' else train_mbmf(SimpleNamespace(MetricConfig=lambda **kw:None,evaluate_publication_model=evaluate),model,data,cfg))
    finally:pub.evaluate_publication_model=old
    with torch.no_grad():pred=(model.predict_raw_pairs(data,data.val_users,data.val_items) if args.method=='MF'
                              else model.predict_raw_pairs(data.val_users,data.val_items))
    pub.save_model_checkpoint_v8(args.output/'checkpoint.pt',model,config=cfg,data=data,train_info=info,extra={'method':args.method,'init_mode':args.init_mode})
    np.savez_compressed(args.output/'validation.npz',pair_id=(data.val_users*data.num_items+data.val_items).numpy(),truth=data.val_ratings_raw.numpy(),prediction=pred.numpy())
    if source!=implementation_signature() or plan['experiment_sha256']!=sha(__file__):raise ValueError('Pilot source changed')
    result=dict(plan=plan,actual_config=asdict(cfg),info=info,validation_RMSE=float((pred-data.val_ratings_raw).square().mean().sqrt()),
                parameter_count=sum(p.numel() for p in model.parameters()),wall_seconds=time.perf_counter()-start,
                checkpoint_sha256=sha(args.output/'checkpoint.pt'),test_labels_poisoned=True,test_evaluated=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(result['validation_RMSE'],flush=True)


if __name__=='__main__':main()
