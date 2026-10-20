"""Check real one-device storage policies without changing an optimizer update.

Synthetic TRAIN targets only. Reports measured update parity and timings;
does not infer GPU speedups from CPU measurements or change training precision.
"""
import argparse,copy,json,sys,time
from dataclasses import asdict,replace
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import torch
    from v5.models import instantiate
    from v5.config import default_models,implementation_signature
    from v5.history import masked_targets
    from v5.training import optimizer_groups,_backward
    from v5.execution import ExecutionConfig,run_training
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0',choices=['cpu','cuda:0'])
    p.add_argument('--methods',default='Method1,Method2,Method3,Method4,Method5,Method6,Method7')
    p.add_argument('--policies',default='resident,auto,host,ssd')
    p.add_argument('--readout',choices=['geometric','coupled','decoder'],default='geometric')
    p.add_argument('--steps',type=int,default=3);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cache-dir',type=Path)
    a=p.parse_args()
    if a.steps<1 or a.threads<1:p.error('Positive steps/threads required')
    if a.device.startswith('cuda') and not torch.cuda.is_available():p.error('CUDA unavailable; run on the target GPU host')
    torch.set_num_threads(a.threads);device=torch.device(a.device);a.output.mkdir(parents=True,exist_ok=True)
    gen=torch.Generator().manual_seed(608);pair=torch.randperm(48*64,generator=gen)[:1800]
    u,i=pair//64,pair%64;y=(1+(u%3)+(i%3)).float().clamp(1,5)
    data=SimpleNamespace(num_users=48,num_items=64,rating_min=1.,rating_max=5.,rating_step=1.,
        train_users=u.to(device),train_items=i.to(device),train_ratings_raw=y.to(device),
        train_timestamps=(torch.arange(len(u),dtype=torch.float64)+1e9).to(device))
    keep,query=masked_targets(data,.2,torch.Generator().manual_seed(19))
    records=[]
    methods=a.methods.split(',');policies=a.policies.split(',')
    if not policies or policies[0]!='resident':p.error('Compare policies against resident first')
    for method in methods:
        if method not in default_models():p.error('Unknown method '+method)
        cfg=replace(default_models()[method],latent_dim=8,hidden_dim=16,feature_dropout=0.,warmup_steps=0,readout_mode=a.readout)
        reference=None
        for policy in policies:
            execution=ExecutionConfig(storage=policy,device=a.device,cache_dir=None if a.cache_dir is None else str(a.cache_dir))
            model=instantiate(data,cfg,31,device);result_state={}
            def callback():
                torch.manual_seed(8821)
                if device.type=='cuda':torch.cuda.manual_seed(8821)
                groups,_=optimizer_groups(model,a.steps);opt=torch.optim.AdamW(groups)
                start=time.perf_counter()
                for _ in range(a.steps):
                    model.train();h=model.history(keep);state=model.prepare(h);model.zero_grad(set_to_none=True)
                    primary,auxiliary,_=model.loss_for_targets(h,state,data.train_users[query],data.train_items[query],data.train_ratings_raw[query])
                    _backward(model,primary,auxiliary)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
                    opt.step();model.clear_cache()
                if device.type=='cuda':torch.cuda.synchronize(device)
                result_state['weights']={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                result_state['optimizer']=copy.deepcopy(opt.state_dict())
                return dict(steps=a.steps,measured_seconds=time.perf_counter()-start)
            result=run_training(model,callback,lambda:instantiate(data,cfg,31,device),execution)
            if reference is None:reference=result_state
            maximum=0.;exact=True
            def compare(x,y):
                nonlocal maximum,exact
                if torch.is_tensor(x):
                    x,y=x.detach().cpu(),y.detach().cpu();exact=exact and torch.equal(x,y)
                    if x.is_floating_point():maximum=max(maximum,float((x-y).abs().max()) if x.numel() else 0.)
                elif isinstance(x,dict):
                    if x.keys()!=y.keys():raise ValueError('Optimizer/model keys differ')
                    for key in x:compare(x[key],y[key])
                elif isinstance(x,list):
                    if len(x)!=len(y):raise ValueError('Optimizer group count differs')
                    for left,right in zip(x,y):compare(left,right)
                elif x!=y:raise ValueError('Optimizer metadata differs')
            compare(reference,result_state)
            records.append(dict(method=method,policy=policy,config=asdict(cfg),bitwise_equal=exact,max_absolute_difference=maximum,**result))
            print(method,policy,'EXACT' if exact else 'DIFFERENT',maximum,flush=True)
    payload=dict(device=str(device),torch=torch.__version__,implementation_sha256=implementation_signature(),records=records,
        scope='synthetic fixed masks, mean optimizer updates incl. optimizer states; not full dataset convergence or a proof for every CUDA kernel',
        precision_policy='unchanged; AMP/TF32 not enabled by this tool',all_bitwise_equal=all(row['bitwise_equal'] for row in records))
    (a.output/'EXECUTION_PARITY.json').write_text(json.dumps(payload,indent=2))
    if not payload['all_bitwise_equal']:raise SystemExit('Parity differs on this host; inspect the measured differences before changing storage in a paper run')
if __name__=='__main__':main()
