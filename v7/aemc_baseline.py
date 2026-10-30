"""Auditable AEMC-NE controls; official implementation parity is not asserted.

Same Eq.(5) objective for full or unbiased uniform-item minibatch gradients.
The optional identity-transfer initialization starts the ReLU scalar network at
h(z)=z. It is an experimental initialization, not a paper-specified detail.
"""
from dataclasses import dataclass, fields
import copy
import math
import time
import torch
from torch import nn


@dataclass
class AEMCConfig:
    main_hidden_dim:int=500
    element_hidden_dim:int=200
    lambda_main:float=100.
    lambda_element:float=100.
    lr:float=.001
    max_epochs:int=1000
    eval_every:int=5
    val_patience:int=20
    main_activation:str="relu"
    element_activation:str="relu"
    initialization:str="torch"
    item_batch_size:int=0
    warm_start_epochs:int=0
    element_skip:float=0.

    def validate(self):
        if isinstance(self.element_skip,bool) or not isinstance(self.element_skip,(int,float)) or not math.isfinite(self.element_skip) or not 0<=self.element_skip<=1:
            raise ValueError("Invalid AEMC fixed element skip")
        if isinstance(self.warm_start_epochs,bool) or not isinstance(self.warm_start_epochs,int) or self.warm_start_epochs<0:
            raise ValueError("Invalid AEMC warm-start budget")
        if self.warm_start_epochs and self.initialization != "identity-transfer":
            raise ValueError("Warm-start control requires identity-transfer initialization")
        for name in ("main_hidden_dim","element_hidden_dim","max_epochs","eval_every","val_patience"):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError("Invalid AEMC "+name)
        if isinstance(self.item_batch_size,bool) or not isinstance(self.item_batch_size,int) or self.item_batch_size<0:raise ValueError("Invalid item batch size")
        for name in ("lambda_main","lambda_element","lr"):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or (name=="lr" and value==0):raise ValueError("Invalid AEMC "+name)
        if self.main_activation not in ("relu","sigmoid","tanh") or self.element_activation not in ("relu","sigmoid","tanh"):raise ValueError("Invalid AEMC activation")
        if self.initialization not in ("torch","xavier","identity-transfer"):raise ValueError("Invalid AEMC initialization")
        if self.initialization=="identity-transfer" and (self.element_activation!="relu" or self.element_hidden_dim<2):raise ValueError("Identity transfer requires at least two ReLU units")
        return self

    @classmethod
    def from_legacy(cls,value,**overrides):
        allowed={field.name for field in fields(cls)}
        return cls(**{**{key:val for key,val in vars(value).items() if key in allowed},**overrides}).validate()


def activate(value,name):
    return {"relu":torch.relu,"sigmoid":torch.sigmoid,"tanh":torch.tanh}[name](value)


class AEMCNE(nn.Module):
    def __init__(self,data,cfg,device):
        super().__init__();self.cfg=copy.deepcopy(cfg).validate()
        self.encoder=nn.Linear(data.num_users,cfg.main_hidden_dim)
        self.decoder=nn.Linear(cfg.main_hidden_dim,data.num_users)
        self.element1=nn.Linear(1,cfg.element_hidden_dim)
        self.element2=nn.Linear(cfg.element_hidden_dim,1)
        if cfg.initialization!="torch":
            for layer in (self.encoder,self.decoder,self.element1,self.element2):
                nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
        if cfg.initialization=="identity-transfer":
            with torch.no_grad():
                self.element1.weight[0].fill_(1.);self.element1.weight[1].fill_(-1.)
                self.element2.weight.zero_();self.element2.weight[0,0]=1.-cfg.element_skip;self.element2.weight[0,1]=-(1.-cfg.element_skip)
        self.to(device);self._eval_hidden=None

    def train(self,mode=True):
        self._eval_hidden=None
        return super().train(mode)

    def load_state_dict(self,*args,**kwargs):
        self._eval_hidden=None
        return super().load_state_dict(*args,**kwargs)

    def item_hidden(self,data,items=None):
        if items is None and not self.training and self._eval_hidden is not None and self._eval_hidden[0]==id(data):
            return self._eval_hidden[1]
        if items is None:
            cache=getattr(data,"_v7_aemc_raw_matrix",None)
            if cache is None or cache.device!=data.train_users.device:
                cache=torch.sparse_coo_tensor(torch.stack([data.train_items,data.train_users]),
                    data.train_ratings_raw,(data.num_items,data.num_users),device=data.train_users.device).coalesce()
                data._v7_aemc_raw_matrix=cache
            raw=cache
        else:
            lookup=torch.full((data.num_items,),-1,dtype=torch.long,device=items.device)
            lookup[items]=torch.arange(len(items),device=items.device)
            local=lookup[data.train_items];keep=local>=0
            raw=torch.sparse_coo_tensor(torch.stack([local[keep],data.train_users[keep]]),
                data.train_ratings_raw[keep],(len(items),data.num_users),device=items.device).coalesce()
        hidden=activate(torch.sparse.mm(raw,self.encoder.weight.T)+self.encoder.bias,self.cfg.main_activation)
        if items is None and not self.training:self._eval_hidden=(id(data),hidden)
        return hidden

    def _h(self,z):
        return self.element2(activate(self.element1(z[:,None]),self.cfg.element_activation))[:,0]+self.cfg.element_skip*z

    def predict_raw_pairs(self,data,u,i):
        hidden=self.item_hidden(data)
        return self._h((hidden[i]*self.decoder.weight[u]).sum(-1)+self.decoder.bias[u])

    def penalty(self):
        main=self.encoder.weight.square().sum()+self.decoder.weight.square().sum()
        element=self.element1.weight.square().sum()+self.element2.weight.square().sum()
        return self.cfg.lambda_main*main+self.cfg.lambda_element*element


def train_aemcne(pub,model,data,cfg,chunk_size=8192):
    from .baselines import _finite_gradients,_snapshot
    cfg.validate()
    if chunk_size<1:raise ValueError("Positive AEMC chunk size required")
    n=len(data.train_users)
    if not n:raise ValueError("Empty AEMC training population")
    from .recovery import EpochRecovery
    recovery=EpochRecovery(model,phase="baseline",config=cfg)
    opt=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    items=torch.unique(data.train_items).cpu()
    generator=torch.Generator(device="cpu").manual_seed(getattr(model,"_v7_model_seed",0)+190081)
    best=math.inf;best_epoch=0;best_state=None;wait=0;steps=0;targets_seen=0;history=[];started=time.perf_counter()
    metric=pub.MetricConfig(compute_testset_ndcg=False,compute_binary_ranking=False,save_test_predictions=False,compute_pearson_spearman=False)
    warm_info = None
    if cfg.warm_start_epochs and recovery.payload is None:
        from .baselines import train_autorec
        from types import SimpleNamespace
        class LinearOutput(nn.Module):
            def __init__(self, original):
                super().__init__(); self.original = original
            @property
            def encoder(self): return self.original.encoder
            @property
            def decoder(self): return self.original.decoder
            def item_hidden(self, d): return self.original.item_hidden(d)
            def predict_raw_pairs(self, d, u, i):
                hidden = self.item_hidden(d)
                return (hidden[i] * self.decoder.weight[u]).sum(-1) + self.decoder.bias[u]
        warm_cfg = SimpleNamespace(optimizer="rprop",lr=.01,max_epochs=cfg.warm_start_epochs,
            eval_every=cfg.eval_every,val_patience=cfg.val_patience,lambda_l2=cfg.lambda_main)
        warm_model=LinearOutput(model)
        if getattr(model,"_v7_artifact_dir",None):
            from pathlib import Path
            warm_model._v7_artifact_dir=Path(model._v7_artifact_dir)/"warm_start"
        warm_model._v7_resume=getattr(model,"_v7_resume",False)
        warm_model._v7_recovery_interval=getattr(model,"_v7_recovery_interval",5)
        warm_info = train_autorec(pub,warm_model,data,warm_cfg,chunk_size)
        model.eval()
        best = float(pub.evaluate_publication_model(model,data,"val",metric_cfg=metric)["RMSE"])
        best_state = _snapshot(model)
        history.append(dict(epoch=0,val_RMSE=best,optimizer_steps=0,phase="linear-head warm start"))
    recovered=recovery.restore(opt,generator)
    epoch=0
    if recovered:
        epoch=int(recovered["epoch"])
        best=recovered["best"]
        best_epoch=recovered["best_epoch"]
        best_state=recovered["best_state"]
        wait=recovered["wait"]
        steps=recovered["steps"]
        targets_seen=recovered["targets_seen"]
        history=recovered["history"]
        warm_info=recovered["warm_info"]
    start_epoch=epoch+1 if wait<cfg.val_patience else cfg.max_epochs+1
    for epoch in range(start_epoch,cfg.max_epochs+1):
        model.train();sse_epoch=0.
        order=items if cfg.item_batch_size==0 else items[torch.randperm(len(items),generator=generator)]
        batch=cfg.item_batch_size or len(items)
        for start in range(0,len(order),batch):
            chosen=order[start:start+batch].to(data.train_items.device)
            lookup=torch.full((data.num_items,),-1,dtype=torch.long,device=chosen.device)
            lookup[chosen]=torch.arange(len(chosen),device=chosen.device)
            local=lookup[data.train_items]
            index=(local>=0).nonzero(as_tuple=False)[:,0]
            if not len(index):continue
            hidden=model.item_hidden(data,chosen);opt.zero_grad(set_to_none=True)
            # Uniform item minibatches estimate the complete observed SSE;
            # normalization is global, not the random number of batch ratings.
            scale=len(items)/len(chosen)
            slices=list(range(0,len(index),chunk_size))
            for number,begin in enumerate(slices):
                ix=index[begin:begin+chunk_size];u=data.train_users[ix]
                z=(hidden[local[ix]]*model.decoder.weight[u]).sum(-1)+model.decoder.bias[u]
                squared=(model._h(z)-data.train_ratings_raw[ix]).square().sum()
                loss=scale*squared/n
                if number==0:loss=loss+model.penalty()/n
                if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite AEMC objective")
                loss.backward(retain_graph=number<len(slices)-1)
                sse_epoch+=float(squared.detach());targets_seen+=len(ix)
            _finite_gradients(model);opt.step();steps+=1
        if epoch==1 or epoch%cfg.eval_every==0 or epoch==cfg.max_epochs:
            model.eval();value=float(pub.evaluate_publication_model(model,data,"val",metric_cfg=metric)["RMSE"])
            if not math.isfinite(value):raise FloatingPointError("Nonfinite AEMC validation")
            history.append(dict(epoch=epoch,loss=sse_epoch/n,val_RMSE=value,optimizer_steps=steps,targets_seen=targets_seen))
            if value<best-1e-8:best,best_epoch,best_state,wait=value,epoch,_snapshot(model),0
            else:wait+=1
            pass
        recovery.save(model,opt,generator,dict(epoch=epoch,best=best,best_epoch=best_epoch,best_state=best_state,wait=wait,steps=steps,targets_seen=targets_seen,history=history,warm_info=warm_info),force=(wait>=cfg.val_patience or epoch==cfg.max_epochs))
        if wait>=cfg.val_patience:break
    if best_state is None:raise RuntimeError("No finite AEMC checkpoint")
    model.load_state_dict(best_state);model.eval()
    warm_steps=warm_info["last_epoch"] if warm_info else 0
    return dict(best_epoch=best_epoch,best_val_RMSE=best,history=history,last_epoch=epoch,
        optimizer_steps=steps+warm_steps,warm_start_optimizer_steps=warm_steps,final_optimizer_steps=steps,
        selected_identity_head_at_warm_start=bool(warm_info and best_epoch==0),
        prediction_targets_seen=targets_seen+warm_steps*n,stopping_reason="validation_patience" if epoch<cfg.max_epochs else "epoch_budget",
        best_checkpoint_at_budget=best_epoch==cfg.max_epochs,elapsed_sec=time.perf_counter()-started,
        warm_start=warm_info,warm_start_epochs=cfg.warm_start_epochs,
        optimization_control="linear AutoRec warm start" if warm_info else "direct Adam",
        optimizer="Adam uniform-item minibatches" if cfg.item_batch_size else "Adam full observed objective",
        implementation="independent; activation/initialization/batch controls are not verified paper implementation details")
