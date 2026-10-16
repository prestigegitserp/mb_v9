"""Validation-only checkpoint trainer, with dataset-size-aware train chunks.

No Test labels are accessed here. Never select a model seed based on Test.
"""
from __future__ import annotations
import copy,time,math
import torch


def train_v2(pub,model,data,cfg,*,chunk_size=125000):
    n=int(data.train_users.numel())
    if n<1:raise ValueError('Empty training split')
    if cfg.eval_every<1 or cfg.val_patience<1:raise ValueError('Invalid eval/patience')
    # Unused legacy angle parameters do not enter V2 optimizer, preventing phantom
    # capacity/optimizer state and ensuring true Cartesian directions are trained.
    active=[p for name,p in model.named_parameters()
            if p.requires_grad and name not in ('Phi','Theta','Phi_user','Theta_user','Phi_item','Theta_item','uv_user_angles','uv_item_angles','iv_user_angles','iv_item_angles')]
    opt=torch.optim.AdamW(active,lr=float(cfg.lr),weight_decay=float(cfg.weight_decay))
    best=float('inf');best_state=None;best_epoch=0;wait=0;hist=[];start=time.perf_counter()
    chunks=max(1,math.ceil(n/max(int(chunk_size),1)))
    mc=pub.MetricConfig(compute_testset_ndcg=False,compute_binary_ranking=False,
                        save_test_predictions=False,compute_pearson_spearman=False)
    for epoch in range(1,int(cfg.max_epochs)+1):
        model.v2_epoch=epoch
        with torch.no_grad():model.v2_phase.fill_(1. if epoch<=model.v2_warmup else 0.)
        model.train();opt.zero_grad(set_to_none=True)
        loss_sum=0.;last_parts={}
        # A shallow copy preserves ORIGINAL full training history and mask buffers,
        # but exposes a row slice to the objective; avoids N*U-I broadcast.
        for idx in range(chunks):
            lo=idx*chunk_size;hi=min(n,(idx+1)*chunk_size)
            d=copy.copy(data)
            d.train_users=data.train_users[lo:hi]
            d.train_items=data.train_items[lo:hi]
            d.train_ratings_raw=data.train_ratings_raw[lo:hi]
            loss,parts=model.loss(d)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite Method V2 loss at epoch={epoch}')
            scaled=loss*((hi-lo)/n)
            scaled.backward()
            loss_sum+=float(scaled.detach());last_parts=parts
        if cfg.grad_clip and cfg.grad_clip>0:
            torch.nn.utils.clip_grad_norm_(active,float(cfg.grad_clip))
        opt.step()
        if epoch==1 or epoch%int(cfg.eval_every)==0 or epoch==cfg.max_epochs:
            # Complete val only; no test evaluation or rankings during training.
            val=pub.evaluate_publication_model(model,data,'val',metric_cfg=mc)
            vr=float(val['RMSE'])
            if not math.isfinite(vr):raise FloatingPointError(f'Nonfinite V2 val RMSE at epoch {epoch}')
            hist.append({'epoch':epoch,'loss':loss_sum,'val_RMSE':vr,**last_parts})
            if vr<best-1e-8:
                best=vr;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());wait=0
            else:
                wait+=1
                if wait>=cfg.val_patience:break
    if best_state is None:raise RuntimeError('No finite validation checkpoint available')
    model.load_state_dict(best_state)
    model.v2_epoch=best_epoch
    return {'best_epoch':best_epoch,'best_val_RMSE':best,'elapsed_sec':time.perf_counter()-start,
            'history':hist,'v2_optimizer':'AdamW', 'v2_train_chunk_size':chunk_size}
