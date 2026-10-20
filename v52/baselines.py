"""V52-only baseline training fixes; frozen V1--V3 sources stay untouched.

Common-protocol MBMF keeps the published factors, objective and adaptive-step
rule, but chooses its checkpoint on the SAME selection population as MF.
The explicit paper policy delegates to the original trainer instead.
"""
from __future__ import annotations

from dataclasses import field, fields, make_dataclass
import math
import time

import torch


def paper_config(config, policy="validation", patience=16, optimizer="paper", adam_lr=.003):
    cls = make_dataclass("V52PaperBaselineConfig", [
        ("checkpoint_policy", str, field(default=policy)),
        ("val_patience", int, field(default=patience)),
        ("optimizer", str, field(default=optimizer)),
        ("adam_lr", float, field(default=adam_lr)),
    ], bases=(type(config),))
    return cls(**{f.name: getattr(config, f.name) for f in fields(config)})


def _snapshot(model,cpu=True):
    return {k: (v.detach().cpu().clone() if cpu else v.detach().clone()) for k, v in model.state_dict().items()}


def _finite_gradients(model):
    checks=[torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    if checks and not bool(torch.stack(checks).all()):
        raise FloatingPointError("Nonfinite baseline gradient; optimizer update was not applied")


def train_mbmf(pub, model, data, cfg, chunk_size=250_000):
    """Exact chunked SSE gradients, accepted-step rule, validation checkpoint.

    A rejected step restores all parameters. Validation changes neither the
    training gradient nor the accept/reject decision. Last-epoch validation is
    mandatory so non-multiple budgets cannot silently omit their last state.
    """
    if cfg.checkpoint_policy != "validation":
        raise ValueError("Use the original trainer for the paper checkpoint policy")
    if cfg.optimizer=='adam':return train_mbmf_adam(pub,model,data,cfg,chunk_size)
    if cfg.optimizer!='paper':raise ValueError('Unknown MBMF optimizer')
    if cfg.objective_reduction not in ("sum", "mean") or chunk_size < 1 or cfg.max_epochs < 1:
        raise ValueError("Invalid MBMF reduction or training budget")
    start = time.perf_counter()
    n = len(data.train_users)
    lr, stable, wait = float(cfg.initial_lr), 0, 0
    best, best_epoch, best_state = math.inf, 0, None
    history = []
    metric = pub.MetricConfig(compute_testset_ndcg=False, compute_binary_ranking=False,
                            save_test_predictions=False, compute_pearson_spearman=False)

    def objective(backward=False):
        w, h = model.factors()
        total = 0.
        slices = list(range(0, n, chunk_size))
        for index, begin in enumerate(slices):
            sl = slice(begin, begin+chunk_size)
            u, i = data.train_users[sl], data.train_items[sl]
            target = model.native_targets(data.train_ratings_raw[sl], u, i)
            squared = ((w[u]*h[i]).sum(-1)-target).square().sum()
            value = squared/(n if cfg.objective_reduction == "mean" else 1)
            if backward and not torch.isfinite(value):
                raise FloatingPointError("Nonfinite V52 MBMF training objective")
            if backward:
                (.5*value).backward(retain_graph=index < len(slices)-1)
            total += float(value.detach())
        return total

    reason = "epoch_budget"
    for epoch in range(1, cfg.max_epochs+1):
        model.train()
        model.zero_grad(set_to_none=True)
        old = objective(backward=True)
        _finite_gradients(model)
        before = _snapshot(model,cpu=False)
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-lr)
            new = objective()
        accepted = math.isfinite(new) and new <= old
        if accepted:
            decrease = old-new
            stable = stable+1 if decrease < cfg.objective_tol else 0
            lr *= cfg.lr_increase
        else:
            model.load_state_dict(before)
            new, decrease, stable = old, 0., 0
            lr = max(cfg.min_lr, lr*cfg.lr_decrease)
        stop_objective = stable >= cfg.objective_patience or (lr <= cfg.min_lr and not accepted)
        if epoch == 1 or epoch % cfg.eval_every == 0 or epoch == cfg.max_epochs or stop_objective:
            model.eval()
            value = float(pub.evaluate_publication_model(model, data, "val", metric_cfg=metric)["RMSE"])
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite MBMF selection RMSE")
            history.append(dict(epoch=epoch, objective=new, objective_decrease=decrease,
                                accepted=accepted, lr=lr, val_RMSE=value))
            if value < best-1e-8:
                best, best_epoch, best_state, wait = value, epoch, _snapshot(model), 0
            else:
                wait += 1
            if wait >= cfg.val_patience:
                reason = "validation_patience"
                break
        if stop_objective:
            reason = "training_objective"
            break
    if best_state is None:
        raise RuntimeError("No finite MBMF validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return dict(best_epoch=best_epoch, best_val_RMSE=best, final_epoch=epoch,
                final_lr=lr, history=history, stopping_reason=reason,
                optimizer='paper_adaptive_SGD',best_checkpoint_at_budget=best_epoch==cfg.max_epochs,
                checkpoint_policy=cfg.checkpoint_policy, elapsed_sec=time.perf_counter()-start)


def train_mbmf_adam(pub,model,data,cfg,chunk_size):
    """Optimizer control on the SAME MBMF factors, radii, targets and mask.

    Explicitly labelled common-protocol optimization; this is not Algorithm 1's
    accepted-step update. Adam may accept non-monotone training objectives.
    """
    if cfg.objective_reduction not in ('sum','mean') or cfg.max_epochs<1 or chunk_size<1:raise ValueError('Invalid MBMF budget')
    opt=torch.optim.Adam(model.parameters(),lr=cfg.adam_lr)
    n=len(data.train_users);best=math.inf;be=0;state=None;wait=0;history=[];start=time.perf_counter()
    mc=pub.MetricConfig(compute_testset_ndcg=False,compute_binary_ranking=False,save_test_predictions=False,compute_pearson_spearman=False)
    for epoch in range(1,cfg.max_epochs+1):
        model.train();opt.zero_grad(set_to_none=True);w,h=model.factors();objective=0.
        chunks=list(range(0,n,chunk_size))
        for index,begin in enumerate(chunks):
            sl=slice(begin,begin+chunk_size);u,i=data.train_users[sl],data.train_items[sl]
            target=model.native_targets(data.train_ratings_raw[sl],u,i)
            squared=((w[u]*h[i]).sum(-1)-target).square().sum()
            value=squared/(n if cfg.objective_reduction=='mean' else 1)
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite MBMF Adam objective')
            (.5*value).backward(retain_graph=index<len(chunks)-1);objective+=float(value.detach())
        _finite_gradients(model);opt.step()
        if epoch==1 or epoch%cfg.eval_every==0 or epoch==cfg.max_epochs:
            model.eval();score=float(pub.evaluate_publication_model(model,data,'val',metric_cfg=mc)['RMSE'])
            if not math.isfinite(score):raise FloatingPointError('Nonfinite MBMF Adam validation')
            history.append(dict(epoch=epoch,objective=objective,val_RMSE=score,lr=cfg.adam_lr))
            if score<best-1e-8:best,be,state,wait=score,epoch,_snapshot(model),0
            else:wait+=1
            if wait>=cfg.val_patience:break
    if state is None:raise RuntimeError('No finite MBMF Adam checkpoint')
    model.load_state_dict(state);model.eval()
    return dict(best_epoch=be,best_val_RMSE=best,final_epoch=epoch,history=history,
        stopping_reason='validation_patience' if epoch<cfg.max_epochs else 'epoch_budget',
        best_checkpoint_at_budget=be==cfg.max_epochs,optimizer='adam_common_protocol',
        checkpoint_policy=cfg.checkpoint_policy,elapsed_sec=time.perf_counter()-start)


def restore_mbmf(pub, core, payload, data, device, strict_data=True):
    """Read extended training metadata without passing it to frozen constructors."""
    if payload.get("frozen_v7_core_sha256") != pub.FROZEN_V7_CORE_SHA256:
        raise ValueError("Checkpoint scientific-core SHA mismatch")
    if strict_data:
        current = pub.data_signature(data)
        for name in ("num_users", "num_items", "rating_min", "rating_max", "train_sha256"):
            if str(payload["data_signature"].get(name)) != str(current[name]):
                raise ValueError("Checkpoint/data mismatch for "+name)
    allowed = {f.name for f in fields(core.PaperBaselineConfig)}
    cfg = core.PaperBaselineConfig(**{k: v for k, v in payload["config"].items() if k in allowed})
    model = pub._make_paper_baseline_v8(data, cfg, payload["variant"], payload["init_mode"], device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def train_autorec(pub, model, data, cfg, chunk_size=250_000):
    """Observed-only AutoRec objective, honoring cfg.lr for BOTH optimizers."""
    if cfg.optimizer not in ("rprop", "adam") or not math.isfinite(cfg.lr) or cfg.lr <= 0:
        raise ValueError("Invalid AutoRec optimizer or learning rate")
    if chunk_size < 1 or cfg.max_epochs < 1:
        raise ValueError("Invalid AutoRec training budget")
    optimizer = (torch.optim.Rprop if cfg.optimizer == "rprop" else torch.optim.Adam)(
        model.parameters(), lr=cfg.lr)
    n = len(data.train_users)
    start = time.perf_counter()
    best, best_epoch, state, wait = math.inf, 0, None, 0
    history = []
    metric = pub.MetricConfig(compute_testset_ndcg=False, compute_binary_ranking=False,
                            save_test_predictions=False, compute_pearson_spearman=False)
    for epoch in range(1, cfg.max_epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        hidden = model.item_hidden(data)
        slices = list(range(0, n, chunk_size))
        sse = 0.
        for index, begin in enumerate(slices):
            sl = slice(begin, begin+chunk_size)
            u, i, y = data.train_users[sl], data.train_items[sl], data.train_ratings_raw[sl]
            pred = (hidden[i]*model.decoder.weight[u]).sum(-1)+model.decoder.bias[u]
            squared = (pred-y).square().sum()
            loss = squared/n
            if index == 0:
                loss = loss+.5*cfg.lambda_l2/n*(model.encoder.weight.square().sum()+model.decoder.weight.square().sum())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite AutoRec training objective")
            loss.backward(retain_graph=index < len(slices)-1)
            sse += float(squared.detach())
        _finite_gradients(model)
        optimizer.step()
        if epoch == 1 or epoch % cfg.eval_every == 0 or epoch == cfg.max_epochs:
            model.eval()
            value = float(pub.evaluate_publication_model(model, data, "val", metric_cfg=metric)["RMSE"])
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite AutoRec selection RMSE")
            history.append(dict(epoch=epoch, loss=sse/n, val_RMSE=value))
            if value < best-1e-8:
                best, best_epoch, state, wait = value, epoch, _snapshot(model), 0
            else:
                wait += 1
                if wait >= cfg.val_patience:
                    break
    if state is None:
        raise RuntimeError("No finite AutoRec validation checkpoint")
    model.load_state_dict(state)
    model.eval()
    return dict(best_epoch=best_epoch, best_val_RMSE=best, history=history,
                last_epoch=epoch, optimizer=cfg.optimizer, effective_initial_lr=cfg.lr,
                best_checkpoint_at_budget=(best_epoch==cfg.max_epochs),
                stopping_reason="validation_patience" if epoch<cfg.max_epochs else "epoch_budget",
                elapsed_sec=time.perf_counter()-start)


def train_biasmf(pub,model,data,cfg):
    """Original MF/BiasMF minibatch objective, including the last validation tick."""
    if cfg.max_epochs<1 or cfg.batch_size<1 or cfg.eval_every<1:raise ValueError("Invalid MF budget")
    with torch.no_grad():model.mu.fill_(float(data.train_ratings_raw.mean()))
    opt=torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    n=data.train_users.numel();best=math.inf;state=None;wait=0;be=0;history=[];steps=0;start=time.perf_counter()
    metric=pub.MetricConfig(compute_testset_ndcg=False,compute_binary_ranking=False,save_test_predictions=False,compute_pearson_spearman=False)
    for epoch in range(1,cfg.max_epochs+1):
        model.train();perm=torch.randperm(n,device='cpu').to(data.train_users.device);total=0.
        for begin in range(0,n,cfg.batch_size):
            index=perm[begin:begin+cfg.batch_size]
            u,i,y=data.train_users[index],data.train_items[index],data.train_ratings_raw[index]
            opt.zero_grad(set_to_none=True)
            loss=(model.forward_pairs(u,i)-y).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite MF objective")
            loss.backward();_finite_gradients(model);opt.step();steps+=1
            total+=float(loss.detach())*len(index)
        if epoch==1 or epoch%cfg.eval_every==0 or epoch==cfg.max_epochs:
            model.eval();value=float(pub.evaluate_publication_model(model,data,"val",metric_cfg=metric)["RMSE"])
            if not math.isfinite(value):raise FloatingPointError("Nonfinite MF validation RMSE")
            history.append(dict(epoch=epoch,loss=total/n,val_RMSE=value,optimizer_steps=steps))
            if value<best-1e-8:best,be,state,wait=value,epoch,_snapshot(model),0
            else:wait+=1
            if wait>=cfg.val_patience:break
    if state is None:raise RuntimeError("No finite MF checkpoint")
    model.load_state_dict(state);model.eval()
    return dict(best_epoch=be,best_val_RMSE=best,history=history,last_epoch=epoch,optimizer_steps=steps,
        best_checkpoint_at_budget=be==cfg.max_epochs,
        stopping_reason="validation_patience" if epoch<cfg.max_epochs else "epoch_budget",elapsed_sec=time.perf_counter()-start)


def train_aemcne(pub,model,data,cfg,chunk_size=200_000):
    """Equation (5) objective with last-epoch selection and finite-gradient checks.

    This is a reimplementation; publication-equivalent performance is a separate
    empirical question. Main and element-wise penalties are applied once.
    """
    if chunk_size<1 or cfg.max_epochs<1:raise ValueError("Invalid AEMC-NE budget")
    opt=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    n=len(data.train_users);best=math.inf;be=0;state=None;wait=0;history=[];start=time.perf_counter()
    metric=pub.MetricConfig(compute_testset_ndcg=False,compute_binary_ranking=False,save_test_predictions=False,compute_pearson_spearman=False)
    for epoch in range(1,cfg.max_epochs+1):
        model.train();opt.zero_grad(set_to_none=True);hidden=model.item_hidden(data);sse=0.
        slices=list(range(0,n,chunk_size))
        for index,begin in enumerate(slices):
            sl=slice(begin,begin+chunk_size);u,i,y=data.train_users[sl],data.train_items[sl],data.train_ratings_raw[sl]
            z=(hidden[i]*model.decoder.weight[u]).sum(-1)+model.decoder.bias[u]
            squared=(model._h(z)-y).square().sum();loss=squared/n
            if index==0:
                main=model.encoder.weight.square().sum()+model.decoder.weight.square().sum()
                element=model.element1.weight.square().sum()+model.element2.weight.square().sum()
                loss=loss+(cfg.lambda_main*main+cfg.lambda_element*element)/n
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite AEMC-NE objective")
            loss.backward(retain_graph=index<len(slices)-1);sse+=float(squared.detach())
        _finite_gradients(model);opt.step()
        if epoch==1 or epoch%cfg.eval_every==0 or epoch==cfg.max_epochs:
            model.eval();value=float(pub.evaluate_publication_model(model,data,"val",metric_cfg=metric)["RMSE"])
            if not math.isfinite(value):raise FloatingPointError("Nonfinite AEMC-NE validation RMSE")
            history.append(dict(epoch=epoch,loss=sse/n,val_RMSE=value))
            if value<best-1e-8:best,be,state,wait=value,epoch,_snapshot(model),0
            else:wait+=1
            if wait>=cfg.val_patience:break
    if state is None:raise RuntimeError("No finite AEMC-NE checkpoint")
    model.load_state_dict(state);model.eval()
    return dict(best_epoch=be,best_val_RMSE=best,history=history,last_epoch=epoch,
        best_checkpoint_at_budget=be==cfg.max_epochs,
        stopping_reason="validation_patience" if epoch<cfg.max_epochs else "epoch_budget",elapsed_sec=time.perf_counter()-start)
