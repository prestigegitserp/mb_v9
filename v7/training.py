"""Masked mean training, honest OOF routing and disjoint probability fitting.

No function in this module reads test tensors. Model selection uses val_* only;
probability fitting uses separately registered calibration/holdout tensors.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time
import json
from pathlib import Path
from .telemetry import TrainingTrace
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn
from . import __version__

from .history import masked_targets, MaskSampler
from .models import instantiate, Method1, Method2, Method3, Method4, Method6, Method7


def _clip_gradients(parameters, max_norm):
    """Keep the same clipping math; classify numerical failure for CPU replay."""
    try:
        return nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    except RuntimeError as exc:
        if 'non-finite' in str(exc).lower() or 'nonfinite' in str(exc).lower():
            raise FloatingPointError('Nonfinite gradient norm; optimizer update was not applied') from exc
        raise


def optimizer_groups(model, total_steps):
    cfg = model.cfg
    total_steps = min(total_steps, cfg.encoder_decay_horizon_epochs * cfg.steps_per_epoch)
    decay = -math.expm1(math.log1p(-cfg.encoder_decay_budget)/max(total_steps, 1))/cfg.lr
    groups = {"encoder": [], "direction": [], "bias": [], "other": []}
    for name, p in model.named_parameters():
        if not p.requires_grad or name == "delta" or name.startswith(("router.", "shape_heads.")):
            continue
        if "direction" in name:
            groups["direction"].append(p)
        elif "bias" in name:
            groups["bias"].append(p)
        elif "encoder" in name and p.ndim > 1:
            groups["encoder"].append(p)
        else:
            groups["other"].append(p)
    out = []
    for name, params in groups.items():
        if params:
            rate = cfg.direction_lr if name == "direction" else cfg.bias_lr if name == "bias" else cfg.lr
            out.append(dict(params=params, lr=rate, weight_decay=decay if name == "encoder" else 0., name=name))
    return out, decay


def _warm_bias(model):
    blocks = [member.bias for member in model.members] if isinstance(model, Method7) else [model.bias]
    data = model._data
    u, i, y = data.train_users, data.train_items, data.train_ratings_raw
    un = torch.bincount(u, minlength=model.nu).float()
    vn = torch.bincount(i, minlength=model.ni).float()
    params = [p for block in blocks for p in block.parameters() if p.requires_grad]
    if model.cfg.warmup_steps and params:
        opt = torch.optim.Adam(params, lr=0.02)
        for _ in range(model.cfg.warmup_steps):
            opt.zero_grad(set_to_none=True)
            loss = sum(((block(u, i, un, vn)-y)/model.width).square().mean() for block in blocks)/len(blocks)
            loss.backward()
            opt.step()
    # The partial-moment population identity assumes a fixed baseline. It stays
    # fixed after warmup in M4; its labels are never reintroduced as input features.
    if isinstance(model, Method4):
        model.bias.requires_grad_(False)
    model.clear_cache()


def _shared_encoder_parameters(model):
    return [p for name, p in model.named_parameters()
            if p.requires_grad and (name.startswith("user_encoder.") or name.startswith("item_encoder."))]


def _backward(model, prediction, auxiliary):
    cfg = model.cfg
    shared = _shared_encoder_parameters(model) if isinstance(model, Method1) else []
    surgery = cfg.gradient_surgery and bool(shared) and auxiliary.requires_grad
    gp = gr = None
    scale = 1.0
    conflict = False
    if surgery:
        gp = torch.autograd.grad(prediction, shared, retain_graph=True, allow_unused=True)
        gr = torch.autograd.grad(auxiliary, shared, retain_graph=True, allow_unused=True)
        pnorm = sum(g.square().sum() for g in gp if g is not None)
        rnorm = sum(g.square().sum() for g in gr if g is not None)
        if float(rnorm) > 0:
            scale = min(1., cfg.auxiliary_gradient_ratio*math.sqrt(float(pnorm)/(float(rnorm)+1e-30)))
        dot = sum((a*b).sum() for a, b in zip(gp, gr) if a is not None and b is not None)
        conflict = float(dot) < 0 and float(pnorm) > 1e-20
    # The gradient budget concerns only the shared encoder coordinates.
    # Decoder reconstruction and other priors keep their declared loss weights.
    loss = prediction+auxiliary
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite V7 training objective")
    loss.backward()
    if surgery:
        coefficient = scale*dot/pnorm if conflict else 0.
        for p, a, b in zip(shared, gp, gr):
            if a is None and b is None:
                continue
            gradient = torch.zeros_like(p)
            if a is not None:
                gradient.add_(a)
            if b is not None:
                gradient.add_(b, alpha=scale)
            if conflict and a is not None:
                gradient.sub_(coefficient*a)
            p.grad = gradient
    # This is an Euclidean gradient-interference filter. Adam momentum and weight
    # decay prevent a claimed guarantee of descent for the actual finite update.
    return float(loss.detach()), scale, float(conflict)


def train_mean(pub, model, data):
    with TrainingTrace(model) as trace:
        return _train_mean(pub,model,data,trace)


def _train_mean(pub, model, data, trace):
    cfg = model.cfg
    start = time.perf_counter()
    from .recovery import EpochRecovery
    recovery=EpochRecovery(model)
    if recovery.payload is None: _warm_bias(model)
    groups, decay = optimizer_groups(model, cfg.max_epochs*cfg.steps_per_epoch)
    opt = torch.optim.AdamW(groups)
    active = [p for group in groups for p in group["params"]]
    mc = pub.MetricConfig(compute_testset_ndcg=False, compute_binary_ranking=False,
                          save_test_predictions=False, compute_pearson_spearman=False)
    rng = torch.Generator().manual_seed(model.model_seed+48611)
    sampler=MaskSampler(data,cfg.mask_fraction,rng,cfg.mask_fraction_jitter)
    best, best_epoch, best_state, wait = float("inf"), 0, None, 0
    history = []
    target_count = 0
    full_target_count = 0
    step_count = 0
    radius_updates = 0
    recovered=recovery.restore(opt,rng)
    epoch=0
    if recovered:
        epoch=int(recovered["epoch"])
        best=recovered["best"]
        best_epoch=recovered["best_epoch"]
        best_state=recovered["best_state"]
        wait=recovered["wait"]
        history=recovered["history"]
        target_count=recovered["target_count"]
        step_count=recovered["step_count"]
        radius_updates=recovered["radius_updates"]
        full_target_count=recovered["full_target_count"]
    start_epoch=epoch+1 if wait<cfg.val_patience else cfg.max_epochs+1
    for epoch in range(start_epoch, cfg.max_epochs+1):
        records = []
        for _ in range(cfg.steps_per_epoch):
            model.train()
            include, query = sampler.sample()
            h = model.history(include)
            state = model.prepare(h)
            u, i, y = data.train_users[query], data.train_items[query], data.train_ratings_raw[query]
            model.zero_grad(set_to_none=True)
            lp, la, parts = model.loss_for_targets(h, state, u, i, y)
            parts["L_masked_prediction"]=float(lp.detach())
            lp=(1-cfg.full_history_weight)*lp
            if isinstance(model, Method2) and cfg.consistency_weight:
                thinner = include & (torch.rand(len(include), generator=rng).to(include.device) >= .15)
                if not thinner.any():
                    thinner[include.nonzero()[0, 0]] = True
                h2 = model.history(thinner)
                s2 = model.prepare(h2)
                a, b = model.experts(h, state, u, i)
                c, d = model.experts(h2, s2, u, i)
                # The more informative context is the stopped teacher; the
                # thinned context learns robustness within its own view.
                consistency=cfg.consistency_weight*.5*(((c-a.detach())/model.width).square().mean()
                                                      +((d-b.detach())/model.width).square().mean())
                la = la+consistency
                parts["L_consistency"]=float(consistency.detach())
                parts["L_auxiliary"]=float(la.detach())
            loss, aux_scale, conflict = _backward(model, lp, la)
            if cfg.full_history_weight:
                # Separate backward releases the masked graph before context-
                # dependent transport/reference buffers are rebuilt. One
                # optimizer update accumulates both TRAIN objectives.
                model.clear_cache()
                full_h=model.history()
                full_state=model.prepare(full_h)
                full_pred=model.predict(full_h,full_state,data.train_users,data.train_items)[0]
                full_loss=((full_pred-data.train_ratings_raw)/model.width).square().mean()
                if isinstance(model, Method3) and cfg.prediction_loss_units == "gaussian":
                    full_loss = .5 * full_loss / cfg.radius_noise**2
                if not torch.isfinite(full_loss):raise FloatingPointError("Nonfinite full-history reconstruction")
                (cfg.full_history_weight*full_loss).backward()
                parts["L_full_observed_reconstruction"]=float(full_loss.detach())
                parts["L_pred"]=float(lp.detach())+cfg.full_history_weight*float(full_loss.detach())
                loss+=cfg.full_history_weight*float(full_loss.detach())
                full_target_count+=len(data.train_users)
            grad_norm=float(_clip_gradients(active,cfg.grad_clip or float("inf")))
            opt.step()
            model.clear_cache()
            step_count += 1
            model._step = step_count
            target_count += len(y)
            if isinstance(model, Method3) and not cfg.context_radius and step_count % cfg.radius_refresh == 0:
                model.refresh_radius(probes=True)
                radius_updates += 1
            records.append(dict(loss=loss, auxiliary_gradient_scale=aux_scale,
                                gradient_conflict=conflict, **parts))
            trace.append(epoch=epoch,optimizer_step=step_count,prediction_targets_seen=target_count,
                context_pairs=len(h.y),query_pairs=len(y),mask_probability=sampler.last_fraction,
                full_observed_targets_seen=full_target_count,full_history_weight=cfg.full_history_weight,
                gradient_norm_before_clip=grad_norm,**records[-1])
        if epoch == 1 or epoch % cfg.eval_every == 0 or epoch == cfg.max_epochs:
            model.eval()
            if isinstance(model, Method3) and not cfg.context_radius:
                model.refresh_radius(probes=True)
                radius_updates += 1
            if isinstance(model, Method6):
                model.refresh_reference()
            model._v7_eval_epoch = epoch
            model._v7_mean_selection = True
            try:
                metrics = pub.evaluate_publication_model(model, data, "val", metric_cfg=mc)
            finally:
                model._v7_mean_selection = False
            value = float(metrics["RMSE"])
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite V7 validation RMSE")
            record = {"epoch": epoch, "optimizer_steps": step_count,
                      "prediction_targets_seen": target_count, "val_RMSE": value,
                      "radius_block_updates": radius_updates,"elapsed_seconds":time.perf_counter()-start}
            logged=getattr(model,"_v7_last_train_rmse",None)
            if logged is not None and logged[0]==epoch:record["train_RMSE"]=logged[1]
            for key in records[0]:
                record[key] = sum(r[key] for r in records)/len(records)
            history.append(record)
            if value < best-1e-8:
                best, best_epoch, wait = value, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                # Persist before acting on the patience boundary.
                pass
        recovery.save(model,opt,rng,dict(epoch=epoch,best=best,best_epoch=best_epoch,best_state=best_state,wait=wait,history=history,target_count=target_count,step_count=step_count,radius_updates=radius_updates,full_target_count=full_target_count),
                      force=(wait>=cfg.val_patience or epoch==cfg.max_epochs))
        if wait>=cfg.val_patience: break
    if best_state is None:
        raise RuntimeError("No valid V7 checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return dict(best_epoch=best_epoch, best_val_RMSE=best, history=history,
                last_epoch=epoch, stopping_reason="validation_patience" if epoch < cfg.max_epochs else "epoch_budget",
                best_checkpoint_at_budget=(best_epoch == cfg.max_epochs),
                mean_optimizer_steps=step_count, prediction_targets_seen=target_count,
                full_observed_targets_seen=full_target_count,
                objective="masked query MSE plus labelled full-TRAIN reconstruction" if cfg.full_history_weight else "masked query MSE",
                bias_warmup_steps=cfg.warmup_steps,
                radius_block_updates=radius_updates,
                radius_block_targets_seen=radius_updates*len(data.train_users),
                encoder_weight_decay=decay,
                realized_encoder_direct_shrinkage=1-(1-cfg.lr*decay)**step_count,
                elapsed_sec=time.perf_counter()-start)


def _training_subset(data, keep):
    out = copy.copy(data)
    for name in ("train_users", "train_items", "train_ratings_raw", "train_timestamps"):
        value = getattr(data, name, None)
        if value is not None:
            setattr(out, name, value[keep])
    return out


def fit_router(pub, model, data):
    cfg = model.cfg
    n = len(data.train_users)
    pairs = data.train_users*model.ni+data.train_items
    unique, inverse = torch.unique(pairs, return_inverse=True)
    folds = min(cfg.router_folds, len(unique))
    if folds < 2:
        raise ValueError("Cannot construct genuine OOF routing")
    rng = torch.Generator().manual_seed(model.model_seed+13007)
    assignment = torch.randperm(len(unique), generator=rng).to(pairs.device) % folds
    fold_index = assignment[inverse]
    oof = data.train_ratings_raw.new_empty(n, 2)
    features = data.train_ratings_raw.new_empty(n, model.router.center.numel())
    total_steps = 0
    for fold in range(folds):
        held = fold_index == fold
        subset = _training_subset(data, ~held)
        fcfg = replace(cfg, router_enabled=False,
                       max_epochs=min(cfg.max_epochs, cfg.router_expert_epochs))
        expert = instantiate(subset, fcfg, model.model_seed+1009*(fold+1), pairs.device)
        print(f"[V7 OOF] Method2 fold {fold+1}/{folds}; held out {int(held.sum())} pairs", flush=True)
        if hasattr(model,"_v7_artifact_dir"):
            expert._v7_artifact_dir=Path(model._v7_artifact_dir)/f"oof_fold_{fold}"
        expert._v7_resume=getattr(model,"_v7_resume",False)
        expert._v7_recovery_interval=getattr(model,"_v7_recovery_interval",5)
        info = train_mean(pub, expert, subset)
        if hasattr(expert,"_v7_artifact_dir"):
            (expert._v7_artifact_dir/"history.json").write_text(json.dumps(info,indent=2))
        total_steps += info["mean_optimizer_steps"]+info["bias_warmup_steps"]
        expert.eval()
        with torch.no_grad():
            h, s = expert.eval_state()
            u, i = data.train_users[held], data.train_items[held]
            p1, p2 = expert.experts(h, s, u, i)
            oof[held] = torch.stack([p1, p2], -1)
            features[held] = expert.routing_features(h, u, i, p1, p2)
        del expert
    model.eval()
    with torch.no_grad():
        h, s = model.eval_state()
        vp1, vp2 = model.experts(h, s, data.val_users, data.val_items)
        difference = oof[:, 0]-oof[:, 1]
        denom = float(difference.square().sum())
        optimal = .5 if denom < 1e-12 else float(((data.train_ratings_raw-oof[:, 1])*difference).sum())/denom
        candidates = [float(torch.sigmoid(model.mix)), min(max(optimal, 0.), 1.), .5]
        def final_mse(prediction):
            point = model.finish_readout(h, s, data.val_users, data.val_items, prediction, {})[0]
            return float((point-data.val_ratings_raw).double().square().mean())
        lambda0 = min(candidates, key=lambda g: final_mse(g*vp1+(1-g)*vp2))
        model.router.lambda0.fill_(lambda0)
        p0 = lambda0*oof[:, 0]+(1-lambda0)*oof[:, 1]
        target = (data.train_ratings_raw-p0)*difference
        # Hold out users for router fitting diagnostics; no IDs enter its inputs.
        users = torch.randperm(model.nu, generator=rng).to(pairs.device)
        hold_users = users[:max(1, model.nu//5)]
        hold = torch.isin(data.train_users, hold_users)
        if hold.all() or not hold.any():
            hold = torch.arange(n, device=pairs.device) % 5 == 0
        fit = ~hold
        router = model.router
        router.center.copy_(features[fit].mean(0))
        router.scale.copy_(features[fit].std(0, unbiased=False).clamp_min(.05))
        router.target_scale.copy_(target[fit].std(unbiased=False).clamp_min(.01))
        router.curvature_scale.copy_(difference[fit].square().mean().clamp_min(.01))
        # This is a model-based sensitivity threshold, NOT a certified
        # conditional confidence interval. No distribution-free safety claim.
        router.noise_floor.copy_(target[fit].std(unbiased=False)/math.sqrt(max(1, int(fit.sum())/50)))
        if cfg.router_counts_only:
            router.target_scale.fill_(1.)
            router.noise_floor.zero_()
    router.requires_grad_(True)
    router_params=list(router.heads.parameters())+(list(router.curvature_heads.parameters()) if cfg.router_curvature and not cfg.router_counts_only else [])
    opt = torch.optim.AdamW(router_params, lr=cfg.lr, weight_decay=.001)
    best, best_state, wait = float("inf"), None, 0
    weights = torch.stack([torch.empty(int(fit.sum()), device=pairs.device).exponential_()
                           for _ in router.heads])
    weights /= weights.mean(-1, keepdim=True)
    xfit, xhold = features[fit], features[hold]
    optimizer_steps = 0
    router_history = []
    if cfg.router_counts_only:
        # Fit eta(counts) directly by OOF point risk. Unlike the full moment
        # router, its inference denominator must NOT use pair predictions.
        residual = data.train_ratings_raw-p0
        fit_target, hold_target = residual[fit], residual[hold]
        fit_difference, hold_difference = difference[fit], difference[hold]
        regression_scale = model.width
    else:
        fit_target, hold_target = target[fit], target[hold]
        fit_difference, hold_difference = torch.ones_like(fit_target), torch.ones_like(hold_target)
        regression_scale = router.target_scale
    for epoch in range(cfg.router_epochs):
        order = torch.randperm(len(xfit), device='cpu').to(pairs.device)
        for start in range(0, len(order), cfg.batch_size):
            index = order[start:start+cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            prediction = router.moments(xfit[index])
            loss = (weights[:, index]*((prediction*fit_difference[index]-fit_target[index])/regression_scale).square()).mean()
            if cfg.router_curvature and not cfg.router_counts_only:
                curvature=router.curvatures(xfit[index]); target2=difference[fit][index].square()
                loss=loss+(weights[:,index]*((curvature-target2)/router.curvature_scale).square()).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite routing moment regression")
            loss.backward()
            _clip_gradients(router_params, cfg.grad_clip or float("inf"))
            opt.step()
            optimizer_steps += 1
        with torch.no_grad():
            total = 0.
            for start in range(0, len(xhold), cfg.batch_size):
                sl = slice(start, start+cfg.batch_size)
                total += float(((router.moments(xhold[sl]).mean(0)*hold_difference[sl]-hold_target[sl])/regression_scale).square().sum())
                if cfg.router_curvature and not cfg.router_counts_only:
                    total+=float(((router.curvatures(xhold[sl]).mean(0)-difference[hold][sl].square())/router.curvature_scale).square().sum())
            value = total/len(xhold)
        router_history.append(dict(epoch=epoch+1,objective=value,optimizer_steps=optimizer_steps))
        if value < best-1e-7:
            best, wait = value, 0
            best_state = copy.deepcopy(router.state_dict())
        else:
            wait += 1
            if wait >= 25:
                break
    if best_state is None:
        raise RuntimeError("No valid router checkpoint")
    router.load_state_dict(best_state)
    router.fitted.fill_(True)
    router.requires_grad_(False)
    with torch.no_grad():
        val_features = model.routing_features(h, data.val_users, data.val_items, vp1, vp2)
        scores = []
        for strength in (0., .25, .5, 1.):
            router.strength.fill_(strength)
            gate, _ = router(val_features, vp1-vp2)
            if cfg.router_support_shrinkage:
                support=(h.user_counts[data.val_users]/(h.user_counts[data.val_users]+cfg.count_tau)*h.item_counts[data.val_items]/(h.item_counts[data.val_items]+cfg.count_tau)).sqrt()
                gate=router.lambda0+support*(gate-router.lambda0)
            scores.append((final_mse(gate*vp1+(1-gate)*vp2), strength))
        selected = min(scores)
        router.strength.fill_(selected[1])
    model.clear_cache()
    if hasattr(model,"_v7_artifact_dir"):
        import pandas as pd
        root=Path(model._v7_artifact_dir);root.mkdir(parents=True,exist_ok=True)
        frame=pd.DataFrame(dict(pair_id=pairs.cpu().numpy(),fold=fold_index.cpu().numpy(),truth=data.train_ratings_raw.cpu().numpy(),
            user_view=oof[:,0].cpu().numpy(),item_view=oof[:,1].cpu().numpy(),benefit_target=target.cpu().numpy(),curvature_target=difference.square().cpu().numpy()))
        for j in range(features.shape[1]):frame[("pair" if cfg.router_context_only and j>=8 else "context")+f"_feature_{j}"]=features[:,j].cpu().numpy()
        frame.to_csv(root/"out_of_fold_risk_targets.csv.gz",index=False)
    return dict(folds=folds, oof_pairs=n, oof_expert_optimizer_steps=total_steps,
                router_optimizer_steps=optimizer_steps, router_history=router_history,
                selected_lambda0=lambda0, selected_router_strength=selected[1],
                selection_head="final_readout", selected_final_val_MSE=selected[0],
                benefit_conditioning="context+pair" if cfg.router_context_only and cfg.router_fine_benefit else "coarse/legacy",
                context_curvature_penalty=cfg.router_context_penalty,
                threshold_status=("count-only OOF point-risk control" if cfg.router_counts_only
                                  else "model-based, not a certified conditional interval"),
                final_refit_distribution_shift="OOF experts use smaller training contexts; strength selected on validation only")


def _mean_signature(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not name.startswith("shape_heads.") and name != "shape_fitted":
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fit_probability(model, data):
    required = [f"v7_calibration_{role}_{part}" for role in ("fit", "hold") for part in ("users", "items", "ratings")]
    if any(not hasattr(data, name) for name in required):
        raise ValueError("M7 needs disjoint calibration fit/hold roles; cannot reuse early-stopping validation")
    model.eval()
    before = _mean_signature(model)
    with torch.no_grad():h, state = model.eval_state()
    # Freeze all mean paths, including shared statistical features and encoders.
    def detached(value):
        if torch.is_tensor(value):return value.detach()
        if isinstance(value,dict):return {key:detached(part) for key,part in value.items()}
        if isinstance(value,list):return [detached(part) for part in value]
        raise TypeError('Unsupported frozen probability state')
    state=detached(state)
    heads = model.shape_heads
    heads.requires_grad_(True)
    opt = torch.optim.AdamW(heads.parameters(), lr=model.cfg.shape_lr, weight_decay=.001)
    fit = [getattr(data, f"v7_calibration_fit_{part}") for part in ("users", "items", "ratings")]
    hold = [getattr(data, f"v7_calibration_hold_{part}") for part in ("users", "items", "ratings")]
    best, best_state, wait = float("inf"), None, 0
    optimizer_steps = 0
    shape_history = []
    for epoch in range(model.cfg.shape_epochs):
        order = torch.randperm(len(fit[0]), device='cpu').to(fit[0].device)
        for start in range(0, len(order), model.cfg.batch_size):
            index = order[start:start+model.cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = model.probability_loss(h, state, *(part[index] for part in fit))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite probability objective")
            loss.backward()
            _clip_gradients(heads.parameters(), model.cfg.grad_clip or float("inf"))
            opt.step()
            optimizer_steps += 1
        with torch.no_grad():
            total = 0.
            for start in range(0, len(hold[0]), model.cfg.batch_size):
                sl = slice(start, start+model.cfg.batch_size)
                total += len(hold[0][sl])*float(model.probability_loss(h, state, *(part[sl] for part in hold)))
            value = total/len(hold[0])
        shape_history.append(dict(epoch=epoch+1,objective=value,optimizer_steps=optimizer_steps))
        if value < best-1e-7:
            best, wait = value, 0
            best_state = copy.deepcopy(heads.state_dict())
        else:
            wait += 1
            if wait >= model.cfg.shape_patience:
                break
    if best_state is None:
        raise RuntimeError("No finite probability checkpoint")
    heads.load_state_dict(best_state)
    heads.requires_grad_(False)
    model.shape_fitted.fill_(True)
    if _mean_signature(model) != before:
        raise AssertionError("Probability fitting changed a protected mean parameter")
    model.clear_cache()
    return dict(best_calibration_hold_objective=best, epochs=epoch+1, mean_parameters_unchanged=True,
                shape_history=shape_history,
                objective="ordinal NLL + weighted RPS" if model.ordinal else "continuous beta-mixture NLL",
                shape_optimizer_steps=optimizer_steps,
                law="ordinal_mean_fibre" if model.ordinal else "same-mean_beta_mixture_endpoint_clipped",
                calibration_fit_size=len(fit[0]), calibration_hold_size=len(hold[0]))


def train_v7(pub, model, data, config, study):
    devices = [data.train_users.device.index or 0] if data.train_users.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(model.model_seed+88169)
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(model.model_seed+88169)
        from .models import LegacyAdapter
        if isinstance(model, LegacyAdapter):
            from .lineage import train_legacy
            result = train_legacy(pub, model, data)
        else:
            result = train_mean(pub, model, data)
        if isinstance(model, Method2) and model.cfg.router_enabled:
            result["v7_routing"] = fit_router(pub, model, data)
        if isinstance(model, Method7):
            if model.cfg.probability_fit:
                result["v7_probability"] = fit_probability(model, data)
            else:
                result["v7_probability"] = dict(status="SKIPPED_POINT_PREDICTION_PROTOCOL",mean_parameters_unchanged=True,
                    calibration_fit_size=0,calibration_hold_size=0,probability_metrics_available=False)
        # A probability law's mean must remain fixed after shape fitting. The
        # current correction is therefore restricted to point-only predictors.
        if model.cfg.method != "Method7" or not model.cfg.probability_fit:
            from .response import fit_response
            result["v7_response"] = fit_response(model, data)
        else:
            result["v7_response"] = dict(status="DISABLED_FIXED_PROBABILITY_MEAN", selected_scale=1.)
    result["architecture_version"] = __version__
    return result
