"""Honest likelihood metrics and archives for the actual V6 distributions."""
from __future__ import annotations

import numpy as np
import torch
from .models import Method7
from .bounds import structural_bounds
from .laws import ordinal_nll


@torch.no_grad()
def distribution_metrics(model, data, split, chunk_size=4096):
    u, i, y = [getattr(data, f"{split}_{part}") for part in ("users", "items", "ratings_raw")]
    h, state = model.eval_state()
    collected, nll = {}, []
    for start in range(0, len(u), chunk_size):
        sl = slice(start, start+chunk_size)
        dist = model.distribution(data, u[sl], i[sl])
        for key in ("variance", "within_variance", "between_variance", "probabilities", "member_means", "member_probabilities"):
            if key in dist:
                collected.setdefault(key, []).append(dist[key].cpu().numpy())
        if "beta_parameters" in dist:
            for index,key in enumerate(("beta_alpha","beta_beta","beta_mixture_weight")):
                collected.setdefault(key,[]).append(torch.stack([parts[index] for parts in dist["beta_parameters"]],1).cpu().numpy())
        if model.ordinal:
            loss = ordinal_nll(dist["probabilities"], y[sl], model.levels)
        else:
            # Mean reduction is weighted by chunk size, including the last chunk.
            loss = model.probability_loss(h, state, u[sl], i[sl], y[sl]).repeat(len(u[sl]))
        nll.append(loss.cpu().numpy())
    extras = {k: np.concatenate(v) for k, v in collected.items()}
    likelihood = "Ordinal_NLL" if model.ordinal else "Beta_NLL_endpoint_clipped"
    metrics = {likelihood: float(np.concatenate(nll).mean()),
               "Probability_NLL": float(np.concatenate(nll).mean()),
               "PredictiveVariance": float(extras["variance"].mean()),
               "WithinMemberVariance": float(extras["within_variance"].mean()),
               "BetweenMemberVariance": float(extras["between_variance"].mean())}
    if model.ordinal:
        prob = extras["probabilities"].astype(np.float64)
        levels = model.levels.cpu().numpy()
        true = y.cpu().numpy()
        index = np.abs(true[:, None]-levels).argmin(-1)
        cdf = prob.cumsum(-1)
        metrics["RankedProbabilityScore"] = float(np.square(cdf[:, :-1]-(levels[None, :-1] >= true[:, None])).sum(-1).mean())
        onehot = np.eye(len(levels))[index]
        metrics["Brier_multiclass"] = float(np.square(prob-onehot).sum(-1).mean())
        for percent in (50, 80, 90, 95):
            tail = .5*(1-percent/100)
            lower = levels[(cdf < tail).sum(-1).clip(max=len(levels)-1)]
            upper = levels[(cdf < 1-tail).sum(-1).clip(max=len(levels)-1)]
            metrics[f"OrdinalCoverage_{percent}"] = float(((true >= lower) & (true <= upper)).mean())
            metrics[f"OrdinalIntervalWidth_{percent}"] = float((upper-lower).mean())
        extras["levels"] = levels
    else:
        metrics["LikelihoodEndpointClip"] = 1e-5
    return metrics, extras


def evaluator(pub, original, archive_cache):
    def evaluate(model, data, split, *, metric_cfg=None, chunk_size=None, return_predictions=False):
        archive_cache.clear()
        if not hasattr(model, "v6_method"):
            return original(model, data, split, metric_cfg=metric_cfg, chunk_size=chunk_size,
                            return_predictions=return_predictions)
        # Avoid computing shape laws during mean checkpoint selection. They are
        # evaluated only after fitting on their two disjoint validation roles.
        training_tick = getattr(model, "_v6_mean_selection", False)
        if training_tick:
            model.eval()
            u, i, y = pub._split_tensors(data, split)
            with torch.no_grad():
                p = pub._predict_raw_chunked(model, data, u, i, chunk_size or 65536)
            return {"RMSE": float((p-y).square().mean().sqrt()), "MAE": float((p-y).abs().mean())}
        metrics, pred = original(model, data, split, metric_cfg=metric_cfg,
                                 chunk_size=chunk_size, return_predictions=True)
        extras = {}
        if return_predictions:
            u, i, _ = pub._split_tensors(data, split)
            records = {}
            with torch.no_grad():
                h, state = model.eval_state()
                for start in range(0, len(u), model.cfg.batch_size):
                    sl = slice(start, start+model.cfg.batch_size)
                    _, parts = model.predict(h, state, u[sl], i[sl])
                    for key, value in parts.items():
                        if torch.is_tensor(value) and value.ndim == 1:
                            records.setdefault(key, []).append(value.cpu().numpy())
            extras.update({k: np.concatenate(v) for k, v in records.items()})
            lower,upper=structural_bounds(model,data,u,i)
            extras.update(structural_lower_bound=lower,structural_upper_bound=upper)
        if model.v6_method == "Method7" and bool(model.shape_fitted):
            more, probability_extras = distribution_metrics(model, data, split, model.cfg.batch_size)
            extras.update(probability_extras)
            metrics.update(more)
        # The save helper has no model argument; retain only the latest exact
        # evaluation, keyed and checked against its prediction vector.
        if return_predictions:
            archive_cache.update(data=id(data), split=split, prediction=pred.copy(), extras=extras)
        return (metrics, pred) if return_predictions else metrics
    return evaluate
