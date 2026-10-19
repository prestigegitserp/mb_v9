"""Train-only, target-excluded histories; no dense U x I rating matrix is needed.

All rating-derived features, priors, timestamp extrema, histogram counts and
encoder inputs are rebuilt from the same context. Missing is never rating zero.
This excludes target *inputs*, not the usual influence of training labels on
learned parameters. Genuine OOF model predictions are produced separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class History:
    u: torch.Tensor
    i: torch.Tensor
    y: torch.Tensor
    user_features: torch.Tensor
    item_features: torch.Tensor
    user_counts: torch.Tensor
    item_counts: torch.Tensor
    user_amplitude: torch.Tensor
    item_amplitude: torch.Tensor
    edge_indices: torch.Tensor


def scatter_sum(x, ids, size):
    shape = (size,) + tuple(x.shape[1:])
    return x.new_zeros(shape).index_add(0, ids, x)


def entity_statistics(ids, y, size, lo, hi, times=None, residual=None, bins=5,counts=None):
    n = (torch.bincount(ids, minlength=size) if counts is None else counts).to(y.dtype)
    safe = n.clamp_min(1)
    t = (y-lo)/(hi-lo)
    gm = t.mean() if t.numel() else t.new_tensor(0.5)
    gs = t.square().mean() if t.numel() else t.new_tensor(0.25)
    mean = scatter_sum(t, ids, size)/safe
    second = scatter_sum(t.square(), ids, size)/safe
    mean = torch.where(n > 0, mean, gm)
    second = torch.where(n > 0, second, gs)
    sd = (second-mean.square()).clamp_min(0).sqrt()
    minimum = t.new_full((size,), float("inf"))
    maximum = t.new_full((size,), -float("inf"))
    minimum.scatter_reduce_(0, ids, t, reduce="amin", include_self=True)
    maximum.scatter_reduce_(0, ids, t, reduce="amax", include_self=True)
    minimum = torch.where(n > 0, minimum, gm)
    maximum = torch.where(n > 0, maximum, gm)
    levels = (t*bins).long().clamp(0, bins-1)
    hist = scatter_sum(F.one_hot(levels, bins).to(y.dtype), ids, size)/safe[:, None]
    global_hist = hist.new_full((bins,), 1/bins)
    if t.numel():
        global_hist = F.one_hot(levels, bins).to(y.dtype).mean(0)
    hist = torch.where((n > 0)[:, None], hist, global_hist[None])
    latest = torch.zeros_like(n)
    span = torch.zeros_like(n)
    time_present = torch.zeros_like(n)
    if times is not None and times.numel() and torch.isfinite(times).all():
        scale = (times.max()-times.min()).clamp_min(1)
        tt = (times-times.min())/scale
        first = tt.new_full((size,), float("inf"))
        last = tt.new_full((size,), -float("inf"))
        first.scatter_reduce_(0, ids, tt, reduce="amin", include_self=True)
        last.scatter_reduce_(0, ids, tt, reduce="amax", include_self=True)
        # Preserve timestamp precision while normalizing, then explicitly match
        # rating/encoder dtype. Unix timestamps may arrive as float64 or int64.
        latest = torch.where(n > 0, last, latest).to(y.dtype)
        span = torch.where(n > 0, last-first, span).to(y.dtype)
        time_present = (n > 0).to(n.dtype)
    residual = t-gm if residual is None else residual
    global_second = residual.square().mean() if residual.numel() else t.new_tensor(0.05)
    local_second = scatter_sum(residual.square(), ids, size)/safe
    trust = n/(n+10)
    amplitude = ((trust*local_second+(1-trust)*global_second)
                 /global_second.clamp_min(1e-4)).clamp(0.05, 4).sqrt()
    features = torch.cat([torch.stack([
        n.log1p()/10, mean, sd, minimum, maximum, latest, span, time_present
    ], -1), hist], -1)
    return features, n, amplitude


def make_history(data, include=None, residual=None,counts=None):
    u, i, y = data.train_users, data.train_items, data.train_ratings_raw
    times = getattr(data, "train_timestamps", None)
    edge_indices = torch.arange(len(u), device=u.device)
    if include is not None:
        u, i, y = u[include], i[include], y[include]
        edge_indices = edge_indices[include]
        if times is not None:
            times = times[include]
        if residual is not None:
            residual = residual[include]
    lo, hi = float(data.rating_min), float(data.rating_max)
    uf, un, ua = entity_statistics(u, y, data.num_users, lo, hi, times, residual,counts=None if counts is None else counts[0])
    vf, vn, va = entity_statistics(i, y, data.num_items, lo, hi, times, residual,counts=None if counts is None else counts[1])
    return History(u, i, y, uf, vf, un, vn, ua, va, edge_indices)


def masked_targets(data, fraction, generator):
    """Exclude all copies of a target pair if the input contains duplicate edges."""
    pairs = data.train_users*int(data.num_items)+data.train_items
    unique, inverse = torch.unique(pairs, return_inverse=True)
    if unique.numel() < 2:
        raise ValueError("Masked training needs at least two distinct observed pairs")
    choose = torch.rand(unique.numel(), generator=generator, device="cpu") < fraction
    if not choose.any():
        choose[0] = True
    if choose.all():
        choose[-1] = False
    query = choose.to(pairs.device)[inverse]
    return ~query, query


class MaskSampler:
    """Cache immutable pair grouping, preserving the original mask RNG exactly."""
    def __init__(self,data,fraction,generator,jitter=0.0):
        pairs=data.train_users*int(data.num_items)+data.train_items
        unique,self.inverse=torch.unique(pairs,return_inverse=True)
        self.size=len(unique);self.device=pairs.device
        self.fraction,self.generator=fraction,generator
        self.jitter=float(jitter)
        if not 0 <= self.jitter < min(fraction,1-fraction):
            raise ValueError("Mask jitter must preserve nonempty context/query probabilities")
        if self.size<2:raise ValueError("Masked training needs two unique pairs")

    def sample(self):
        fraction=self.fraction
        if self.jitter:
            fraction += self.jitter*(2*float(torch.rand((),generator=self.generator))-1)
        self.last_fraction=fraction
        choose=torch.rand(self.size,generator=self.generator,device="cpu")<fraction
        if not choose.any():choose[0]=True
        if choose.all():choose[-1]=False
        query=choose.to(self.device)[self.inverse]
        return ~query,query


class HistoryEncoder(nn.Module):
    """Sparse equivalent of value+mask first-layer encoding, with fixed hidden width."""
    def __init__(self, inputs, hidden, k):
        super().__init__()
        self.value = nn.Embedding(inputs, hidden)
        self.mask = nn.Embedding(inputs, hidden)
        self.network = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(),
                                     nn.Linear(hidden, k))
        nn.init.normal_(self.value.weight, std=0.03)
        nn.init.normal_(self.mask.weight, std=0.03)

    def forward(self, owners, counterparts, residual, count, entities):
        values = self.value(counterparts)*residual[:, None]+self.mask(counterparts)
        pooled = scatter_sum(values, owners, entities)/count.clamp_min(1).sqrt()[:, None]
        return self.network(pooled)
