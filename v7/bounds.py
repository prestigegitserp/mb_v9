"""Structural support of the point predictor; these are not rating CIs."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from .models import Method1,Method2,Method3,Method4,Method6,Method7
from .numerics import smooth_gain


@torch.no_grad()
def structural_bounds(model,data,u,i):
    if hasattr(model, "inner"):
        if bool(model.response_fitted) and float(model.response_scale) != 1.:
            return np.full(tuple(u.shape), model.lo), np.full(tuple(u.shape), model.hi)
        if model.cfg.lineage == "v3":
            return np.full(tuple(u.shape), model.lo), np.full(tuple(u.shape), model.hi)
        from importlib import import_module
        return import_module(model.cfg.lineage + ".bounds").structural_bounds(model.inner, data, u, i)
    if bool(model.response_fitted) and float(model.response_scale) != 1.:
        return np.full(tuple(u.shape), model.lo), np.full(tuple(u.shape), model.hi)
    if model.cfg.readout_mode != "geometric":
        return np.full(tuple(u.shape),model.lo),np.full(tuple(u.shape),model.hi)
    if len(u)>model.cfg.batch_size:
        parts=[structural_bounds(model,data,u[s:s+model.cfg.batch_size],i[s:s+model.cfg.batch_size])
               for s in range(0,len(u),model.cfg.batch_size)]
        return tuple(np.concatenate([p[j] for p in parts]) for j in range(2))
    h,state=model.eval_state()
    def interval(base,gain,up=1.,down=1.):
        return base-gain*down*(base-model.lo),base+gain*up*(model.hi-base)
    if isinstance(model,(Method1,Method3)):
        _,extra=model.predict(h,state,u,i);lower,upper=interval(model.base(h,u,i),extra["gain"])
    elif isinstance(model,Method2):
        _,extra=model.predict(h,state,u,i);gate=extra["mix_gate"];base=model.base(h,u,i)
        g1=smooth_gain(state["ru"][u],F.softplus(model.counter_item(i)).squeeze(-1)+1e-4,model.half,model.gain_offset)
        g2=smooth_gain(F.softplus(model.counter_user(u)).squeeze(-1)+1e-4,state["ri"][i],model.half,model.gain_offset)
        a,b=interval(base,g1),interval(base,g2)
        lower,upper=gate*a[0]+(1-gate)*b[0],gate*a[1]+(1-gate)*b[1]
    elif isinstance(model,Method6):
        base=model.base(h,u,i);parts=[]
        for view,gain in enumerate(model._gains(state,u,i)):
            if model.cfg.transport_enabled:
                group=model._reference_groups(h,u,i,(base-model.lo)/model.width,gain)
                scales=state["transport_balance"][view,group]
                parts.append(interval(base,gain,scales[:,0],scales[:,1]))
            else:parts.append(interval(base,gain))
        gate=torch.sigmoid(model.mix)
        lower=gate*parts[0][0]+(1-gate)*parts[1][0];upper=gate*parts[0][1]+(1-gate)*parts[1][1]
    elif isinstance(model,Method7):
        lows,highs=[],[]
        support=(h.user_counts[u]/(h.user_counts[u]+model.cfg.count_tau)*h.item_counts[i]/(h.item_counts[i]+model.cfg.count_tau)).sqrt()
        for member,s in zip(model.members,state["members"]):
            base=member.bias(u,i,h.user_counts,h.item_counts)
            gain=torch.sigmoid(member.trust)*s["ru"][u]*s["ri"][i]/model.half*support
            a,b=interval(base,gain);eps=model.cfg.mean_interior_epsilon
            lows.append(model.lo+model.width*(eps+(1-2*eps)*(a-model.lo)/model.width))
            highs.append(model.lo+model.width*(eps+(1-2*eps)*(b-model.lo)/model.width))
        lower,upper=torch.stack(lows).mean(0),torch.stack(highs).mean(0)
    else:
        if isinstance(model,Method4) and model.cfg.moment_refinement and not model.cfg.project_moments and model.cfg.moment_refinement_link=="linear":
            return np.full(tuple(u.shape),-np.inf),np.full(tuple(u.shape),np.inf)
        margin=model.width*model.cfg.moment_refinement_scale if isinstance(model,Method4) and model.cfg.signed_moments and model.cfg.moment_refinement and not model.cfg.project_moments else 0.
        lower=data.train_ratings_raw.new_full(u.shape,model.lo-margin);upper=data.train_ratings_raw.new_full(u.shape,model.hi+margin)
    return lower.cpu().numpy(),upper.cpu().numpy()
