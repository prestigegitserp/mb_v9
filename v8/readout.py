"""Target-excluded reliability coupling with an exact decoder-only null.

The six features contain only TRAIN-context support and predictions. Zero
initialization reproduces the former fixed gate without consuming random draws.
"""
import torch


def point_baseline(model, history, users, items):
    if model.bias is None:
        return torch.stack([member.bias(users, items, history.user_counts,
                                       history.item_counts)
                            for member in model.members]).mean(0)
    return model.base(history, users, items)


def reliability(model, history, users, items, base, geometric, decoder):
    if not model.cfg.adaptive_readout or not hasattr(model, 'readout_reliability'):
        return torch.ones_like(base)
    geometric = geometric.mean(0) if geometric.ndim > 1 else geometric
    tau = model.cfg.readout_count_tau
    cu, ci = history.user_counts[users], history.item_counts[items]
    su, si = cu/(cu+tau), ci/(ci+tau)
    dg = ((geometric-base)/model.width).detach()
    dd = ((decoder-base)/model.width).detach()
    features = torch.stack((torch.ones_like(base), su, si, dg, dd, dg*dd), -1).detach()
    return 2*torch.sigmoid(features @ model.readout_reliability)


def context_mean(model, history, state, users, items, geometric):
    cfg = model.cfg
    base = point_baseline(model,history,users,items)
    scale = model.geometry_scale
    if cfg.readout_mode == 'geometric':
        gate = reliability(model,history,users,items,base,geometric,base).clamp(max=1)*scale
        return base+gate*(geometric-base), dict(context_readout_gate=gate,
                            geometric_residual=(geometric-base)/model.width)
    if cfg.wide_context_readout:
        du,di=model.wide_user_decoder,model.wide_item_decoder
        zu,zi=history.wide_user_code,history.wide_item_code
    else:
        du,di=getattr(model,'user_decoder',None),getattr(model,'item_decoder',None)
        if du is None:du,di=model.context_user_decoder,model.context_item_decoder
        zu,zi=state['zu'],state['zi']
    from .aggregation import pair_dot
    chunk=getattr(cfg,'pair_dot_chunk',32768)
    signal=.5*(pair_dot(zu,du.weight,users,items,chunk)+pair_dot(zi,di.weight,items,users,chunk))
    decoder=base+model.width*signal
    anchor=decoder.clamp(model.lo,model.hi)
    if cfg.readout_mode == 'decoder':
        return anchor.expand_as(geometric), dict(context_readout_gate=torch.ones_like(signal),decoder_mean=decoder)
    tau=cfg.readout_count_tau
    support=((history.user_counts[users]/(history.user_counts[users]+tau)
              *history.item_counts[items]/(history.item_counts[items]+tau)).sqrt()
             if cfg.readout_count_adaptive else torch.ones_like(signal))
    gain=reliability(model,history,users,items,base,geometric,decoder)
    gate=(cfg.readout_weight*support*gain).clamp(0,1)
    if cfg.readout_mode == 'residual':
        correction=torch.tanh((geometric-base)/model.half)
        slack=torch.where(correction>=0,model.hi-anchor,anchor-model.lo)
        return anchor+scale*gate*slack*correction, dict(
            context_readout_gate=scale*gate,decoder_mean=decoder,geometric_residual=correction,
            readout_reliability=gain)
    coupled=((1-gate)*geometric+gate*decoder).clamp(model.lo,model.hi)
    return anchor+scale*(coupled-anchor), dict(context_readout_gate=gate,
                            decoder_mean=decoder,readout_reliability=gain)
