"""A falsifiable decoder/geometry coupling, with its decoder-only null exposed.

The same target-excluded codes serve the geometry and a dual linear residual
decoder. This is an architectural candidate, not a novelty or accuracy claim.
No validation labels enter prediction. Support shrinkage uses context counts.
"""
import torch


def context_mean(model, history, state, users, items, geometric):
    cfg = model.cfg
    if cfg.readout_mode == "geometric":
        return geometric, {}
    if cfg.wide_context_readout:
        du,di=model.wide_user_decoder,model.wide_item_decoder
        zu,zi=history.wide_user_code,history.wide_item_code
    else:
        du,di=getattr(model,"user_decoder",None),getattr(model,"item_decoder",None)
        if du is None:du,di=model.context_user_decoder,model.context_item_decoder
        zu,zi=state["zu"],state["zi"]
    signal=.5*((zu[users]*du(items)).sum(-1)+(zi[items]*di(users)).sum(-1))
    if model.bias is None:
        base = torch.stack([member.bias(users, items, history.user_counts, history.item_counts)
                            for member in model.members]).mean(0)
    else:
        base = model.base(history, users, items)
    decoder = base+model.width*signal
    if cfg.readout_mode == "decoder":
        return decoder.clamp(model.lo,model.hi).expand_as(geometric), dict(context_readout_gate=torch.ones_like(signal),decoder_mean=decoder)
    else:
        tau = cfg.readout_count_tau
        support = ((history.user_counts[users]/(history.user_counts[users]+tau)
                   *history.item_counts[items]/(history.item_counts[items]+tau)).sqrt()
                   if cfg.readout_count_adaptive else torch.ones_like(signal))
        gate = cfg.readout_weight*support
    if cfg.readout_mode == "residual":
        # Decoder is the anchor; geometry supplies a support-bounded correction.
        # A zero readout weight is exactly the exposed decoder-only control.
        anchor=decoder.clamp(model.lo,model.hi)
        correction=torch.tanh((geometric-base)/model.half)
        slack=torch.where(correction>=0,model.hi-anchor,anchor-model.lo)
        return anchor+gate*slack*correction, dict(context_readout_gate=gate,decoder_mean=decoder,geometric_residual=correction)
    # Project AFTER combining, rather than saturating the decoder before the
    # objective sees it. M4 applies its full moment-cone projection afterwards.
    mean = ((1-gate)*geometric+gate*decoder).clamp(model.lo, model.hi)
    return mean, dict(context_readout_gate=gate, decoder_mean=decoder)
