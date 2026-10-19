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
    du = getattr(model, "user_decoder", None)
    di = getattr(model, "item_decoder", None)
    if du is None:
        du, di = model.context_user_decoder, model.context_item_decoder
    signal = .5*((state["zu"][users]*du(items)).sum(-1)
                 +(state["zi"][items]*di(users)).sum(-1))
    if model.bias is None:
        base = torch.stack([member.bias(users, items, history.user_counts, history.item_counts)
                            for member in model.members]).mean(0)
    else:
        base = model.base(history, users, items)
    decoder = base+model.width*signal
    if cfg.readout_mode == "decoder":
        gate = torch.ones_like(signal)
    else:
        tau = cfg.readout_count_tau
        support = ((history.user_counts[users]/(history.user_counts[users]+tau)
                   *history.item_counts[items]/(history.item_counts[items]+tau)).sqrt()
                   if cfg.readout_count_adaptive else torch.ones_like(signal))
        gate = cfg.readout_weight*support
    # Project AFTER combining, rather than saturating the decoder before the
    # objective sees it. M4 applies its full moment-cone projection afterwards.
    mean = ((1-gate)*geometric+gate*decoder).clamp(model.lo, model.hi)
    return mean, dict(context_readout_gate=gate, decoder_mean=decoder)
