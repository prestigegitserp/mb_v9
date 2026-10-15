"""Range-safe, bias-centered spherical residuals with explicit guarantees."""
import torch
import torch.nn.functional as F


def unit(x, eps=1e-8):
    return F.normalize(x, p=2, dim=-1, eps=eps)


def cosine_pairs(u, i, users, items):
    return (unit(u)[users] * unit(i)[items]).sum(-1).clamp(-1., 1.)


def bounded_residual(base, score, gain, rating_min, rating_max):
    """For base in [min,max], score in [-1,1], gain in [0,1] -> valid rating.

    Unlike symmetric residual radius, negative and positive excursions use the
    available distance to each separate endpoint. No output clipping required.
    """
    up = rating_max - base
    down = base - rating_min
    return base + gain * score * torch.where(score >= 0., up, down)


def monotone_base(center_logit, u_bias, i_bias, users, items, rmin, rmax):
    return rmin + (rmax-rmin) * torch.sigmoid(
        center_logit + u_bias(users).squeeze(-1) + i_bias(items).squeeze(-1))


def radial_gain(r1, r2, reference, logit_gain, floor=.15):
    """A differentiable amplitude with nonzero floor that prevents dead angles.

    For nonnegative radii and reference, clip ratio without hard-clipping
    gradients in the small-radii regime using sigmoid-like ratio/(1+ratio).
    """
    ratio = (r1 * r2).clamp_min(0.) / (reference + 1e-6)
    soft = 2.0 * ratio / (1.0 + ratio)
    weight = torch.sigmoid(logit_gain)
    return floor + (1.0-floor) * weight * soft.clamp(max=1.0)


def gradient_cosine(a, b, eps=1e-12):
    x = torch.cat([p.reshape(-1) for p in a if p is not None]) if any(p is not None for p in a) else None
    y = torch.cat([p.reshape(-1) for p in b if p is not None]) if any(p is not None for p in b) else None
    if x is None or y is None or x.numel()!=y.numel():return float('nan')
    return float((x @ y/(x.norm()*y.norm()+eps)).detach())
