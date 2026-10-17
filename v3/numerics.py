"""Small differentiable geometric and matrix-free inference kernels."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def unit(x):
    return F.normalize(x, p=2, dim=-1, eps=1e-8)


def sphere_exp(q, v, max_angle=0.5):
    q = unit(q)
    v = v-(q*v).sum(-1, keepdim=True)*q
    length = v.norm(dim=-1, keepdim=True)
    v = v*(max_angle/length.clamp_min(max_angle)).clamp_max(1)
    length = v.norm(dim=-1, keepdim=True)
    sinc = torch.sinc(length/math.pi)
    return unit(torch.cos(length)*q+sinc*v)


def bounded_residual(base, cosine, gain, lo, hi):
    c = cosine.clamp(-1, 1)
    g = gain.clamp(0, 1)
    return base+g*c*torch.where(c >= 0, hi-base, base-lo)


def smooth_gain(ru, ri, half, offset):
    log_product = (ru.clamp_min(1e-8).log()+ri.clamp_min(1e-8).log()
                   -math.log(half))
    return torch.sigmoid(log_product+offset)


def spectral_filter(hessian, proposal, ridge):
    eye = torch.eye(hessian.shape[-1], device=hessian.device, dtype=hessian.dtype)
    return torch.linalg.solve(hessian+ridge*eye,
                              (hessian@proposal.unsqueeze(-1))).squeeze(-1)


def conjugate_gradient(operator, rhs, steps=25, tolerance=1e-6):
    """SPD solve. Stops safely on a zero RHS; returns a diagnostic residual."""
    x = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    rr = (residual*residual).sum()
    initial = rr.sqrt()
    if initial <= 1e-12:
        return x, 0.0
    for _ in range(steps):
        ad = operator(direction)
        denom = (direction*ad).sum()
        if not torch.isfinite(denom) or denom <= 0:
            raise FloatingPointError("Non-SPD or nonfinite conjugate-gradient operator")
        alpha = rr/denom
        x = x+alpha*direction
        residual = residual-alpha*ad
        new_rr = residual.square().sum()
        if new_rr.sqrt() <= tolerance*initial:
            rr = new_rr
            break
        direction = residual+(new_rr/rr)*direction
        rr = new_rr
    return x, float((rr.sqrt()/initial).detach())


def signed_moment_projection(point, base, lo, hi, step):
    """Euclidean projection onto the exact feasible partial-moment polygon.

    A continuous bounded response has a triangle; an ordinal response has the
    quadrilateral between its extreme and nearest-neighbour rating levels.
    This projection preserves feasible moments, not necessarily point RMSE.
    """
    zero = torch.zeros_like(base)
    if step is None:
        lower = upper = base
    else:
        scaled = (base-lo)/float(step)
        lower = (lo+scaled.floor()*step).clamp(lo, hi)
        upper = (lo+scaled.ceil()*step).clamp(lo, hi)
    vertices = torch.stack([
        torch.stack([zero, base-lo], -1),
        torch.stack([hi-base, zero], -1),
        torch.stack([(upper-base).clamp_min(0), zero], -1),
        torch.stack([zero, (base-lower).clamp_min(0)], -1),
    ], -2)
    end = vertices.roll(-1, dims=-2)
    edge = end-vertices
    rel = point.unsqueeze(-2)-vertices
    cross = edge[..., 0]*rel[..., 1]-edge[..., 1]*rel[..., 0]
    inside = (cross <= 1e-6).all(-1)
    position = (rel*edge).sum(-1)/edge.square().sum(-1).clamp_min(1e-12)
    candidates = vertices+position.clamp(0, 1).unsqueeze(-1)*edge
    index = (candidates-point.unsqueeze(-2)).square().sum(-1).argmin(-1)
    nearest = candidates.gather(-2, index[..., None, None].expand(*index.shape, 1, 2)).squeeze(-2)
    return torch.where(inside[..., None], point, nearest)


def gaussian_bilinear_moments(mu, mi, lu, li, correlation=1.0):
    """Exact bilinear moments with Cov(z,d)=correlation*lu@li.T.

    lu/li have shape batch x K x rank. This is an explicitly low-rank joint
    Gaussian approximation, not an exact posterior for a nonlinear network.
    Correlation 1 shares the Gaussian draw; 0 makes the two marginals independent.
    """
    cross_mean = correlation*(lu*li).sum((-1, -2))
    mean = (mu*mi).sum(-1)+cross_mean
    a = torch.einsum("nkr,nk->nr", lu, mi)
    b = torch.einsum("nkr,nk->nr", li, mu)
    cross = torch.einsum("nkr,nks->nrs", lu, li)
    # A convex mixture of joint and independent covariance keeps both marginals
    # fixed and is PSD for every correlation in [0,1].
    variance = (a.square()+b.square()+2*correlation*a*b).sum(-1)
    variance = variance+cross.square().sum((-1, -2))+correlation**2*(cross*cross.transpose(-1, -2)).sum((-1, -2))
    return mean, variance, cross_mean
