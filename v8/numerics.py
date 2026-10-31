"""Small differentiable geometric and matrix-free inference kernels."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def unit(x):
    return F.normalize(x, p=2, dim=-1, eps=1e-8)


def sphere_exp(q, v, max_angle=0.5):
    q = unit(q)
    if max_angle == 0:
        # Keep an explicit zero derivative to v, including a zero-information
        # cold entity. Multiplying v by 0/||v|| would be undefined at v=0.
        return q+v*0
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


def residual_balance(residual, gain, groups, count):
    """Bounded sign scales giving zero gain-weighted reference residual per cell.

    The larger aggregate side is reduced (both may shrink below the shared
    numerical floor). Both scales are in [0,1],
    retaining range bounds, angular order and the zero-residual anchor. A
    one-sided nonzero reference necessarily becomes flat on that side; an
    empty or identically zero cell retains identity scales. This is a finite
    reference identity, not an assertion about unseen rating calibration.
    """
    weighted = gain.double()*residual.double()
    positive = weighted.new_zeros(count).index_add(0, groups, weighted.clamp_min(0))
    negative = weighted.new_zeros(count).index_add(0, groups, (-weighted).clamp_min(0))
    # A shared floor also preserves P*s+ = N*s- when BOTH sides are tiny.
    # Avoid overflowing the inactive ratio and its backward expression.
    floor = 1e-12
    up = torch.where(positive > 0, (negative/positive.clamp_min(floor)).clamp_max(1), torch.ones_like(positive))
    down = torch.where(negative > 0, (positive/negative.clamp_min(floor)).clamp_max(1), torch.ones_like(negative))
    return torch.stack([up, down], -1).to(residual.dtype)


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


def signed_moment_projection(point, base, lo, hi, step, dispersion_weight=1.0):
    """Metric projection onto the exact feasible partial-moment polygon.

    A continuous bounded response has a triangle; an ordinal response has the
    quadrilateral between its extreme and nearest-neighbour rating levels.
    The metric weights squared mean-coordinate error by 1 and squared
    dispersion-coordinate error by dispersion_weight. Weight 1 retains the
    original Euclidean arithmetic. Positive weights give a unique projection;
    this preserves feasibility, not necessarily point RMSE.
    """
    if not math.isfinite(dispersion_weight) or dispersion_weight <= 0:
        raise ValueError("Moment projection weight must be finite and positive")
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
    def metric_inner(a,b):
        if dispersion_weight == 1.0:
            return (a*b).sum(-1)
        return ((a[...,0]-a[...,1])*(b[...,0]-b[...,1])
                +dispersion_weight*(a[...,0]+a[...,1])*(b[...,0]+b[...,1]))
    position = metric_inner(rel,edge)/metric_inner(edge,edge).clamp_min(1e-12)
    candidates = vertices+position.clamp(0, 1).unsqueeze(-1)*edge
    distance=candidates-point.unsqueeze(-2)
    index = metric_inner(distance,distance).argmin(-1)
    nearest = candidates.gather(-2, index[..., None, None].expand(*index.shape, 1, 2)).squeeze(-2)
    return torch.where(inside[..., None], point, nearest)


def signed_moment_priority_projection(point, base, lo, hi, step):
    """Unique lexicographic projection: mean first, then dispersion.

    A fixed feasible mean has a closed interval of feasible absolute moments.
    Clipping dispersion onto that interval preserves the selected mean and the
    continuous/ordinal polygon. This is a defined tie-break at zero mean-metric
    weight, not an SPD projection with a singular metric.
    """
    mean=(point[...,0]-point[...,1]).clamp(min=lo-base,max=hi-base)
    expected=base+mean
    minimum=mean.abs()
    if step is not None:
        lower=(lo+((base-lo)/step).floor()*step).clamp(lo,hi)
        upper=(lo+((base-lo)/step).ceil()*step).clamp(lo,hi)
        gap=(upper-lower).clamp_min(1e-12)
        chord=((upper-expected)*(base-lower)+(expected-lower)*(upper-base))/gap
        minimum=torch.where((expected>=lower)&(expected<=upper)&(upper>lower),chord,minimum)
    maximum=(2*(hi-base)*(base-lo)+(hi+lo-2*base)*mean)/(hi-lo)
    minimum=torch.minimum(minimum.clamp_min(0),maximum)
    dispersion=torch.maximum(minimum,torch.minimum(maximum,point.sum(-1)))
    return torch.stack([.5*(dispersion+mean),.5*(dispersion-mean)],-1)


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
@torch.no_grad()
def preconditioned_cg(operator, rhs, precondition, steps=64, tolerance=1e-4, check_every=4):
    """Matrix-free SPD solve, with an actual residual stopping certificate.

    Vectors retain their dtype. Scalar inner products accumulate in FP64;
    this is an explicit scientific solver choice, not an execution policy.
    """
    if steps<1 or check_every<1 or not 0<tolerance<1:raise ValueError("Invalid PCG budget")
    dot=lambda a,b:(a.double()*b.double()).sum()
    norm=float(dot(rhs,rhs).sqrt())
    if norm==0:return torch.zeros_like(rhs),0.,0
    x=torch.zeros_like(rhs);r=rhs.clone();z=precondition(r);p=z.clone();rz=dot(r,z)
    relative=1.
    for iteration in range(1,steps+1):
        ap=operator(p);denom=dot(p,ap)
        if not torch.isfinite(denom) or denom<0:raise FloatingPointError("Non-SPD/nonfinite PCG operator")
        alpha=rz/denom.clamp_min(torch.finfo(torch.float64).tiny)
        x=x+alpha*p;r=r-alpha*ap
        if iteration==1 or iteration%check_every==0 or iteration==steps:
            true=rhs-operator(x);relative=float(dot(true,true).sqrt())/norm
            if not math.isfinite(relative):raise FloatingPointError("Nonfinite PCG residual")
            if relative<=tolerance:return x,relative,iteration
        z=precondition(r);new=dot(r,z)
        beta=new/rz.clamp_min(torch.finfo(torch.float64).tiny)
        p=z+beta*p;rz=new
    return x,relative,iteration
