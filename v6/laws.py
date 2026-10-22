"""A full ordinal probability fibre with a prescribed mean.

Exponential tilting and mean/shape orthogonality are known statistical tools.
This implementation does not claim their invention. The implicit Jacobian
preserves the mean and avoids differentiating the bisection branching decisions.
"""
from __future__ import annotations

import torch


class _MeanTilt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, support):
        work = logits.double()
        r = support.double()
        if not torch.isfinite(work).all() or not torch.isfinite(target).all():
            raise ValueError("Mean tilt requires finite logits and means")
        mu = target.double().clamp(float(r[0])+1e-9, float(r[-1])-1e-9)
        # Remove the affine-logit gauge before solving. It has no effect on
        # the mean-constrained law, but can otherwise exhaust a fixed bracket.
        affine = (work[:, -1]-work[:, 0])/(r[-1]-r[0])
        work = work-work[:, :1]-affine[:, None]*(r-r[0])
        # This bracket dominates every adjacent logit slope. At either end all
        # scores are ordered toward the corresponding extreme support point.
        slopes = work.diff(dim=-1)/r.diff()
        gap = float(r.diff().min())
        bound = slopes.abs().amax(-1)+64/gap
        low, high = -bound, bound
        for _ in range(64):
            alpha = .5*(low+high)
            p = torch.softmax(work+alpha[:, None]*r, -1)
            mean = (p*r).sum(-1)
            low = torch.where(mean < mu, alpha, low)
            high = torch.where(mean < mu, high, alpha)
        p = torch.softmax(work+(.5*(low+high))[:, None]*r, -1).to(logits.dtype)
        if not torch.allclose((p.double()*r).sum(-1), mu, atol=3e-6, rtol=0):
            raise FloatingPointError("Mean-constrained law did not converge")
        ctx.save_for_backward(p, support)
        return p

    @staticmethod
    def backward(ctx, grad):
        p, support = ctx.saved_tensors
        mean = (p*support).sum(-1, keepdim=True)
        v = p*(support-mean)
        variance = (v*(support-mean)).sum(-1).clamp_min(1e-12)
        adjustment = (v*grad).sum(-1)/variance
        grad_logits = p*(grad-(p*grad).sum(-1, keepdim=True))-v*adjustment[:, None]
        return grad_logits, adjustment, None


def mean_tilt(logits, raw_mean, levels):
    midpoint = .5*(levels[0]+levels[-1])
    half = .5*(levels[-1]-levels[0])
    return _MeanTilt.apply(logits, (raw_mean-midpoint)/half,
                           (levels-midpoint)/half)


def ordinal_nll(probabilities, ratings, levels):
    if len(levels) < 2:
        raise ValueError("An ordinal law needs at least two levels")
    index = torch.round((ratings-levels[0])/(levels[1]-levels[0])).long()
    if (index < 0).any() or (index >= len(levels)).any():
        raise ValueError("Rating outside the declared ordinal support")
    if ((levels[index]-ratings).abs() > 2e-5).any():
        raise ValueError("Continuous labels cannot be silently treated as ordinal")
    return -probabilities.gather(1, index[:, None]).squeeze(-1).clamp_min(1e-30).log()


def mixture_variance(probabilities, means, levels):
    """Exact total-variance identity for an ensemble of bounded ordinal laws."""
    member_variance = (probabilities*(levels[None, None, :]-means[..., None]).square()).sum(-1)
    within = member_variance.mean(0)
    between = means.var(0, unbiased=False)
    return within+between, within, between
