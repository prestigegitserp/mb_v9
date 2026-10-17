"""Mean-anchored monotone angular transport with an explicit finite slope budget.

The learned object is interval probability mass dQ, not an unconstrained MLP.
Projection enforces total mass, left-half mass, first moment and slope caps.
Linear interpolation across baseline anchors preserves all three equalities.
Guarantees concern the declared angular reference, not unknown catalogue ratings.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize, linprog, brentq, Bounds, LinearConstraint
from scipy.special import betainc, softmax


def _constraints(nodes, anchors, margin):
    width = np.diff(nodes)
    middle = 0.5*(nodes[1:]+nodes[:-1])
    left = (middle < 0.5).astype(float)
    matrix = np.stack([np.ones_like(middle), left, middle])
    targets = np.stack([np.ones_like(anchors), anchors, 1-anchors], -1)
    lipschitz = margin*(anchors**2+(1-anchors)**2)/np.minimum(anchors, 1-anchors)
    caps = lipschitz[:, None]*width[None]
    initial = []
    for anchor, b, cap in zip(anchors, targets, caps):
        fit = linprog(np.zeros_like(width), A_eq=matrix, b_eq=b,
                      bounds=list(zip(np.zeros_like(width), cap)), method="highs")
        if not fit.success:
            raise ValueError("Infeasible transport grid/slope/anchor combination")
        # A basic LP vertex has no useful feasible gradient directions. Start
        # from an entropy interior instead, projecting only if its slope is too
        # large. This prevents a numerically "learned" spline that cannot learn.
        l = middle < .5
        def mass(beta):
            d = np.empty_like(width)
            d[l] = anchor*softmax(np.log(width[l])+beta*middle[l])
            d[~l] = (1-anchor)*softmax(np.log(width[~l])+beta*middle[~l])
            return d
        beta = brentq(lambda z: mass(z)@middle-(1-anchor), -1e5, 1e5)
        interior = mass(beta)
        if np.max(interior-cap) > 1e-9:
            center = minimize(lambda z: .5*np.square(z-interior).sum(), fit.x,
                              jac=lambda z: z-interior, method="SLSQP",
                              constraints=[LinearConstraint(matrix, b, b)],
                              bounds=Bounds(np.zeros_like(cap), cap),
                              options={"ftol": 1e-12, "maxiter": 200})
            if not center.success:
                raise ValueError("Cannot initialize a feasible transport interior")
            interior = center.x
        initial.append(interior)
    return matrix, targets, caps, np.asarray(initial)


class _MomentProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw, matrix, target, cap, initial):
        a = matrix.detach().cpu().double().numpy()
        b = target.detach().cpu().double().numpy()
        upper = cap.detach().cpu().double().numpy()
        start = initial.detach().cpu().double().numpy()
        values = raw.detach().cpu().double().numpy()
        out = []
        projectors = []
        for value, rhs, ceiling, x0 in zip(values, b, upper, start):
            fit = minimize(lambda z: 0.5*np.square(z-value).sum(), x0,
                           jac=lambda z: z-value, method="SLSQP",
                           constraints=[LinearConstraint(a, rhs, rhs)],
                           bounds=Bounds(np.zeros_like(ceiling), ceiling),
                           options={"ftol": 1e-11, "maxiter": 150})
            z = fit.x
            if (not fit.success or np.max(np.abs(a@z-rhs)) > 2e-7
                    or np.min(z) < -2e-8 or np.max(z-ceiling) > 2e-8):
                raise FloatingPointError(f"Transport projection failed: {fit.message}")
            out.append(z)
            free = (z > 2e-7) & (z < ceiling-2e-7)
            af = a[:, free]
            pf = np.eye(int(free.sum()))-af.T@np.linalg.pinv(af@af.T)@af
            projector = np.zeros((len(z), len(z)))
            projector[np.ix_(free, free)] = pf
            projectors.append(projector)
        p = torch.as_tensor(np.asarray(projectors), device=raw.device, dtype=raw.dtype)
        ctx.save_for_backward(p)
        return torch.as_tensor(np.asarray(out), device=raw.device, dtype=raw.dtype)

    @staticmethod
    def backward(ctx, grad):
        (projector,) = ctx.saved_tensors
        return torch.einsum("bij,bj->bi", projector, grad), None, None, None, None


class AngularTransport(nn.Module):
    def __init__(self, k, nodes=33, slope_margin=3.0, grid=257, learned=True):
        super().__init__()
        self.learned = learned
        positions = 0.5*(1-np.cos(np.linspace(0, np.pi, nodes)))
        positions[nodes//2] = 0.5
        anchors = np.asarray([.02, .10, .25, .40, .50, .60, .75, .90, .98])
        a, b, cap, initial = _constraints(positions, anchors, slope_margin)
        for name, value in [("nodes", positions), ("anchors", anchors), ("matrix", a),
                            ("target", b), ("cap", cap), ("feasible_start", initial)]:
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
        self.mass = nn.Parameter(torch.tensor(initial, dtype=torch.float32), requires_grad=learned)
        angle_grid = np.linspace(-1, 1, grid)
        cdf = betainc((k-1)/2, (k-1)/2, (angle_grid+1)/2)
        self.register_buffer("angle_grid", torch.tensor(angle_grid, dtype=torch.float32))
        self.register_buffer("isotropic_cdf", torch.tensor(cdf, dtype=torch.float32))
        self.register_buffer("reference_cdf", torch.tensor(cdf, dtype=torch.float32).repeat(2, 4, 1))
        self.register_buffer("reference_sizes", torch.zeros(2, 4, dtype=torch.long))

    def projected_values(self):
        masses = _MomentProjection.apply(self.mass, self.matrix, self.target,
                                         self.cap, self.feasible_start)
        return torch.cat([masses.new_zeros((len(masses), 1)), masses.cumsum(-1)], -1)

    @staticmethod
    def power_quantile(v, p):
        # Defined for strictly interior bases; endpoint inputs evaluated stably.
        p = p.clamp(1e-7, 1-1e-7)
        # Both branches must stay finite: torch.where does not prevent an
        # unused overflowing branch from contaminating its backward pass.
        left = p*torch.exp(((1-p)/p)*torch.log((2*v).clamp(1e-30, 1)))
        right = 1-(1-p)*torch.exp((p/(1-p))*torch.log((2*(1-v)).clamp(1e-30, 1)))
        value = torch.where(v <= .5, left, right)
        return torch.where(v <= 0, torch.zeros_like(value),
                           torch.where(v >= 1, torch.ones_like(value), value))

    @staticmethod
    def boundary_quantile(v, p):
        """Finite-slope analytic closure, with Q(.5)=integral(Q)=p.

        A linear left half plus a terminal ramp has these identities for p<=.5;
        reflection supplies p>=.5. Its slope is finite for every interior p.
        """
        p = p.clamp(1e-7, 1-1e-7)
        reflect = p > .5
        small = torch.minimum(p, 1-p)
        t = torch.where(reflect, 1-v, v)
        width = small/(2*(1-small))
        value = 2*small*t.clamp(0, .5)+(1-small)*((t-1+width)/width).clamp(0, 1)
        return torch.where(reflect, 1-value, value)

    def quantile(self, v, p, values=None):
        if not self.learned:
            return self.power_quantile(v, p)
        values = self.projected_values() if values is None else values
        a = torch.searchsorted(self.anchors, p.contiguous()).clamp(1, len(self.anchors)-1)
        weight = (p-self.anchors[a-1])/(self.anchors[a]-self.anchors[a-1])
        j = torch.searchsorted(self.nodes, v.contiguous()).clamp(1, len(self.nodes)-1)
        interpolation = (v-self.nodes[j-1])/(self.nodes[j]-self.nodes[j-1])
        low = values[a-1, j-1]+interpolation*(values[a-1, j]-values[a-1, j-1])
        high = values[a, j-1]+interpolation*(values[a, j]-values[a, j-1])
        learned = low+weight*(high-low)
        # Close the extreme-baseline regions continuously, without the infinite
        # endpoint slopes of the original power quantile. Each convex component
        # has the required endpoint/half/mean identities. Their mixing weight
        # solves the target mean exactly and converges to 1 at the bank edge.
        low = p < self.anchors[0]
        high = p > self.anchors[-1]
        small = torch.where(low, p, 1-p).clamp_min(1e-7)
        anchor = self.anchors[0]
        weight = (.25*small/(anchor-.75*small).clamp_min(1e-7)).clamp(0, 1)
        extreme_p = torch.where(low, .75*p, 1-.75*(1-p))
        boundary = self.boundary_quantile(v, extreme_p)
        first = values[0, j-1]+interpolation*(values[0, j]-values[0, j-1])
        last = values[-1, j-1]+interpolation*(values[-1, j]-values[-1, j-1])
        edge = torch.where(low, first, last)
        outside = weight*edge+(1-weight)*boundary
        return torch.where(low | high, outside, learned).clamp(0, 1)

    def angular_cdf(self, cosine, view, group, reference=None):
        c = cosine.clamp(-1, 1)
        z = (c+1)*0.5*(len(self.angle_grid)-1)
        j = z.floor().long().clamp(0, len(self.angle_grid)-2)
        t = z-j.to(z.dtype)
        bank = (self.reference_cdf if reference is None else reference)[view]
        return bank[group, j]+t*(bank[group, j+1]-bank[group, j])

    @torch.no_grad()
    def update_reference(self, scores, groups, view, weight):
        for group in range(4):
            c = scores[groups == group].sort().values
            self.reference_sizes[view, group] = len(c)
            if len(c) < 32:
                self.reference_cdf[view, group].copy_(self.isotropic_cdf)
                continue
            ecdf = torch.searchsorted(c, self.angle_grid, right=True).to(c.dtype)/len(c)
            cdf = (1-weight)*self.isotropic_cdf+weight*ecdf
            cdf[0] = 0
            cdf[-1] = 1
            self.reference_cdf[view, group].copy_(cdf)
