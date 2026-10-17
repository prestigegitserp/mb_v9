"""Seven independent V3 replacements, sharing only train-history plumbing.

Every model exposes the V8 pair-prediction interface. No legacy angle model is
allocated and then left unused. Derived uncertainty is explicitly conditional
on the working linearization; it is not labelled a calibrated neural posterior.
"""
from __future__ import annotations

import copy
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig
from .history import make_history, HistoryEncoder, scatter_sum
from .numerics import (unit, sphere_exp, bounded_residual, smooth_gain,
                       spectral_filter, conjugate_gradient,
                       signed_moment_projection, gaussian_bilinear_moments, residual_balance)
from .transport import AngularTransport
from .laws import mean_tilt, ordinal_nll, mixture_variance


def direction_table(size, k):
    layer = nn.Embedding(size, k)
    with torch.no_grad():
        layer.weight.copy_(unit(torch.randn(size, k)))
    return layer


class BiasBlock(nn.Module):
    def __init__(self, nu, ni, lo, hi, mean, tau):
        super().__init__()
        self.lo, self.hi, self.tau = lo, hi, tau
        p = min(max((mean-lo)/(hi-lo), 1e-4), 1-1e-4)
        self.center = nn.Parameter(torch.tensor(math.log(p/(1-p))))
        self.user = nn.Embedding(nu, 1)
        self.item = nn.Embedding(ni, 1)
        nn.init.zeros_(self.user.weight)
        nn.init.zeros_(self.item.weight)

    def forward(self, u, i, un, vn):
        a = un[u]/(un[u]+self.tau)
        b = vn[i]/(vn[i]+self.tau)
        logit = self.center+a*self.user(u).squeeze(-1)+b*self.item(i).squeeze(-1)
        return self.lo+(self.hi-self.lo)*torch.sigmoid(logit)


class BaseModel(nn.Module):
    def __init__(self, data, cfg, seed, own_bias=True):
        super().__init__()
        cfg.validate()
        self.cfg, self.model_seed = copy.deepcopy(cfg), int(seed)
        self.v3_method = cfg.method
        self.lo, self.hi = float(data.rating_min), float(data.rating_max)
        self.width = self.hi-self.lo
        self.mid, self.half = .5*(self.lo+self.hi), .5*self.width
        self.nu, self.ni, self.k = data.num_users, data.num_items, cfg.latent_dim
        self.rating_step = getattr(data, "rating_step", None)
        if self.rating_step is not None:
            raw_levels = (data.train_ratings_raw-self.lo)/self.rating_step
            if not torch.allclose(raw_levels, raw_levels.round(), atol=2e-5, rtol=0):
                self.rating_step = None
        self._data = data
        self._eval_cache = None
        self._step = 0
        mean = float(data.train_ratings_raw.mean())
        self.bias = BiasBlock(self.nu, self.ni, self.lo, self.hi, mean, cfg.count_tau) if own_bias else None

    def add_encoders(self):
        self.user_encoder = HistoryEncoder(self.ni, self.cfg.hidden_dim, self.k)
        self.item_encoder = HistoryEncoder(self.nu, self.cfg.hidden_dim, self.k)

    def clear_cache(self):
        self._eval_cache = None

    def train(self, mode=True):
        self.clear_cache()
        return super().train(mode)

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        self.clear_cache()
        return result

    def history(self, include=None):
        data = self._data
        u, i = data.train_users, data.train_items
        chosen_u = u if include is None else u[include]
        chosen_i = i if include is None else i[include]
        un = torch.bincount(chosen_u, minlength=self.nu).float()
        vn = torch.bincount(chosen_i, minlength=self.ni).float()
        with torch.no_grad():
            base = self.bias(u, i, un, vn) if self.bias is not None else self.mid
            residual = (data.train_ratings_raw-base)/self.width
        return make_history(data, include, residual)

    def base(self, h, u, i):
        return self.bias(u, i, h.user_counts, h.item_counts)

    def encode(self, h):
        residual = ((h.y-self.base(h, h.u, h.i))/self.width).detach()
        zu = self.user_encoder(h.u, h.i, residual, h.user_counts, self.nu)
        zi = self.item_encoder(h.i, h.u, residual, h.item_counts, self.ni)
        return zu, zi

    def eval_state(self):
        if self._eval_cache is None or self.training:
            h = self.history()
            state = self.prepare(h)
            if not self.training:
                self._eval_cache = (h, state)
            return h, state
        return self._eval_cache

    def predict_raw_pairs(self, data, u, i):
        h, state = self.eval_state()
        return self.predict(h, state, u, i)[0]

    @torch.no_grad()
    def score_catalog(self, data, users, chunk_size=65536):
        """V3-correct full-catalog scores; no old angular-factor cache is used."""
        self.eval()
        out = []
        for user in users:
            values = []
            for start in range(0, self.ni, chunk_size):
                item = torch.arange(start, min(start+chunk_size, self.ni), device=user.device)
                values.append(self.predict_raw_pairs(data, user.expand_as(item), item))
            out.append(torch.cat(values))
        return torch.stack(out)

    def common_prior(self, state):
        zero = next(self.parameters()).new_zeros(())
        penalty = zero
        n = max(1, self._data.train_users.numel())
        if self.bias is not None:
            penalty = penalty+self.cfg.bias_prior*(self.bias.user.weight.square().sum()
                       +self.bias.item.weight.square().sum())/n
        for name, value in state.items():
            if name in ("zu", "zi"):
                penalty = penalty+self.cfg.code_prior*value.square().sum()/n
        for name, module in self.named_modules():
            if isinstance(module, nn.Embedding) and "direction" in name:
                initial = getattr(self, "initial_"+name.replace(".", "_"), None)
                if initial is not None:
                    penalty = penalty+self.cfg.direction_prior*(1-(unit(module.weight)*initial).sum(-1)).sum()/n
        return penalty

    def loss_for_targets(self, h, state, u, i, y):
        pred, _ = self.predict(h, state, u, i)
        lp = ((pred-y)/self.width).square().mean()
        aux = self.auxiliary(h, state, u, i, y)+self.common_prior(state)
        return lp, aux, {"L_pred": float(lp.detach()), "L_auxiliary": float(aux.detach())}

    def auxiliary(self, h, state, u, i, y):
        return y.new_zeros(())

    @torch.no_grad()
    def diagnostics(self, data, u, i):
        h, state = self.eval_state()
        prediction, extra = self.predict(h, state, u, i)
        out = {"v3_range_violation": float((F.relu(self.lo-prediction)+F.relu(prediction-self.hi)).max()),
               "v3_prediction_std": float(prediction.std(unbiased=False)),
               "v3_parameter_count_total": sum(p.numel() for p in self.parameters())}
        for name in ("last_cg_residual", "last_graph_cg_residual", "posterior_cg_residual"):
            if hasattr(self, name):
                out[f"v3_{name}"] = float(getattr(self, name))
        for name, value in extra.items():
            if torch.is_tensor(value) and value.numel():
                out[f"v3_{name}_mean"] = float(value.float().mean())
        return out


class Method1(BaseModel):
    def __init__(self, data, cfg, seed):
        super().__init__(data, cfg, seed)
        self.add_encoders()
        self.user_direction = direction_table(self.nu, self.k)
        self.item_direction = direction_table(self.ni, self.k)
        self.user_decoder = nn.Embedding(self.ni, self.k)
        self.item_decoder = nn.Embedding(self.nu, self.k)
        nn.init.normal_(self.user_decoder.weight, std=.05)
        nn.init.normal_(self.item_decoder.weight, std=.05)
        self.user_radius = nn.Linear(self.k, 1)
        self.item_radius = nn.Linear(self.k, 1)
        self.user_proposal = nn.Linear(self.k, self.k, bias=False)
        self.item_proposal = nn.Linear(self.k, self.k, bias=False)
        self.gain_offset = nn.Parameter(torch.tensor(0.0))

    def _steer(self, q, z, radius, owners, counterparts, decoder, proposal, entities):
        if not self.cfg.steering_enabled:
            return unit(q)
        # Population/amortized prior survives even in a locally unobserved direction.
        q = unit(q+self.cfg.context_strength*unit(z))
        v = proposal(z)
        v = v-(v*q).sum(-1, keepdim=True)*q
        if self.cfg.information_filter:
            rows = decoder(counterparts).detach()
            gram = sum_outer(rows, owners, entities, self.cfg.batch_size)
            eye = torch.eye(self.k, device=z.device, dtype=z.dtype)
            tangent = eye-q[..., :, None]*q[..., None, :]
            info = (radius.square()[..., None, None]*(tangent@gram@tangent)).detach()
            v = spectral_filter(info, v, self.cfg.information_ridge)
        return sphere_exp(q, v, self.cfg.tangent_limit)

    def prepare(self, h):
        zu, zi = self.encode(h)
        ru = math.sqrt(self.half)*(F.softplus(self.user_radius(zu)).squeeze(-1)+1e-4)
        ri = math.sqrt(self.half)*(F.softplus(self.item_radius(zi)).squeeze(-1)+1e-4)
        if not self.cfg.radius_enabled:
            ru, ri = torch.full_like(ru, math.sqrt(self.half)*.5), torch.full_like(ri, math.sqrt(self.half)*.5)
        qu = self._steer(self.user_direction.weight, zu, ru, h.u, h.i,
                         self.user_decoder, self.user_proposal, self.nu)
        qi = self._steer(self.item_direction.weight, zi, ri, h.i, h.u,
                         self.item_decoder, self.item_proposal, self.ni)
        return dict(zu=zu, zi=zi, ru=ru, ri=ri, qu=qu, qi=qi)

    def predict(self, h, s, u, i):
        cosine = (s["qu"][u]*s["qi"][i]).sum(-1)
        gain = smooth_gain(s["ru"][u], s["ri"][i], self.half, self.gain_offset)
        return bounded_residual(self.base(h, u, i), cosine, gain, self.lo, self.hi), {"gain": gain}

    def auxiliary(self, h, s, u, i, y):
        target = ((y-self.base(h, u, i))/self.width).detach()
        a = (s["qu"][u]*s["ru"][u, None]*self.user_decoder(i)).sum(-1)
        b = (s["qi"][i]*s["ri"][i, None]*self.item_decoder(u)).sum(-1)
        return self.cfg.reconstruction_weight*.5*((a-target).square().mean()+(b-target).square().mean())


class RiskRouter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(8, cfg.router_width), nn.GELU(),
                                                nn.Linear(cfg.router_width, 1))
                                    for _ in range(cfg.router_replicates)])
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.register_buffer("center", torch.zeros(8))
        self.register_buffer("scale", torch.ones(8))
        self.register_buffer("target_scale", torch.tensor(1.0))
        self.register_buffer("noise_floor", torch.tensor(0.0))
        self.register_buffer("lambda0", torch.tensor(0.5))
        self.register_buffer("strength", torch.tensor(0.0))
        self.register_buffer("fitted", torch.tensor(False))
        self.cfg = cfg

    def moments(self, features):
        x = (features-self.center)/self.scale
        if self.cfg.router_counts_only:
            # No prediction, disagreement, bias or rating-spread input survives
            # into this control's gate. Its OOF objective fits the gate directly.
            x = torch.cat([x[..., :2], torch.zeros_like(x[..., 2:])], -1)
        return torch.stack([head(x).squeeze(-1) for head in self.heads])*self.target_scale

    def forward(self, features, difference):
        estimates = self.moments(features)
        mean = estimates.mean(0)
        if self.cfg.router_counts_only:
            eta = torch.maximum(-self.lambda0, torch.minimum(1-self.lambda0, mean))
            return self.lambda0+self.strength*eta, torch.zeros_like(mean)
        beta = self.cfg.router_uncertainty*estimates.std(0, unbiased=False)+self.noise_floor
        soft = mean.sign()*F.relu(mean.abs()-beta)
        eta = soft/(difference.square()+self.cfg.router_ridge)
        eta = torch.maximum(-self.lambda0, torch.minimum(1-self.lambda0, eta))
        return (self.lambda0+self.strength*eta).clamp(0, 1), beta


class Method2(BaseModel):
    def __init__(self, data, cfg, seed):
        super().__init__(data, cfg, seed)
        self.add_encoders()
        self.uv_user_direction = direction_table(self.nu, self.k)
        self.uv_item_direction = direction_table(self.ni, self.k)
        self.iv_user_direction = direction_table(self.nu, self.k)
        self.iv_item_direction = direction_table(self.ni, self.k)
        self.user_radius = nn.Linear(self.k, 1)
        self.item_radius = nn.Linear(self.k, 1)
        self.counter_user = nn.Embedding(self.nu, 1)
        self.counter_item = nn.Embedding(self.ni, 1)
        nn.init.zeros_(self.counter_user.weight)
        nn.init.zeros_(self.counter_item.weight)
        self.mix = nn.Parameter(torch.tensor(0.0))
        self.gain_offset = nn.Parameter(torch.tensor(0.0))
        self.router = RiskRouter(cfg)
        self.router.requires_grad_(False)

    def prepare(self, h):
        zu, zi = self.encode(h)
        return dict(zu=zu, zi=zi,
                    qu=unit(self.uv_user_direction.weight+self.cfg.context_strength*unit(zu)),
                    qi=unit(self.uv_item_direction.weight),
                    pu=unit(self.iv_user_direction.weight),
                    pi=unit(self.iv_item_direction.weight+self.cfg.context_strength*unit(zi)),
                    ru=F.softplus(self.user_radius(zu)).squeeze(-1)+1e-4,
                    ri=F.softplus(self.item_radius(zi)).squeeze(-1)+1e-4)

    def experts(self, h, s, u, i):
        base = self.base(h, u, i)
        c1 = (s["qu"][u]*s["qi"][i]).sum(-1)
        c2 = (s["pu"][u]*s["pi"][i]).sum(-1)
        g1 = smooth_gain(s["ru"][u], F.softplus(self.counter_item(i)).squeeze(-1)+1e-4,
                         self.half, self.gain_offset)
        g2 = smooth_gain(F.softplus(self.counter_user(u)).squeeze(-1)+1e-4, s["ri"][i],
                         self.half, self.gain_offset)
        return (bounded_residual(base, c1, g1, self.lo, self.hi),
                bounded_residual(base, c2, g2, self.lo, self.hi))

    def routing_features(self, h, u, i, p1, p2):
        b = self.base(h, u, i)
        return torch.stack([h.user_counts[u].log1p()/10, h.item_counts[i].log1p()/10,
                            (p1-self.mid)/self.width, (p2-self.mid)/self.width,
                            (p1-p2)/self.width, (b-self.mid)/self.width,
                            h.user_features[u, 2], h.item_features[i, 2]], -1)

    def predict(self, h, s, u, i):
        p1, p2 = self.experts(h, s, u, i)
        beta = torch.zeros_like(p1)
        if self.cfg.router_enabled and bool(self.router.fitted):
            gate, beta = self.router(self.routing_features(h, u, i, p1, p2), p1-p2)
        else:
            gate = torch.sigmoid(self.mix).expand_as(p1)
        return gate*p1+(1-gate)*p2, {"mix_gate": gate, "router_beta": beta,
                                     "user_view_prediction": p1, "item_view_prediction": p2}

    def auxiliary(self, h, s, u, i, y):
        p1, p2 = self.experts(h, s, u, i)
        # No direct cross-view agreement: diversity is not penalised blindly.
        return self.cfg.auxiliary_weight*.5*(((p1-y)/self.width).square().mean()
                                           +((p2-y)/self.width).square().mean())


class RadiusBase(BaseModel):
    def __init__(self, data, cfg, seed):
        super().__init__(data, cfg, seed)
        self.add_encoders()
        self.user_decoder = nn.Embedding(self.ni, self.k)
        self.item_decoder = nn.Embedding(self.nu, self.k)
        nn.init.normal_(self.user_decoder.weight, std=.05)
        nn.init.normal_(self.item_decoder.weight, std=.05)

    def reconstruction(self, h, s, u, i, y):
        target = ((y-self.base(h, u, i))/self.width).detach()
        a = (s["zu"][u]*self.user_decoder(i)).sum(-1)
        b = (s["zi"][i]*self.item_decoder(u)).sum(-1)
        return .5*((a-target).square().mean()+(b-target).square().mean())


class Method3(RadiusBase):
    def __init__(self, data, cfg, seed):
        super().__init__(data, cfg, seed)
        self.user_direction = direction_table(self.nu, self.k)
        self.item_direction = direction_table(self.ni, self.k)
        self.user_log_prior = nn.Linear(self.k, 1)
        self.item_log_prior = nn.Linear(self.k, 1)
        self.delta = nn.Parameter(torch.zeros(self.nu+self.ni))
        self.gain_offset = nn.Parameter(torch.tensor(0.0))
        self.last_cg_residual = 0.0

    def prepare(self, h):
        zu, zi = self.encode(h)
        prior = self.cfg.log_radius_bound*torch.cat([
            torch.tanh(self.user_log_prior(zu)).squeeze(-1),
            torch.tanh(self.item_log_prior(zi)).squeeze(-1)])
        state = dict(zu=zu, zi=zi, prior=prior, qu=unit(self.user_direction.weight),
                    qi=unit(self.item_direction.weight),
                    r0u=math.sqrt(self.half)*h.user_amplitude,
                    r0i=math.sqrt(self.half)*h.item_amplitude)
        if self.cfg.graph_uncertainty:
            state["graph_samples"] = self._graph_draws(h, state)
        return state

    @torch.no_grad()
    def _graph_draws(self, h, s):
        """Conditional working covariance from this context, including its prior.

        Fixed probes make repeated inference/checkpoint restoration deterministic.
        No full-history covariance is reused for masked training queries.
        """
        u, i = h.u, h.i
        cosine = (s["qu"][u]*s["qi"][i]).sum(-1)
        base = self.base(h, u, i)
        slack = torch.where(cosine >= 0, self.hi-base, base-self.lo)
        gain = torch.sigmoid(self._logits(s, u, i))
        derivative = cosine*slack*gain*(1-gain)/self.width
        precision, noise = self.cfg.radius_precision, self.cfg.radius_noise
        def at(values):
            return torch.cat([scatter_sum(values, u, self.nu), scatter_sum(values, i, self.ni)])
        def operator(vector):
            return at(derivative.square()/noise**2*(vector[u]+vector[self.nu+i]))+precision*vector
        gen = torch.Generator(device=u.device).manual_seed(self.model_seed+19231)
        columns = []
        self.last_graph_cg_residual = 0.
        for _ in range(self.cfg.graph_probes):
            rhs = at(derivative*torch.randn(len(u), generator=gen, device=u.device)/noise)
            rhs = rhs+math.sqrt(precision)*torch.randn(len(self.delta), generator=gen, device=u.device)
            draw, residual = conjugate_gradient(operator, rhs, self.cfg.cg_steps)
            self.last_graph_cg_residual = max(self.last_graph_cg_residual, residual)
            columns.append(draw)
        return torch.stack(columns, -1)

    def _logits(self, s, u, i, delta=None):
        delta = self.delta if delta is None else delta
        log0 = s["r0u"][u].log()+s["r0i"][i].log()-math.log(self.half)
        return log0+delta[u]+delta[self.nu+i]+self.gain_offset

    def predict(self, h, s, u, i):
        cosine = (s["qu"][u]*s["qi"][i]).sum(-1)
        logits = self._logits(s, u, i)
        variance = torch.zeros_like(logits)
        if self.cfg.graph_uncertainty:
            draws = s["graph_samples"]
            variance = (draws[u]+draws[self.nu+i]).square().mean(-1)
            nodes, weights = np.polynomial.hermite.hermgauss(5)
            x = torch.as_tensor(nodes, device=logits.device, dtype=logits.dtype)
            w = torch.as_tensor(weights/math.sqrt(math.pi), device=logits.device, dtype=logits.dtype)
            gain = (torch.sigmoid(logits[:, None]+(2*variance).clamp_min(0).sqrt()[:, None]*x)*w).sum(-1)
        else:
            gain = torch.sigmoid(logits)
        return bounded_residual(self.base(h, u, i), cosine, gain, self.lo, self.hi), {"gain": gain, "graph_logradius_variance": variance}

    def auxiliary(self, h, s, u, i, y):
        n = max(1, self._data.train_users.numel())
        anchor = .5*self.cfg.radius_precision*(self.delta-s["prior"]).square().sum()/n
        return self.cfg.reconstruction_weight*self.reconstruction(h, s, u, i, y)+anchor

    def loss_for_targets(self, h, state, u, i, y):
        prediction, auxiliary, _ = super().loss_for_targets(h, state, u, i, y)
        prediction = .5*prediction/self.cfg.radius_noise**2
        return prediction, auxiliary, {"L_pred": float(prediction.detach()),
                                       "L_auxiliary": float(auxiliary.detach())}

    @torch.no_grad()
    def refresh_radius(self, probes=True):
        h = self.history()
        s = self.prepare(h)
        u, i, y = h.u, h.i, h.y
        cosine = (s["qu"][u]*s["qi"][i]).sum(-1)
        base = self.base(h, u, i)
        slack = torch.where(cosine >= 0, self.hi-base, base-self.lo)
        variance = ((s["graph_samples"][u]+s["graph_samples"][self.nu+i]).square().mean(-1)
                    if self.cfg.graph_uncertainty else torch.zeros_like(y))
        nodes, weights = np.polynomial.hermite.hermgauss(5)
        x = torch.as_tensor(nodes, device=y.device, dtype=y.dtype)
        w = torch.as_tensor(weights/math.sqrt(math.pi), device=y.device, dtype=y.dtype)
        shift = (2*variance).clamp_min(0).sqrt()[:, None]*x
        def integrated_gain(delta):
            sigmoid = torch.sigmoid(self._logits(s, u, i, delta)[:, None]+shift)
            return (sigmoid*w).sum(-1), (sigmoid*(1-sigmoid)*w).sum(-1)
        gain, gain_derivative = integrated_gain(self.delta)
        pred = base+cosine*slack*gain
        e = (pred-y)/self.width
        derivative = cosine*slack*gain_derivative/self.width
        precision = self.cfg.radius_precision
        noise = self.cfg.radius_noise
        def at(values):
            return torch.cat([scatter_sum(values, u, self.nu), scatter_sum(values, i, self.ni)])
        def incidence(vector):
            return vector[u]+vector[self.nu+i]
        weight = derivative.square()/noise**2
        def operator(vector):
            return at(weight*incidence(vector))+precision*vector
        rhs = -at(derivative*e/noise**2)-precision*(self.delta-s["prior"])
        step, residual = conjugate_gradient(operator, rhs, self.cfg.cg_steps)
        step *= min(1., self.cfg.radius_trust_step/max(float(step.abs().max()), 1e-12))
        def objective(delta):
            p = base+cosine*slack*integrated_gain(delta)[0]
            return .5*((p-y)/(self.width*noise)).square().sum()+.5*precision*(delta-s["prior"]).square().sum()
        before = objective(self.delta)
        self.last_radius_objective_before = float(before)
        for scale in (1., .5, .25, .125, .0625):
            proposed = (self.delta+scale*step).clamp(-self.cfg.log_radius_bound, self.cfg.log_radius_bound)
            if objective(proposed) <= before:
                self.delta.copy_(proposed)
                break
        self.last_radius_objective_after = float(objective(self.delta))
        self.last_cg_residual = residual
        self.clear_cache()


class Method4(RadiusBase):
    def __init__(self, data, cfg, seed):
        super().__init__(data, cfg, seed)
        self.user_direction = direction_table(self.nu, self.k)
        self.item_direction = direction_table(self.ni, self.k)
        self.user_shift = nn.Linear(self.k, 2)
        self.item_shift = nn.Linear(self.k, 2)
        nn.init.zeros_(self.user_shift.weight)
        nn.init.zeros_(self.user_shift.bias)
        nn.init.zeros_(self.item_shift.weight)
        nn.init.zeros_(self.item_shift.bias)

    def _interior(self, raw):
        tau = raw.detach().median().clamp_min(.05)*(1-self.cfg.prior_interior_median)/self.cfg.prior_interior_median
        return (raw/(tau+raw)).clamp(1e-5, 1-1e-5)

    def prepare(self, h):
        zu, zi = self.encode(h)
        pu, pi = self._interior(h.user_amplitude), self._interior(h.item_amplitude)
        du = self.cfg.log_radius_bound*torch.tanh(self.user_shift(zu))
        di = self.cfg.log_radius_bound*torch.tanh(self.item_shift(zi))
        ru = math.sqrt(self.half)*torch.sigmoid(torch.logit(pu)[:, None]+du)
        ri = math.sqrt(self.half)*torch.sigmoid(torch.logit(pi)[:, None]+di)
        return dict(zu=zu, zi=zi, ru=ru, ri=ri, du=du, di=di,
                    qu=self.user_direction.weight+self.cfg.context_strength*zu,
                    qi=self.item_direction.weight+self.cfg.context_strength*zi)

    def predict(self, h, s, u, i):
        base = self.base(h, u, i)
        split = max(1, self.k//2)
        cp = (unit(s["qu"][u, :split])*unit(s["qi"][i, :split])).sum(-1)
        cn = (unit(s["qu"][u, split:])*unit(s["qi"][i, split:])).sum(-1)
        gp = s["ru"][u, 0]*s["ri"][i, 0]/self.half
        gn = s["ru"][u, 1]*s["ri"][i, 1]/self.half
        if not self.cfg.signed_moments:
            cosine = (unit(s["qu"][u])*unit(s["qi"][i])).sum(-1)
            return bounded_residual(base, cosine, gp, self.lo, self.hi), {"gain": gp}
        proposal = torch.stack([(self.hi-base)*gp*(1+cp)*.5,
                                (base-self.lo)*gn*(1+cn)*.5], -1)
        moments = signed_moment_projection(proposal, base, self.lo, self.hi, self.rating_step) if self.cfg.project_moments else proposal
        return base+moments[:, 0]-moments[:, 1], {"positive_moment": moments[:, 0], "negative_moment": moments[:, 1],
                                               "radius_positive": s["ru"][u, 0]}

    def auxiliary(self, h, s, u, i, y):
        _, parts = self.predict(h, s, u, i)
        base = self.base(h, u, i).detach()
        aux = self.cfg.reconstruction_weight*self.reconstruction(h, s, u, i, y)
        if self.cfg.signed_moments:
            aux = aux+self.cfg.auxiliary_weight*(
                ((parts["positive_moment"]-F.relu(y-base))/self.width).square().mean()
                +((parts["negative_moment"]-F.relu(base-y))/self.width).square().mean())
        n = max(1, self._data.train_users.numel())
        return aux+.5*self.cfg.radius_precision*(s["du"].square().sum()+s["di"].square().sum())/n


def sum_outer(rows, owners, entities, chunk=4096):
    k = rows.shape[-1]
    out = rows.new_zeros(entities, k, k)
    for start in range(0, len(rows), chunk):
        values = rows[start:start+chunk]
        out.index_add_(0, owners[start:start+chunk], values[:, :, None]*values[:, None, :])
    return out


class DualDecoderBase(BaseModel):
    def __init__(self, data, cfg, seed):
        super().__init__(data, cfg, seed)
        self.add_encoders()
        self.user_decoder = nn.Embedding(self.ni, self.k)
        self.item_decoder = nn.Embedding(self.nu, self.k)
        nn.init.normal_(self.user_decoder.weight, std=.03)
        nn.init.normal_(self.item_decoder.weight, std=.03)
        self.mix = nn.Parameter(torch.tensor(0.0))

    def auxiliary(self, h, s, u, i, y):
        target = ((y-self.base(h, u, i))/self.width).detach()
        a = (s["zu"][u]*s["du"][i]).sum(-1)
        b = (s["zi"][i]*s["di"][u]).sum(-1)
        rec = .5*((a-target).square().mean()+(b-target).square().mean())
        n = max(1, self._data.train_users.numel())
        prior = self.cfg.code_prior*(self.user_decoder.weight.square().sum()
                                     +self.item_decoder.weight.square().sum())/n
        return self.cfg.reconstruction_weight*rec+prior


class Method5(DualDecoderBase):
    def __init__(self, data, cfg, seed):
        super().__init__(data, cfg, seed)
        p = min(max(cfg.posterior_correlation_init, 1e-5), 1-1e-5)
        self.correlation = nn.Parameter(torch.tensor(math.log(p/(1-p))))

    def _ridge(self, z, prior, owners, counterparts, target, entities):
        precision, noise = self.cfg.posterior_precision, self.cfg.posterior_noise
        gram = sum_outer(z[owners], counterparts, entities, self.cfg.batch_size)/noise**2
        eye = torch.eye(self.k, device=z.device, dtype=z.dtype)
        hessian = gram+precision*eye
        rhs = scatter_sum(z[owners]*target[:, None], counterparts, entities)/noise**2+precision*prior
        return torch.linalg.solve(hessian, rhs.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def _joint_response(self, z, d, owners, counterparts, view):
        """Matrix-free, context-specific Gaussian working posterior samples.

        The random generator is isolated and repeatable, making checkpoint
        predictions deterministic. No full-history covariance is reused for a
        masked query. Encoder NETWORK weights are conditioned on, not integrated.
        """
        nz, nd = len(z), len(d)
        precision, noise = self.cfg.posterior_precision, self.cfg.posterior_noise
        z, d = z.detach(), d.detach()
        zedge, dedge = z[owners], d[counterparts]
        def at(value):
            a = scatter_sum(dedge*value[:, None], owners, nz)
            b = scatter_sum(zedge*value[:, None], counterparts, nd)
            return torch.cat([a, b])
        def operator(value):
            dz, dd = value[:nz], value[nz:]
            score = (dz[owners]*dedge).sum(-1)+(dd[counterparts]*zedge).sum(-1)
            return at(score/noise**2)+precision*value
        generator = torch.Generator(device=z.device).manual_seed(self.model_seed+104729*(view+1))
        samples = []
        for _ in range(self.cfg.posterior_probes):
            eps = torch.randn(len(owners), generator=generator, device=z.device, dtype=z.dtype)
            prior_noise = torch.randn(nz+nd, self.k, generator=generator, device=z.device, dtype=z.dtype)
            rhs = at(eps/noise)+math.sqrt(precision)*prior_noise
            value, residual = conjugate_gradient(operator, rhs, self.cfg.posterior_cg_steps)
            self.posterior_cg_residual = max(self.posterior_cg_residual, residual)
            samples.append(value)
        factors = torch.stack(samples, -1)/math.sqrt(len(samples))
        return factors[:nz], factors[nz:]

    @torch.no_grad()
    def _independent_response(self, z, d, owners, counterparts):
        """Exact covariance factors for a block-diagonal working precision.

        Each code conditions on its decoder mean and vice versa. This actually
        removes the cross blocks BEFORE inversion; simply zeroing a joint
        covariance afterwards retains different marginals and is a separate
        ablation. Network weights and the plug-in means stay fixed in both.
        """
        precision, noise = self.cfg.posterior_precision, self.cfg.posterior_noise
        eye = torch.eye(self.k, device=z.device, dtype=z.dtype)
        hz = sum_outer(d[counterparts], owners, len(z), self.cfg.batch_size)/noise**2+precision*eye
        hd = sum_outer(z[owners], counterparts, len(d), self.cfg.batch_size)/noise**2+precision*eye
        def factor(hessian):
            lower = torch.linalg.cholesky(hessian)
            inverse = torch.linalg.solve_triangular(lower, eye.expand_as(lower), upper=False)
            return inverse.transpose(-1, -2)
        return factor(hz), factor(hd)

    def prepare(self, h):
        zu, zi = self.encode(h)
        self.posterior_cg_residual = 0.
        target = ((h.y-self.base(h, h.u, h.i))/self.width).detach()
        du = self._ridge(zu, self.user_decoder.weight, h.u, h.i, target, self.ni)
        di = self._ridge(zi, self.item_decoder.weight, h.i, h.u, target, self.nu)
        if self.cfg.joint_posterior:
            lu, ld = self._joint_response(zu, du, h.u, h.i, 0)
            li, le = self._joint_response(zi, di, h.i, h.u, 1)
        else:
            lu, ld = self._independent_response(zu, du, h.u, h.i)
            li, le = self._independent_response(zi, di, h.i, h.u)
        return dict(zu=zu, zi=zi, du=du, di=di, lu=lu, ld=ld, li=li, le=le)

    def _view(self, h, z, d, lz, ld, u, i, base, reverse=False):
        corr = torch.sigmoid(self.correlation) if self.cfg.joint_posterior and self.cfg.posterior_cross_covariance else 0.0
        mean, variance, correction = gaussian_bilinear_moments(z, d, lz, ld, corr)
        p = ((base-self.lo)/self.width).clamp(1e-6, 1-1e-6)
        anchor = math.sqrt(2)*torch.erfinv(2*p-1)
        signal = mean/torch.sqrt(self.cfg.score_temperature**2+variance.clamp_min(0))
        pred = self.lo+self.width*.5*(1+torch.erf((anchor+signal)/math.sqrt(2)))
        return pred, variance, correction

    def predict(self, h, s, u, i):
        base = self.base(h, u, i)
        a, va, ca = self._view(h, s["zu"][u], s["du"][i], s["lu"][u], s["ld"][i], u, i, base)
        b, vb, cb = self._view(h, s["zi"][i], s["di"][u], s["li"][i], s["le"][u], u, i, base)
        gate = torch.sigmoid(self.mix)
        return gate*a+(1-gate)*b, {"latent_epistemic_variance": .5*(va+vb),
                                   "joint_score_correction": .5*(ca+cb)}


class Method6(DualDecoderBase):
    def __init__(self, data, cfg, seed):
        super().__init__(data, cfg, seed)
        self.uv_user_direction = direction_table(self.nu, self.k)
        self.uv_item_direction = direction_table(self.ni, self.k)
        self.iv_user_direction = direction_table(self.nu, self.k)
        self.iv_item_direction = direction_table(self.ni, self.k)
        self.counter_user = nn.Embedding(self.nu, 1)
        self.counter_item = nn.Embedding(self.ni, 1)
        nn.init.zeros_(self.counter_user.weight)
        nn.init.zeros_(self.counter_item.weight)
        self.coupling = nn.Parameter(torch.tensor(0.0))
        self.reference_groups = 4*(cfg.reference_baseline_bins*cfg.reference_gain_bins if cfg.conditional_reference else 1)
        self.transport = AngularTransport(self.k, cfg.transport_nodes,
                                           cfg.transport_slope_margin, cfg.angular_grid,
                                           cfg.learned_transport and cfg.transport_enabled, self.reference_groups)

    def _radius(self, x):
        if not self.cfg.radius_enabled:
            return x.new_full(x.shape[:-1], math.sqrt(self.half)*.5)
        temperature = self.cfg.radius_temperature or .5*math.sqrt(self.k)
        return math.sqrt(self.half)*torch.tanh(x.norm(dim=-1)/temperature)

    def prepare(self, h):
        zu, zi = self.encode(h)
        du, di = self.user_decoder.weight, self.item_decoder.weight
        eta = torch.sigmoid(self.coupling)
        cap = math.sqrt(self.half)
        ru, ri = self._radius(zu), self._radius(zi)
        # Both operands of the convex radius coupling have the SAME valid cap.
        rd = eta*self._radius(du)+(1-eta)*cap*torch.sigmoid(self.counter_item.weight[:, 0])
        re = eta*self._radius(di)+(1-eta)*cap*torch.sigmoid(self.counter_user.weight[:, 0])
        if not self.cfg.decoder_coupling:
            rd, re = cap*torch.sigmoid(self.counter_item.weight[:, 0]), cap*torch.sigmoid(self.counter_user.weight[:, 0])
        if not self.cfg.radius_enabled:
            rd, re = torch.full_like(rd, cap*.5), torch.full_like(re, cap*.5)
        state = dict(zu=zu, zi=zi, du=du, di=di, ru=ru, ri=ri, rd=rd, re=re,
                    qu=unit(self.uv_user_direction.weight+self.cfg.context_strength*zu),
                    qi=unit(self.uv_item_direction.weight+(self.cfg.context_strength*du if self.cfg.decoder_coupling else 0)),
                    pu=unit(self.iv_user_direction.weight+(self.cfg.context_strength*di if self.cfg.decoder_coupling else 0)),
                    pi=unit(self.iv_item_direction.weight+self.cfg.context_strength*zi),
                    transport_values=self.transport.projected_values() if self.transport.learned else None)
        if self.cfg.transport_enabled:
            u, i = self._update_reference(h, state)
            state["reference_users"], state["reference_items"] = u, i
            state["reference_cdf"] = self.transport.reference_cdf.clone()
            state["reference_sizes"] = self.transport.reference_sizes.clone()
            if self.cfg.balance_transport:
                _, residuals, gains, groups = self._transport_residuals(h, state, u, i)
                state["transport_balance"] = torch.stack([
                    residual_balance(residual, gain, group, self.reference_groups)
                    for residual, gain, group in zip(residuals, gains, groups)])
            else:
                state["transport_balance"] = zu.new_ones(2, self.reference_groups, 2)
        return state

    @staticmethod
    def groups(h, u, i):
        return (h.user_counts[u] < 20).long()*2+(h.item_counts[i] < 5).long()

    def angular_scores(self, s, u, i):
        return ((s["qu"][u]*s["qi"][i]).sum(-1).clamp(-1, 1),
                (s["pu"][u]*s["pi"][i]).sum(-1).clamp(-1, 1))

    def _gains(self, s, u, i):
        return ((s["ru"][u]*s["rd"][i]/self.half).clamp(0, 1),
                (s["ri"][i]*s["re"][u]/self.half).clamp(0, 1))

    def _reference_groups(self, h, u, i, p, gain):
        degree = self.groups(h, u, i)
        if not self.cfg.conditional_reference:
            return degree
        bp, bg = self.cfg.reference_baseline_bins, self.cfg.reference_gain_bins
        baseline_cell = (p.detach()*bp).long().clamp(0, bp-1)
        gain_cell = (gain.detach()*bg).long().clamp(0, bg-1)
        return (degree*bp+baseline_cell)*bg+gain_cell

    def _transport_residuals(self, h, s, u, i):
        base = self.base(h, u, i)
        p = (base-self.lo)/self.width
        gains = self._gains(s, u, i)
        groups = [self._reference_groups(h, u, i, p, gain) for gain in gains]
        residuals = [self.transport.anchored_residual(cosine, p, base, self.lo, self.hi,
                     view, group, s["reference_cdf"], s["transport_values"])
                     for view, (cosine, group) in enumerate(zip(self.angular_scores(s, u, i), groups))]
        return base, residuals, gains, groups

    def predict(self, h, s, u, i):
        if self.cfg.transport_enabled:
            base, residuals, gains, groups = self._transport_residuals(h, s, u, i)
            predictions = []
            for view, (residual, gain, group) in enumerate(zip(residuals, gains, groups)):
                scale = s["transport_balance"][view, group, (residual < 0).long()]
                predictions.append(base+gain*scale*residual)
            p1, p2 = predictions
        else:
            base = self.base(h, u, i)
            c1, c2 = self.angular_scores(s, u, i)
            gains = self._gains(s, u, i)
            a, b = gains
            p1 = bounded_residual(base, c1, a, self.lo, self.hi)
            p2 = bounded_residual(base, c2, b, self.lo, self.hi)
        gate = torch.sigmoid(self.mix)
        return gate*p1+(1-gate)*p2, {"gain": .5*(gains[0]+gains[1]), "radius_user": s["ru"][u]}

    @torch.no_grad()
    def refresh_reference(self):
        h = self.history()
        self.prepare(h)
        self.clear_cache()

    @torch.no_grad()
    def _update_reference(self, h, s):
        gen = torch.Generator(device=h.u.device).manual_seed(self.model_seed+277)
        n = self.cfg.angular_reference_pairs
        u = torch.randint(self.nu, (n,), generator=gen, device=h.u.device)
        i = torch.randint(self.ni, (n,), generator=gen, device=h.u.device)
        p = (self.base(h, u, i)-self.lo)/self.width
        for view, (cosine, gain) in enumerate(zip(self.angular_scores(s, u, i), self._gains(s, u, i))):
            groups = self._reference_groups(h, u, i, p, gain)
            self.transport.update_reference(cosine, groups, view, self.cfg.angular_reference_weight)
        return u, i


class MeanMember(nn.Module):
    def __init__(self, data, cfg):
        super().__init__()
        self.cfg = cfg
        self.lo, self.hi = float(data.rating_min), float(data.rating_max)
        self.half = .5*(self.hi-self.lo)
        self.bias = BiasBlock(data.num_users, data.num_items, self.lo, self.hi,
                              float(data.train_ratings_raw.mean()), cfg.count_tau)
        self.user_direction = direction_table(data.num_users, cfg.latent_dim)
        self.item_direction = direction_table(data.num_items, cfg.latent_dim)
        def encoder():
            return nn.Sequential(nn.Linear(13, cfg.hidden_dim), nn.GELU(),
                                 nn.Linear(cfg.hidden_dim, cfg.latent_dim+1))
        self.user_encoder, self.item_encoder = encoder(), encoder()
        self.trust = nn.Parameter(torch.tensor(0.0))

    def entity(self, features, directions, encoder):
        features = F.dropout(features, self.cfg.feature_dropout, self.training)
        code = encoder(features)
        q = unit(directions+self.cfg.context_strength*code[:, :-1])
        radius = math.sqrt(self.half)*torch.sigmoid(code[:, -1])
        if not self.cfg.radius_enabled:
            radius = torch.full_like(radius, math.sqrt(self.half)*.5)
        return q, radius

    def prepare(self, h):
        qu, ru = self.entity(h.user_features, self.user_direction.weight, self.user_encoder)
        qi, ri = self.entity(h.item_features, self.item_direction.weight, self.item_encoder)
        return dict(qu=qu, qi=qi, ru=ru, ri=ri)

    def predict(self, h, state, u, i):
        base = self.bias(u, i, h.user_counts, h.item_counts)
        cosine = (state["qu"][u]*state["qi"][i]).sum(-1)
        support = (h.user_counts[u]/(h.user_counts[u]+self.cfg.count_tau)
                   *h.item_counts[i]/(h.item_counts[i]+self.cfg.count_tau)).sqrt()
        amplitude = torch.sigmoid(self.trust)*state["ru"][u]*state["ri"][i]/self.half*support
        prediction = bounded_residual(base, cosine, amplitude, self.lo, self.hi)
        # The trained point mean itself is strictly interior. The continuous
        # probability phase can then retain that mean exactly, even if a bias
        # sigmoid numerically saturates at an endpoint.
        eps = self.cfg.mean_interior_epsilon
        return self.lo+(self.hi-self.lo)*(eps+(1-2*eps)*(prediction-self.lo)/(self.hi-self.lo))


class Method7(BaseModel):
    def __init__(self, data, cfg, seed):
        super().__init__(data, cfg, seed, own_bias=False)
        self.members = nn.ModuleList([MeanMember(data, cfg) for _ in range(cfg.ensemble_members)])
        step = self.rating_step
        # Synthetic's declared step does not imply its generated labels are
        # quantized. Validate TRAIN labels before choosing an ordinal likelihood.
        if step is not None:
            t = (data.train_ratings_raw-self.lo)/step
            if not torch.allclose(t, t.round(), atol=2e-5, rtol=0):
                step = None
        self.ordinal = step is not None
        levels = (torch.arange(self.lo, self.hi+step*.5, step, device=data.train_users.device)
                  if self.ordinal else torch.linspace(self.lo, self.hi, 129, device=data.train_users.device))
        self.register_buffer("levels", levels)
        shape_dim = len(levels) if self.ordinal and cfg.shape_full else (1 if self.ordinal or not cfg.shape_full else 3)
        # Pair features: two 13-dimensional clean histories + mean and cosine.
        self.shape_heads = nn.ModuleList([nn.Sequential(nn.Linear(28, 32), nn.GELU(),
                                                        nn.Linear(32, shape_dim))
                                          for _ in self.members])
        for head in self.shape_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.shape_heads.requires_grad_(False)
        self.register_buffer("shape_fitted", torch.tensor(False))

    def prepare(self, h):
        return {"members": [member.prepare(h) for member in self.members]}

    def member_predictions(self, h, state, u, i):
        return torch.stack([member.predict(h, s, u, i)
                            for member, s in zip(self.members, state["members"])])

    def predict(self, h, s, u, i):
        means = self.member_predictions(h, s, u, i)
        return means.mean(0), {"member_mean_disagreement": means.var(0, unbiased=False)}

    def loss_for_targets(self, h, s, u, i, y):
        means = self.member_predictions(h, s, u, i)
        # Separate MSE members. Their mixture is also reported; probability loss
        # never updates a mean parameter. Deterministic bootstrap weights keep
        # seeds distinct without dropping entire users from an ensemble member.
        pair = u*self.ni+i
        weights = []
        for member in range(len(means)):
            key = (pair+(member+1)*1000003+self.model_seed*65537) & 0x7fffffff
            key = ((key ^ (key >> 16))*0x45d9f3b) & 0x7fffffff
            key = ((key ^ (key >> 16))*0x45d9f3b) & 0x7fffffff
            hashed = (key ^ (key >> 16)).double()
            uniform = (hashed+.5)/2147483647
            weights.append((-uniform.log()).float())
        weight = torch.stack(weights)
        weight = weight/weight.mean(-1, keepdim=True).clamp_min(1e-5)
        lp = (weight*((means-y[None])/self.width).square()).mean()
        aux = y.new_zeros(())
        n = max(1, self._data.train_users.numel())
        for member in self.members:
            aux = aux+self.cfg.bias_prior*(member.bias.user.weight.square().sum()
                  +member.bias.item.weight.square().sum())/n/len(means)
        aux = aux+self.common_prior(s)
        return lp, aux, {"L_pred": float(lp.detach()), "L_auxiliary": float(aux.detach())}

    def _shape_features(self, h, state, mean, u, i):
        cosine = (state["qu"][u]*state["qi"][i]).sum(-1)
        return torch.cat([h.user_features[u], h.item_features[i],
                           ((mean-self.mid)/self.width)[:, None], cosine[:, None]], -1).detach()

    def _shape_states(self, h, state, u, i):
        means = self.member_predictions(h, state, u, i)
        shapes = [head(self._shape_features(h, member_state, means[j], u, i))
                  for j, (head, member_state) in enumerate(zip(self.shape_heads, state["members"]))]
        return means, shapes

    def _beta_parameters(self, shape, mean):
        eps = torch.finfo(mean.dtype).eps
        p = ((mean.detach()-self.lo)/self.width).clamp(eps, 1-eps)
        concentration = F.softplus(shape[:, :2])+0.1
        if not self.cfg.shape_full:
            concentration = concentration.expand(-1, 2)
        logit = shape[:, 2] if self.cfg.shape_full else torch.zeros_like(p)
        # Never recover a log-weight from a rounded sigmoid. Sigmoid(17) is
        # already exactly 1 in float32, while these log-weights and gradients
        # remain finite for either sign and for strongly saturated logits.
        log_weights = torch.stack([F.logsigmoid(logit), F.logsigmoid(-logit)], -1)
        return p[:, None]*concentration, (1-p)[:, None]*concentration, torch.sigmoid(logit), log_weights

    def probability_members(self, h, state, u, i):
        means, shapes = self._shape_states(h, state, u, i)
        probabilities, beta_parameters = [], []
        for j, shape in enumerate(shapes):
            if self.ordinal:
                if not self.cfg.shape_full:
                    r = (self.levels-self.mid)/self.half
                    shape = shape*r.square()[None]
                probabilities.append(mean_tilt(shape, means[j].detach(), self.levels))
            else:
                # Continuous fallback: two beta components have the SAME mean.
                # It is a different, explicitly labelled law, not a fictitious PMF
                # for continuous Jester/synthetic labels.
                alpha, beta, mix, _ = self._beta_parameters(shape, means[j])
                beta_parameters.append((alpha, beta, mix))
        return means, probabilities, beta_parameters

    def distribution(self, data, u, i):
        h, state = self.eval_state()
        means, prob, beta = self.probability_members(h, state, u, i)
        if self.ordinal:
            stack = torch.stack(prob)
            total, within, between = mixture_variance(stack, means.detach(), self.levels)
            return dict(mean=means.mean(0), probabilities=stack.mean(0), variance=total,
                        within_variance=within, between_variance=between, levels=self.levels)
        member_var = []
        for a, b, weight in beta:
            variance = self.width**2*a*b/((a+b).square()*(a+b+1))
            member_var.append(weight*variance[:, 0]+(1-weight)*variance[:, 1])
        within = torch.stack(member_var).mean(0)
        between = means.var(0, unbiased=False)
        return dict(mean=means.mean(0), variance=within+between, within_variance=within,
                    between_variance=between, beta_parameters=beta)

    def probability_loss(self, h, state, u, i, y):
        if self.ordinal:
            _, prob, _ = self.probability_members(h, state, u, i)
            return ordinal_nll(torch.stack(prob).mean(0), y, self.levels).mean()
        means, shapes = self._shape_states(h, state, u, i)
        t = ((y-self.lo)/self.width).clamp(1e-5, 1-1e-5)
        logs = []
        for mean, shape in zip(means, shapes):
            a, b, _, log_weights = self._beta_parameters(shape, mean)
            logpdf = ((a-1)*t[:, None].log()+(b-1)*torch.log1p(-t[:, None])
                      -torch.lgamma(a)-torch.lgamma(b)+torch.lgamma(a+b)-math.log(self.width))
            logs.append(torch.logsumexp(logpdf+log_weights, -1))
        return -(torch.logsumexp(torch.stack(logs), 0)-math.log(len(logs))).mean()


MODEL_CLASSES = {f"Method{i}": cls for i, cls in enumerate(
    [Method1, Method2, Method3, Method4, Method5, Method6, Method7], 1)}


def instantiate(data, config, seed=1, device=None):
    cfg = copy.deepcopy(config)
    cfg.validate()
    device = data.train_users.device if device is None else device
    # Isolated construction avoids contaminating caller RNG when checking a saved
    # checkpoint. CUDA callers still seed their model explicitly in the runtime.
    target_device = torch.device(device)
    devices = ([target_device.index if target_device.index is not None else torch.cuda.current_device()]
               if target_device.type == "cuda" else [])
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(int(seed))
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(int(seed))
        model = MODEL_CLASSES[cfg.method](data, cfg, seed).to(device)
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and "direction" in name:
            model.register_buffer("initial_"+name.replace(".", "_"), unit(module.weight.detach().clone()))
    return model
