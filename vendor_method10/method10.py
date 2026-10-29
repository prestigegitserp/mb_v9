"""Method 10: dual autoencoders with a bounded spherical final decoder.

Only the last dot product is changed. Directions come from the actual hidden
activations and output weights; there are no independent angular embeddings.
The objective is the observed-entry norm of the *fused* reconstruction error.
This extension does not modify, import, or monkey-patch the frozen V7/V8 models.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

METHOD10_VERSION = "1.0.0"


@dataclass(frozen=True)
class Method10Config:
    latent_dim: int = 32
    hidden_dim: int = 128
    decoder_dim: int = 128
    radius_temperature: float = 1.0
    loss_reduction: str = "frobenius"
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 200
    eval_every: int = 1
    val_patience: int = 20
    grad_clip: float = 5.0
    chunk_size: int = 65_536

    def __post_init__(self):
        for name in ("latent_dim", "hidden_dim", "decoder_dim", "max_epochs",
                     "eval_every", "val_patience", "chunk_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("radius_temperature", "lr", "grad_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.loss_reduction not in ("frobenius", "squared_frobenius", "mse"):
            raise ValueError("unknown loss_reduction")


def bounded_spherical(v: torch.Tensor, cap: float, temperature: float) -> torch.Tensor:
    """rho(v) * direction(v), with rho = cap*tanh(||v||/temperature).

    This is the spherical computation without explicit atan/acos, which have
    coordinate singularities. A small-norm Taylor expansion gives the correct
    continuous limit and nonzero Jacobian cap/temperature at the zero vector.
    The direction is unchanged and the norm is at most cap (roundoff aside).
    """
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    z = norm / temperature
    # Safe denominator also in the inactive branch: no hidden division by zero.
    regular = cap * torch.tanh(z) / norm.clamp_min(temperature * 1e-4)
    small_z = z.clamp_max(1e-4)  # Keep the inactive Taylor branch finite, too.
    near_zero = (cap / temperature) * (1 - small_z.square() / 3 + 2 * small_z.pow(4) / 15)
    scale = torch.where(z < 1e-4, near_zero, regular)
    return v * scale


def observed_fusion_loss(user_view: torch.Tensor, item_view: torch.Tensor,
                         target: torch.Tensor, reduction: str = "frobenius") -> torch.Tensor:
    """All arguments are vectors of the same OBSERVED user-item pairs.

    Missing entries never occur here. For matrix predictions, select mask==1
    first. Averaging two separate losses would define a different objective.
    """
    if user_view.shape != item_view.shape or user_view.shape != target.shape:
        raise ValueError("views and target must have identical shapes")
    if not target.numel():
        raise ValueError("at least one observed rating is required")
    residual = (user_view + item_view) * 0.5 - target
    if reduction == "frobenius":
        return torch.linalg.vector_norm(residual.reshape(-1))
    if reduction == "squared_frobenius":
        return residual.square().sum()
    if reduction == "mse":
        return residual.square().mean()
    raise ValueError(f"unknown reduction: {reduction}")


class HistoryAutoencoder(nn.Module):
    """d -> hidden(ReLU) -> K(ReLU) -> D(linear) -> d(spherical).

    The first linear layer accepts sparse histories exactly as its dense
    equivalent would. The decoder's final output weight has shape [d, D].
    """
    def __init__(self, input_dim: int, cfg: Method10Config):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, cfg.hidden_dim)
        self.encoder = nn.Sequential(nn.ReLU(), nn.Linear(cfg.hidden_dim, cfg.latent_dim), nn.ReLU())
        self.decoder_hidden = nn.Linear(cfg.latent_dim, cfg.decoder_dim)
        self.output_weight = nn.Parameter(torch.empty(input_dim, cfg.decoder_dim))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_weight)

    def penultimate(self, history: torch.Tensor) -> torch.Tensor:
        if history.layout != torch.strided:
            x = torch.sparse.mm(history, self.input_layer.weight.T) + self.input_layer.bias
        else:
            x = self.input_layer(history)
        return self.decoder_hidden(self.encoder(x))


class Method10(nn.Module):
    """Two complete AEs; only their final output dot products are spherical.

    `data` follows the project's dense/sparse MatrixData interface. Only TRAIN
    triples enter the encoder histories, including during val/test prediction.
    """
    def __init__(self, data: Any, cfg: Method10Config | None = None,
                 device: torch.device | str | None = None):
        super().__init__()
        self.cfg = cfg or Method10Config()
        self.num_users, self.num_items = int(data.num_users), int(data.num_items)
        self.rating_min, self.rating_max = float(data.rating_min), float(data.rating_max)
        if min(self.num_users, self.num_items) <= 0:
            raise ValueError("user/item counts must be positive")
        if not (math.isfinite(self.rating_min) and math.isfinite(self.rating_max)
                and self.rating_min < self.rating_max):
            raise ValueError("a finite documented rating_min < rating_max is required")
        self.midpoint = (self.rating_min + self.rating_max) / 2
        self.half_range = (self.rating_max - self.rating_min) / 2
        self.radius_cap = math.sqrt(self.half_range)
        device = torch.device(device) if device is not None else data.train_ratings_raw.device
        u = data.train_users.detach().to(device=device, dtype=torch.long)
        i = data.train_items.detach().to(device=device, dtype=torch.long)
        r = data.train_ratings_raw.detach().to(device=device, dtype=torch.float32)
        if r.ndim != 1 or u.shape != r.shape or i.shape != r.shape or not r.numel():
            raise ValueError("nonempty one-dimensional training triples are required")
        if not torch.isfinite(r).all() or (r < self.rating_min).any() or (r > self.rating_max).any():
            raise ValueError("training ratings must be finite and inside the documented range")
        if (u < 0).any() or (u >= self.num_users).any() or (i < 0).any() or (i >= self.num_items).any():
            raise ValueError("training indices are outside the matrix")
        x = torch.sparse_coo_tensor(torch.stack((u, i)), (r - self.midpoint) / self.half_range,
                                    (self.num_users, self.num_items), device=device,
                                    check_invariants=True).coalesce()
        if x._nnz() != r.numel():
            raise ValueError("duplicate training user-item pairs are not allowed")
        self.register_buffer("user_history", x, persistent=False)
        self.register_buffer("item_history", x.transpose(0, 1).coalesce(), persistent=False)
        self.user_ae = HistoryAutoencoder(self.num_items, self.cfg)
        self.item_ae = HistoryAutoencoder(self.num_users, self.cfg)
        self.to(device)

    def factors(self) -> tuple[torch.Tensor, ...]:
        """Bounded (H_user, W_user, H_item, W_item); all are [entities, D]."""
        def sphere(x):
            return bounded_spherical(x, self.radius_cap, self.cfg.radius_temperature)
        return (sphere(self.user_ae.penultimate(self.user_history)),
                sphere(self.user_ae.output_weight),
                sphere(self.item_ae.penultimate(self.item_history)),
                sphere(self.item_ae.output_weight))

    def predict_views_pairs(self, users: torch.Tensor, items: torch.Tensor,
                           factors: tuple[torch.Tensor, ...] | None = None):
        hu, wu, hi, wi = self.factors() if factors is None else factors
        ru = self.midpoint + (hu[users] * wu[items]).sum(-1)
        ri = self.midpoint + (hi[items] * wi[users]).sum(-1)
        return ru, ri

    def predict_raw_pairs(self, data: Any, users: torch.Tensor, items: torch.Tensor):
        """Compatibility with the project's evaluation API; history stays TRAIN-only."""
        ru, ri = self.predict_views_pairs(users, items)
        return (ru + ri) * 0.5

    def reconstruct_views(self):
        """For small matrices only: user output [U,I], item output [I,U]."""
        hu, wu, hi, wi = self.factors()
        return self.midpoint + hu @ wu.T, self.midpoint + hi @ wi.T

    def forward(self):
        """Full fused reconstruction [U,I]; training uses memory-safe pairs instead."""
        ru, ri = self.reconstruct_views()
        return (ru + ri.T) * 0.5


def _slices(n: int, size: int):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def backward_observed_objective(model: Method10, users: torch.Tensor,
                                items: torch.Tensor, target: torch.Tensor) -> float:
    """Exact full-observed-set gradient, without materialising U x I predictions.

    Accumulate chunk gradients at detached factor leaves, then backpropagate
    through the two AEs once. The global Frobenius norm is NOT computed as an
    average of chunk norms (that would be a different objective).
    Call optimizer.zero_grad() before and optimizer.step() after this function.
    """
    if not target.numel():
        raise ValueError("empty training observations")
    factors = model.factors()
    with torch.no_grad():
        squared = torch.zeros((), device=target.device, dtype=torch.float64)
        for sl in _slices(target.numel(), model.cfg.chunk_size):
            ru, ri = model.predict_views_pairs(users[sl], items[sl], factors)
            squared += (((ru + ri) * 0.5 - target[sl]).double().square()).sum()
        sse = squared.item()
    if not math.isfinite(sse):
        raise FloatingPointError("nonfinite training loss")
    reduction = model.cfg.loss_reduction
    if reduction == "frobenius":
        value = math.sqrt(sse)
        coefficient = 0.5 / value if value > 0 else 0.0
    elif reduction == "mse":
        value, coefficient = sse / target.numel(), 1.0 / target.numel()
    else:
        value, coefficient = sse, 1.0
    leaves = tuple(f.detach().requires_grad_(True) for f in factors)
    for sl in _slices(target.numel(), model.cfg.chunk_size):
        ru, ri = model.predict_views_pairs(users[sl], items[sl], leaves)
        residual = (ru + ri) * 0.5 - target[sl]
        (coefficient * residual.square().sum()).backward()
    torch.autograd.backward(factors, tuple(leaf.grad for leaf in leaves))
    return value


@torch.no_grad()
def evaluate_method10(model: Method10, users: torch.Tensor, items: torch.Tensor,
                      target: torch.Tensor, *, return_predictions: bool = False):
    if not target.numel():
        raise ValueError("cannot evaluate an empty split")
    model.eval()
    factors = model.factors()
    squared = absolute = violations = 0.0
    columns = []
    for sl in _slices(target.numel(), model.cfg.chunk_size):
        ru, ri = model.predict_views_pairs(users[sl], items[sl], factors)
        pred = (ru + ri) * 0.5
        error = (pred - target[sl]).double()
        squared += error.square().sum().item()
        absolute += error.abs().sum().item()
        violations += ((pred < model.rating_min - 1e-6) | (pred > model.rating_max + 1e-6)).sum().item()
        if return_predictions:
            columns.append(torch.stack((ru, ri, pred), -1).cpu())
    n = target.numel()
    metrics = {"RMSE": math.sqrt(squared / n), "MAE": absolute / n,
               "Frobenius": math.sqrt(squared), "out_of_range_fraction": violations / n, "n": n}
    return (metrics, torch.cat(columns)) if return_predictions else metrics


def train_method10(model: Method10, data: Any, *, verbose: bool = True):
    """Adam; select the best epoch on validation RMSE, then restore its weights.

    No test tensor is read here. Default weight_decay=0 gives exactly the user's
    fused reconstruction objective, with no extra reconstruction/consistency loss.
    """
    if not data.val_ratings_raw.numel():
        raise ValueError("a nonempty validation split is required for epoch selection")
    cfg = model.cfg
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history, best_state = [], None
    best_val, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        value = backward_observed_objective(model, data.train_users, data.train_items, data.train_ratings_raw)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        row = {"epoch": epoch, "train_objective_before_step": value}
        if epoch % cfg.eval_every == 0 or epoch == cfg.max_epochs:
            metrics = evaluate_method10(model, data.val_users, data.val_items, data.val_ratings_raw)
            row.update({f"val_{key}": val for key, val in metrics.items()})
            if not math.isfinite(metrics["RMSE"]):
                raise FloatingPointError("nonfinite validation RMSE")
            if metrics["RMSE"] < best_val:
                best_val, best_epoch, stale = metrics["RMSE"], epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            if verbose:
                print(f"Method10 epoch={epoch} loss={value:.6f} val_RMSE={metrics['RMSE']:.6f}", flush=True)
        history.append(row)
        if stale >= cfg.val_patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "best_val_RMSE": best_val,
            "epochs_completed": len(history), "history": history}


def _history_signature(data: Any):
    """Match the exact training history and dimensions when restoring a model."""
    order = torch.argsort(data.train_users * int(data.num_items) + data.train_items)
    digest = hashlib.sha256()
    for values in (data.train_users.long(), data.train_items.long(), data.train_ratings_raw.float()):
        digest.update(values[order].detach().cpu().contiguous().numpy().tobytes())
    return {"shape": [int(data.num_users), int(data.num_items)],
            "rating_range": [float(data.rating_min), float(data.rating_max)],
            "train_sha256": digest.hexdigest()}


def save_method10(path: str | Path, model: Method10, data: Any, train_info: dict):
    torch.save({"method": "Method10", "version": METHOD10_VERSION,
                "config": asdict(model.cfg), "data_signature": _history_signature(data),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "train_info": train_info}, path)


def load_method10(path: str | Path, data: Any, device: torch.device | str = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("method") != "Method10" or payload.get("version") != METHOD10_VERSION:
        raise ValueError("unsupported Method10 checkpoint")
    if payload["data_signature"] != _history_signature(data):
        raise ValueError("checkpoint training history, IDs, or rating scale does not match")
    model = Method10(data, Method10Config(**payload["config"]), device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
