"""Explicit, serializable architecture and training choices.

Defaults are research hypotheses, not MovieLens-tuned optima.  An epoch contains
several independently masked prediction steps; both steps and target counts are
reported, so these epochs must not be equated with V2 full-batch updates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from . import __version__


def implementation_signature():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


@dataclass
class ModelConfig:
    method: str = "Method1"
    latent_dim: int = 20
    hidden_dim: int = 128
    max_epochs: int = 300
    eval_every: int = 5
    val_patience: int = 16
    steps_per_epoch: int = 5
    mask_fraction: float = 0.20
    warmup_steps: int = 20
    lr: float = 0.001
    direction_lr: float = 0.002
    bias_lr: float = 0.003
    grad_clip: float = 5.0
    encoder_decay_budget: float = 0.05
    code_prior: float = 0.0001
    bias_prior: float = 0.001
    direction_prior: float = 0.0001
    reconstruction_weight: float = 0.05
    auxiliary_weight: float = 0.20
    consistency_weight: float = 0.02
    count_tau: float = 10.0
    context_strength: float = 0.35
    tangent_limit: float = 0.50
    steering_enabled: bool = True
    radius_enabled: bool = True
    information_filter: bool = True
    information_ridge: float = 1.0
    gradient_surgery: bool = True
    auxiliary_gradient_ratio: float = 0.30
    router_folds: int = 3
    router_expert_epochs: int = 150
    router_epochs: int = 200
    router_width: int = 16
    router_replicates: int = 3
    router_ridge: float = 0.02
    router_uncertainty: float = 1.0
    router_enabled: bool = True
    router_counts_only: bool = False
    log_radius_bound: float = 2.0
    radius_precision: float = 2.0
    radius_noise: float = 0.25
    radius_refresh: int = 5
    radius_trust_step: float = 0.20
    cg_steps: int = 25
    graph_probes: int = 12
    graph_uncertainty: bool = True
    signed_moments: bool = True
    project_moments: bool = True
    prior_interior_median: float = 0.60
    posterior_precision: float = 20.0
    posterior_noise: float = 0.25
    posterior_probes: int = 8
    posterior_cg_steps: int = 12
    posterior_correlation_init: float = 0.05
    joint_posterior: bool = True
    posterior_cross_covariance: bool = True
    score_temperature: float = 0.40
    radius_temperature: float = 0.0
    transport_enabled: bool = True
    learned_transport: bool = True
    transport_nodes: int = 33
    transport_slope_margin: float = 3.0
    angular_grid: int = 257
    angular_reference_pairs: int = 8192
    angular_reference_weight: float = 0.50
    conditional_reference: bool = True
    reference_baseline_bins: int = 3
    reference_gain_bins: int = 3
    balance_transport: bool = True
    decoder_coupling: bool = True
    ensemble_members: int = 3
    mean_interior_epsilon: float = 1e-5
    feature_dropout: float = 0.15
    shape_epochs: int = 200
    shape_lr: float = 0.001
    shape_patience: int = 20
    shape_full: bool = True
    batch_size: int = 4096

    def validate(self):
        flags = {"information_filter", "gradient_surgery", "router_enabled", "router_counts_only",
                 "steering_enabled", "radius_enabled", "posterior_cross_covariance",
                 "conditional_reference", "balance_transport", "decoder_coupling", "graph_uncertainty",
                 "signed_moments", "project_moments", "joint_posterior", "transport_enabled",
                 "learned_transport", "shape_full"}
        for name, value in vars(self).items():
            if name == "method":
                continue
            if name in flags:
                if not isinstance(value, bool):
                    raise ValueError(f"{name} must be a boolean")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        integer_fields = ("latent_dim", "hidden_dim", "max_epochs", "eval_every", "val_patience",
                          "steps_per_epoch", "warmup_steps", "router_folds", "router_expert_epochs",
                          "router_epochs", "router_width", "router_replicates", "radius_refresh",
                          "cg_steps", "graph_probes", "posterior_probes", "posterior_cg_steps",
                          "transport_nodes", "angular_grid", "angular_reference_pairs", "reference_baseline_bins",
                          "reference_gain_bins", "ensemble_members",
                          "shape_epochs", "shape_patience", "batch_size")
        for name in integer_fields:
            if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be an integer")
        if self.method not in {f"Method{i}" for i in range(1, 8)}:
            raise ValueError("Unknown V3 method")
        if self.latent_dim < 2 or self.hidden_dim < 4:
            raise ValueError("V3 requires K >= 2 and hidden_dim >= 4")
        if self.method == "Method4" and self.signed_moments and self.latent_dim < 4:
            raise ValueError("M4 signed moments need K >= 4 to avoid one-dimensional sign-only channels")
        for name in ("max_epochs", "eval_every", "val_patience", "steps_per_epoch",
                     "cg_steps", "graph_probes", "posterior_probes", "batch_size",
                     "router_epochs", "router_expert_epochs", "router_width", "router_replicates",
                     "shape_epochs", "shape_patience", "ensemble_members", "posterior_cg_steps",
                     "radius_refresh", "angular_reference_pairs", "angular_grid", "reference_baseline_bins",
                     "reference_gain_bins"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.mask_fraction < 1:
            raise ValueError("mask_fraction must be strictly between 0 and 1")
        if not 0 <= self.encoder_decay_budget < 1:
            raise ValueError("encoder_decay_budget must be in [0,1)")
        for name in ("lr", "direction_lr", "bias_lr", "shape_lr", "radius_precision", "posterior_precision",
                     "radius_noise", "posterior_noise", "count_tau", "information_ridge", "log_radius_bound",
                     "radius_trust_step", "score_temperature", "router_ridge", "transport_slope_margin"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("code_prior", "bias_prior", "direction_prior", "reconstruction_weight",
                     "auxiliary_weight", "consistency_weight", "context_strength", "router_uncertainty",
                     "auxiliary_gradient_ratio", "radius_temperature", "tangent_limit", "warmup_steps", "grad_clip"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 <= self.angular_reference_weight <= 1 or not 0 <= self.feature_dropout < 1:
            raise ValueError("Invalid reference mixture or feature dropout")
        if not 0 < self.prior_interior_median < 1 or not 0 < self.posterior_correlation_init < 1:
            raise ValueError("Interior radius prior and initial correlation must be strictly inside (0,1)")
        if not 0 < self.mean_interior_epsilon < .01:
            raise ValueError("mean_interior_epsilon must be inside (0,.01)")
        if self.transport_nodes < 17 or self.transport_nodes % 2 != 1:
            raise ValueError("transport_nodes must be odd and >= 17")
        if self.angular_grid < 3:
            raise ValueError("angular_grid must be >= 3")
        if self.router_folds < 2:
            raise ValueError("At least two folds are needed for honest routing")
        return self


def default_models():
    out = {}
    for i in range(1, 8):
        c = ModelConfig(method=f"Method{i}")
        if i in (2, 3, 4):
            c.hidden_dim = 32
        if i in (3, 4):
            c.max_epochs = 450  # identical budgets for the two radius mechanisms
        if i == 7:
            c.hidden_dim = 64
        out[c.method] = c
    return out


@dataclass
class StudyConfig:
    architecture_version: str = __version__
    implementation_sha256: str = field(default_factory=implementation_signature)
    profile: str = "balanced"
    role_seed: int = 73019
    validation_selection_fraction: float = 0.60
    validation_calibration_fraction: float = 0.20
    models: dict[str, ModelConfig] = field(default_factory=default_models)

    def validate(self):
        if not isinstance(self.role_seed, int) or isinstance(self.role_seed, bool) or self.role_seed < 0:
            raise ValueError("role_seed must be a nonnegative integer")
        if not 0 < self.validation_selection_fraction < 1:
            raise ValueError("Invalid validation selection fraction")
        if not 0 < self.validation_calibration_fraction < 1-self.validation_selection_fraction:
            raise ValueError("A disjoint calibration holdout is required")
        for model in self.models.values():
            model.validate()
        if set(self.models) != {f"Method{i}" for i in range(1, 8)}:
            raise ValueError("A V3 study must declare all seven model configurations")
        if any(name != model.method for name, model in self.models.items()):
            raise ValueError("Model configuration identity mismatch")
        return self
