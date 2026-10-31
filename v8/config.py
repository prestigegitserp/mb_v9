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
    presentation = {"export.py","figures.py","publication.py","reporting.py","search_report.py","fidelity.py","secondary_report.py"}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if path.name in presentation:
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    root = Path(__file__).resolve().parents[1]
    dependencies = [root / "examples/v8/historical_incumbents.json", root / "vendor_method10/method10.py", root / "examples/v8/validated_candidates.json"]
    for namespace in ("v3", "v4", "v52", "v7"):
        dependencies.extend(sorted((root / namespace).glob("*.py")))
    for path in dependencies:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


@dataclass
class ModelConfig:
    method: str = "Method1"
    lineage: str = "v8"
    response_shrinkage: bool = True
    encoder_decay_horizon_epochs: int = 80
    prediction_loss_units: str = "mse"
    fused_history: bool = True
    pair_dot_chunk: int = 32768
    history_signal: str = "residual"
    history_gate_support_shrinkage: bool = False
    latent_dim: int = 20
    hidden_dim: int = 128
    max_epochs: int = 300
    eval_every: int = 5
    val_patience: int = 16
    steps_per_epoch: int = 5
    mask_fraction: float = 0.20
    mask_fraction_jitter: float = 0.0
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
    posterior_cg_steps: int = 64
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
    readout_mode: str = "geometric"
    readout_weight: float = 0.5
    readout_count_tau: float = 10.0
    readout_count_adaptive: bool = True
    moment_refinement_scale: float = 0.25
    moment_refinement_link: str = "tanh"
    collaborative_l2: float = 0.0
    history_activation: str = "gelu"
    history_normalization: str = "layernorm"
    history_pooling_power: float = 0.5
    wide_context_readout: bool = False
    mechanism_auxiliary_weight: float = 1.0
    full_history_weight: float = 0.0
    probability_fit: bool = True
    history_degree_gate: bool = True
    history_residual_skip: bool = True
    cavity_degree_normalization: bool = True
    radius_line_search: bool = True
    moment_context_prior: bool = True
    posterior_degree_precision: bool = True
    transport_support_shrinkage: bool = True
    router_support_shrinkage: bool = True
    ensemble_bootstrap_temperature: float = 0.5

    tangent_cavity: bool = True
    cavity_strength: float = 0.5
    router_curvature: bool = True
    router_context_only: bool = True
    router_fine_benefit: bool = True
    router_context_penalty: float = 0.25
    context_radius: bool = True
    graph_context_directions: bool = True
    moment_refinement: bool = True
    orthogonal_moment_loss: bool = True
    moment_dispersion_weight: float = 0.1
    moment_projection_weight: float = 1.0
    moment_projection_mode: str = "metric"
    posterior_mean_steps: int = 1
    posterior_relaxation: float = 0.25
    posterior_preconditioned: bool = True
    posterior_solver_tolerance: float = 1e-4
    posterior_cg_check_every: int = 4
    hierarchical_reference: bool = True
    hard_transport_balance: bool = False
    transport_moment_weight: float = 0.02
    identity_history: bool = True
    ensemble_mean_weight: float = 0.25
    probability_rps_weight: float = 0.05
    adaptive_readout: bool = True
    readout_gate_prior: float = 0.001
    calibrate_geometry: bool = True
    continuous_response: bool = True
    gradient_diagnostics: bool = False
    gradient_diagnostics_every: int = 10

    def validate(self):
        if not isinstance(self.fused_history,bool) or self.history_signal not in ("residual","raw","centered","gated","mixed"):
            raise ValueError("Invalid history computation/signal")
        if isinstance(self.pair_dot_chunk,bool) or not isinstance(self.pair_dot_chunk,int) or self.pair_dot_chunk<1:
            raise ValueError("Positive pair-dot computational chunk required")
        if self.lineage not in ("v3", "v4", "v52", "v8"):
            raise ValueError("Unknown scientific model lineage")
        if self.prediction_loss_units not in ("mse", "gaussian"):
            raise ValueError("Unknown prediction loss units")
        if not isinstance(self.response_shrinkage, bool):
            raise ValueError("response_shrinkage must be boolean")
        if isinstance(self.encoder_decay_horizon_epochs, bool) or not isinstance(self.encoder_decay_horizon_epochs, int) or self.encoder_decay_horizon_epochs < 1:
            raise ValueError("Positive encoder decay horizon required")
        flags = {"information_filter", "gradient_surgery", "router_enabled", "router_counts_only",
                 "steering_enabled", "radius_enabled", "posterior_cross_covariance",
                 "conditional_reference", "balance_transport", "decoder_coupling", "graph_uncertainty",
                 "signed_moments", "project_moments", "joint_posterior", "transport_enabled",
                 "learned_transport", "shape_full"}
        flags.update({"tangent_cavity", "router_curvature", "router_context_only", "router_fine_benefit",
                      "context_radius", "graph_context_directions", "moment_refinement", "orthogonal_moment_loss",
                      "posterior_preconditioned", "hierarchical_reference", "hard_transport_balance", "identity_history"})
        flags.update({"readout_count_adaptive", "probability_fit", "wide_context_readout",
                      "adaptive_readout", "calibrate_geometry", "continuous_response", "gradient_diagnostics"})
        flags.update({"history_degree_gate", "history_residual_skip", "history_gate_support_shrinkage", "cavity_degree_normalization",
                      "radius_line_search", "moment_context_prior", "posterior_degree_precision",
                      "transport_support_shrinkage", "router_support_shrinkage"})
        for name, value in vars(self).items():
            if name in ("method", "lineage", "response_shrinkage", "prediction_loss_units", "fused_history", "history_signal", "moment_projection_mode", "readout_mode", "moment_refinement_link", "history_activation", "history_normalization"):
                continue
            if name in flags:
                if not isinstance(value, bool):
                    raise ValueError(f"{name} must be a boolean")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("posterior_mean_steps", "posterior_cg_check_every"):
            if isinstance(getattr(self,name),bool) or not isinstance(getattr(self,name),int):
                raise ValueError(name+" must be an integer")
        if self.posterior_mean_steps<0 or self.posterior_cg_check_every<1 or not 0<self.posterior_relaxation<=1:
            raise ValueError("Invalid posterior inference configuration")
        if not 0<self.posterior_solver_tolerance<1 or not 0<=self.ensemble_mean_weight<=1:
            raise ValueError("Invalid solver tolerance or mean loss weight")
        if self.moment_projection_weight <= 0:
            raise ValueError("moment_projection_weight must be strictly positive")
        if self.moment_projection_mode not in ("metric", "mean-priority"):
            raise ValueError("Invalid moment_projection_mode")
        if self.history_activation not in ("gelu", "sigmoid", "relu", "tanh") or self.history_normalization not in ("layernorm", "none"):
            raise ValueError("Invalid history activation/normalization")
        if not 0 <= self.mechanism_auxiliary_weight <= 1:
            raise ValueError("Invalid mechanism auxiliary weight")
        if not 0 <= self.history_pooling_power <= 1 or not 0 <= self.full_history_weight < 1:
            raise ValueError("Invalid history pooling or hybrid reconstruction weight")
        if self.readout_mode not in ("geometric", "coupled", "decoder", "residual"):
            raise ValueError("Invalid readout_mode")
        if not 0 <= self.readout_weight <= 1 or self.readout_count_tau <= 0:
            raise ValueError("Invalid readout weight or count prior")
        if self.moment_refinement_link not in ("tanh", "linear") or self.moment_refinement_scale < 0:
            raise ValueError("Invalid moment refinement link or scale")
        if self.collaborative_l2<0:raise ValueError('collaborative_l2 must be nonnegative')
        if self.readout_gate_prior < 0:
            raise ValueError('readout_gate_prior must be nonnegative')
        if isinstance(self.gradient_diagnostics_every, bool) or not isinstance(self.gradient_diagnostics_every, int) or self.gradient_diagnostics_every < 1:
            raise ValueError('gradient_diagnostics_every must be a positive integer')
        if not 0 <= self.ensemble_bootstrap_temperature <= 1:
            raise ValueError('ensemble_bootstrap_temperature must be in [0,1]')
        if self.method == "Method7" and self.readout_mode != "geometric" and not self.identity_history:
            raise ValueError("M7 context readout requires identity histories")
        if self.method == "Method3" and self.readout_mode != "geometric" and not self.context_radius:
            raise ValueError("M3 context readout requires context-specific radii")
        for name in ("cavity_strength", "router_context_penalty", "moment_dispersion_weight", "transport_moment_weight", "probability_rps_weight"):
            if getattr(self,name)<0:raise ValueError(name+" must be nonnegative")
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
            raise ValueError("Unknown V8 method")
        if self.latent_dim < 2 or self.hidden_dim < 4:
            raise ValueError("V8 requires K >= 2 and hidden_dim >= 4")
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
        if not 0 <= self.mask_fraction_jitter < min(self.mask_fraction, 1-self.mask_fraction):
            raise ValueError("mask_fraction_jitter must keep every context fraction inside (0,1)")
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


def _previous_defaults():
    out = {}
    for i in range(1, 8):
        c = ModelConfig(method=f"Method{i}")
        if i in (2, 3, 4):
            c.hidden_dim = 32
        if i in (3, 4):
            c.max_epochs = 450  # identical budgets for the two radius mechanisms
        if i == 7:
            c.hidden_dim = 64
        if i in (1,6):c.hidden_dim=64
        if i==5:c.eval_every=1
        if i==4:
            c.moment_projection_weight=c.moment_dispersion_weight
            c.moment_projection_mode="mean-priority"
            c.moment_refinement_link="linear"
            c.moment_refinement_scale=1.0
            c.collaborative_l2=1000.0
        if i in (1,2,3,4,5,7):c.mask_fraction_jitter=.15
        if i==1:
            # The normalized-information hypothesis lost the matched-budget
            # validation comparison; keep it as an explicit experiment.
            c.cavity_degree_normalization=False
            c.history_degree_gate=False
        if i==5:
            c.posterior_degree_precision=False
            c.history_degree_gate=False
            c.collaborative_l2=0.0
        if i in (2,3,5,6):c.readout_mode="coupled"
        if i==2:c.collaborative_l2=10.0
        if i==6:
            c.collaborative_l2=1000.0
            c.context_strength=2.0
            c.angular_reference_weight=0.0
        if i==7:c.ensemble_mean_weight=.5
        out[c.method] = c
    return out


def default_models():
    from .lineage import historical_models
    return historical_models()


@dataclass
class StudyConfig:
    architecture_version: str = __version__
    implementation_sha256: str = field(default_factory=implementation_signature)
    method8: dict = field(default_factory=dict)
    search_seed: int = 1
    profile: str = "balanced"
    role_seed: int = 73019
    validation_selection_fraction: float = 0.60
    validation_calibration_fraction: float = 0.20
    search_jobs_per_method: int = 0
    split_validation_roles: bool = True
    protocol: str = "common"
    mbmf_checkpoint_policy: str = "validation"
    mbmf_val_patience: int = 16
    mf_max_epochs: int = 1000
    mbmf_max_epochs: int = 5000
    mbmf_optimizer: str = "paper"
    mbmf_adam_lr: float = 0.003
    aemc_main_activation: str = "sigmoid"
    aemc_element_activation: str = "relu"
    aemc_initialization: str = "identity-transfer"
    aemc_item_batch_size: int = 0
    aemc_element_skip: float = 0.
    analysis_anchor: dict = field(default_factory=dict)
    artifact_policy: str = "selected"
    baseline_overrides: dict = field(default_factory=dict)
    search_profile: str = "development"
    models: dict[str, ModelConfig] = field(default_factory=default_models)

    def validate(self):
        from vendor_method10.method10 import Method10Config
        Method10Config(**self.method8)
        if isinstance(self.search_seed,bool) or not isinstance(self.search_seed,int) or self.search_seed<0: raise ValueError("Invalid search seed")
        if isinstance(self.aemc_element_skip,bool) or not isinstance(self.aemc_element_skip,(int,float)) or not math.isfinite(self.aemc_element_skip) or not 0<=self.aemc_element_skip<=1:
            raise ValueError("Invalid AEMC element skip")
        if self.artifact_policy not in ("selected", "all"):
            raise ValueError("artifact_policy must be selected or all")
        if self.search_profile not in ("development", "comprehensive"):
            raise ValueError("Unknown search profile")
        if not isinstance(self.baseline_overrides, dict) or set(self.baseline_overrides) - {"autorec", "aemcne", "biasmf"}:
            raise ValueError("Invalid baseline overrides")
        if not isinstance(self.analysis_anchor,dict):raise ValueError("Invalid analysis anchor")
        if self.aemc_main_activation not in ("relu", "sigmoid", "tanh") or self.aemc_element_activation not in ("relu", "sigmoid", "tanh"):
            raise ValueError("Invalid AEMC activation control")
        if self.aemc_initialization not in ("torch", "xavier", "identity-transfer"):
            raise ValueError("Invalid AEMC initialization control")
        if self.aemc_initialization=="identity-transfer" and self.aemc_element_activation!="relu":
            raise ValueError("Exact identity transfer requires the ReLU scalar network")
        if isinstance(self.aemc_item_batch_size,bool) or not isinstance(self.aemc_item_batch_size,int) or self.aemc_item_batch_size<0:
            raise ValueError("AEMC item batch size must be a nonnegative integer")
        if self.mbmf_checkpoint_policy not in ("validation", "paper"):
            raise ValueError("Invalid MBMF checkpoint policy")
        if self.mbmf_optimizer not in ('paper','adam') or not isinstance(self.mbmf_adam_lr,(int,float)) or not math.isfinite(self.mbmf_adam_lr) or self.mbmf_adam_lr<=0:
            raise ValueError('Invalid MBMF optimizer')
        if self.mbmf_checkpoint_policy=='paper' and self.mbmf_optimizer!='paper':
            raise ValueError('Paper update policy requires the paper optimizer')
        if isinstance(self.mbmf_val_patience, bool) or not isinstance(self.mbmf_val_patience, int) or self.mbmf_val_patience < 1:
            raise ValueError("mbmf_val_patience must be a positive integer")
        if isinstance(self.mf_max_epochs,bool) or not isinstance(self.mf_max_epochs,int) or self.mf_max_epochs<1:
            raise ValueError("mf_max_epochs must be a positive integer")
        if isinstance(self.mbmf_max_epochs,bool) or not isinstance(self.mbmf_max_epochs,int) or self.mbmf_max_epochs<1:
            raise ValueError("mbmf_max_epochs must be a positive integer")
        if not isinstance(self.role_seed, int) or isinstance(self.role_seed, bool) or self.role_seed < 0:
            raise ValueError("role_seed must be a nonnegative integer")
        if not 0 < self.validation_selection_fraction < 1:
            raise ValueError("Invalid validation selection fraction")
        if not 0 < self.validation_calibration_fraction < 1-self.validation_selection_fraction:
            raise ValueError("A disjoint calibration holdout is required")
        if isinstance(self.search_jobs_per_method,bool) or not isinstance(self.search_jobs_per_method,int) or self.search_jobs_per_method<0:
            raise ValueError("Invalid search budget")
        if self.protocol not in ("common","aemc-paper","autorec-paper") or not isinstance(self.split_validation_roles,bool):
            raise ValueError("Invalid evaluation protocol")
        for model in self.models.values():
            model.validate()
        if set(self.models) != {f"Method{i}" for i in range(1, 8)}:
            raise ValueError("A V8 study must declare all seven model configurations")
        if any(name != model.method for name, model in self.models.items()):
            raise ValueError("Model configuration identity mismatch")
        return self
