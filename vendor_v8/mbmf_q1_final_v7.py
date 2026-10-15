"""
MBMF / thesis experimental pipeline -- revision v7
==================================================

Families implemented
--------------------
Baselines:
  * MF-Bias : standard biased matrix factorization (integrated from the redesign bundle)
  * MBMF-F  : fixed magnitudes + global centring (paper Eq. 38-39)
  * MBMF-C  : historical magnitudes + individual centring (paper Eq. 40, 42-44)
  * MBMF-N  : historical magnitudes + non-negative projected factors (paper Eq. 36-37, 42-43)

Proposed / integrated models:
  * Method 1: dual-autoencoder latent-magnitude MBMF (original thesis model)
  * Method 2: two-view magnitude MBMF fusion (original thesis cell3/cell4 model, corrected)
  * Method 3: historical-prior residual-magnitude AE-MBMF (multiplicative log-radius residual)
  * Method 4: bounded prior-residual MBMF (new v7 logit-radius residual formulation)
  * Method 5: decoder-coupled dual-view MBMF with raw encoder/decoder magnitudes
  * Method 6: range-safe decoder-coupled dual-view MBMF
  * Method 7: uncertainty-aware inequality-bounded MF (revised UC-IBMF integration)

Publication-safety properties
-----------------------------
  * movieId is the item key, never title.
  * train/validation/test masks are disjoint.
  * rating scale must be declared explicitly by default; it is not learned from test labels.
  * held-out ratings are never fed to encoders.
  * a regression test corrupts held-out labels and verifies predictions are unchanged.
  * test is never used for checkpoint selection or hyperparameter selection.
  * correct K-dimensional hyperspherical map uses K-1 angles and preserves ||x||=r.
  * no output clamp is used by any primary model.
  * original thesis formulas can be reproduced via radius_mode='raw_norm'.
  * optional bounded radius maps are explicit ablations, not silently substituted.
  * Method-2's old free K-vectors can be reproduced, or quotient-reparameterised as positive scalars.
  * best checkpoints are deep-copied.
  * MBMF-C uses the minimum of the current factorising TRAIN matrix in Eq. 40, not the
    documented global rating minimum.
  * raw-prediction range/OOB diagnostics, structural-bound violations, seeds, splits and config are saved.

Important fidelity note
-----------------------
The paper repository currently exposes no implementation beyond the README. Therefore this file
follows the equations/algorithms in the paper, with PyTorch autograd for the spherical Jacobian.
For MBMF-N, the paper projects W,H after spherical conversion but still updates angles using the
spherical derivative. `project_nonnegative_straight_through` reproduces that algorithmic ordering:
forward values are projected while the backward derivative treats the projection as identity.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =============================================================================
# 0. Types and configuration
# =============================================================================

InitMode = Literal["legacy_uniform_2pi", "canonical_uniform", "isotropic_gaussian"]
SplitMode = Literal["random_per_user", "temporal_per_user"]
BaselineVariant = Literal["MBMF-F", "MBMF-C", "MBMF-N"]
InputMode = Literal["values_only", "values_plus_mask"]
ModelSpace = Literal["legacy_unit", "centered_raw"]
LatentRadiusMode = Literal["raw_norm", "bounded_tanh"]
CounterpartRadiusMode = Literal["scalar_softplus", "vector_norm", "bounded_sigmoid"]
CFeasibilityPolicy = Literal["global_scale", "raise", "report_only"]
PriorMode = Literal["legacy_local_mean_std", "eq42_shrinkage"]
ResidualTransform = Literal["hard_clamp", "smooth_tanh"]
AELossMode = Literal["averaged", "separate"]
LossForm = Literal["mse", "frobenius"]


@dataclass
class DataConfig:
    data_dir: str = "/content/drive/MyDrive/Thesis/dataset"

    # Q1-safe default: force the experimenter to state the documented rating range.
    # MovieLens latest-small: 0.5, 5.0
    # MovieLens 1M:          1.0, 5.0
    # (The dense AE implementation is intended for small/1M-scale matrices, not MovieLens-20M.)
    rating_min: Optional[float] = None
    rating_max: Optional[float] = None
    require_explicit_rating_scale: bool = True

    split_mode: SplitMode = "random_per_user"
    train_fraction: float = 0.80
    val_fraction: float = 0.10
    test_fraction: float = 0.10
    split_seed: int = 2026
    min_train_per_user: int = 1
    enforce_train_support_for_all_items: bool = True


@dataclass
class PaperBaselineConfig:
    latent_dim: int = 20
    variants: Tuple[BaselineVariant, ...] = ("MBMF-F", "MBMF-C", "MBMF-N")

    # Three-init ablation is run for these variants only. The other variants use default_init.
    init_ablation_variants: Tuple[BaselineVariant, ...] = ("MBMF-F", "MBMF-C", "MBMF-N")
    init_modes: Tuple[InitMode, ...] = (
        "legacy_uniform_2pi",
        "canonical_uniform",
        "isotropic_gaussian",
    )
    default_init: InitMode = "isotropic_gaussian"

    # Paper Sec. 3.5.3 dynamic step settings.
    max_epochs: int = 500
    initial_lr: float = 0.1
    lr_increase: float = 1.1
    lr_decrease: float = 0.5
    min_lr: float = 1e-12
    objective_tol: float = 1e-5
    objective_patience: int = 10
    eval_every: int = 5

    # The paper objective is ||Z*(V-WH)||_F^2.  The code reports/stops on that
    # exact scale, while using 0.5*objective only for the gradient step because
    # the factor 2 is absorbed into the paper's learning coefficient update.
    objective_reduction: Literal["sum", "mean"] = "sum"

    # Paper Eq. 43 used rho=0.05 in reported real-data experiments.
    rho: float = 0.05

    # Eq. 40 can be infeasible unless magnitudes are modified. The paper states
    # they must be modified but does not prescribe an algorithm. `global_scale`
    # is an explicit, reproducible rank-1-preserving adjustment.
    c_feasibility_policy: CFeasibilityPolicy = "global_scale"
    c_feasibility_margin: float = 1e-6


@dataclass
class Method1Config:
    latent_dim: int = 20
    hidden_dim: int = 128
    input_mode: InputMode = "values_plus_mask"
    # centered_raw is the mathematically MBMF-consistent Q1 formulation.
    # legacy_unit reproduces the old notebook's [0,1] modelling space (without its known bugs).
    model_space: ModelSpace = "centered_raw"
    # Thesis code used Uniform(0,2pi) angles. Keep that as the primary setting;
    # change explicitly for an initialization ablation.
    angle_init: InitMode = "legacy_uniform_2pi"

    # raw_norm reproduces the thesis/code formula r=||z||+eps.
    # bounded_tanh is an explicit globally-range-safe ablation.
    radius_mode: LatentRadiusMode = "raw_norm"
    radius_eps: float = 1e-6

    lr: float = 0.0051            # old notebook setting
    weight_decay: float = 0.0
    max_epochs: int = 2000
    eval_every: int = 15
    val_patience: int = 20
    grad_clip: float = 1.0

    # Old code instantiated alpha=0.7, beta=0.35.
    # L_AE is L_user + L_item (not their average), matching the thesis equation.
    alpha_ae: float = 0.70
    beta_mbmf: float = 0.35


@dataclass
class Method2Config:
    latent_dim: int = 20
    input_mode: InputMode = "values_plus_mask"
    model_space: ModelSpace = "centered_raw"
    angle_init: InitMode = "legacy_uniform_2pi"
    radius_mode: LatentRadiusMode = "raw_norm"
    radius_eps: float = 1e-6

    # The old code used trainable K-vectors and consumed only their norms.
    # scalar_softplus is the quotient reparameterisation: same attainable positive
    # scalar radii with no unidentifiable angular degrees of freedom.
    counterpart_radius_mode: CounterpartRadiusMode = "vector_norm"
    bounded_counterpart_init_fraction: float = 0.50

    lr: float = 0.00615           # old cell-3 setting
    weight_decay: float = 0.0
    max_epochs: int = 2000
    eval_every: int = 15
    val_patience: int = 20
    grad_clip: float = 1.0

    fusion_weight_user_view: float = 0.50
    consistency_weight: float = 0.0


@dataclass
class Method3Config:
    """Historical-prior residual-magnitude AE-MBMF.

    This is the revised form of model-2 from ``mbmf_two_models_complete.ipynb``.
    It is deliberately named Method 3 so it cannot be confused with the two thesis methods.
    """
    latent_dim: int = 20
    input_mode: InputMode = "values_plus_mask"
    model_space: ModelSpace = "centered_raw"
    angle_init: InitMode = "legacy_uniform_2pi"

    # Prior is computed from TRAIN observations in unit-normalised rating space.
    # eq42_shrinkage adds the paper's global/local shrinkage (rho).
    prior_mode: PriorMode = "eq42_shrinkage"
    rho: float = 0.05

    # r = r0 * exp(delta).  smooth_tanh avoids the zero-gradient walls of hard clamp.
    residual_transform: ResidualTransform = "smooth_tanh"
    max_abs_log_scale: float = 2.0
    delta_l2: float = 0.0

    # Legacy source code averaged the two AE outputs before the AE loss.  The revised
    # default uses separate reconstruction losses to prevent error cancellation.
    ae_loss_mode: AELossMode = "separate"
    loss_form: LossForm = "mse"
    alpha_ae: float = 0.50
    beta_mbmf: float = 0.50

    lr: float = 5e-4
    weight_decay: float = 0.0
    max_epochs: int = 2000
    eval_every: int = 15
    val_patience: int = 20
    grad_clip: float = 1.0


@dataclass
class ExperimentConfigV6:
    data: DataConfig = field(default_factory=DataConfig)
    paper: PaperBaselineConfig = field(default_factory=PaperBaselineConfig)
    method1: Method1Config = field(default_factory=Method1Config)
    method2: Method2Config = field(default_factory=Method2Config)
    method3: Method3Config = field(default_factory=Method3Config)

    # Repeated data splits and parameter initialisations are separated so their
    # sources of variance can be reported independently.
    split_seeds: Tuple[int, ...] = (2026,)
    model_seeds: Tuple[int, ...] = (11, 22, 33, 44, 55)
    latent_dims: Tuple[int, ...] = (20,)
    fast_debug: bool = False
    output_dir: str = "./mbmf_q1_results_v7"




def apply_thesis_reproduction_preset(cfg: ExperimentConfig) -> ExperimentConfig:
    """Corrected reproduction of the two original thesis-code methods only.

    It restores the old values-only/[0,1] modelling choices but does NOT restore
    known bugs: held-out leakage, wrong spherical indexing, output clipping, the
    missing square in Method-2 loss, or test-based early stopping.
    """
    cfg.method1.input_mode = "values_only"
    cfg.method1.model_space = "legacy_unit"
    cfg.method1.angle_init = "legacy_uniform_2pi"
    cfg.method1.radius_mode = "raw_norm"
    cfg.method1.alpha_ae = 0.70
    cfg.method1.beta_mbmf = 0.35

    cfg.method2.input_mode = "values_only"
    cfg.method2.model_space = "legacy_unit"
    cfg.method2.angle_init = "legacy_uniform_2pi"
    cfg.method2.radius_mode = "raw_norm"
    cfg.method2.counterpart_radius_mode = "vector_norm"
    cfg.method2.fusion_weight_user_view = 0.50
    cfg.method2.consistency_weight = 0.0
    cfg.method2.grad_clip = 0.0  # old cell3/cell4 did not clip gradients
    return cfg


def apply_method3_legacy_reproduction_preset(cfg: ExperimentConfig) -> ExperimentConfig:
    """Corrected reproduction of model-2 from mbmf_two_models_complete.ipynb.

    The source architecture/loss space is retained while its held-out-input leakage
    and evaluation clipping are intentionally removed.
    """
    cfg.method3.input_mode = "values_only"
    cfg.method3.model_space = "legacy_unit"
    cfg.method3.angle_init = "legacy_uniform_2pi"
    cfg.method3.prior_mode = "legacy_local_mean_std"
    cfg.method3.residual_transform = "hard_clamp"
    cfg.method3.max_abs_log_scale = 6.0
    cfg.method3.ae_loss_mode = "averaged"
    cfg.method3.loss_form = "frobenius"
    cfg.method3.alpha_ae = 0.50
    cfg.method3.beta_mbmf = 0.50
    cfg.method3.delta_l2 = 0.0
    cfg.method3.lr = 5e-4
    cfg.method3.max_epochs = 100
    cfg.method3.eval_every = 1
    cfg.method3.val_patience = 20
    cfg.method3.grad_clip = 5.0
    return cfg


apply_method3_other_chat_reproduction_preset = apply_method3_legacy_reproduction_preset


def apply_all_corrected_reproduction_presets(cfg: ExperimentConfig) -> ExperimentConfig:
    apply_thesis_reproduction_preset(cfg)
    apply_method3_legacy_reproduction_preset(cfg)
    return cfg


def apply_q1_safe_preset_v6_core(cfg: ExperimentConfigV6) -> ExperimentConfigV6:
    """Recommended primary preprocessing while preserving both thesis model equations.

    Missingness is supplied explicitly to the encoders.  Raw latent-norm radii remain
    the primary thesis formula; bounded radii stay a separately named ablation.
    """
    cfg.method1.input_mode = "values_plus_mask"
    cfg.method1.model_space = "centered_raw"
    cfg.method1.angle_init = "legacy_uniform_2pi"
    cfg.method1.radius_mode = "raw_norm"

    cfg.method2.input_mode = "values_plus_mask"
    cfg.method2.model_space = "centered_raw"
    cfg.method2.angle_init = "legacy_uniform_2pi"
    cfg.method2.radius_mode = "raw_norm"
    # The old K-vector counterpart has K-1 unidentifiable angular degrees because
    # only its norm is consumed.  The Q1 preset uses the quotient scalar parameterisation;
    # the legacy preset above retains vector_norm for strict reproduction.
    cfg.method2.counterpart_radius_mode = "scalar_softplus"
    cfg.method2.consistency_weight = 0.0
    cfg.method2.grad_clip = 1.0

    cfg.method3.input_mode = "values_plus_mask"
    cfg.method3.model_space = "centered_raw"
    cfg.method3.angle_init = "legacy_uniform_2pi"
    cfg.method3.prior_mode = "eq42_shrinkage"
    cfg.method3.rho = cfg.paper.rho
    cfg.method3.residual_transform = "smooth_tanh"
    cfg.method3.max_abs_log_scale = 2.0
    cfg.method3.ae_loss_mode = "separate"
    cfg.method3.loss_form = "mse"
    cfg.method3.delta_l2 = 0.0
    cfg.method3.lr = 5e-4
    cfg.method3.max_epochs = 2000
    cfg.method3.eval_every = 15
    cfg.method3.val_patience = 20
    cfg.method3.grad_clip = 1.0
    return cfg


# =============================================================================
# 1. Reproducibility
# =============================================================================


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # older PyTorch without warn_only
        torch.use_deterministic_algorithms(True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 2. Data loading and leakage-safe splitting
# =============================================================================


def load_movielens(data_dir: str) -> pd.DataFrame:
    root = Path(data_dir)
    csv_path = root / "ratings.csv"
    dat_path = root / "ratings.dat"

    if csv_path.exists():
        ratings = pd.read_csv(csv_path)
        required = {"userId", "movieId", "rating"}
        missing = required - set(ratings.columns)
        if missing:
            raise ValueError(f"ratings.csv missing columns: {sorted(missing)}")
        if "timestamp" not in ratings.columns:
            ratings["timestamp"] = np.nan
    elif dat_path.exists():
        ratings = pd.read_csv(
            dat_path,
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
        )
    else:
        raise FileNotFoundError(f"No ratings.csv or ratings.dat under {root}")

    ratings = ratings[["userId", "movieId", "rating", "timestamp"]].copy()
    ratings["rating"] = ratings["rating"].astype(float)

    before = len(ratings)
    if ratings["timestamp"].notna().any():
        ratings = ratings.sort_values("timestamp")
    ratings = ratings.drop_duplicates(["userId", "movieId"], keep="last").reset_index(drop=True)
    removed = before - len(ratings)
    if removed:
        print(f"[data] removed {removed} duplicated user-item rows (latest retained)")
    return ratings


def _allocate_user_split(
    rows: pd.DataFrame,
    cfg: DataConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(rows)
    idx = rows.index.to_numpy().copy()
    temporal = cfg.split_mode == "temporal_per_user" and rows["timestamp"].notna().all()
    if temporal:
        idx = rows.sort_values("timestamp").index.to_numpy()
    else:
        rng.shuffle(idx)

    if n <= cfg.min_train_per_user:
        return idx, np.array([], dtype=int), np.array([], dtype=int)

    n_val = int(round(cfg.val_fraction * n))
    n_test = int(round(cfg.test_fraction * n))
    max_holdout = max(0, n - cfg.min_train_per_user)
    n_test = min(n_test, max_holdout)
    n_val = min(n_val, max_holdout - n_test)

    # Ensure a useful held-out set for users with enough ratings.
    if n >= cfg.min_train_per_user + 2 and n_test == 0:
        n_test = 1
    if n >= cfg.min_train_per_user + 3 and n_val == 0 and n_test < max_holdout:
        n_val = 1

    if temporal:
        a = n - n_val - n_test
        b = n - n_test
        return idx[:a], idx[a:b], idx[b:]

    return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]


def make_safe_split(ratings: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    if not math.isclose(cfg.train_fraction + cfg.val_fraction + cfg.test_fraction, 1.0, abs_tol=1e-10):
        raise ValueError("train + val + test fractions must sum to 1")

    if cfg.split_mode == "temporal_per_user" and ratings["timestamp"].isna().any():
        raise ValueError("temporal_per_user requested but one or more timestamps are missing")

    if cfg.split_mode == "temporal_per_user" and cfg.enforce_train_support_for_all_items:
        raise ValueError(
            "Temporal splitting and forcibly moving held-out interactions into train are incompatible. "
            "Set enforce_train_support_for_all_items=False for a chronology-preserving temporal experiment."
        )

    work = ratings.copy().reset_index(drop=True)
    work["split"] = "train"
    rng = np.random.default_rng(cfg.split_seed)

    for _, rows in work.groupby("userId", sort=False):
        tr, va, te = _allocate_user_split(rows, cfg, rng)
        work.loc[tr, "split"] = "train"
        work.loc[va, "split"] = "val"
        work.loc[te, "split"] = "test"

    if cfg.enforce_train_support_for_all_items:
        train_items = set(work.loc[work.split == "train", "movieId"])
        missing = sorted(set(work["movieId"]) - train_items)
        moved = 0
        for item in missing:
            # Preserve the locked test set whenever validation can provide support.
            cand_val = work.index[(work.movieId == item) & (work.split == "val")].to_numpy()
            cand_test = work.index[(work.movieId == item) & (work.split == "test")].to_numpy()
            cand = cand_val if len(cand_val) else cand_test
            if len(cand):
                work.loc[int(cand[0]), "split"] = "train"
                moved += 1
        if moved:
            print(f"[split] moved {moved} held-out interactions to train for item support")

    train_user_counts = work[work.split == "train"].groupby("userId").size()
    if len(train_user_counts) != work["userId"].nunique():
        raise RuntimeError("at least one user has no training interaction")
    if (train_user_counts < cfg.min_train_per_user).any():
        raise RuntimeError("at least one user has insufficient train support")

    counts = work.split.value_counts().to_dict()
    if counts.get("val", 0) == 0 or counts.get("test", 0) == 0:
        raise RuntimeError(f"validation/test split is empty: {counts}")
    print(f"[split] {counts}")
    return work


def dataset_manifest(data_dir: str, ratings: pd.DataFrame) -> Dict[str, object]:
    """Reproducibility manifest for the exact ratings file used."""
    root = Path(data_dir)
    source = root / "ratings.csv" if (root / "ratings.csv").exists() else root / "ratings.dat"
    h = hashlib.sha256()
    with open(source, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {
        "source_file": str(source),
        "sha256": h.hexdigest(),
        "rows_after_duplicate_resolution": int(len(ratings)),
        "num_users": int(ratings.userId.nunique()),
        "num_items": int(ratings.movieId.nunique()),
        "observed_rating_min": float(ratings.rating.min()),
        "observed_rating_max": float(ratings.rating.max()),
        "timestamp_available": bool(ratings.timestamp.notna().any()),
    }


@dataclass
class MatrixData:
    num_users: int
    num_items: int
    user_to_index: Dict[int, int]
    item_to_index: Dict[int, int]
    index_to_user: np.ndarray
    index_to_item: np.ndarray

    rating_min: float
    rating_max: float
    midpoint: float
    half_range: float

    train_values_centered: torch.Tensor
    train_mask: torch.Tensor

    train_users: torch.Tensor
    train_items: torch.Tensor
    val_users: torch.Tensor
    val_items: torch.Tensor
    test_users: torch.Tensor
    test_items: torch.Tensor

    train_ratings_raw: torch.Tensor
    val_ratings_raw: torch.Tensor
    test_ratings_raw: torch.Tensor

    user_train_mean_raw: torch.Tensor


def build_matrix_data_v6_core(split: pd.DataFrame, cfg: DataConfig, device: torch.device) -> MatrixData:
    users = np.sort(split.userId.unique())
    items = np.sort(split.movieId.unique())
    u2i = {int(v): i for i, v in enumerate(users)}
    i2i = {int(v): i for i, v in enumerate(items)}

    work = split.copy()
    work["u"] = work.userId.map(u2i).astype(int)
    work["i"] = work.movieId.map(i2i).astype(int)

    if cfg.require_explicit_rating_scale and (cfg.rating_min is None or cfg.rating_max is None):
        raise ValueError(
            "For publication runs set DataConfig.rating_min/rating_max from dataset documentation "
            "(e.g. latest-small: 0.5,5.0; MovieLens-1M: 1.0,5.0)."
        )

    observed_min = float(work.rating.min())
    observed_max = float(work.rating.max())
    rmin = float(observed_min if cfg.rating_min is None else cfg.rating_min)
    rmax = float(observed_max if cfg.rating_max is None else cfg.rating_max)
    if rmax <= rmin:
        raise ValueError("invalid rating range")
    if observed_min < rmin - 1e-8 or observed_max > rmax + 1e-8:
        raise ValueError(f"observed ratings [{observed_min},{observed_max}] outside [{rmin},{rmax}]")

    midpoint = 0.5 * (rmin + rmax)
    half = 0.5 * (rmax - rmin)
    n, m = len(users), len(items)

    def pairs(name: str):
        p = work[work.split == name]
        u = torch.as_tensor(p.u.to_numpy(), dtype=torch.long, device=device)
        i = torch.as_tensor(p.i.to_numpy(), dtype=torch.long, device=device)
        y = torch.as_tensor(p.rating.to_numpy(), dtype=torch.float32, device=device)
        return u, i, y

    tr_u, tr_i, tr_y = pairs("train")
    va_u, va_i, va_y = pairs("val")
    te_u, te_i, te_y = pairs("test")

    train_values = torch.zeros((n, m), dtype=torch.float32, device=device)
    train_mask = torch.zeros((n, m), dtype=torch.float32, device=device)
    train_values[tr_u, tr_i] = tr_y - midpoint
    train_mask[tr_u, tr_i] = 1.0

    # Each row has exactly one split label.  Checking duplicate user-item keys is
    # sufficient and avoids three large Python sets on MovieLens-1M.
    if work.duplicated(["u", "i"]).any():
        raise RuntimeError("duplicate user-item key detected after duplicate resolution")
    if not set(work["split"].unique()).issubset({"train", "val", "test"}):
        raise RuntimeError("unknown split label detected")

    global_mean = tr_y.mean()
    sums = torch.zeros(n, device=device)
    cnt = torch.zeros(n, device=device)
    sums.scatter_add_(0, tr_u, tr_y)
    cnt.scatter_add_(0, tr_u, torch.ones_like(tr_y))
    user_mean = torch.full((n,), float(global_mean.item()), device=device)
    ok = cnt > 0
    user_mean[ok] = sums[ok] / cnt[ok]

    print(
        f"[data] users={n:,}, items={m:,}, interactions={len(work):,}, "
        f"documented_range=[{rmin},{rmax}]"
    )

    return MatrixData(
        n, m, u2i, i2i, users, items,
        rmin, rmax, midpoint, half,
        train_values, train_mask,
        tr_u, tr_i, va_u, va_i, te_u, te_i,
        tr_y, va_y, te_y,
        user_mean,
    )


# =============================================================================
# 3. Exact hyperspherical geometry
# =============================================================================


def spherical_to_cartesian(angles: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    """Eq. (10) in zero-based form.

    x0 = r cos(phi0)
    xk = r [prod_{j=0}^{k-1} sin(phi_j)] cos(phi_k), 1<=k<=K-2
    xK-1 = r prod_{j=0}^{K-2} sin(phi_j)
    """
    if angles.ndim != 2:
        raise ValueError("angles must be [N,K-1]")
    if radii.ndim == 1:
        radii = radii[:, None]
    if radii.ndim != 2 or radii.shape != (angles.shape[0], 1):
        raise ValueError("radii must be [N] or [N,1]")

    n, km1 = angles.shape
    k = km1 + 1
    if k < 2:
        raise ValueError("K must be >=2")

    sin_phi = torch.sin(angles)
    cos_phi = torch.cos(angles)
    prefix = torch.cumprod(sin_phi, dim=1)

    out = [cos_phi[:, :1]]
    if k > 2:
        out.append(prefix[:, :-1] * cos_phi[:, 1:])
    out.append(prefix[:, -1:])
    unit = torch.cat(out, dim=1)
    return radii * unit


def cartesian_unit_to_angles(unit: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    unit = unit / unit.norm(dim=1, keepdim=True).clamp_min(eps)
    n, k = unit.shape
    if k < 2:
        raise ValueError("K must be >=2")
    a = torch.zeros((n, k - 1), dtype=unit.dtype, device=unit.device)
    if k == 2:
        a[:, 0] = torch.remainder(torch.atan2(unit[:, 1], unit[:, 0]), 2 * math.pi)
        return a
    for j in range(k - 2):
        tail = torch.linalg.vector_norm(unit[:, j + 1 :], dim=1)
        a[:, j] = torch.atan2(tail, unit[:, j])
    a[:, -1] = torch.remainder(torch.atan2(unit[:, -1], unit[:, -2]), 2 * math.pi)
    return a


def initialise_angles(n: int, k: int, mode: InitMode, device: torch.device) -> torch.Tensor:
    if k < 2:
        raise ValueError("latent dimension must be >=2")
    if mode == "legacy_uniform_2pi":
        # The paper says only "random angle matrices"; this is the old notebook's interpretation.
        return torch.rand((n, k - 1), device=device) * (2 * math.pi)
    if mode == "canonical_uniform":
        a = torch.empty((n, k - 1), device=device)
        if k > 2:
            a[:, :-1] = torch.rand((n, k - 2), device=device) * math.pi
        a[:, -1] = torch.rand(n, device=device) * (2 * math.pi)
        return a
    if mode == "isotropic_gaussian":
        q = torch.randn((n, k), device=device)
        q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return cartesian_unit_to_angles(q)
    raise ValueError(mode)


def geometry_self_test(device: torch.device) -> None:
    for k in (2, 3, 5, 20):
        for mode in ("legacy_uniform_2pi", "canonical_uniform", "isotropic_gaussian"):
            a = initialise_angles(257, k, mode, device)
            r = 0.01 + 3.0 * torch.rand(257, 1, device=device)
            x = spherical_to_cartesian(a, r)
            e = (x.norm(dim=1) - r[:, 0]).abs().max().item()
            if e > 3e-5:
                raise AssertionError(f"geometry failed K={k} mode={mode}: {e}")

    # Explicit K=3 formula regression: catches the old cos(phi[k-1]) indexing bug.
    a = torch.tensor([[0.7, 1.2]], dtype=torch.float64, device=device)
    r = torch.tensor([[2.3]], dtype=torch.float64, device=device)
    x = spherical_to_cartesian(a, r)[0]
    expected = torch.stack([
        r[0, 0] * torch.cos(a[0, 0]),
        r[0, 0] * torch.sin(a[0, 0]) * torch.cos(a[0, 1]),
        r[0, 0] * torch.sin(a[0, 0]) * torch.sin(a[0, 1]),
    ])
    if not torch.allclose(x, expected, atol=1e-12, rtol=1e-12):
        raise AssertionError("explicit K=3 spherical formula regression")

    # Round trip for non-singular random directions.
    q = torch.randn(251, 7, dtype=torch.float64, device=device)
    q = q / q.norm(dim=1, keepdim=True)
    ar = cartesian_unit_to_angles(q)
    qr = spherical_to_cartesian(ar, torch.ones((len(q), 1), dtype=q.dtype, device=device))
    if not torch.allclose(q, qr, atol=2e-10, rtol=2e-10):
        raise AssertionError("cartesian->angles->cartesian round-trip regression")
    print("[self-test] hyperspherical geometry: PASS")


def project_nonnegative_straight_through(x: torch.Tensor) -> torch.Tensor:
    """Forward = max(x,0), backward = identity.

    This mirrors Algorithm 2's ordering more closely than differentiating through ReLU:
    the paper projects W/H after spherical conversion but angle updates still use the
    spherical derivative.
    """
    projected = torch.clamp_min(x, 0.0)
    return x + (projected - x).detach()


# =============================================================================
# 4. Metrics
# =============================================================================


def regression_metrics_raw(pred_raw: torch.Tensor, true_raw: torch.Tensor, rmin: float, rmax: float) -> Dict[str, float]:
    if true_raw.numel() == 0:
        return {k: float("nan") for k in ("MAE", "RMSE", "OOB_rate", "max_bound_violation", "pred_min", "pred_max")}
    err = pred_raw - true_raw
    violation = torch.maximum(torch.relu(torch.as_tensor(rmin, device=pred_raw.device) - pred_raw),
                              torch.relu(pred_raw - torch.as_tensor(rmax, device=pred_raw.device)))
    return {
        "MAE": float(torch.mean(torch.abs(err)).item()),
        "RMSE": float(torch.sqrt(torch.mean(err ** 2)).item()),
        "OOB_rate": float(torch.mean(((pred_raw < rmin - 1e-6) | (pred_raw > rmax + 1e-6)).float()).item()),
        "max_bound_violation": float(violation.max().item()),
        "pred_min": float(pred_raw.min().item()),
        "pred_max": float(pred_raw.max().item()),
    }


def train_threshold_f1(
    users: torch.Tensor,
    true_raw: torch.Tensor,
    pred_raw: torch.Tensor,
    user_train_mean_raw: torch.Tensor,
) -> Dict[str, float]:
    """Leakage-safe analogue of the paper's per-user average threshold.

    The threshold is computed from training ratings only; held-out labels never choose it.
    """
    if true_raw.numel() == 0:
        return {"F1_trainThreshold_macro_user": float("nan"), "F1_trainThreshold_micro": float("nan")}
    thr = user_train_mean_raw[users]
    yt = true_raw > thr
    yp = pred_raw > thr

    tp = (yt & yp).sum().float()
    fp = ((~yt) & yp).sum().float()
    fn = (yt & (~yp)).sum().float()
    p = tp / (tp + fp).clamp_min(1.0)
    r = tp / (tp + fn).clamp_min(1.0)
    micro = (2 * p * r / (p + r).clamp_min(1e-12)).item()

    vals = []
    for u in torch.unique(users):
        q = users == u
        ytu, ypu = yt[q], yp[q]
        tpu = (ytu & ypu).sum().float()
        fpu = ((~ytu) & ypu).sum().float()
        fnu = (ytu & (~ypu)).sum().float()
        pu = tpu / (tpu + fpu).clamp_min(1.0)
        ru = tpu / (tpu + fnu).clamp_min(1.0)
        vals.append(2 * pu * ru / (pu + ru).clamp_min(1e-12))
    return {
        "F1_trainThreshold_macro_user": float(torch.stack(vals).mean().item()),
        "F1_trainThreshold_micro": float(micro),
    }


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# 5. Paper historical magnitudes Eq. 42-44
# =============================================================================


def _group_mean_std(
    index: torch.Tensor,
    values: torch.Tensor,
    size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sums = torch.zeros(size, dtype=values.dtype, device=values.device)
    sums2 = torch.zeros_like(sums)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, index, values)
    sums2.scatter_add_(0, index, values * values)
    counts.scatter_add_(0, index, torch.ones_like(values))
    mean = torch.zeros_like(sums)
    std = torch.zeros_like(sums)
    ok = counts > 0
    mean[ok] = sums[ok] / counts[ok]
    var = torch.zeros_like(sums)
    var[ok] = (sums2[ok] / counts[ok] - mean[ok] ** 2).clamp_min(0)
    std[ok] = torch.sqrt(var[ok])  # population std; paper does not specify ddof
    return mean, std, counts


@torch.no_grad()
def historical_magnitudes_eq42(data: MatrixData, rho: float) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    if not (0 < rho <= 1):
        raise ValueError("rho must be in (0,1]")
    y = data.train_ratings_raw
    if torch.any(y < 0):
        raise ValueError("Eq.42 requires non-negative historical ratings")

    gmean = y.mean()
    gstd = torch.sqrt(torch.mean((y - gmean) ** 2))
    gmag = torch.sqrt((gmean + gstd).clamp_min(1e-12))

    um, us, uc = _group_mean_std(data.train_users, y, data.num_users)
    im, iss, ic = _group_mean_std(data.train_items, y, data.num_items)

    local_u = torch.sqrt((um + us).clamp_min(0))
    local_i = torch.sqrt((im + iss).clamp_min(0))

    omega_u = torch.clamp(uc / (rho * data.num_items), max=1.0)
    omega_i = torch.clamp(ic / (rho * data.num_users), max=1.0)
    omega_u[uc == 0] = 0
    omega_i[ic == 0] = 0

    rW = omega_u * local_u + (1 - omega_u) * gmag
    rH = omega_i * local_i + (1 - omega_i) * gmag

    return rW[:, None], rH[:, None], {
        "global_magnitude": float(gmag.item()),
        "mean_omega_user": float(omega_u.mean().item()),
        "mean_omega_item": float(omega_i.mean().item()),
    }


@torch.no_grad()
def prepare_paper_radii(
    data: MatrixData,
    variant: BaselineVariant,
    cfg: PaperBaselineConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    if variant == "MBMF-F":
        r = math.sqrt(data.half_range)
        return (
            torch.full((data.num_users, 1), r, device=data.train_mask.device),
            torch.full((data.num_items, 1), r, device=data.train_mask.device),
            {"c_feasibility_scale": 1.0},
        )

    rWN, rHN, info = historical_magnitudes_eq42(data, cfg.rho)
    if variant == "MBMF-N":
        return rWN, rHN, {**info, "c_feasibility_scale": 1.0}

    if variant != "MBMF-C":
        raise ValueError(variant)

    # Paper Eq. 44: each C magnitude = corresponding N magnitude / sqrt(2).
    rW = rWN / math.sqrt(2.0)
    rH = rHN / math.sqrt(2.0)

    # Eq. 40 uses min(V) of the CURRENT factorising matrix, not the externally
    # documented rating minimum.  Under the common protocol the current matrix is TRAIN.
    c_current_min = float(data.train_ratings_raw.min().item())
    # Feasibility requires V - min(V) - b <= b => b >= (V-min(V))/2.
    product = (rW[data.train_users, 0] * rH[data.train_items, 0]).clamp_min(1e-12)
    required = ((data.train_ratings_raw - c_current_min) / 2.0).clamp_min(0)
    ratio = required / product
    max_ratio = float(ratio.max().item()) if ratio.numel() else 1.0
    scale = 1.0

    if max_ratio > 1.0 + 1e-7:
        if cfg.c_feasibility_policy == "raise":
            raise RuntimeError(
                f"MBMF-C Eq.40 infeasible for train data; max required/current product ratio={max_ratio:.6g}. "
                "Use global_scale or explicitly define another magnitude-adjustment rule."
            )
        if cfg.c_feasibility_policy == "global_scale":
            # Scaling each vector by sqrt(max_ratio) scales every pair product by max_ratio.
            scale = math.sqrt(max_ratio * (1.0 + cfg.c_feasibility_margin))
            rW = rW * scale
            rH = rH * scale
        elif cfg.c_feasibility_policy != "report_only":
            raise ValueError(cfg.c_feasibility_policy)

    return rW, rH, {
        **info,
        "c_max_required_product_ratio_before_adjustment": max_ratio,
        "c_vector_scale": scale,
        "c_product_scale": scale * scale,
        "c_feasibility_scale": scale,
        "c_current_matrix_min": c_current_min,
    }


# =============================================================================
# 6. MBMF-F / MBMF-C / MBMF-N
# =============================================================================


class PaperMBMF(nn.Module):
    def __init__(
        self,
        data: MatrixData,
        cfg: PaperBaselineConfig,
        variant: BaselineVariant,
        init_mode: InitMode,
        device: torch.device,
    ):
        super().__init__()
        self.variant = variant
        self.data_meta = (data.rating_min, data.midpoint, float(data.train_ratings_raw.min().item()))
        rW, rH, rinfo = prepare_paper_radii(data, variant, cfg)
        self.register_buffer("rW", rW.to(device))
        self.register_buffer("rH", rH.to(device))
        self.radius_info = rinfo
        self.Phi = nn.Parameter(initialise_angles(data.num_users, cfg.latent_dim, init_mode, device))
        self.Theta = nn.Parameter(initialise_angles(data.num_items, cfg.latent_dim, init_mode, device))

    def factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        W0 = spherical_to_cartesian(self.Phi, self.rW)
        H0 = spherical_to_cartesian(self.Theta, self.rH)
        if self.variant == "MBMF-N":
            return project_nonnegative_straight_through(W0), project_nonnegative_straight_through(H0)
        return W0, H0

    def native_predict_pairs(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        W, H = self.factors()
        return torch.sum(W[users] * H[items], dim=1)

    def native_targets(self, raw: torch.Tensor, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        rmin, midpoint, c_current_min = self.data_meta
        if self.variant == "MBMF-F":
            return raw - midpoint
        if self.variant == "MBMF-C":
            b = self.rW[users, 0] * self.rH[items, 0]
            return raw - c_current_min - b
        if self.variant == "MBMF-N":
            return raw
        raise ValueError(self.variant)

    def predict_raw_pairs(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        p = self.native_predict_pairs(users, items)
        rmin, midpoint, c_current_min = self.data_meta
        if self.variant == "MBMF-F":
            return p + midpoint
        if self.variant == "MBMF-C":
            return p + c_current_min + self.rW[users, 0] * self.rH[items, 0]
        if self.variant == "MBMF-N":
            return p
        raise ValueError(self.variant)

    @torch.no_grad()
    def diagnostics(self, users: torch.Tensor, items: torch.Tensor) -> Dict[str, float]:
        W, H = self.factors()
        pred_native = torch.sum(W[users] * H[items], dim=1)
        bound = self.rW[users, 0] * self.rH[items, 0]
        if self.variant == "MBMF-N":
            violation = torch.relu(pred_native - bound)
        else:
            violation = torch.relu(torch.abs(pred_native) - bound)
        return {
            "max_user_norm_over_radius": float((W.norm(dim=1) / self.rW[:, 0].clamp_min(1e-12)).max().item()),
            "max_item_norm_over_radius": float((H.norm(dim=1) / self.rH[:, 0].clamp_min(1e-12)).max().item()),
            "max_structural_bound_violation": float(violation.max().item()) if violation.numel() else 0.0,
            "mean_rW": float(self.rW.mean().item()),
            "mean_rH": float(self.rH.mean().item()),
            "max_pair_bound_eval": float(bound.max().item()) if bound.numel() else float("nan"),
            **self.radius_info,
        }


def paper_objective(error: torch.Tensor, reduction: str) -> torch.Tensor:
    """Paper Eq. 22 objective value ||error||_F^2 (or mean analogue for debug)."""
    if reduction == "sum":
        return torch.sum(error ** 2)
    if reduction == "mean":
        return torch.mean(error ** 2)
    raise ValueError(reduction)


def paper_gradient_loss(error: torch.Tensor, reduction: str) -> torch.Tensor:
    """Half objective used only for gradients, matching the factor-2-absorbed update equations."""
    return 0.5 * paper_objective(error, reduction)


def train_paper_mbmf(model: PaperMBMF, data: MatrixData, cfg: PaperBaselineConfig) -> Dict[str, object]:
    """Paper-style dynamic-step training.

    Validation is LOGGED ONLY. It does not choose a checkpoint, preserving the paper's
    objective-based termination rather than silently turning the baseline into a
    validation-selected model.
    """
    lr = cfg.initial_lr
    stable = 0
    history: List[Dict[str, object]] = []
    start = time.perf_counter()
    final_epoch = 0

    for epoch in range(1, cfg.max_epochs + 1):
        final_epoch = epoch
        model.train()
        model.zero_grad(set_to_none=True)
        target = model.native_targets(data.train_ratings_raw, data.train_users, data.train_items)
        pred = model.native_predict_pairs(data.train_users, data.train_items)
        grad_loss = paper_gradient_loss(pred - target, cfg.objective_reduction)
        old_objective = paper_objective(pred - target, cfg.objective_reduction).detach()
        if not torch.isfinite(grad_loss):
            raise FloatingPointError(f"{model.variant}: non-finite objective at epoch {epoch}")
        grad_loss.backward()

        before = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)
            new_pred = model.native_predict_pairs(data.train_users, data.train_items)
            new_objective = paper_objective(new_pred - target, cfg.objective_reduction)

        accepted = bool(torch.isfinite(new_objective) and new_objective <= old_objective)
        if accepted:
            dec = float((old_objective - new_objective).item())
            stable = stable + 1 if dec < cfg.objective_tol else 0
            lr *= cfg.lr_increase
        else:
            model.load_state_dict(before)
            new_objective = old_objective
            dec = 0.0
            stable = 0
            lr = max(cfg.min_lr, lr * cfg.lr_decrease)

        if epoch == 1 or epoch % cfg.eval_every == 0:
            with torch.no_grad():
                vp = model.predict_raw_pairs(data.val_users, data.val_items)
                vm = regression_metrics_raw(vp, data.val_ratings_raw, data.rating_min, data.rating_max)
            history.append({
                "epoch": epoch,
                "objective": float(new_objective.item()),
                "objective_decrease": dec,
                "accepted": accepted,
                "lr": lr,
                "val_RMSE_log_only": vm["RMSE"],
            })

        if stable >= cfg.objective_patience:
            break
        if lr <= cfg.min_lr and not accepted:
            break

    return {
        "final_epoch": final_epoch,
        "elapsed_sec": time.perf_counter() - start,
        "final_lr": lr,
        "history": history,
    }


# =============================================================================
# 7. Shared components for thesis models
# =============================================================================


def model_space_train_matrix(data: MatrixData, space: ModelSpace) -> torch.Tensor:
    """Training-only dense matrix in the modelling space; missing entries are zero.

    legacy_unit:
        (V-rmin)/(rmax-rmin), matching the old notebook's numeric space.
    centered_raw:
        V-midpoint in raw rating units, matching the MBMF centring argument.
    """
    if space == "centered_raw":
        return data.train_values_centered * data.train_mask
    if space == "legacy_unit":
        width = data.rating_max - data.rating_min
        raw_observed = data.train_values_centered + data.midpoint * data.train_mask
        return ((raw_observed - data.rating_min * data.train_mask) / width) * data.train_mask
    raise ValueError(space)


def model_space_targets(data: MatrixData, raw: torch.Tensor, space: ModelSpace) -> torch.Tensor:
    if space == "centered_raw":
        return raw - data.midpoint
    if space == "legacy_unit":
        return (raw - data.rating_min) / (data.rating_max - data.rating_min)
    raise ValueError(space)


def model_space_to_raw(data: MatrixData, pred: torch.Tensor, space: ModelSpace) -> torch.Tensor:
    if space == "centered_raw":
        return pred + data.midpoint
    if space == "legacy_unit":
        return data.rating_min + (data.rating_max - data.rating_min) * pred
    raise ValueError(space)


def model_pair_bound_scale(data: MatrixData, space: ModelSpace) -> float:
    """Desired symmetric |p| bound used only by bounded-radius ablations."""
    if space == "centered_raw":
        return data.half_range
    if space == "legacy_unit":
        # Symmetric [-1,1] bound in unit space. This does NOT guarantee p>=0.
        return 1.0
    raise ValueError(space)


def build_encoder_inputs(
    data: MatrixData,
    mode: InputMode,
    space: ModelSpace,
) -> Tuple[torch.Tensor, torch.Tensor]:
    values = model_space_train_matrix(data, space)
    # Scale centred raw values to [-1,1] for the encoder only; targets stay in raw centred units.
    encoded_values = values / data.half_range if space == "centered_raw" else values
    if mode == "values_only":
        return encoded_values, encoded_values.T
    if mode == "values_plus_mask":
        return (
            torch.cat([encoded_values, data.train_mask], dim=1),
            torch.cat([encoded_values.T, data.train_mask.T], dim=1),
        )
    raise ValueError(mode)


def input_dims(data: MatrixData, mode: InputMode) -> Tuple[int, int]:
    mult = 2 if mode == "values_plus_mask" else 1
    return mult * data.num_items, mult * data.num_users


def latent_radius(z: torch.Tensor, mode: LatentRadiusMode, half_range: float, eps: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(z, dim=1, keepdim=True)
    if mode == "raw_norm":
        return norm + eps
    if mode == "bounded_tanh":
        cap = math.sqrt(half_range)
        if not (0.0 < eps < cap):
            raise ValueError("radius_eps must satisfy 0 < eps < sqrt(half_range)")
        # Exact cap: eps <= r < cap.  Do not add eps after cap*tanh, which
        # would technically violate the claimed global radius bound.
        return eps + (cap - eps) * torch.tanh(norm / math.sqrt(z.shape[1]))
    raise ValueError(mode)


class AEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class PairDecoder(nn.Module):
    """K -> hidden -> full output, evaluated only at requested output indices."""
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def hidden(self, z: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc1(z))

    def selected(self, z: torch.Tensor, sample_index: torch.Tensor, output_index: torch.Tensor) -> torch.Tensor:
        h = self.hidden(z)[sample_index]
        w = self.fc2.weight[output_index]
        b = self.fc2.bias[output_index]
        return torch.sum(h * w, dim=1) + b


class TwoStageFeatureExtractor(nn.Module):
    """Old Method-2 architecture: input -> K -> K/2 -> K."""
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        bottleneck = max(1, latent_dim // 2)
        self.encoder1 = nn.Linear(input_dim, latent_dim)
        self.encoder2 = nn.Linear(latent_dim, bottleneck)
        self.decoder1 = nn.Linear(bottleneck, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.encoder2(torch.relu(self.encoder1(x)))
        return self.decoder1(q)


class PositiveCounterpartRadii(nn.Module):
    """Reparameterises the old Method-2 free-vector norm.

    vector_norm: strict old-code parameterisation D in R^{n x K}, radius=||D_row||+eps.
    scalar_softplus: quotient parameterisation with one positive scalar per entity.
    bounded_sigmoid: explicit range-safe ablation.
    """
    def __init__(
        self, n: int, k: int, mode: CounterpartRadiusMode, half_range: float, eps: float,
        bounded_init_fraction: float = 0.50,
    ):
        super().__init__()
        self.mode, self.half_range, self.eps = mode, half_range, eps
        self.bounded_init_fraction = float(bounded_init_fraction)
        if mode == "vector_norm":
            self.vectors = nn.Parameter(torch.randn(n, k))
        elif mode in ("scalar_softplus", "bounded_sigmoid"):
            # Start near sqrt(half_range), a neutral MBMF-F-sized radius.
            target = max(math.sqrt(half_range) - eps, 1e-4)
            if mode == "scalar_softplus":
                inv = math.log(math.expm1(target)) if target < 20 else target
            else:
                cap = math.sqrt(half_range)
                frac = self.bounded_init_fraction
                if not 0.0 < frac < 1.0:
                    raise ValueError("bounded_counterpart_init_fraction must be in (0,1)")
                inv = math.log(frac / (1 - frac))
            self.logits = nn.Parameter(torch.full((n, 1), float(inv)))
        else:
            raise ValueError(mode)

    def forward(self) -> torch.Tensor:
        if self.mode == "vector_norm":
            return torch.linalg.vector_norm(self.vectors, dim=1, keepdim=True) + self.eps
        if self.mode == "scalar_softplus":
            return torch.nn.functional.softplus(self.logits) + self.eps
        cap = math.sqrt(self.half_range)
        if not (0.0 < self.eps < cap):
            raise ValueError("radius_eps must satisfy 0 < eps < sqrt(half_range)")
        return self.eps + (cap - self.eps) * torch.sigmoid(self.logits)


# =============================================================================
# 8. Thesis Method 1 -- Dual Autoencoder Latent-Magnitude MBMF
# =============================================================================


class ThesisMethod1(nn.Module):
    """Reconstruction of the first model in the uploaded notebook.

    z_u = E_u(X_u), z_i = E_i(X_i)
    r^W_u = ||z_u|| + eps, r^H_i = ||z_i|| + eps          [raw_norm mode]
    W_u = Spherical(Phi_u, r^W_u), H_i = Spherical(Theta_i, r^H_i)
    p_ui = W_u^T H_i
    L_AE = MSE(D_u(z_u),V) + MSE(D_i(z_i),V)
    L = alpha L_AE + beta MSE(p,V)
    """
    def __init__(self, data: MatrixData, cfg: Method1Config, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.half_range = data.half_range
        uin, iin = input_dims(data, cfg.input_mode)
        self.encoder_user = AEEncoder(uin, cfg.hidden_dim, cfg.latent_dim)
        self.decoder_user = PairDecoder(cfg.latent_dim, cfg.hidden_dim, data.num_items)
        self.encoder_item = AEEncoder(iin, cfg.hidden_dim, cfg.latent_dim)
        self.decoder_item = PairDecoder(cfg.latent_dim, cfg.hidden_dim, data.num_users)
        self.Phi = nn.Parameter(initialise_angles(data.num_users, cfg.latent_dim, cfg.angle_init, device))
        self.Theta = nn.Parameter(initialise_angles(data.num_items, cfg.latent_dim, cfg.angle_init, device))
        self.to(device)

    def encoded(self, data: MatrixData):
        xu, xi = build_encoder_inputs(data, self.cfg.input_mode, self.cfg.model_space)
        zu = self.encoder_user(xu)
        zi = self.encoder_item(xi)
        bound_scale = model_pair_bound_scale(data, self.cfg.model_space)
        rW = latent_radius(zu, self.cfg.radius_mode, bound_scale, self.cfg.radius_eps)
        rH = latent_radius(zi, self.cfg.radius_mode, bound_scale, self.cfg.radius_eps)
        W = spherical_to_cartesian(self.Phi, rW)
        H = spherical_to_cartesian(self.Theta, rH)
        return zu, zi, rW, rH, W, H

    def predict_centered_pairs(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        *_, W, H = self.encoded(data)
        return torch.sum(W[users] * H[items], dim=1)

    def predict_raw_pairs(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return model_space_to_raw(data, self.predict_centered_pairs(data, users, items), self.cfg.model_space)

    def loss(self, data: MatrixData) -> Tuple[torch.Tensor, Dict[str, float]]:
        zu, zi, _, _, W, H = self.encoded(data)
        u, i = data.train_users, data.train_items
        target = model_space_targets(data, data.train_ratings_raw, self.cfg.model_space)

        # User AE: sample=user u, output=item i.
        rec_u = self.decoder_user.selected(zu, u, i)
        # Item AE: sample=item i, output=user u.
        rec_i = self.decoder_item.selected(zi, i, u)
        l_user = torch.mean((rec_u - target) ** 2)
        l_item = torch.mean((rec_i - target) ** 2)
        l_ae = l_user + l_item

        pred = torch.sum(W[u] * H[i], dim=1)
        l_mbmf = torch.mean((pred - target) ** 2)
        total = self.cfg.alpha_ae * l_ae + self.cfg.beta_mbmf * l_mbmf
        return total, {
            "L_user_AE": float(l_user.detach().item()),
            "L_item_AE": float(l_item.detach().item()),
            "L_AE": float(l_ae.detach().item()),
            "L_MBM": float(l_mbmf.detach().item()),
        }

    @torch.no_grad()
    def diagnostics(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> Dict[str, float]:
        _, _, rW, rH, W, H = self.encoded(data)
        p = torch.sum(W[users] * H[items], dim=1)
        b = rW[users, 0] * rH[items, 0]
        return {
            "max_structural_bound_violation": float(torch.relu(torch.abs(p) - b).max().item()),
            "mean_rW": float(rW.mean().item()),
            "mean_rH": float(rH.mean().item()),
            "max_pair_bound_eval": float(b.max().item()),
        }


# =============================================================================
# 9. Thesis Method 2 -- Two-view magnitude fusion
# =============================================================================


class ThesisMethod2(nn.Module):
    """Reconstruction of the second, genuinely different model in old cell3/cell4.

    User-anchored view:
      h_u = D1_u(E_u(X_u)) in R^K
      a_u = ||h_u||+eps
      b_i = positive trainable counterpart radius (old code: ||D2_u[i,:]||+eps)
      p^U_ui = <Spherical(Phi_U[u],a_u), Spherical(Theta_U[i],b_i)>

    Item-anchored view:
      h_i = D1_i(E_i(X_i)) in R^K
      c_i = ||h_i||+eps
      d_u = positive trainable counterpart radius (old code: ||D2_i[u,:]||+eps)
      p^I_ui = <Spherical(Phi_I[i],c_i), Spherical(Theta_I[u],d_u)>

    Fused prediction:
      p_ui = gamma p^U_ui + (1-gamma) p^I_ui

    Corrected training loss:
      L = MSE(p,V) + lambda_consistency MSE(p^U,p^I)

    The old notebook accidentally omitted the square in the prediction loss; this implementation
    intentionally fixes that bug. There is no AE reconstruction loss because the old second model
    commented out its reconstructions and never used weight_ae/weight_mbmf.
    """
    def __init__(self, data: MatrixData, cfg: Method2Config, device: torch.device):
        super().__init__()
        self.cfg = cfg
        uin, iin = input_dims(data, cfg.input_mode)
        k = cfg.latent_dim
        self.user_features = TwoStageFeatureExtractor(uin, k)
        self.item_features = TwoStageFeatureExtractor(iin, k)

        self.item_counterpart = PositiveCounterpartRadii(
            data.num_items, k, cfg.counterpart_radius_mode, model_pair_bound_scale(data, cfg.model_space),
            cfg.radius_eps, cfg.bounded_counterpart_init_fraction
        )
        self.user_counterpart = PositiveCounterpartRadii(
            data.num_users, k, cfg.counterpart_radius_mode, model_pair_bound_scale(data, cfg.model_space),
            cfg.radius_eps, cfg.bounded_counterpart_init_fraction
        )

        # User-anchored directions.
        self.Phi_user = nn.Parameter(initialise_angles(data.num_users, k, cfg.angle_init, device))
        self.Theta_user = nn.Parameter(initialise_angles(data.num_items, k, cfg.angle_init, device))
        # Item-anchored directions (kept independent, as in old Phi/Theta vs Phi2/Theta2).
        self.Phi_item = nn.Parameter(initialise_angles(data.num_items, k, cfg.angle_init, device))
        self.Theta_item = nn.Parameter(initialise_angles(data.num_users, k, cfg.angle_init, device))
        self.to(device)

    def factors(self, data: MatrixData):
        xu, xi = build_encoder_inputs(data, self.cfg.input_mode, self.cfg.model_space)
        hu = self.user_features(xu)
        hi = self.item_features(xi)

        bound_scale = model_pair_bound_scale(data, self.cfg.model_space)
        a_u = latent_radius(hu, self.cfg.radius_mode, bound_scale, self.cfg.radius_eps)
        b_i = self.item_counterpart()
        c_i = latent_radius(hi, self.cfg.radius_mode, bound_scale, self.cfg.radius_eps)
        d_u = self.user_counterpart()

        WU = spherical_to_cartesian(self.Phi_user, a_u)       # users
        HU = spherical_to_cartesian(self.Theta_user, b_i)     # items
        WI = spherical_to_cartesian(self.Phi_item, c_i)       # items
        HI = spherical_to_cartesian(self.Theta_item, d_u)     # users
        return a_u, b_i, c_i, d_u, WU, HU, WI, HI

    def predict_centered_views(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor):
        a_u, b_i, c_i, d_u, WU, HU, WI, HI = self.factors(data)
        pU = torch.sum(WU[users] * HU[items], dim=1)
        # Old code formed [item,user] and transposed; this is the same scalar dot product.
        pI = torch.sum(WI[items] * HI[users], dim=1)
        g = float(self.cfg.fusion_weight_user_view)
        if not 0 <= g <= 1:
            raise ValueError("fusion_weight_user_view must be in [0,1]")
        p = g * pU + (1 - g) * pI
        return pU, pI, p, (a_u, b_i, c_i, d_u)

    def predict_raw_pairs(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return model_space_to_raw(data, self.predict_centered_views(data, users, items)[2], self.cfg.model_space)

    def loss(self, data: MatrixData) -> Tuple[torch.Tensor, Dict[str, float]]:
        pU, pI, p, _ = self.predict_centered_views(data, data.train_users, data.train_items)
        target = model_space_targets(data, data.train_ratings_raw, self.cfg.model_space)
        l_pred = torch.mean((p - target) ** 2)
        l_cons = torch.mean((pU - pI) ** 2)
        total = l_pred + self.cfg.consistency_weight * l_cons
        return total, {"L_pred": float(l_pred.detach().item()), "L_consistency": float(l_cons.detach().item())}

    @torch.no_grad()
    def diagnostics(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> Dict[str, float]:
        pU, pI, p, (a, b, c, d) = self.predict_centered_views(data, users, items)
        g = float(self.cfg.fusion_weight_user_view)
        bU = a[users, 0] * b[items, 0]
        bI = c[items, 0] * d[users, 0]
        bf = g * bU + (1 - g) * bI
        return {
            "max_user_view_bound_violation": float(torch.relu(torch.abs(pU) - bU).max().item()),
            "max_item_view_bound_violation": float(torch.relu(torch.abs(pI) - bI).max().item()),
            "max_structural_bound_violation": float(torch.relu(torch.abs(p) - bf).max().item()),
            "max_pair_bound_eval": float(bf.max().item()),
            "mean_user_encoded_radius": float(a.mean().item()),
            "mean_item_counterpart_radius": float(b.mean().item()),
            "mean_item_encoded_radius": float(c.mean().item()),
            "mean_user_counterpart_radius": float(d.mean().item()),
        }


# =============================================================================
# 10. Proposed Method 3 -- Historical-prior residual-magnitude AE-MBMF
# =============================================================================


def _historical_unit_magnitudes(
    data: MatrixData,
    rho: float,
    prior_mode: PriorMode,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Initial radii from TRAIN ratings normalised to [0,1].

    ``legacy_local_mean_std`` matches the separate notebook's initial magnitude idea:
    sqrt(local mean + local population std), with global fallback only when empty.

    ``eq42_shrinkage`` applies the MBMF Eq.42/43 global-local shrinkage to those
    unit-normalised historical ratings.  It is a proposed stabilisation, not a claim that
    the paper itself applies Eq.42 to normalised MovieLens values.
    """
    if not (0 < rho <= 1):
        raise ValueError("rho must be in (0,1]")
    width = data.rating_max - data.rating_min
    y = ((data.train_ratings_raw - data.rating_min) / width).clamp(0.0, 1.0)

    gmean = y.mean()
    gstd = torch.sqrt(torch.mean((y - gmean) ** 2))
    gmag = torch.sqrt((gmean + gstd).clamp_min(1e-8))

    um, us, uc = _group_mean_std(data.train_users, y, data.num_users)
    im, iss, ic = _group_mean_std(data.train_items, y, data.num_items)
    local_u = torch.sqrt((um + us).clamp_min(1e-8))
    local_i = torch.sqrt((im + iss).clamp_min(1e-8))

    if prior_mode == "legacy_local_mean_std":
        rW = torch.where((uc > 0), local_u, gmag.expand_as(local_u))
        rH = torch.where((ic > 0), local_i, gmag.expand_as(local_i))
        omega_u = (uc > 0).to(y.dtype)
        omega_i = (ic > 0).to(y.dtype)
    elif prior_mode == "eq42_shrinkage":
        omega_u = torch.clamp(uc / (rho * data.num_items), max=1.0)
        omega_i = torch.clamp(ic / (rho * data.num_users), max=1.0)
        omega_u[uc == 0] = 0
        omega_i[ic == 0] = 0
        rW = omega_u * local_u + (1 - omega_u) * gmag
        rH = omega_i * local_i + (1 - omega_i) * gmag
    else:
        raise ValueError(prior_mode)

    return rW[:, None], rH[:, None], {
        "prior_global_unit_magnitude": float(gmag.item()),
        "prior_mean_omega_user": float(omega_u.mean().item()),
        "prior_mean_omega_item": float(omega_i.mean().item()),
    }


@torch.no_grad()
def method3_initial_radii(
    data: MatrixData,
    cfg: Method3Config,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    rW, rH, info = _historical_unit_magnitudes(data, cfg.rho, cfg.prior_mode)
    # Unit-space prior products are dimensionless.  For centered raw-space prediction,
    # scale each radius by sqrt(half_range), so pair products have rating-unit scale.
    scale = 1.0 if cfg.model_space == "legacy_unit" else math.sqrt(data.half_range)
    rW = (rW * scale).clamp_min(1e-8)
    rH = (rH * scale).clamp_min(1e-8)
    return rW, rH, {**info, "prior_radius_space_scale": float(scale)}


class ResidualRadiusAEBranch(nn.Module):
    """Legacy residual-radius branch, made pairwise/memory-safe.

    input -> K -> K/2 bottleneck -> K -> output
    bottleneck -> zero-initialised scalar log-radius correction
    """
    def __init__(self, input_dim: int, output_dim: int, latent_dim: int):
        super().__init__()
        bottleneck = max(1, latent_dim // 2)
        self.encoder1 = nn.Linear(input_dim, latent_dim)
        self.encoder2 = nn.Linear(latent_dim, bottleneck)
        self.decoder_hidden = nn.Linear(bottleneck, latent_dim)
        self.decoder_output = nn.Linear(latent_dim, output_dim)
        self.magnitude_head = nn.Linear(bottleneck, 1)
        nn.init.zeros_(self.magnitude_head.weight)
        nn.init.zeros_(self.magnitude_head.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder2(torch.relu(self.encoder1(x)))

    def selected_reconstruction(
        self, z: torch.Tensor, sample_index: torch.Tensor, output_index: torch.Tensor
    ) -> torch.Tensor:
        h = torch.relu(self.decoder_hidden(z))[sample_index]
        w = self.decoder_output.weight[output_index]
        b = self.decoder_output.bias[output_index]
        return torch.sum(h * w, dim=1) + b

    def raw_log_correction(self, z: torch.Tensor) -> torch.Tensor:
        return self.magnitude_head(z)


def _bounded_log_correction(raw: torch.Tensor, cfg: Method3Config) -> torch.Tensor:
    b = float(cfg.max_abs_log_scale)
    if b < 0:
        raise ValueError("max_abs_log_scale must be non-negative")
    if b == 0:
        return torch.zeros_like(raw)
    if cfg.residual_transform == "hard_clamp":
        return torch.clamp(raw, -b, b)
    if cfg.residual_transform == "smooth_tanh":
        return b * torch.tanh(raw / b)
    raise ValueError(cfg.residual_transform)


def _observed_loss(pred: torch.Tensor, target: torch.Tensor, form: LossForm) -> torch.Tensor:
    e2 = (pred - target) ** 2
    if form == "mse":
        return torch.mean(e2)
    if form == "frobenius":
        return torch.sqrt(torch.sum(e2) + 1e-8)
    raise ValueError(form)


class ProposedMethod3(nn.Module):
    r"""Historical-prior residual-magnitude AE-MBMF.

    Revised from model-2 in ``mbmf_two_models_complete.ipynb``:

      z_u = E_u(X_u),    delta_u = T(g_u(z_u))
      z_i = E_i(X_i),    delta_i = T(g_i(z_i))
      r_u = r0_u exp(delta_u),   r_i = r0_i exp(delta_i)
      W_u = Spherical(Phi_u,r_u), H_i = Spherical(Theta_i,r_i)
      p_ui = W_u^T H_i

    Q1 default uses separate user/item AE losses.  The legacy preset can reproduce
    the legacy source code's averaged-AE Frobenius loss while retaining leakage-safe
    train-only encoder input and raw (unclipped) evaluation.
    """
    def __init__(self, data: MatrixData, cfg: Method3Config, device: torch.device):
        super().__init__()
        self.cfg = cfg
        uin, iin = input_dims(data, cfg.input_mode)
        rW0, rH0, pinfo = method3_initial_radii(data, cfg)
        self.register_buffer("rW0", rW0.to(device))
        self.register_buffer("rH0", rH0.to(device))
        self.prior_info = pinfo
        self.user_branch = ResidualRadiusAEBranch(uin, data.num_items, cfg.latent_dim)
        self.item_branch = ResidualRadiusAEBranch(iin, data.num_users, cfg.latent_dim)
        self.Phi = nn.Parameter(initialise_angles(data.num_users, cfg.latent_dim, cfg.angle_init, device))
        self.Theta = nn.Parameter(initialise_angles(data.num_items, cfg.latent_dim, cfg.angle_init, device))
        self.to(device)

    def encoded(self, data: MatrixData):
        xu, xi = build_encoder_inputs(data, self.cfg.input_mode, self.cfg.model_space)
        zu = self.user_branch.encode(xu)
        zi = self.item_branch.encode(xi)
        du = _bounded_log_correction(self.user_branch.raw_log_correction(zu), self.cfg)
        di = _bounded_log_correction(self.item_branch.raw_log_correction(zi), self.cfg)
        rW = self.rW0 * torch.exp(du)
        rH = self.rH0 * torch.exp(di)
        W = spherical_to_cartesian(self.Phi, rW)
        H = spherical_to_cartesian(self.Theta, rH)
        return zu, zi, du, di, rW, rH, W, H

    def predict_model_pairs(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        *_, W, H = self.encoded(data)
        return torch.sum(W[users] * H[items], dim=1)

    def predict_raw_pairs(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return model_space_to_raw(data, self.predict_model_pairs(data, users, items), self.cfg.model_space)

    def loss(self, data: MatrixData) -> Tuple[torch.Tensor, Dict[str, float]]:
        zu, zi, du, di, _, _, W, H = self.encoded(data)
        u, i = data.train_users, data.train_items
        target = model_space_targets(data, data.train_ratings_raw, self.cfg.model_space)

        rec_u = self.user_branch.selected_reconstruction(zu, u, i)
        rec_i = self.item_branch.selected_reconstruction(zi, i, u)
        l_user = _observed_loss(rec_u, target, self.cfg.loss_form)
        l_item = _observed_loss(rec_i, target, self.cfg.loss_form)
        if self.cfg.ae_loss_mode == "separate":
            l_ae = l_user + l_item
        elif self.cfg.ae_loss_mode == "averaged":
            l_ae = _observed_loss(0.5 * (rec_u + rec_i), target, self.cfg.loss_form)
        else:
            raise ValueError(self.cfg.ae_loss_mode)

        pred = torch.sum(W[u] * H[i], dim=1)
        l_mbmf = _observed_loss(pred, target, self.cfg.loss_form)
        l_delta = torch.mean(du ** 2) + torch.mean(di ** 2)
        total = self.cfg.alpha_ae * l_ae + self.cfg.beta_mbmf * l_mbmf + self.cfg.delta_l2 * l_delta
        return total, {
            "L_user_AE": float(l_user.detach().item()),
            "L_item_AE": float(l_item.detach().item()),
            "L_AE": float(l_ae.detach().item()),
            "L_MBM": float(l_mbmf.detach().item()),
            "L_delta": float(l_delta.detach().item()),
        }

    @torch.no_grad()
    def diagnostics(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> Dict[str, float]:
        _, _, du, di, rW, rH, W, H = self.encoded(data)
        p = torch.sum(W[users] * H[items], dim=1)
        b = rW[users, 0] * rH[items, 0]
        return {
            "max_structural_bound_violation": float(torch.relu(torch.abs(p) - b).max().item()) if b.numel() else 0.0,
            "mean_prior_rW": float(self.rW0.mean().item()),
            "mean_prior_rH": float(self.rH0.mean().item()),
            "mean_rW": float(rW.mean().item()),
            "mean_rH": float(rH.mean().item()),
            "min_log_correction_user": float(du.min().item()),
            "max_log_correction_user": float(du.max().item()),
            "min_log_correction_item": float(di.min().item()),
            "max_log_correction_item": float(di.max().item()),
            "max_pair_bound_eval": float(b.max().item()) if b.numel() else float("nan"),
            **self.prior_info,
        }


# =============================================================================
# 11. Training / evaluation for proposed models
# =============================================================================


def evaluate_model_v6_legacy(model: nn.Module, data: MatrixData, split: Literal["train", "val", "test"]) -> Dict[str, float]:
    if split == "train":
        u, i, y = data.train_users, data.train_items, data.train_ratings_raw
    elif split == "val":
        u, i, y = data.val_users, data.val_items, data.val_ratings_raw
    elif split == "test":
        u, i, y = data.test_users, data.test_items, data.test_ratings_raw
    else:
        raise ValueError(split)

    model.eval()
    with torch.no_grad():
        if isinstance(model, PaperMBMF):
            p = model.predict_raw_pairs(u, i)
        else:
            # Encoders always consume TRAIN data internally through MatrixData.
            p = model.predict_raw_pairs(data, u, i)
        out = regression_metrics_raw(p, y, data.rating_min, data.rating_max)
        out.update(train_threshold_f1(u, y, p, data.user_train_mean_raw))
    return out


def train_thesis_model(
    model: nn.Module,
    data: MatrixData,
    *,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    eval_every: int,
    val_patience: int,
    grad_clip: float,
) -> Dict[str, object]:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    wait = 0
    history: List[Dict[str, object]] = []
    start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, parts = model.loss(data)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite thesis-model loss at epoch {epoch}")
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if epoch == 1 or epoch % eval_every == 0:
            val = evaluate_model(model, data, "val")
            if not math.isfinite(val["RMSE"]):
                raise FloatingPointError(f"non-finite validation RMSE at epoch {epoch}")
            history.append({"epoch": epoch, "loss": float(loss.item()), "val_RMSE": val["RMSE"], **parts})
            if val["RMSE"] < best - 1e-8:
                best = val["RMSE"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= val_patience:
                    break

    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_val_RMSE": best,
        "elapsed_sec": time.perf_counter() - start,
        "history": history,
    }


# =============================================================================
# 12. Experiment wrappers
# =============================================================================


def run_paper_once(
    data: MatrixData,
    cfg: PaperBaselineConfig,
    variant: BaselineVariant,
    init_mode: InitMode,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    set_global_seed(seed)
    model = PaperMBMF(data, cfg, variant, init_mode, device)
    info = train_paper_mbmf(model, data, cfg)
    val = evaluate_model(model, data, "val")
    test = evaluate_model(model, data, "test")
    diag = model.diagnostics(data.test_users, data.test_items)
    return {
        "model": variant,
        "protocol": "common_train_val_test",
        "init": init_mode,
        "seed": seed,
        "latent_dim": cfg.latent_dim,
        "parameters": parameter_count(model),
        "magnitude_source": "known_global_bounds" if variant == "MBMF-F" else "training_split_only (Eq.42; protocol adaptation)",
        "global_rating_bound_guaranteed": bool(variant == "MBMF-F"),
        **{f"val_{k}": v for k, v in val.items()},
        **{f"test_{k}": v for k, v in test.items()},
        **diag,
        **{k: v for k, v in info.items() if k != "history"},
        "_history": info.get("history", []),
    }


def run_method1_once(data: MatrixData, cfg: Method1Config, seed: int, device: torch.device) -> Dict[str, object]:
    set_global_seed(seed)
    model = ThesisMethod1(data, cfg, device)
    info = train_thesis_model(
        model, data, max_epochs=cfg.max_epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
        eval_every=cfg.eval_every, val_patience=cfg.val_patience, grad_clip=cfg.grad_clip,
    )
    val = evaluate_model(model, data, "val")
    test = evaluate_model(model, data, "test")
    diag = _model_diagnostics(model, data, data.test_users, data.test_items)
    return {
        "model": "Thesis-Method-1-AE-LatentMagnitude",
        "protocol": "common_train_val_test",
        "init": cfg.angle_init,
        "seed": seed,
        "latent_dim": cfg.latent_dim,
        "input_mode": cfg.input_mode,
        "model_space": cfg.model_space,
        "radius_mode": cfg.radius_mode,
        "parameters": parameter_count(model),
        "target_space": cfg.model_space,
        "global_rating_bound_guaranteed": bool(cfg.model_space == "centered_raw" and cfg.radius_mode == "bounded_tanh"),
        **{f"val_{k}": v for k, v in val.items()},
        **{f"test_{k}": v for k, v in test.items()},
        **diag,
        **{k: v for k, v in info.items() if k != "history"},
        "_history": info.get("history", []),
    }


def run_method2_once(data: MatrixData, cfg: Method2Config, seed: int, device: torch.device) -> Dict[str, object]:
    set_global_seed(seed)
    model = ThesisMethod2(data, cfg, device)
    info = train_thesis_model(
        model, data, max_epochs=cfg.max_epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
        eval_every=cfg.eval_every, val_patience=cfg.val_patience, grad_clip=cfg.grad_clip,
    )
    val = evaluate_model(model, data, "val")
    test = evaluate_model(model, data, "test")
    diag = _model_diagnostics(model, data, data.test_users, data.test_items)
    return {
        "model": "Thesis-Method-2-DualViewFusion",
        "protocol": "common_train_val_test",
        "init": cfg.angle_init,
        "seed": seed,
        "latent_dim": cfg.latent_dim,
        "input_mode": cfg.input_mode,
        "model_space": cfg.model_space,
        "radius_mode": cfg.radius_mode,
        "counterpart_radius_mode": cfg.counterpart_radius_mode,
        "parameters": parameter_count(model),
        "target_space": cfg.model_space,
        "global_rating_bound_guaranteed": bool(
            cfg.model_space == "centered_raw" and cfg.radius_mode == "bounded_tanh" and cfg.counterpart_radius_mode == "bounded_sigmoid"
        ),
        **{f"val_{k}": v for k, v in val.items()},
        **{f"test_{k}": v for k, v in test.items()},
        **diag,
        **{k: v for k, v in info.items() if k != "history"},
        "_history": info.get("history", []),
    }


def run_method3_once(data: MatrixData, cfg: Method3Config, seed: int, device: torch.device) -> Dict[str, object]:
    set_global_seed(seed)
    model = ProposedMethod3(data, cfg, device)
    info = train_thesis_model(
        model, data, max_epochs=cfg.max_epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
        eval_every=cfg.eval_every, val_patience=cfg.val_patience, grad_clip=cfg.grad_clip,
    )
    val = evaluate_model(model, data, "val")
    test = evaluate_model(model, data, "test")
    diag = _model_diagnostics(model, data, data.test_users, data.test_items)
    return {
        "model": "Proposed-Method-3-PriorResidualMagnitude",
        "protocol": "common_train_val_test",
        "init": cfg.angle_init,
        "seed": seed,
        "latent_dim": cfg.latent_dim,
        "input_mode": cfg.input_mode,
        "model_space": cfg.model_space,
        "prior_mode": cfg.prior_mode,
        "residual_transform": cfg.residual_transform,
        "max_abs_log_scale": cfg.max_abs_log_scale,
        "ae_loss_mode": cfg.ae_loss_mode,
        "loss_form": cfg.loss_form,
        "parameters": parameter_count(model),
        "target_space": cfg.model_space,
        "global_rating_bound_guaranteed": False,
        **{f"val_{k}": v for k, v in val.items()},
        **{f"test_{k}": v for k, v in test.items()},
        **diag,
        **{k: v for k, v in info.items() if k != "history"},
        "_history": info.get("history", []),
    }


def summarise_results(df: pd.DataFrame) -> pd.DataFrame:
    groups = [c for c in [
        "model", "init", "latent_dim", "input_mode", "model_space", "radius_mode",
        "counterpart_radius_mode", "prior_mode", "residual_transform", "ae_loss_mode", "loss_form"
    ] if c in df.columns]
    metrics = [c for c in [
        "test_MAE", "test_RMSE", "test_F1_trainThreshold_macro_user",
        "test_F1_trainThreshold_micro", "test_OOB_rate", "elapsed_sec"
    ] if c in df.columns]
    return df.groupby(groups, dropna=False)[metrics].agg(["mean", "std", "count"])



def heldout_label_invariance_test(device: torch.device) -> None:
    """Regression guard against the leakage bug in the earlier notebooks.

    For thesis methods, changing validation/test *labels* while keeping the training
    observations fixed must not change predictions.  The encoders are allowed to see
    only MatrixData.train_values_centered/train_mask.
    """
    data = synthetic_data(device)

    m1cfg = Method1Config(latent_dim=4, hidden_dim=8, max_epochs=1)
    m2cfg = Method2Config(latent_dim=4, max_epochs=1)
    m3cfg = Method3Config(latent_dim=4, max_epochs=1)

    set_global_seed(1234)
    m1 = ThesisMethod1(data, m1cfg, device).eval()
    set_global_seed(1234)
    m2 = ThesisMethod2(data, m2cfg, device).eval()
    set_global_seed(1234)
    m3 = ProposedMethod3(data, m3cfg, device).eval()

    with torch.no_grad():
        p1_before = m1.predict_raw_pairs(data, data.val_users, data.val_items).clone()
        p2_before = m2.predict_raw_pairs(data, data.val_users, data.val_items).clone()
        p3_before = m3.predict_raw_pairs(data, data.val_users, data.val_items).clone()

    altered = copy.deepcopy(data)
    # Deliberately corrupt held-out labels only.  Training values/mask are unchanged.
    altered.val_ratings_raw = data.rating_min + data.rating_max - data.val_ratings_raw
    altered.test_ratings_raw = data.rating_min + data.rating_max - data.test_ratings_raw

    with torch.no_grad():
        p1_after = m1.predict_raw_pairs(altered, altered.val_users, altered.val_items)
        p2_after = m2.predict_raw_pairs(altered, altered.val_users, altered.val_items)
        p3_after = m3.predict_raw_pairs(altered, altered.val_users, altered.val_items)

    if not torch.equal(p1_before, p1_after):
        raise AssertionError("Method 1 predictions depend on held-out labels: leakage regression.")
    if not torch.equal(p2_before, p2_after):
        raise AssertionError("Method 2 predictions depend on held-out labels: leakage regression.")
    if not torch.equal(p3_before, p3_after):
        raise AssertionError("Method 3 predictions depend on held-out labels: leakage regression.")
    print("[self-test] held-out label invariance: PASS")


def validation_selected_runs_v6_legacy(runs: pd.DataFrame) -> pd.DataFrame:
    """Select K/initialisation *only from validation RMSE*, separately per outer split/model.

    Model seed is treated as replication, not a tunable hyperparameter.  The returned
    frame contains all seed replicates of the validation-selected configuration and can
    therefore be used for the locked-test table without test-driven cherry-picking.
    """
    if runs.empty:
        return runs.copy()

    required = {"split_seed", "model", "val_RMSE", "test_RMSE"}
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"runs is missing columns required for selection: {sorted(missing)}")

    # Fields that define a model configuration.  Missing/NaN values are retained.
    candidates = [
        "protocol", "init", "latent_dim", "input_mode", "model_space", "radius_mode",
        "counterpart_radius_mode", "prior_mode", "residual_transform",
        "max_abs_log_scale", "ae_loss_mode", "loss_form",
    ]
    cfg_cols = [c for c in candidates if c in runs.columns]

    selected_parts: List[pd.DataFrame] = []
    for (split_seed, model_name), g in runs.groupby(["split_seed", "model"], dropna=False):
        group_cols = cfg_cols
        means = (
            g.groupby(group_cols, dropna=False, as_index=False)["val_RMSE"]
            .mean()
            .sort_values(["val_RMSE"] + group_cols, kind="mergesort")
        )
        best = means.iloc[0]

        mask = torch.ones(len(g), dtype=torch.bool).numpy()
        for c in group_cols:
            bv = best[c]
            if pd.isna(bv):
                mask &= g[c].isna().to_numpy()
            else:
                mask &= (g[c].to_numpy() == bv)
        chosen = g.loc[mask].copy()
        chosen["selection_val_RMSE_mean"] = float(best["val_RMSE"])
        selected_parts.append(chosen)

    return pd.concat(selected_parts, ignore_index=True) if selected_parts else runs.iloc[0:0].copy()


def summarise_selected_test_v6_legacy(selected: pd.DataFrame) -> pd.DataFrame:
    """Publication-facing test summary after validation-only configuration selection."""
    if selected.empty:
        return pd.DataFrame()
    metrics = [
        c for c in [
            "test_MAE", "test_RMSE", "test_F1_trainThreshold_macro_user",
            "test_F1_trainThreshold_micro", "test_OOB_rate", "elapsed_sec"
        ] if c in selected.columns
    ]
    # Aggregate model-seed replicates and outer split replicates.
    return selected.groupby(["model"], dropna=False)[metrics].agg(["mean", "std", "count"])


def run_all_experiments_v6_legacy(cfg: ExperimentConfigV6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full study under explicitly separated split and model seeds.

    `split_seeds` changes which interactions are held out. `model_seeds` changes only
    parameter/angle initialisation. `latent_dims` is shared by every family so K is never
    tuned asymmetrically. Test labels are not consulted until each fitted run is finished. Publication-facing configuration selection is then performed from validation RMSE only; see selected_by_validation_*.csv.
    """
    device = get_device()
    print(f"[device] {device}")
    geometry_self_test(device)
    heldout_label_invariance_test(device)
    ratings = load_movielens(cfg.data.data_dir)
    data_manifest = dataset_manifest(cfg.data.data_dir, ratings)

    base_pcfg = copy.deepcopy(cfg.paper)
    base_m1cfg = copy.deepcopy(cfg.method1)
    base_m2cfg = copy.deepcopy(cfg.method2)
    base_m3cfg = copy.deepcopy(cfg.method3)
    split_seeds: Sequence[int] = cfg.split_seeds
    model_seeds: Sequence[int] = cfg.model_seeds
    latent_dims: Sequence[int] = cfg.latent_dims

    if cfg.fast_debug:
        split_seeds = split_seeds[:1]
        model_seeds = model_seeds[:1]
        latent_dims = latent_dims[:1]

    rows: List[Dict[str, object]] = []
    histories: List[Dict[str, object]] = []
    split_frames: List[pd.DataFrame] = []

    def collect(row: Dict[str, object], *, split_seed: int, latent_dim: int) -> None:
        hist = row.pop("_history", [])
        row["split_seed"] = int(split_seed)
        row["latent_dim"] = int(latent_dim)
        histories.append({
            "model": row.get("model"),
            "init": row.get("init"),
            "model_seed": row.get("seed"),
            "split_seed": int(split_seed),
            "latent_dim": int(latent_dim),
            "history": hist,
        })
        rows.append(row)

    for split_seed in split_seeds:
        dcfg = copy.deepcopy(cfg.data)
        dcfg.split_seed = int(split_seed)
        print(f"\n######## split_seed={split_seed} ########")
        split = make_safe_split(ratings, dcfg)
        sf = split[["userId", "movieId", "rating", "timestamp", "split"]].copy()
        sf.insert(0, "split_seed", int(split_seed))
        split_frames.append(sf)
        data = build_matrix_data(split, dcfg, device)

        for latent_dim in latent_dims:
            pcfg = copy.deepcopy(base_pcfg)
            m1cfg = copy.deepcopy(base_m1cfg)
            m2cfg = copy.deepcopy(base_m2cfg)
            m3cfg = copy.deepcopy(base_m3cfg)
            pcfg.latent_dim = int(latent_dim)
            m1cfg.latent_dim = int(latent_dim)
            m2cfg.latent_dim = int(latent_dim)
            m3cfg.latent_dim = int(latent_dim)

            if cfg.fast_debug:
                pcfg.max_epochs = min(20, pcfg.max_epochs)
                m1cfg.max_epochs = min(20, m1cfg.max_epochs)
                m2cfg.max_epochs = min(20, m2cfg.max_epochs)
                m3cfg.max_epochs = min(20, m3cfg.max_epochs)
                pcfg.eval_every = m1cfg.eval_every = m2cfg.eval_every = m3cfg.eval_every = 2

            for variant in pcfg.variants:
                modes: Sequence[InitMode] = (
                    pcfg.init_modes if variant in pcfg.init_ablation_variants else (pcfg.default_init,)
                )
                for mode in modes:
                    for model_seed in model_seeds:
                        print(
                            f"\n=== {variant} | K={latent_dim} | init={mode} | "
                            f"split={split_seed} | model_seed={model_seed} ==="
                        )
                        row = run_paper_once(data, pcfg, variant, mode, int(model_seed), device)
                        collect(row, split_seed=int(split_seed), latent_dim=int(latent_dim))

            for model_seed in model_seeds:
                print(
                    f"\n=== Thesis Method 1 | K={latent_dim} | split={split_seed} | "
                    f"model_seed={model_seed} ==="
                )
                row = run_method1_once(data, m1cfg, int(model_seed), device)
                collect(row, split_seed=int(split_seed), latent_dim=int(latent_dim))

            for model_seed in model_seeds:
                print(
                    f"\n=== Thesis Method 2 | K={latent_dim} | split={split_seed} | "
                    f"model_seed={model_seed} ==="
                )
                row = run_method2_once(data, m2cfg, int(model_seed), device)
                collect(row, split_seed=int(split_seed), latent_dim=int(latent_dim))

            for model_seed in model_seeds:
                print(
                    f"\n=== Proposed Method 3 | K={latent_dim} | split={split_seed} | "
                    f"model_seed={model_seed} ==="
                )
                row = run_method3_once(data, m3cfg, int(model_seed), device)
                collect(row, split_seed=int(split_seed), latent_dim=int(latent_dim))

    runs = pd.DataFrame(rows)
    summary = summarise_results(runs)
    selected_runs = validation_selected_runs(runs)
    selected_summary = summarise_selected_test(selected_runs)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)
    summary.to_csv(out / "summary.csv")
    selected_runs.to_csv(out / "selected_by_validation_runs.csv", index=False)
    selected_summary.to_csv(out / "selected_by_validation_summary.csv")
    if not runs.empty and "split_seed" in runs.columns:
        by_split_groups = [
            c for c in [
                "split_seed", "model", "init", "latent_dim", "input_mode", "model_space",
                "radius_mode", "counterpart_radius_mode", "prior_mode", "residual_transform",
                "ae_loss_mode", "loss_form"
            ]
            if c in runs.columns
        ]
        by_split_metrics = [
            c for c in [
                "test_MAE", "test_RMSE", "test_F1_trainThreshold_macro_user",
                "test_F1_trainThreshold_micro", "test_OOB_rate"
            ]
            if c in runs.columns
        ]
        runs.groupby(by_split_groups, dropna=False)[by_split_metrics].agg(["mean", "std", "count"]).to_csv(
            out / "summary_by_split.csv"
        )
    if split_frames:
        pd.concat(split_frames, ignore_index=True).to_csv(out / "split_assignments.csv", index=False)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
    }
    with open(out / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)
    with open(out / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(data_manifest, f, ensure_ascii=False, indent=2)
    with open(out / "training_histories.json", "w", encoding="utf-8") as f:
        json.dump(histories, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out.resolve()}")
    return runs, summary


# =============================================================================
# 13. Synthetic tests: geometry, 3 MBMF baselines, and 3 proposed methods
# =============================================================================


def synthetic_data(device: torch.device) -> MatrixData:
    set_global_seed(123)
    n, m = 9, 13
    rmin, rmax = 0.5, 5.0
    midpoint, half = (rmin + rmax) / 2, (rmax - rmin) / 2
    rng = np.random.default_rng(123)

    # Deterministically guarantee every user AND every item has training support,
    # then fill the remaining training quota at random.  The previous smoke test
    # merely shuffled all pairs and *usually* had support; that was not a guarantee.
    all_pairs = [(u, i) for u in range(n) for i in range(m)]
    mandatory = []
    seen = set()
    for u in range(n):
        pair = (u, u % m)
        mandatory.append(pair); seen.add(pair)
    for i in range(m):
        pair = (i % n, i)
        if pair not in seen:
            mandatory.append(pair); seen.add(pair)
    remaining = [p for p in all_pairs if p not in seen]
    rng.shuffle(remaining)
    tr = mandatory + remaining[: 80 - len(mandatory)]
    rest = remaining[80 - len(mandatory):]
    va = rest[:18]
    te = rest[18:]
    if {u for u, _ in tr} != set(range(n)) or {i for _, i in tr} != set(range(m)):
        raise AssertionError("synthetic split failed train-support invariant")

    def pack(ps):
        us, it, yy = [], [], []
        for u, i in ps:
            us.append(u); it.append(i); yy.append(float(rng.choice([0.5, 1.0, 2.5, 4.0, 5.0])))
        return (
            torch.tensor(us, dtype=torch.long, device=device),
            torch.tensor(it, dtype=torch.long, device=device),
            torch.tensor(yy, dtype=torch.float32, device=device),
        )

    tu, ti, ty = pack(tr); vu, vi, vy = pack(va); eu, ei, ey = pack(te)
    valmat = torch.zeros(n, m, device=device)
    mask = torch.zeros(n, m, device=device)
    valmat[tu, ti] = ty - midpoint
    mask[tu, ti] = 1

    sums = torch.zeros(n, device=device); cnt = torch.zeros(n, device=device)
    sums.scatter_add_(0, tu, ty); cnt.scatter_add_(0, tu, torch.ones_like(ty))
    means = torch.full((n,), float(ty.mean().item()), device=device)
    ok = cnt > 0; means[ok] = sums[ok] / cnt[ok]

    return MatrixData(
        n, m, {i:i for i in range(n)}, {i:i for i in range(m)}, np.arange(n), np.arange(m),
        rmin, rmax, midpoint, half,
        valmat, mask,
        tu, ti, vu, vi, eu, ei,
        ty, vy, ey,
        means,
    )


def synthetic_smoke_test(device: Optional[torch.device] = None) -> None:
    device = device or get_device()
    geometry_self_test(device)
    d = synthetic_data(device)

    pcfg = PaperBaselineConfig(latent_dim=4, max_epochs=3, eval_every=1, objective_reduction="mean")
    for v in ("MBMF-F", "MBMF-C", "MBMF-N"):
        set_global_seed(7)
        m = PaperMBMF(d, pcfg, v, "isotropic_gaussian", device)
        target = m.native_targets(d.train_ratings_raw, d.train_users, d.train_items)
        pred = m.native_predict_pairs(d.train_users, d.train_items)
        loss = paper_gradient_loss(pred - target, "mean")
        loss.backward()
        if not torch.isfinite(loss):
            raise AssertionError(v)
        diag = m.diagnostics(d.test_users, d.test_items)
        if diag["max_structural_bound_violation"] > 5e-5:
            raise AssertionError((v, diag))

    c1 = Method1Config(latent_dim=4, hidden_dim=8, radius_mode="raw_norm", max_epochs=2)
    m1 = ThesisMethod1(d, c1, device)
    l1, _ = m1.loss(d); l1.backward()
    if not torch.isfinite(l1): raise AssertionError("method1")
    if m1.diagnostics(d, d.test_users, d.test_items)["max_structural_bound_violation"] > 5e-5:
        raise AssertionError("method1 bound")

    c2 = Method2Config(latent_dim=4, radius_mode="raw_norm", counterpart_radius_mode="scalar_softplus", max_epochs=2)
    m2 = ThesisMethod2(d, c2, device)
    l2, _ = m2.loss(d); l2.backward()
    if not torch.isfinite(l2): raise AssertionError("method2")
    if m2.diagnostics(d, d.test_users, d.test_items)["max_structural_bound_violation"] > 5e-5:
        raise AssertionError("method2 bound")

    c3 = Method3Config(latent_dim=4, max_epochs=2)
    m3 = ProposedMethod3(d, c3, device)
    with torch.no_grad():
        _, _, du0, di0, rw0, rh0, _, _ = m3.encoded(d)
        if du0.abs().max() > 1e-8 or di0.abs().max() > 1e-8:
            raise AssertionError("method3 zero-initialised residual head")
        if not torch.allclose(rw0, m3.rW0, atol=1e-7, rtol=1e-7):
            raise AssertionError("method3 user prior identity at init")
        if not torch.allclose(rh0, m3.rH0, atol=1e-7, rtol=1e-7):
            raise AssertionError("method3 item prior identity at init")
    l3, _ = m3.loss(d); l3.backward()
    if not torch.isfinite(l3): raise AssertionError("method3")
    if m3.diagnostics(d, d.test_users, d.test_items)["max_structural_bound_violation"] > 5e-5:
        raise AssertionError("method3 bound")

    heldout_label_invariance_test(device)
    print("[smoke-test] PASS: F/C/N + Method1 + Method2 + Method3 finite, leakage-safe, and structurally bounded")


# =============================================================================
# 14. Entry point
# =============================================================================


def main_v6_legacy() -> None:
    """Frozen v6-style entry point retained only for provenance; do not use for publication runs."""
    cfg = ExperimentConfigV6()

    # REQUIRED for real runs. Choose the one matching your dataset:
    # cfg.data.rating_min, cfg.data.rating_max = 0.5, 5.0   # MovieLens latest-small
    # cfg.data.rating_min, cfg.data.rating_max = 1.0, 5.0   # MovieLens 1M
    # cfg.data.data_dir = "/content/drive/MyDrive/Thesis/dataset"
    # cfg.fast_debug = True

    synthetic_smoke_test()

    # Recommended primary formulation for real experiments:
    apply_q1_safe_preset_v6_core(cfg)

    if cfg.data.rating_min is None or cfg.data.rating_max is None:
        print("[main] synthetic tests passed. Set cfg.data.rating_min/rating_max before real experiments.")
        return
    run_all_experiments_v6_legacy(cfg)



# =============================================================================
# V7 EXTENSIONS -- Methods 4-7, strict locked-test protocol, common ranking,
# uncertainty diagnostics, ablation orchestration, and publication plots.
# =============================================================================

import os
import shutil
from dataclasses import asdict as _asdict
from itertools import combinations
from typing import Any


# Preserve the v6-safe builders/preset before overriding their public names.
_build_matrix_data_v6 = build_matrix_data_v6_core
_apply_q1_safe_preset_v6 = apply_q1_safe_preset_v6_core


@dataclass
class MFConfig:
    """Standard biased MF baseline kept outside the seven proposed methods."""
    latent_dim: int = 20
    lr: float = 2e-3
    weight_decay: float = 1e-5
    max_epochs: int = 1000
    eval_every: int = 5
    val_patience: int = 20
    grad_clip: float = 5.0


@dataclass
class Method4Config:
    """Bounded prior-residual MBMF (new in v7).

    A train-only historical radius prior is projected into the feasible open interval
    (eps, sqrt(half_range)-eps).  The neural residual acts additively in logit-radius
    space, so every learned radius remains strictly below sqrt(half_range):

        q0 = r0_safe / c, c=sqrt(h)
        q  = sigmoid(logit(q0) + delta), |delta|<=B
        r  = c*q

    Thus r_u r_i < h and the centered prediction is analytically inside the declared
    rating interval without output clipping.
    """
    latent_dim: int = 20
    input_mode: InputMode = "values_plus_mask"
    model_space: ModelSpace = "centered_raw"
    angle_init: InitMode = "legacy_uniform_2pi"
    prior_mode: PriorMode = "eq42_shrinkage"
    rho: float = 0.05
    residual_transform: ResidualTransform = "smooth_tanh"
    max_abs_logit_shift: float = 2.0
    prior_floor_fraction: float = 1e-4
    delta_l2: float = 0.0
    ae_loss_mode: AELossMode = "separate"
    loss_form: LossForm = "mse"
    alpha_ae: float = 0.50
    beta_mbmf: float = 0.50
    lr: float = 5e-4
    weight_decay: float = 0.0
    max_epochs: int = 2000
    eval_every: int = 15
    val_patience: int = 20
    grad_clip: float = 1.0


@dataclass
class Method5Config:
    """Decoder-coupled dual-view MBMF with raw encoder/decoder norms.

    This imports the central idea of ``02_dv_mbmf_faithful`` but fixes the ambiguous
    zero-fill encoding: by default a missingness mask is concatenated to the values.
    The user-view decoder column and item-view decoder column are not decorative; their
    norms directly parameterize the opposite-side MBMF magnitudes.
    """
    latent_dim: int = 20
    hidden_dim: int = 128
    input_mode: InputMode = "values_plus_mask"
    angle_init: InitMode = "legacy_uniform_2pi"
    radius_mode: str = "raw_norm"
    radius_temperature: float = 0.0  # <=0 => sqrt(K), relevant only for range_safe
    radius_eps: float = 1e-6
    branch_mode: str = "both"  # both | user | item
    fusion_weight_user_view: float = 0.50
    prediction_weight: float = 1.0
    user_reconstruction_weight: float = 1.0
    item_reconstruction_weight: float = 1.0
    branch_consistency_weight: float = 0.0
    l2_code_weight: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 2000
    eval_every: int = 15
    val_patience: int = 20
    grad_clip: float = 1.0


@dataclass
class Method6Config(Method5Config):
    """Range-safe decoder-coupled dual-view MBMF.

    The radius map is c*tanh(||z||/tau), c=sqrt(half_range).  V7 defaults tau to
    sqrt(K) (when radius_temperature<=0), avoiding the near-immediate saturation caused
    by a fixed tau=1 as K grows.  The default epsilon is zero to preserve the source
    range-safe formula exactly; a positive floor remains available as an explicit setting.
    """
    radius_mode: str = "range_safe"
    radius_eps: float = 0.0


@dataclass
class Method7Config:
    """Uncertainty-aware inequality-bounded MF imported/revised from UC-IBMF.

    We intentionally avoid claiming calibration by name in v7.  Calibration is measured
    (Gaussian NLL, PICP, MPIW) rather than assumed.  The point predictor is range-safe by
    construction because it is a convex mixture of two experts in [-1,1].
    """
    latent_dim: int = 64
    hidden_dim: int = 64
    histogram_bins: int = 5
    feature_dropout: float = 0.15
    min_scale: float = 0.05
    tau_init: float = 20.0
    context_strength: float = 0.25
    nll_weight: float = 0.5
    huber_weight: float = 0.5
    consistency_weight: float = 0.05
    min_variance: float = 1e-4
    huber_beta_normalized: float = 0.25
    radius_mode: str = "adaptive"  # adaptive | fixed
    use_context: bool = True
    use_shrinkage: bool = True
    use_bias_expert: bool = True
    heteroscedastic: bool = True
    loss_mode: str = "robust"  # robust | mse
    lr: float = 1e-3
    weight_decay: float = 1e-6
    max_epochs: int = 1000
    batch_size: int = 4096
    eval_every: int = 5
    val_patience: int = 20
    grad_clip: float = 5.0


@dataclass
class EvaluationConfig:
    ranking_ks: Tuple[int, ...] = (10,)
    ranking_item_chunk_size: int = 2048
    ranking_max_users: Optional[int] = None
    bootstrap_reference_model: str = "MBMF-F"
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 2026
    generate_plots: bool = True
    keep_candidate_checkpoints: bool = False
    save_per_user_metrics: bool = True


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    paper: PaperBaselineConfig = field(default_factory=PaperBaselineConfig)
    mf: MFConfig = field(default_factory=MFConfig)
    method1: Method1Config = field(default_factory=Method1Config)
    method2: Method2Config = field(default_factory=Method2Config)
    method3: Method3Config = field(default_factory=Method3Config)
    method4: Method4Config = field(default_factory=Method4Config)
    method5: Method5Config = field(default_factory=Method5Config)
    method6: Method6Config = field(default_factory=Method6Config)
    method7: Method7Config = field(default_factory=Method7Config)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    split_seeds: Tuple[int, ...] = (2026,)
    model_seeds: Tuple[int, ...] = (11, 22, 33, 44, 55)
    latent_dims: Tuple[int, ...] = (20,)
    enable_mf_baseline: bool = True
    enabled_baselines: Tuple[BaselineVariant, ...] = ("MBMF-F", "MBMF-C", "MBMF-N")
    enabled_methods: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    strict_locked_test: bool = True
    fast_debug: bool = False
    run_tag: str = "primary"
    output_dir: str = "./mbmf_q1_results_v7"


def build_matrix_data(split: pd.DataFrame, cfg: DataConfig, device: torch.device) -> MatrixData:
    """V7 wrapper: retain the v6 MatrixData contract and attach train-only timestamps.

    Dynamic attributes are used to avoid breaking the serialized v6 dataclass layout.
    Methods that do not consume time remain completely unaffected.
    """
    data = _build_matrix_data_v6(split, cfg, device)
    work = split.copy()
    work["u"] = work.userId.map(data.user_to_index).astype(int)
    work["i"] = work.movieId.map(data.item_to_index).astype(int)
    tr = work[work.split == "train"]
    if "timestamp" in tr and tr.timestamp.notna().any():
        ts = torch.as_tensor(tr.timestamp.fillna(0).to_numpy(), dtype=torch.float32, device=device)
    else:
        ts = torch.zeros(len(tr), dtype=torch.float32, device=device)
    data.train_timestamps = ts
    return data


def apply_q1_safe_preset(cfg: ExperimentConfig) -> ExperimentConfig:
    """V7 Q1 preset: corrected primary settings for all seven proposed methods."""
    _apply_q1_safe_preset_v6(cfg)  # methods 1-3

    # Method 2: for squared loss this coefficient has an exact decomposition:
    # gamma L_U + (1-gamma)L_I = L_fused + gamma(1-gamma)L_consistency.
    g = float(cfg.method2.fusion_weight_user_view)
    cfg.method2.consistency_weight = g * (1.0 - g)

    cfg.method4.input_mode = "values_plus_mask"
    cfg.method4.model_space = "centered_raw"
    cfg.method4.prior_mode = "eq42_shrinkage"
    cfg.method4.rho = cfg.paper.rho
    cfg.method4.residual_transform = "smooth_tanh"

    # Fix the old dual-view zero/missing collision by making missingness explicit.
    cfg.method5.input_mode = "values_plus_mask"
    cfg.method5.radius_mode = "raw_norm"
    cfg.method6.input_mode = "values_plus_mask"
    cfg.method6.radius_mode = "range_safe"
    cfg.method6.radius_temperature = 0.0  # dimension-aware sqrt(K)

    # UC-IBMF is uncertainty-aware; calibration is evaluated rather than assumed.
    cfg.method7.radius_mode = "adaptive"
    cfg.method7.use_context = True
    cfg.method7.use_shrinkage = True
    cfg.method7.use_bias_expert = True
    cfg.method7.heteroscedastic = True
    cfg.method7.loss_mode = "robust"
    return cfg


# -----------------------------------------------------------------------------
# Auxiliary baseline: standard biased matrix factorization
# -----------------------------------------------------------------------------


class BiasedMatrixFactorization(nn.Module):
    """Strong conventional MF baseline; not counted among Methods 1-7."""
    def __init__(self, data: MatrixData, cfg: MFConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        k = cfg.latent_dim
        self.user_factors = nn.Embedding(data.num_users, k)
        self.item_factors = nn.Embedding(data.num_items, k)
        self.user_bias = nn.Embedding(data.num_users, 1)
        self.item_bias = nn.Embedding(data.num_items, 1)
        self.global_mean = nn.Parameter(data.train_ratings_raw.mean().detach().clone())
        nn.init.normal_(self.user_factors.weight, std=0.05)
        nn.init.normal_(self.item_factors.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        self.to(device)

    def predict_raw_pairs(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (
            self.global_mean
            + self.user_bias(users).squeeze(-1)
            + self.item_bias(items).squeeze(-1)
            + torch.sum(self.user_factors(users) * self.item_factors(items), dim=1)
        )

    def loss(self, data: MatrixData) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred = self.predict_raw_pairs(data, data.train_users, data.train_items)
        mse = torch.mean((pred - data.train_ratings_raw).square())
        return mse, {"L_MSE": float(mse.detach())}

    @torch.no_grad()
    def diagnostics(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> Dict[str, float]:
        pred = self.predict_raw_pairs(data, users, items)
        violation = torch.maximum(torch.relu(data.rating_min - pred), torch.relu(pred - data.rating_max))
        return {
            "max_rating_bound_violation": float(violation.max().item()) if pred.numel() else 0.0,
            "mean_user_factor_norm": float(self.user_factors.weight.norm(dim=1).mean().item()),
            "mean_item_factor_norm": float(self.item_factors.weight.norm(dim=1).mean().item()),
        }


# -----------------------------------------------------------------------------
# Method 4: bounded prior-residual MBMF
# -----------------------------------------------------------------------------


def _bounded_shift(raw: torch.Tensor, bound: float, transform: ResidualTransform) -> torch.Tensor:
    b = float(bound)
    if b < 0:
        raise ValueError("residual bound must be non-negative")
    if b == 0:
        return torch.zeros_like(raw)
    if transform == "hard_clamp":
        return torch.clamp(raw, -b, b)
    if transform == "smooth_tanh":
        return b * torch.tanh(raw / b)
    raise ValueError(transform)


class ProposedMethod4(nn.Module):
    """Bounded prior-residual magnitude AE-MBMF."""
    def __init__(self, data: MatrixData, cfg: Method4Config, device: torch.device):
        super().__init__()
        if cfg.model_space != "centered_raw":
            raise ValueError("Method 4 requires centered_raw for its rating-range theorem")
        if not 0.0 < cfg.prior_floor_fraction < 0.5:
            raise ValueError("prior_floor_fraction must be in (0, 0.5)")
        self.cfg = cfg
        uin, iin = input_dims(data, cfg.input_mode)
        self.user_branch = ResidualRadiusAEBranch(uin, data.num_items, cfg.latent_dim)
        self.item_branch = ResidualRadiusAEBranch(iin, data.num_users, cfg.latent_dim)
        rW_raw, rH_raw, pinfo = method3_initial_radii(data, Method3Config(
            latent_dim=cfg.latent_dim, input_mode=cfg.input_mode, model_space=cfg.model_space,
            angle_init=cfg.angle_init, prior_mode=cfg.prior_mode, rho=cfg.rho,
        ))
        cap = math.sqrt(data.half_range)
        eps = cap * cfg.prior_floor_fraction
        rW0 = rW_raw.clamp(eps, cap - eps)
        rH0 = rH_raw.clamp(eps, cap - eps)
        qW0 = (rW0 / cap).clamp(cfg.prior_floor_fraction, 1.0 - cfg.prior_floor_fraction)
        qH0 = (rH0 / cap).clamp(cfg.prior_floor_fraction, 1.0 - cfg.prior_floor_fraction)
        logitW0 = torch.log(qW0) - torch.log1p(-qW0)
        logitH0 = torch.log(qH0) - torch.log1p(-qH0)
        self.register_buffer("rW0_raw", rW_raw.to(device))
        self.register_buffer("rH0_raw", rH_raw.to(device))
        self.register_buffer("rW0", rW0.to(device))
        self.register_buffer("rH0", rH0.to(device))
        self.register_buffer("logitW0", logitW0.to(device))
        self.register_buffer("logitH0", logitH0.to(device))
        self.radius_cap = float(cap)
        self.prior_info = {
            **pinfo,
            "prior_projection_rate_user": float((rW_raw != rW0).float().mean().item()),
            "prior_projection_rate_item": float((rH_raw != rH0).float().mean().item()),
            "radius_cap": float(cap),
        }
        self.Phi = nn.Parameter(initialise_angles(data.num_users, cfg.latent_dim, cfg.angle_init, device))
        self.Theta = nn.Parameter(initialise_angles(data.num_items, cfg.latent_dim, cfg.angle_init, device))
        self.to(device)

    def encoded(self, data: MatrixData):
        xu, xi = build_encoder_inputs(data, self.cfg.input_mode, self.cfg.model_space)
        zu = self.user_branch.encode(xu)
        zi = self.item_branch.encode(xi)
        du = _bounded_shift(
            self.user_branch.raw_log_correction(zu), self.cfg.max_abs_logit_shift,
            self.cfg.residual_transform,
        )
        di = _bounded_shift(
            self.item_branch.raw_log_correction(zi), self.cfg.max_abs_logit_shift,
            self.cfg.residual_transform,
        )
        rW = self.radius_cap * torch.sigmoid(self.logitW0 + du)
        rH = self.radius_cap * torch.sigmoid(self.logitH0 + di)
        W = spherical_to_cartesian(self.Phi, rW)
        H = spherical_to_cartesian(self.Theta, rH)
        return zu, zi, du, di, rW, rH, W, H

    def predict_model_pairs(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        *_, W, H = self.encoded(data)
        return torch.sum(W[users] * H[items], dim=1)

    def predict_raw_pairs(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return self.predict_model_pairs(data, users, items) + data.midpoint

    def loss(self, data: MatrixData) -> Tuple[torch.Tensor, Dict[str, float]]:
        zu, zi, du, di, _, _, W, H = self.encoded(data)
        u, i = data.train_users, data.train_items
        target = data.train_ratings_raw - data.midpoint
        rec_u = self.user_branch.selected_reconstruction(zu, u, i)
        rec_i = self.item_branch.selected_reconstruction(zi, i, u)
        l_user = _observed_loss(rec_u, target, self.cfg.loss_form)
        l_item = _observed_loss(rec_i, target, self.cfg.loss_form)
        if self.cfg.ae_loss_mode == "separate":
            l_ae = l_user + l_item
        elif self.cfg.ae_loss_mode == "averaged":
            l_ae = _observed_loss(0.5 * (rec_u + rec_i), target, self.cfg.loss_form)
        else:
            raise ValueError(self.cfg.ae_loss_mode)
        pred = torch.sum(W[u] * H[i], dim=1)
        l_mbmf = _observed_loss(pred, target, self.cfg.loss_form)
        l_delta = torch.mean(du.square()) + torch.mean(di.square())
        total = self.cfg.alpha_ae * l_ae + self.cfg.beta_mbmf * l_mbmf + self.cfg.delta_l2 * l_delta
        return total, {
            "L_user_AE": float(l_user.detach()), "L_item_AE": float(l_item.detach()),
            "L_AE": float(l_ae.detach()), "L_MBM": float(l_mbmf.detach()),
            "L_logit_shift": float(l_delta.detach()),
        }

    @torch.no_grad()
    def diagnostics(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> Dict[str, float]:
        _, _, du, di, rW, rH, W, H = self.encoded(data)
        p = torch.sum(W[users] * H[items], dim=1)
        pair_bound = rW[users, 0] * rH[items, 0]
        raw = p + data.midpoint
        return {
            "max_structural_bound_violation": float(torch.relu(torch.abs(p) - pair_bound).max().item()) if p.numel() else 0.0,
            "max_rating_bound_violation": float(torch.maximum(
                torch.relu(torch.tensor(data.rating_min, device=raw.device) - raw),
                torch.relu(raw - data.rating_max),
            ).max().item()) if raw.numel() else 0.0,
            "max_pair_bound_eval": float(pair_bound.max().item()) if pair_bound.numel() else float("nan"),
            "mean_rW": float(rW.mean().item()), "mean_rH": float(rH.mean().item()),
            "min_logit_shift_user": float(du.min().item()), "max_logit_shift_user": float(du.max().item()),
            "min_logit_shift_item": float(di.min().item()), "max_logit_shift_item": float(di.max().item()),
            **self.prior_info,
        }


# -----------------------------------------------------------------------------
# Methods 5/6: decoder-coupled dual-view MBMF (raw / range-safe)
# -----------------------------------------------------------------------------


class SparseHistoryEncoderV7(nn.Module):
    """Sparse first layer exactly equivalent to Linear([values, mask]) -> ReLU -> Linear.

    The old DV package was sparse but encoded missing entries as numeric zero; that collides
    with a minimum rating after [0,1] normalization.  V7 keeps sparse computation while, in
    ``values_plus_mask`` mode, giving observedness its own weight matrix.  For a history row x
    with mask m the preactivation is x @ W_value + m @ W_mask + b, exactly the same value as
    a dense Linear on concat([x,m]) with the corresponding block weights.
    """
    def __init__(
        self,
        row_index: torch.Tensor,
        col_index: torch.Tensor,
        values: torch.Tensor,
        rows: int,
        cols: int,
        hidden_dim: int,
        latent_dim: int,
        input_mode: InputMode,
    ) -> None:
        super().__init__()
        if input_mode not in {"values_only", "values_plus_mask"}:
            raise ValueError(input_mode)
        indices = torch.stack([row_index.long(), col_index.long()], dim=0)
        value_sparse = torch.sparse_coo_tensor(
            indices, values.float(), size=(rows, cols), device=values.device
        ).coalesce()
        self.register_buffer("value_history", value_sparse)
        if input_mode == "values_plus_mask":
            mask_sparse = torch.sparse_coo_tensor(
                indices, torch.ones_like(values, dtype=torch.float32), size=(rows, cols), device=values.device
            ).coalesce()
        else:
            mask_sparse = None
        self.register_buffer("mask_history", mask_sparse)
        self.value_weight = nn.Parameter(torch.empty(cols, hidden_dim, device=values.device))
        self.mask_weight = (
            nn.Parameter(torch.empty(cols, hidden_dim, device=values.device))
            if input_mode == "values_plus_mask" else None
        )
        self.input_bias = nn.Parameter(torch.zeros(hidden_dim, device=values.device))
        self.output = nn.Linear(hidden_dim, latent_dim).to(values.device)
        nn.init.xavier_uniform_(self.value_weight)
        if self.mask_weight is not None:
            nn.init.xavier_uniform_(self.mask_weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward_all(self) -> torch.Tensor:
        hidden = torch.sparse.mm(self.value_history, self.value_weight)
        if self.mask_weight is not None:
            hidden = hidden + torch.sparse.mm(self.mask_history, self.mask_weight)
        hidden = torch.relu(hidden + self.input_bias)
        return self.output(hidden)


class DecoderCoupledDualViewMBMF(nn.Module):
    def __init__(self, data: MatrixData, cfg: Method5Config, device: torch.device):
        super().__init__()
        if cfg.radius_mode not in {"raw_norm", "range_safe", "fixed"}:
            raise ValueError("dual-view radius_mode must be raw_norm, range_safe, or fixed")
        if cfg.branch_mode not in {"both", "user", "item"}:
            raise ValueError("branch_mode must be both, user, or item")
        if not 0.0 <= cfg.fusion_weight_user_view <= 1.0:
            raise ValueError("fusion_weight_user_view must be in [0,1]")
        self.cfg = cfg
        k = cfg.latent_dim
        width = max(data.rating_max - data.rating_min, 1e-12)
        unit_values = ((data.train_ratings_raw - data.rating_min) / width).clamp(0.0, 1.0)
        self.user_encoder = SparseHistoryEncoderV7(
            data.train_users, data.train_items, unit_values,
            data.num_users, data.num_items, cfg.hidden_dim, k, cfg.input_mode,
        )
        self.item_encoder = SparseHistoryEncoderV7(
            data.train_items, data.train_users, unit_values,
            data.num_items, data.num_users, cfg.hidden_dim, k, cfg.input_mode,
        )
        # Linear decoder columns: reconstruction parameters AND counterpart magnitude sources.
        self.user_view_item_decoder = nn.Embedding(data.num_items, k)
        self.item_view_user_decoder = nn.Embedding(data.num_users, k)
        self.user_view_item_bias = nn.Embedding(data.num_items, 1)
        self.item_view_user_bias = nn.Embedding(data.num_users, 1)
        nn.init.xavier_uniform_(self.user_view_item_decoder.weight)
        nn.init.xavier_uniform_(self.item_view_user_decoder.weight)
        nn.init.zeros_(self.user_view_item_bias.weight)
        nn.init.zeros_(self.item_view_user_bias.weight)
        self.uv_user_angles = nn.Parameter(initialise_angles(data.num_users, k, cfg.angle_init, device))
        self.uv_item_angles = nn.Parameter(initialise_angles(data.num_items, k, cfg.angle_init, device))
        self.iv_user_angles = nn.Parameter(initialise_angles(data.num_users, k, cfg.angle_init, device))
        self.iv_item_angles = nn.Parameter(initialise_angles(data.num_items, k, cfg.angle_init, device))
        self.to(device)

    def _radius(self, vectors: torch.Tensor, data: MatrixData) -> torch.Tensor:
        norm = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        if self.cfg.radius_mode == "raw_norm":
            return norm
        cap = math.sqrt(data.half_range)
        if self.cfg.radius_mode == "fixed":
            return torch.full_like(norm, cap)
        eps = float(self.cfg.radius_eps)
        if not 0 <= eps < cap:
            raise ValueError("radius_eps must be in [0,sqrt(half_range))")
        temp = float(self.cfg.radius_temperature)
        if temp <= 0:
            temp = math.sqrt(self.cfg.latent_dim)
        return eps + (cap - eps) * torch.tanh(norm / max(temp, 1e-8))

    @staticmethod
    def _unit(angles: torch.Tensor) -> torch.Tensor:
        r = torch.ones((*angles.shape[:-1], 1), dtype=angles.dtype, device=angles.device)
        return spherical_to_cartesian(angles, r)

    def codes(self, data: MatrixData):
        # Sparse computation is algebraically equivalent to the dense [values,mask]
        # first layer but never materializes the full concatenated history matrices.
        zu = self.user_encoder.forward_all()
        zi = self.item_encoder.forward_all()
        du = self.user_view_item_decoder.weight
        di = self.item_view_user_decoder.weight
        return zu, zi, du, di

    def _all_state(self, data: MatrixData):
        zu, zi, du, di = self.codes(data)
        ru_u = self._radius(zu, data)
        ru_i = self._radius(du, data)
        ri_i = self._radius(zi, data)
        ri_u = self._radius(di, data)
        qu = self._unit(self.uv_user_angles)
        pu = self._unit(self.uv_item_angles)
        qi = self._unit(self.iv_user_angles)
        pi = self._unit(self.iv_item_angles)
        return zu, zi, du, di, ru_u, ru_i, ri_i, ri_u, qu, pu, qi, pi

    def predict_views(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor):
        state = self._all_state(data)
        zu, zi, du, di, ru_u, ru_i, ri_i, ri_u, qu, pu, qi, pi = state
        uv_cos = torch.sum(qu[users] * pu[items], dim=1)
        iv_cos = torch.sum(qi[users] * pi[items], dim=1)
        uv = data.midpoint + ru_u[users, 0] * ru_i[items, 0] * uv_cos
        iv = data.midpoint + ri_u[users, 0] * ri_i[items, 0] * iv_cos
        if self.cfg.branch_mode == "user":
            pred = uv
        elif self.cfg.branch_mode == "item":
            pred = iv
        else:
            g = float(self.cfg.fusion_weight_user_view)
            pred = g * uv + (1.0 - g) * iv
        return uv, iv, pred, state

    def predict_raw_pairs(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return self.predict_views(data, users, items)[2]

    def loss(self, data: MatrixData) -> Tuple[torch.Tensor, Dict[str, float]]:
        u, i, y = data.train_users, data.train_items, data.train_ratings_raw
        uv, iv, pred, state = self.predict_views(data, u, i)
        zu, zi, du, di = state[:4]
        width = data.rating_max - data.rating_min
        target_unit = (y - data.rating_min) / width
        pred_unit = (pred - data.rating_min) / width
        rec_u = torch.sum(zu[u] * du[i], dim=1) + self.user_view_item_bias(i).squeeze(-1)
        rec_i = torch.sum(zi[i] * di[u], dim=1) + self.item_view_user_bias(u).squeeze(-1)
        l_pred = torch.mean((pred_unit - target_unit).square())
        l_ru = torch.mean((rec_u - target_unit).square())
        l_ri = torch.mean((rec_i - target_unit).square())
        l_cons = torch.mean(((uv - iv) / width).square())
        l2 = 0.25 * (zu.square().mean() + zi.square().mean() + du.square().mean() + di.square().mean())
        total = (
            self.cfg.prediction_weight * l_pred
            + self.cfg.user_reconstruction_weight * l_ru
            + self.cfg.item_reconstruction_weight * l_ri
            + self.cfg.branch_consistency_weight * l_cons
            + self.cfg.l2_code_weight * l2
        )
        return total, {
            "L_pred": float(l_pred.detach()), "L_user_reconstruction": float(l_ru.detach()),
            "L_item_reconstruction": float(l_ri.detach()), "L_branch_consistency": float(l_cons.detach()),
            "L_code": float(l2.detach()),
        }

    @torch.no_grad()
    def diagnostics(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> Dict[str, float]:
        uv, iv, pred, state = self.predict_views(data, users, items)
        _, _, _, _, ru_u, ru_i, ri_i, ri_u, *_ = state
        bu = ru_u[users, 0] * ru_i[items, 0]
        bi = ri_u[users, 0] * ri_i[items, 0]
        g = float(self.cfg.fusion_weight_user_view)
        if self.cfg.branch_mode == "user":
            bf = bu
        elif self.cfg.branch_mode == "item":
            bf = bi
        else:
            bf = g * bu + (1.0 - g) * bi
        centered = pred - data.midpoint
        # Raw-mode shared envelopes also upper-bound the bias-free linear decoder terms.
        return {
            "max_user_view_bound_violation": float(torch.relu(torch.abs(uv - data.midpoint) - bu).max().item()) if bu.numel() else 0.0,
            "max_item_view_bound_violation": float(torch.relu(torch.abs(iv - data.midpoint) - bi).max().item()) if bi.numel() else 0.0,
            "max_structural_bound_violation": float(torch.relu(torch.abs(centered) - bf).max().item()) if bf.numel() else 0.0,
            "max_pair_bound_eval": float(bf.max().item()) if bf.numel() else float("nan"),
            "mean_user_view_user_radius": float(ru_u.mean().item()),
            "mean_user_view_item_radius": float(ru_i.mean().item()),
            "mean_item_view_user_radius": float(ri_u.mean().item()),
            "mean_item_view_item_radius": float(ri_i.mean().item()),
        }


# -----------------------------------------------------------------------------
# Method 7: uncertainty-aware inequality-bounded MF
# -----------------------------------------------------------------------------


def _history_axis_features(
    indices: torch.Tensor,
    ratings_raw: torch.Tensor,
    size: int,
    rating_min: float,
    rating_max: float,
    histogram_bins: int,
    timestamps: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if histogram_bins < 2:
        raise ValueError("histogram_bins must be >=2")
    idx = indices.detach().cpu().numpy().astype(np.int64)
    y = ratings_raw.detach().cpu().numpy().astype(np.float64)
    span = max(float(rating_max - rating_min), 1e-12)
    yn = np.clip((y - rating_min) / span, 0.0, 1.0)
    counts = np.bincount(idx, minlength=size).astype(np.float64)
    sums = np.bincount(idx, weights=yn, minlength=size)
    sums2 = np.bincount(idx, weights=yn * yn, minlength=size)
    den = np.maximum(counts, 1.0)
    mean = sums / den
    var = np.maximum(sums2 / den - mean * mean, 0.0)
    std = np.sqrt(var)
    mins = np.full(size, np.inf, dtype=np.float64)
    maxs = np.full(size, -np.inf, dtype=np.float64)
    np.minimum.at(mins, idx, yn)
    np.maximum.at(maxs, idx, yn)
    mins[~np.isfinite(mins)] = 0.0
    maxs[~np.isfinite(maxs)] = 0.0
    log_count = np.log1p(counts)
    log_count /= max(float(log_count.max()), 1.0)
    hist = np.zeros((size, histogram_bins), dtype=np.float64)
    bins = np.minimum((yn * histogram_bins).astype(np.int64), histogram_bins - 1)
    np.add.at(hist, (idx, bins), 1.0)
    hist /= den[:, None]
    recency = np.zeros(size, dtype=np.float64)
    history_span = np.zeros(size, dtype=np.float64)
    if timestamps is not None and timestamps.numel() == len(idx):
        ts = timestamps.detach().cpu().numpy().astype(np.float64)
        if np.isfinite(ts).any() and float(np.nanmax(ts) - np.nanmin(ts)) > 0:
            t0, t1 = float(np.nanmin(ts)), float(np.nanmax(ts))
            tn = (np.nan_to_num(ts, nan=t0) - t0) / max(t1 - t0, 1.0)
            tmin = np.full(size, np.inf, dtype=np.float64)
            tmax = np.full(size, -np.inf, dtype=np.float64)
            np.minimum.at(tmin, idx, tn)
            np.maximum.at(tmax, idx, tn)
            valid = np.isfinite(tmin) & np.isfinite(tmax)
            recency[valid] = tmax[valid]
            history_span[valid] = tmax[valid] - tmin[valid]
    base = np.column_stack([log_count, mean, std, mins, maxs, recency, history_span])
    feats = np.concatenate([base, hist], axis=1).astype(np.float32)
    device = ratings_raw.device
    return torch.from_numpy(feats).to(device), torch.from_numpy(counts.astype(np.float32)).to(device)


def build_method7_history_features(data: MatrixData, bins: int):
    ts = getattr(data, "train_timestamps", None)
    uf, uc = _history_axis_features(
        data.train_users, data.train_ratings_raw, data.num_users,
        data.rating_min, data.rating_max, bins, ts,
    )
    it, ic = _history_axis_features(
        data.train_items, data.train_ratings_raw, data.num_items,
        data.rating_min, data.rating_max, bins, ts,
    )
    return uf, it, uc, ic


def _inverse_softplus_scalar(x: float) -> float:
    if x <= 0:
        raise ValueError("inverse-softplus target must be positive")
    return x if x > 20 else math.log(math.expm1(x))


class ProposedMethod7(nn.Module):
    """Uncertainty-aware inequality-bounded factor model."""
    def __init__(self, data: MatrixData, cfg: Method7Config, device: torch.device):
        super().__init__()
        if cfg.radius_mode not in {"adaptive", "fixed"}:
            raise ValueError("radius_mode must be adaptive or fixed")
        if cfg.loss_mode not in {"robust", "mse"}:
            raise ValueError("loss_mode must be robust or mse")
        if not 0 <= cfg.min_scale < 1:
            raise ValueError("min_scale must be in [0,1)")
        if cfg.min_variance <= 0:
            raise ValueError("min_variance must be positive")
        if cfg.nll_weight < 0 or cfg.huber_weight < 0 or cfg.consistency_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if cfg.loss_mode == "robust" and cfg.nll_weight + cfg.huber_weight <= 0:
            raise ValueError("robust loss requires a positive NLL or Huber weight")
        self.cfg = cfg
        self.rating_min, self.rating_max = data.rating_min, data.rating_max
        self.midpoint, self.half_range = data.midpoint, data.half_range
        uf, itf, uc, ic = build_method7_history_features(data, cfg.histogram_bins)
        self.register_buffer("user_features", uf)
        self.register_buffer("item_features", itf)
        self.register_buffer("user_counts", uc)
        self.register_buffer("item_counts", ic)
        fd = uf.shape[1]
        d, h = cfg.latent_dim, cfg.hidden_dim
        self.user_direction = nn.Embedding(data.num_users, d)
        self.item_direction = nn.Embedding(data.num_items, d)
        self.user_bias = nn.Embedding(data.num_users, 1)
        self.item_bias = nn.Embedding(data.num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(()))
        self.user_encoder = self._encoder(fd, h, d)
        self.item_encoder = self._encoder(fd, h, d)
        self.user_radius = self._head(fd, h)
        self.item_radius = self._head(fd, h)
        self.user_logvariance = self._head(fd, h)
        self.item_logvariance = self._head(fd, h)
        self.user_radius_prior = nn.Parameter(torch.zeros(()))
        self.item_radius_prior = nn.Parameter(torch.zeros(()))
        invtau = _inverse_softplus_scalar(float(cfg.tau_init))
        self.log_tau_user = nn.Parameter(torch.tensor(invtau))
        self.log_tau_item = nn.Parameter(torch.tensor(invtau))
        target_var = max(0.15 ** 2, cfg.min_variance)
        self.log_variance_prior = nn.Parameter(torch.tensor(_inverse_softplus_scalar(target_var)))
        nn.init.normal_(self.user_direction.weight, std=1.0 / math.sqrt(d))
        nn.init.normal_(self.item_direction.weight, std=1.0 / math.sqrt(d))
        nn.init.zeros_(self.user_bias.weight); nn.init.zeros_(self.item_bias.weight)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.to(device)

    @staticmethod
    def _encoder(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Linear(hidden, output_dim))

    @staticmethod
    def _head(input_dim: int, hidden: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def _entity_state(self, ids, features, counts, table, encoder, radius_head, prior_logit, log_tau):
        cfg = self.cfg
        x = features[ids]
        dropped = torch.nn.functional.dropout(x, p=cfg.feature_dropout, training=self.training)
        context = torch.tanh(encoder(dropped)) if cfg.use_context else torch.zeros((len(ids), cfg.latent_dim), device=x.device)
        direction = torch.nn.functional.normalize(table(ids) + cfg.context_strength * context, dim=-1, eps=1e-12)
        if cfg.radius_mode == "fixed":
            scale = torch.ones(len(ids), device=x.device, dtype=x.dtype)
            reliability = torch.ones_like(scale)
        else:
            local = torch.sigmoid(radius_head(dropped).squeeze(-1))
            prior = torch.sigmoid(prior_logit)
            if cfg.use_shrinkage:
                tau = torch.nn.functional.softplus(log_tau) + 1e-6
                reliability = counts[ids] / (counts[ids] + tau)
            else:
                reliability = torch.ones_like(local)
            mixed = reliability * local + (1.0 - reliability) * prior
            scale = cfg.min_scale + (1.0 - cfg.min_scale) * mixed
        return direction, scale, reliability

    def forward_pairs(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor, return_aux=False):
        ud, us, ur = self._entity_state(users, self.user_features, self.user_counts, self.user_direction,
                                        self.user_encoder, self.user_radius, self.user_radius_prior, self.log_tau_user)
        idr, ins, ir = self._entity_state(items, self.item_features, self.item_counts, self.item_direction,
                                          self.item_encoder, self.item_radius, self.item_radius_prior, self.log_tau_item)
        gate = us * ins
        cosine = torch.sum(ud * idr, dim=-1).clamp(-1.0, 1.0)
        if self.cfg.use_bias_expert:
            bias = torch.tanh(self.global_bias + self.user_bias(users).squeeze(-1) + self.item_bias(items).squeeze(-1))
        else:
            bias = torch.zeros_like(cosine)
        zn = (1.0 - gate) * bias + gate * cosine
        pred = self.midpoint + self.half_range * zn
        if not return_aux:
            return pred
        if self.cfg.heteroscedastic:
            ulv = self.user_logvariance(self.user_features[users]).squeeze(-1)
            ilv = self.item_logvariance(self.item_features[items]).squeeze(-1)
            variance_logit = self.log_variance_prior + ulv + ilv
        else:
            # Correct homoscedastic ablation: one trainable global variance, rather
            # than untrained user/item variance heads leaking into evaluation.
            variance_logit = self.log_variance_prior.expand_as(pred)
        vnorm = torch.nn.functional.softplus(variance_logit) + self.cfg.min_variance
        return pred, {
            "variance_normalized": vnorm, "variance_raw": vnorm * self.half_range ** 2,
            "gate": gate, "user_scale": us, "item_scale": ins,
            "user_reliability": ur, "item_reliability": ir, "cosine": cosine, "bias": bias,
        }

    def predict_raw_pairs(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return self.forward_pairs(data, users, items, return_aux=False)

    def predict_raw_with_uncertainty(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor):
        p, aux = self.forward_pairs(data, users, items, return_aux=True)
        return p, aux["variance_raw"]

    def loss_pairs(
        self, data: MatrixData, u: torch.Tensor, i: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Sparse edge-batch objective used by the Method-7 trainer."""
        pred, aux = self.forward_pairs(data, u, i, return_aux=True)
        err = pred - y
        norm_err = err / data.half_range
        norm_mse = norm_err.square().mean()
        if self.cfg.loss_mode == "mse":
            nll = torch.zeros_like(norm_mse); huber = torch.zeros_like(norm_mse); total = norm_mse
        else:
            # Heteroscedastic mode uses user/item variance heads; homoscedastic mode
            # uses the single trainable global variance prepared in forward_pairs.
            v = aux["variance_normalized"]
            nll = 0.5 * (norm_err.square() / v + torch.log(v)).mean()
            pnorm = (pred - data.midpoint) / data.half_range
            ynorm = (y - data.midpoint) / data.half_range
            huber = torch.nn.functional.smooth_l1_loss(pnorm, ynorm, beta=self.cfg.huber_beta_normalized)
            total = self.cfg.nll_weight * nll + self.cfg.huber_weight * huber
        consistency = torch.zeros_like(total)
        if self.training and self.cfg.consistency_weight > 0 and self.cfg.use_context:
            second = self.forward_pairs(data, u, i, return_aux=False)
            first_norm = (pred.detach() - data.midpoint) / data.half_range
            second_norm = (second - data.midpoint) / data.half_range
            consistency = torch.mean((second_norm - first_norm).square())
            total = total + self.cfg.consistency_weight * consistency
        return total, {
            "L_total": float(total.detach()), "L_normalized_mse": float(norm_mse.detach()),
            "L_nll": float(nll.detach()), "L_huber": float(huber.detach()),
            "L_consistency": float(consistency.detach()), "mean_gate": float(aux["gate"].mean().detach()),
        }

    def loss(self, data: MatrixData) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Full-edge wrapper kept for unit tests and diagnostics. Publication training uses
        # train_method7_minibatch so Method 7 retains the sparse edge-training semantics of
        # its source redesign rather than materialising all edge activations at once.
        return self.loss_pairs(data, data.train_users, data.train_items, data.train_ratings_raw)

    @torch.no_grad()
    def diagnostics(self, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> Dict[str, float]:
        pred, aux = self.forward_pairs(data, users, items, return_aux=True)
        violation = torch.maximum(torch.relu(data.rating_min - pred), torch.relu(pred - data.rating_max))
        return {
            "max_rating_bound_violation": float(violation.max().item()) if violation.numel() else 0.0,
            "mean_gate": float(aux["gate"].mean().item()) if users.numel() else float("nan"),
            "min_gate": float(aux["gate"].min().item()) if users.numel() else float("nan"),
            "max_gate": float(aux["gate"].max().item()) if users.numel() else float("nan"),
            "mean_user_reliability": float(aux["user_reliability"].mean().item()) if users.numel() else float("nan"),
            "mean_item_reliability": float(aux["item_reliability"].mean().item()) if users.numel() else float("nan"),
            "mean_variance_raw": float(aux["variance_raw"].mean().item()) if users.numel() else float("nan"),
        }


def train_method7_minibatch(model: ProposedMethod7, data: MatrixData, cfg: Method7Config) -> Dict[str, object]:
    """AdamW edge-mini-batch trainer for Method 7 with validation-only checkpointing."""
    if cfg.batch_size <= 0:
        raise ValueError("Method7 batch_size must be positive")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    wait = 0
    history: List[Dict[str, object]] = []
    n = int(data.train_users.numel())
    device = data.train_users.device
    start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        total_examples = 0
        component_sums: Dict[str, float] = {}
        for start_idx in range(0, n, cfg.batch_size):
            idx = perm[start_idx : start_idx + cfg.batch_size]
            u = data.train_users[idx]; i = data.train_items[idx]; y = data.train_ratings_raw[idx]
            opt.zero_grad(set_to_none=True)
            loss, parts = model.loss_pairs(data, u, i, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Method-7 loss at epoch {epoch}")
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            bs = int(idx.numel())
            total_loss += float(loss.detach().item()) * bs
            total_examples += bs
            for key, value in parts.items():
                component_sums[key] = component_sums.get(key, 0.0) + float(value) * bs
        if epoch == 1 or epoch % cfg.eval_every == 0:
            val = evaluate_model(model, data, "val")
            if not math.isfinite(val["RMSE"]):
                raise FloatingPointError(f"non-finite Method-7 validation RMSE at epoch {epoch}")
            point: Dict[str, object] = {
                "epoch": epoch,
                "loss": total_loss / max(total_examples, 1),
                "val_RMSE": val["RMSE"],
            }
            point.update({k: v / max(total_examples, 1) for k, v in component_sums.items()})
            history.append(point)
            if val["RMSE"] < best - 1e-8:
                best = val["RMSE"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= cfg.val_patience:
                    break
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_val_RMSE": best,
        "elapsed_sec": time.perf_counter() - start,
        "history": history,
        "training_mode": "edge_minibatch_adamw",
        "batch_size": int(cfg.batch_size),
    }


# -----------------------------------------------------------------------------
# Common V7 evaluation, exact ranking, and statistical helpers
# -----------------------------------------------------------------------------


def _predict_raw_generic(model: nn.Module, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
    if isinstance(model, PaperMBMF):
        return model.predict_raw_pairs(users, items)
    return model.predict_raw_pairs(data, users, items)


@torch.no_grad()
def build_eval_prediction_cache(model: nn.Module, data: MatrixData) -> Dict[str, Any]:
    """Precompute deterministic entity state for repeated evaluation/ranking queries.

    This is mathematically identical to ``predict_raw_pairs`` in eval mode, but avoids
    recomputing every encoder and spherical conversion once per user/item chunk.
    """
    model.eval()
    if isinstance(model, PaperMBMF):
        W, H = model.factors()
        return {"kind": "paper", "model": model, "W": W, "H": H}
    if isinstance(model, BiasedMatrixFactorization):
        return {
            "kind": "mf", "model": model,
            "U": model.user_factors.weight, "I": model.item_factors.weight,
            "ub": model.user_bias.weight[:, 0], "ib": model.item_bias.weight[:, 0],
            "gm": model.global_mean.detach(),
        }
    if isinstance(model, ThesisMethod1):
        *_, W, H = model.encoded(data)
        return {"kind": "m1", "model": model, "W": W, "H": H}
    if isinstance(model, ThesisMethod2):
        _, _, _, _, WU, HU, WI, HI = model.factors(data)
        return {"kind": "m2", "model": model, "WU": WU, "HU": HU, "WI": WI, "HI": HI}
    if isinstance(model, ProposedMethod3):
        *_, W, H = model.encoded(data)
        return {"kind": "m3", "model": model, "W": W, "H": H}
    if isinstance(model, ProposedMethod4):
        *_, W, H = model.encoded(data)
        return {"kind": "m4", "model": model, "W": W, "H": H}
    if isinstance(model, DecoderCoupledDualViewMBMF):
        state = model._all_state(data)
        _, _, _, _, ru_u, ru_i, ri_i, ri_u, qu, pu, qi, pi = state
        return {
            "kind": "dv", "model": model, "ru_u": ru_u, "ru_i": ru_i,
            "ri_i": ri_i, "ri_u": ri_u, "qu": qu, "pu": pu, "qi": qi, "pi": pi,
        }
    if isinstance(model, ProposedMethod7):
        dev = data.train_users.device
        uids = torch.arange(data.num_users, dtype=torch.long, device=dev)
        iids = torch.arange(data.num_items, dtype=torch.long, device=dev)
        ud, us, _ = model._entity_state(
            uids, model.user_features, model.user_counts, model.user_direction,
            model.user_encoder, model.user_radius, model.user_radius_prior, model.log_tau_user,
        )
        idr, ins, _ = model._entity_state(
            iids, model.item_features, model.item_counts, model.item_direction,
            model.item_encoder, model.item_radius, model.item_radius_prior, model.log_tau_item,
        )
        ub = model.user_bias.weight[:, 0] if model.cfg.use_bias_expert else torch.zeros(data.num_users, device=dev)
        ib = model.item_bias.weight[:, 0] if model.cfg.use_bias_expert else torch.zeros(data.num_items, device=dev)
        return {
            "kind": "m7", "model": model, "ud": ud, "us": us, "idr": idr, "ins": ins,
            "ub": ub, "ib": ib, "gb": model.global_bias.detach(),
        }
    return {"kind": "generic", "model": model}


@torch.no_grad()
def _predict_raw_cached(cache: Dict[str, Any], data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
    kind = cache["kind"]
    model = cache["model"]
    if kind == "paper":
        native = torch.sum(cache["W"][users] * cache["H"][items], dim=1)
        rmin, midpoint, cmin = model.data_meta
        if model.variant == "MBMF-F":
            return native + midpoint
        if model.variant == "MBMF-C":
            return native + cmin + model.rW[users, 0] * model.rH[items, 0]
        return native
    if kind == "mf":
        return cache["gm"] + cache["ub"][users] + cache["ib"][items] + torch.sum(cache["U"][users] * cache["I"][items], dim=1)
    if kind in {"m1", "m3", "m4"}:
        native = torch.sum(cache["W"][users] * cache["H"][items], dim=1)
        space = "centered_raw" if kind == "m4" else model.cfg.model_space
        return model_space_to_raw(data, native, space)
    if kind == "m2":
        p_u = torch.sum(cache["WU"][users] * cache["HU"][items], dim=1)
        p_i = torch.sum(cache["WI"][items] * cache["HI"][users], dim=1)
        g = float(model.cfg.fusion_weight_user_view)
        return model_space_to_raw(data, g * p_u + (1.0 - g) * p_i, model.cfg.model_space)
    if kind == "dv":
        uv = data.midpoint + cache["ru_u"][users, 0] * cache["ru_i"][items, 0] * torch.sum(cache["qu"][users] * cache["pu"][items], dim=1)
        iv = data.midpoint + cache["ri_u"][users, 0] * cache["ri_i"][items, 0] * torch.sum(cache["qi"][users] * cache["pi"][items], dim=1)
        if model.cfg.branch_mode == "user":
            return uv
        if model.cfg.branch_mode == "item":
            return iv
        g = float(model.cfg.fusion_weight_user_view)
        return g * uv + (1.0 - g) * iv
    if kind == "m7":
        gate = cache["us"][users] * cache["ins"][items]
        cosine = torch.sum(cache["ud"][users] * cache["idr"][items], dim=1).clamp(-1.0, 1.0)
        if model.cfg.use_bias_expert:
            bias = torch.tanh(cache["gb"] + cache["ub"][users] + cache["ib"][items])
        else:
            bias = torch.zeros_like(cosine)
        zn = (1.0 - gate) * bias + gate * cosine
        return data.midpoint + data.half_range * zn
    return _predict_raw_generic(model, data, users, items)


def evaluate_model(model: nn.Module, data: MatrixData, split: Literal["train", "val", "test"]) -> Dict[str, float]:
    if split == "train":
        u, i, y = data.train_users, data.train_items, data.train_ratings_raw
    elif split == "val":
        u, i, y = data.val_users, data.val_items, data.val_ratings_raw
    elif split == "test":
        u, i, y = data.test_users, data.test_items, data.test_ratings_raw
    else:
        raise ValueError(split)
    model.eval()
    with torch.no_grad():
        p = _predict_raw_generic(model, data, u, i)
        out = regression_metrics_raw(p, y, data.rating_min, data.rating_max)
        out.update(train_threshold_f1(u, y, p, data.user_train_mean_raw))
        uncertainty_trained = hasattr(model, "predict_raw_with_uncertainty") and not (
            isinstance(model, ProposedMethod7) and model.cfg.loss_mode == "mse"
        )
        if uncertainty_trained:
            pu, var = model.predict_raw_with_uncertainty(data, u, i)
            err = pu - y
            v = var.clamp_min(1e-12)
            out["Gaussian_NLL"] = float((0.5 * (err.square() / v + torch.log(2.0 * math.pi * v))).mean().item())
            # Several central Gaussian intervals are reported so "calibration" is measured,
            # not inferred from a single 90% coverage number.  z-values are fixed constants to
            # avoid a SciPy dependency in the standalone research bundle.
            coverage_z = {
                50: 0.6744897501960817,
                80: 1.2815515655446004,
                90: 1.6448536269514722,
                95: 1.959963984540054,
            }
            gaps = []
            width = max(data.rating_max - data.rating_min, 1e-12)
            for level, zcrit in coverage_z.items():
                hw = zcrit * torch.sqrt(v)
                picp = float(((y >= pu - hw) & (y <= pu + hw)).float().mean().item())
                mpiw = float((2.0 * hw).mean().item())
                nominal = level / 100.0
                out[f"PICP{level}"] = picp
                out[f"PICP{level}_abs_gap"] = abs(picp - nominal)
                out[f"MPIW{level}"] = mpiw
                out[f"MPIW{level}_normalized"] = mpiw / width
                gaps.append(abs(picp - nominal))
            out["coverage_calibration_MAE"] = float(np.mean(gaps))
    return out


@torch.no_grad()
def exact_ranking_metrics_v7(
    model: nn.Module,
    data: MatrixData,
    k: int,
    item_chunk_size: int = 2048,
    max_users: Optional[int] = None,
    seed: int = 2026,
) -> Dict[str, float]:
    if k <= 0 or k > data.num_items:
        return {f"Precision@{k}": float("nan"), f"Recall@{k}": float("nan"), f"NDCG@{k}": float("nan"), "ranking_users": 0.0}
    # Relevant test items are defined with the same train-only user threshold used by F1.
    rel: Dict[int, set[int]] = {}
    for u in torch.unique(data.test_users).tolist():
        q = data.test_users == int(u)
        threshold = float(data.user_train_mean_raw[int(u)].item())
        items = data.test_items[q][data.test_ratings_raw[q] > threshold].detach().cpu().tolist()
        if items:
            rel[int(u)] = set(map(int, items))
    users = np.asarray(sorted(rel), dtype=np.int64)
    if max_users is not None and len(users) > max_users:
        users = np.sort(np.random.default_rng(seed).choice(users, size=max_users, replace=False))
    # At final test time both train and validation interactions are already known;
    # neither should be eligible recommendation candidates.
    seen: Dict[int, set[int]] = {}
    for us, it in ((data.train_users, data.train_items), (data.val_users, data.val_items)):
        for u in torch.unique(us).tolist():
            q = us == int(u)
            seen.setdefault(int(u), set()).update(map(int, it[q].detach().cpu().tolist()))
    pv, rv, nv = [], [], []
    device = data.train_users.device
    model.eval()
    prediction_cache = build_eval_prediction_cache(model, data)
    for u in users.tolist():
        scores = np.empty(data.num_items, dtype=np.float32)
        for start in range(0, data.num_items, item_chunk_size):
            end = min(start + item_chunk_size, data.num_items)
            it = torch.arange(start, end, dtype=torch.long, device=device)
            us = torch.full_like(it, int(u))
            scores[start:end] = _predict_raw_cached(prediction_cache, data, us, it).detach().cpu().numpy()
        if u in seen and seen[u]:
            scores[np.fromiter(seen[u], dtype=np.int64)] = -np.inf
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        hits = np.fromiter((int(x) in rel[u] for x in top), dtype=np.float64)
        pv.append(float(hits.sum() / k))
        rv.append(float(hits.sum() / len(rel[u])))
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = float(np.sum(hits * discounts))
        ideal = float(np.sum(discounts[:min(k, len(rel[u]))]))
        nv.append(dcg / max(ideal, 1e-12))
    if not pv:
        return {f"Precision@{k}": float("nan"), f"Recall@{k}": float("nan"), f"NDCG@{k}": float("nan"), "ranking_users": 0.0}
    return {f"Precision@{k}": float(np.mean(pv)), f"Recall@{k}": float(np.mean(rv)), f"NDCG@{k}": float(np.mean(nv)), "ranking_users": float(len(pv))}


@torch.no_grad()
def per_user_error_frame(model: nn.Module, data: MatrixData, model_name: str, split_seed: int, model_seed: int) -> pd.DataFrame:
    p = _predict_raw_generic(model, data, data.test_users, data.test_items)
    e = p - data.test_ratings_raw
    frame = pd.DataFrame({
        "user": data.test_users.detach().cpu().numpy(),
        "abs_error": e.abs().detach().cpu().numpy(),
        "sq_error": e.square().detach().cpu().numpy(),
    })
    out = frame.groupby("user", as_index=False).agg(user_MAE=("abs_error", "mean"), user_MSE=("sq_error", "mean"))
    out.insert(0, "model_seed", int(model_seed)); out.insert(0, "split_seed", int(split_seed)); out.insert(0, "model", model_name)
    return out


def paired_bootstrap_vs_reference(per_user: pd.DataFrame, reference: str, samples: int, seed: int) -> pd.DataFrame:
    """Paired user bootstrap on user-level MSE, separately within each outer split."""
    rows = []
    rng = np.random.default_rng(seed)
    mean_seed = per_user.groupby(["split_seed", "model", "user"], as_index=False)["user_MSE"].mean()
    for split_seed, g in mean_seed.groupby("split_seed"):
        ref = g[g.model == reference][["user", "user_MSE"]].rename(columns={"user_MSE": "ref"})
        if ref.empty:
            continue
        for name in sorted(set(g.model) - {reference}):
            cur = g[g.model == name][["user", "user_MSE"]].rename(columns={"user_MSE": "cur"})
            merged = ref.merge(cur, on="user", how="inner")
            if merged.empty:
                continue
            d = merged["cur"].to_numpy() - merged["ref"].to_numpy()  # negative => method better
            n = len(d)
            draws = np.empty(samples, dtype=np.float64)
            for s in range(samples):
                draws[s] = d[rng.integers(0, n, size=n)].mean()
            rows.append({
                "split_seed": int(split_seed), "reference": reference, "model": name,
                "delta_user_MSE": float(d.mean()), "ci95_low": float(np.quantile(draws, .025)),
                "ci95_high": float(np.quantile(draws, .975)),
                "p_two_sided": float(min(1.0, 2.0 * min(np.mean(draws <= 0), np.mean(draws >= 0)))),
                "paired_users": int(n),
            })
    return pd.DataFrame(rows)


def hierarchical_paired_bootstrap_vs_reference(
    per_user: pd.DataFrame, reference: str, samples: int, seed: int
) -> pd.DataFrame:
    """Hierarchical paired bootstrap across outer splits and users.

    Model seeds are averaged first for each (split, model, user).  Each bootstrap draw
    resamples outer split IDs with replacement and then resamples paired users within each
    selected split.  The outer split therefore remains the top-level experimental unit,
    while within-split user pairing preserves the recommender comparison structure.
    Negative delta_user_MSE means the candidate beats the reference.
    """
    if per_user.empty or samples <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    mean_seed = per_user.groupby(["split_seed", "model", "user"], as_index=False)["user_MSE"].mean()
    models = sorted(set(mean_seed.model) - {reference})
    rows = []
    for name in models:
        split_deltas: Dict[int, np.ndarray] = {}
        for ss, g in mean_seed.groupby("split_seed"):
            ref = g[g.model == reference][["user", "user_MSE"]].rename(columns={"user_MSE": "ref"})
            cur = g[g.model == name][["user", "user_MSE"]].rename(columns={"user_MSE": "cur"})
            merged = ref.merge(cur, on="user", how="inner")
            if not merged.empty:
                split_deltas[int(ss)] = (merged["cur"].to_numpy() - merged["ref"].to_numpy()).astype(np.float64)
        if not split_deltas:
            continue
        split_ids = np.asarray(sorted(split_deltas), dtype=np.int64)
        observed = float(np.mean([split_deltas[int(ss)].mean() for ss in split_ids]))
        draws = np.empty(samples, dtype=np.float64)
        for b in range(samples):
            chosen = rng.choice(split_ids, size=len(split_ids), replace=True)
            split_means = []
            for ss in chosen:
                d = split_deltas[int(ss)]
                split_means.append(float(d[rng.integers(0, len(d), size=len(d))].mean()))
            draws[b] = float(np.mean(split_means))
        rows.append({
            "reference": reference, "model": name, "delta_user_MSE": observed,
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "p_two_sided": float(min(1.0, 2.0 * min(np.mean(draws <= 0), np.mean(draws >= 0)))),
            "outer_splits": int(len(split_ids)),
            "paired_users_total": int(sum(len(x) for x in split_deltas.values())),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Strict locked-test candidate/checkpoint machinery
# -----------------------------------------------------------------------------


def _cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _config_key(kind: str, cfg_dict: Dict[str, Any], **extra: Any) -> str:
    return json.dumps({"kind": kind, "config": cfg_dict, **extra}, sort_keys=True, ensure_ascii=True, default=str)


def _stage_payload(path: Path, kind: str, model: nn.Module, cfg_obj: Any, **extra: Any) -> None:
    payload = {"kind": kind, "config": _asdict(cfg_obj), "state_dict": _cpu_state_dict(model), **extra}
    torch.save(payload, path)



def _model_diagnostics(model: nn.Module, data: MatrixData, users: torch.Tensor, items: torch.Tensor) -> Dict[str, float]:
    if isinstance(model, PaperMBMF):
        return model.diagnostics(users, items)
    return model.diagnostics(data, users, items)

def _common_candidate_row(model_name: str, seed: int, cfg_obj: Any, val: Dict[str, float], info: Dict[str, object], diag: Dict[str, float], **extra):
    cd = _asdict(cfg_obj)
    return {
        "model": model_name, "protocol": "strict_train_val_locked_test_v7", "seed": int(seed),
        "latent_dim": int(cd.get("latent_dim", extra.get("latent_dim", 0))),
        "parameters": int(extra.pop("parameters")),
        "config_key": extra.pop("config_key"),
        "global_rating_bound_guaranteed": bool(extra.pop("global_rating_bound_guaranteed", False)),
        **{k: v for k, v in cd.items() if isinstance(v, (str, int, float, bool, type(None)))},
        **extra,
        **{f"val_{k}": v for k, v in val.items()},
        **{f"valdiag_{k}": v for k, v in diag.items()},
        **{k: v for k, v in info.items() if k != "history"},
        "_history": info.get("history", []),
    }


def _train_candidate(kind: str, data: MatrixData, cfg_obj: Any, seed: int, device: torch.device,
                     ckpt: Path, *, variant: Optional[str] = None, init_mode: Optional[str] = None) -> Dict[str, object]:
    set_global_seed(seed)
    if kind == "paper":
        model = PaperMBMF(data, cfg_obj, variant, init_mode, device)
        info = train_paper_mbmf(model, data, cfg_obj)
        model_name = str(variant)
        key = _config_key(kind, _asdict(cfg_obj), variant=variant, init_mode=init_mode)
        guaranteed = bool(variant == "MBMF-F")
        _stage_payload(ckpt, kind, model, cfg_obj, variant=variant, init_mode=init_mode)
    elif kind == "mf":
        model = BiasedMatrixFactorization(data, cfg_obj, device)
        info = train_thesis_model(
            model, data, max_epochs=cfg_obj.max_epochs, lr=cfg_obj.lr,
            weight_decay=cfg_obj.weight_decay, eval_every=cfg_obj.eval_every,
            val_patience=cfg_obj.val_patience, grad_clip=cfg_obj.grad_clip,
        )
        model_name = "MF-Bias"
        key = _config_key(kind, _asdict(cfg_obj))
        guaranteed = False
        _stage_payload(ckpt, kind, model, cfg_obj)
    else:
        cls_map = {
            "m1": ThesisMethod1, "m2": ThesisMethod2, "m3": ProposedMethod3, "m4": ProposedMethod4,
            "m5": DecoderCoupledDualViewMBMF, "m6": DecoderCoupledDualViewMBMF, "m7": ProposedMethod7,
        }
        name_map = {
            "m1": "Method-1-AE-LatentMagnitude", "m2": "Method-2-DualViewFusion",
            "m3": "Method-3-PriorResidualMagnitude", "m4": "Method-4-BoundedPriorResidual",
            "m5": "Method-5-DVMBMF-Raw", "m6": "Method-6-DVMBMF-RangeSafe",
            "m7": "Method-7-UAIBMF",
        }
        model = cls_map[kind](data, cfg_obj, device)
        if kind == "m7":
            info = train_method7_minibatch(model, data, cfg_obj)
        else:
            info = train_thesis_model(
                model, data, max_epochs=cfg_obj.max_epochs, lr=cfg_obj.lr, weight_decay=cfg_obj.weight_decay,
                eval_every=cfg_obj.eval_every, val_patience=cfg_obj.val_patience, grad_clip=cfg_obj.grad_clip,
            )
        model_name = name_map[kind]
        key = _config_key(kind, _asdict(cfg_obj))
        guaranteed = (
            kind == "m4" or (kind in {"m5", "m6"} and cfg_obj.radius_mode in {"range_safe", "fixed"}) or kind == "m7"
            or (kind == "m1" and cfg_obj.model_space == "centered_raw" and cfg_obj.radius_mode == "bounded_tanh")
            or (kind == "m2" and cfg_obj.model_space == "centered_raw" and cfg_obj.radius_mode == "bounded_tanh" and cfg_obj.counterpart_radius_mode == "bounded_sigmoid")
        )
        _stage_payload(ckpt, kind, model, cfg_obj)
    val = evaluate_model(model, data, "val")
    diag = _model_diagnostics(model, data, data.val_users, data.val_items)
    row = _common_candidate_row(
        model_name, seed, cfg_obj, val, info, diag, parameters=parameter_count(model),
        config_key=key, global_rating_bound_guaranteed=guaranteed,
        init=(init_mode if kind == "paper" else getattr(cfg_obj, "angle_init", None)),
        variant=variant,
    )
    row["checkpoint_path"] = str(ckpt)
    return row


def validation_selected_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """V7: select configuration from validation only; test columns need not exist."""
    if runs.empty:
        return runs.copy()
    required = {"split_seed", "model", "val_RMSE", "config_key"}
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"candidate runs missing {sorted(missing)}")
    parts = []
    for (split_seed, model_name), g in runs.groupby(["split_seed", "model"], dropna=False):
        means = g.groupby("config_key", as_index=False)["val_RMSE"].mean().sort_values(["val_RMSE", "config_key"], kind="mergesort")
        best = means.iloc[0]
        chosen = g[g.config_key == best.config_key].copy()
        chosen["selection_val_RMSE_mean"] = float(best.val_RMSE)
        parts.append(chosen)
    return pd.concat(parts, ignore_index=True) if parts else runs.iloc[:0].copy()


def _rebuild_from_payload(payload: Dict[str, Any], data: MatrixData, device: torch.device) -> nn.Module:
    kind = payload["kind"]; cd = payload["config"]
    if kind == "paper":
        cfg = PaperBaselineConfig(**cd)
        model = PaperMBMF(data, cfg, payload["variant"], payload["init_mode"], device)
    elif kind == "mf": model = BiasedMatrixFactorization(data, MFConfig(**cd), device)
    elif kind == "m1": model = ThesisMethod1(data, Method1Config(**cd), device)
    elif kind == "m2": model = ThesisMethod2(data, Method2Config(**cd), device)
    elif kind == "m3": model = ProposedMethod3(data, Method3Config(**cd), device)
    elif kind == "m4": model = ProposedMethod4(data, Method4Config(**cd), device)
    elif kind == "m5": model = DecoderCoupledDualViewMBMF(data, Method5Config(**cd), device)
    elif kind == "m6": model = DecoderCoupledDualViewMBMF(data, Method6Config(**cd), device)
    elif kind == "m7": model = ProposedMethod7(data, Method7Config(**cd), device)
    else: raise ValueError(kind)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device); model.eval()
    return model


def _locked_test_selected(selected: pd.DataFrame, split_frames: Dict[int, pd.DataFrame], cfg: ExperimentConfig,
                          device: torch.device) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, user_frames = [], []
    cache: Dict[int, MatrixData] = {}
    for _, row in selected.iterrows():
        ss = int(row.split_seed)
        if ss not in cache:
            cache[ss] = build_matrix_data(split_frames[ss], cfg.data, device)
        data = cache[ss]
        payload = torch.load(row.checkpoint_path, map_location="cpu", weights_only=False)
        model = _rebuild_from_payload(payload, data, device)
        test = evaluate_model(model, data, "test")
        diag = _model_diagnostics(model, data, data.test_users, data.test_items)
        rank = {}
        for k in cfg.evaluation.ranking_ks:
            rank.update(exact_ranking_metrics_v7(
                model, data, int(k), cfg.evaluation.ranking_item_chunk_size,
                cfg.evaluation.ranking_max_users, cfg.evaluation.bootstrap_seed,
            ))
        out = row.drop(labels=[c for c in ["checkpoint_path"] if c in row.index]).to_dict()
        out.update({f"test_{k}": v for k, v in test.items()})
        out.update({f"testdiag_{k}": v for k, v in diag.items()})
        out.update({f"test_{k}": v for k, v in rank.items()})
        rows.append(out)
        if cfg.evaluation.save_per_user_metrics:
            user_frames.append(per_user_error_frame(model, data, str(row.model), ss, int(row.seed)))
    return pd.DataFrame(rows), (pd.concat(user_frames, ignore_index=True) if user_frames else pd.DataFrame())


def summarise_selected_test(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    metrics = [c for c in selected.columns if c.startswith("test_") and pd.api.types.is_numeric_dtype(selected[c])]
    metrics += [c for c in ["elapsed_sec", "parameters"] if c in selected.columns]
    metrics = list(dict.fromkeys(metrics))
    return selected.groupby("model", dropna=False)[metrics].agg(["mean", "std", "count"])


def summarise_selected_test_by_split(selected: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Primary Q1 summary treating the outer data split, not random initialization, as unit.

    Model-seed replicates are averaged within each split first.  The returned summary then
    reports between-split mean/std/count, avoiding pseudo-replication of initialization seeds.
    """
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()
    metrics = [c for c in selected.columns if c.startswith("test_") and pd.api.types.is_numeric_dtype(selected[c])]
    metrics += [c for c in ["elapsed_sec", "parameters"] if c in selected.columns]
    metrics = list(dict.fromkeys(metrics))
    split_means = selected.groupby(["model", "split_seed"], as_index=False)[metrics].mean()
    summary = split_means.groupby("model", dropna=False)[metrics].agg(["mean", "std", "count"])
    return split_means, summary


# -----------------------------------------------------------------------------
# Plots and ablations
# -----------------------------------------------------------------------------


def generate_publication_plots(locked: pd.DataFrame, histories: List[Dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plots] skipped: matplotlib unavailable ({exc})")
        return
    pdir = output_dir / "plots"; pdir.mkdir(parents=True, exist_ok=True)
    if locked.empty:
        return
    def bar(metric: str, filename: str, xlabel: str):
        if metric not in locked.columns: return
        g = locked.groupby("model")[metric].agg(["mean", "std"]).sort_values("mean")
        fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(g))))
        ax.barh(g.index, g["mean"], xerr=g["std"].fillna(0.0))
        ax.set_xlabel(xlabel); ax.set_ylabel("Model"); ax.set_title(xlabel + " by model")
        fig.tight_layout(); fig.savefig(pdir / filename, dpi=180); plt.close(fig)
    bar("test_RMSE", "test_rmse.png", "Test RMSE")
    bar("test_MAE", "test_mae.png", "Test MAE")
    bar("test_OOB_rate", "test_oob_rate.png", "Out-of-bound rate")
    bar("test_max_bound_violation", "test_max_bound_violation.png", "Maximum raw rating-bound violation")
    bar("test_F1_trainThreshold_macro_user", "test_f1_macro_user.png", "Macro-user F1 (train-derived threshold)")
    bar("test_F1_trainThreshold_micro", "test_f1_micro.png", "Micro F1 (train-derived threshold)")
    for c in locked.columns:
        if c.startswith(("test_NDCG@", "test_Precision@", "test_Recall@")):
            bar(c, c.replace("@", "_at_") + ".png", c.replace("test_", ""))
    if "testdiag_max_rating_bound_violation" in locked.columns:
        bar("testdiag_max_rating_bound_violation", "rating_bound_violation.png", "Maximum analytical rating-bound violation")
    if "testdiag_max_structural_bound_violation" in locked.columns:
        bar("testdiag_max_structural_bound_violation", "structural_bound_violation.png", "Maximum structural-bound violation")
    # Every model-specific numeric diagnostic gets the same publication-style treatment.
    # This keeps Methods 4-7 first-class citizens instead of silently omitting their new state.
    diag_dir = pdir / "diagnostics"; diag_dir.mkdir(parents=True, exist_ok=True)
    for c in locked.columns:
        if not c.startswith("testdiag_") or not pd.api.types.is_numeric_dtype(locked[c]):
            continue
        g = locked.groupby("model")[c].agg(["mean", "std"]).dropna(subset=["mean"]).sort_values("mean")
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(g))))
        ax.barh(g.index, g["mean"], xerr=g["std"].fillna(0.0))
        label = c.replace("testdiag_", "").replace("_", " ")
        ax.set_xlabel(label); ax.set_ylabel("Model"); ax.set_title(label + " by model")
        fig.tight_layout(); fig.savefig(diag_dir / (c.replace("testdiag_", "") + ".png"), dpi=180); plt.close(fig)
    bar("test_Gaussian_NLL", "uncertainty_gaussian_nll.png", "Gaussian NLL")
    for level in (50, 80, 90, 95):
        bar(f"test_PICP{level}_abs_gap", f"uncertainty_picp{level}_gap.png", f"|PICP{level} - {level/100:.2f}|")
        bar(f"test_MPIW{level}_normalized", f"uncertainty_mpiw{level}_normalized.png", f"Normalized {level}% interval width")
    bar("test_coverage_calibration_MAE", "uncertainty_coverage_calibration_mae.png", "Mean absolute coverage calibration error")
    bar("elapsed_sec", "training_time.png", "Training time (seconds)")
    if "parameters" in locked.columns and "test_RMSE" in locked.columns:
        g = locked.groupby("model", as_index=False).agg(parameters=("parameters", "mean"), rmse=("test_RMSE", "mean"))
        fig, ax = plt.subplots(figsize=(8, 6)); ax.scatter(g.parameters, g.rmse)
        for _, r in g.iterrows(): ax.annotate(str(r.model), (r.parameters, r.rmse), fontsize=7)
        ax.set_xscale("log"); ax.set_xlabel("Trainable parameters (log scale)"); ax.set_ylabel("Test RMSE")
        ax.set_title("Model size vs test RMSE"); fig.tight_layout(); fig.savefig(pdir / "parameters_vs_rmse.png", dpi=180); plt.close(fig)
    # Mean validation trajectories for the validation-selected checkpoints only.
    selected_ids = set(locked["candidate_id"].astype(int).tolist()) if "candidate_id" in locked.columns else set()
    hrows = []
    for h in histories:
        if selected_ids and int(h.get("candidate_id", -1)) not in selected_ids:
            continue
        for point in h.get("history", []):
            if "val_RMSE" in point:
                hrows.append({"model": h.get("model"), "epoch": point.get("epoch"), "val_RMSE": point.get("val_RMSE")})
    if hrows:
        hf = pd.DataFrame(hrows).dropna()
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, g in hf.groupby("model"):
            curve = g.groupby("epoch", as_index=False).val_RMSE.mean().sort_values("epoch")
            ax.plot(curve.epoch, curve.val_RMSE, label=name)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Validation RMSE"); ax.set_title("Validation trajectories")
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(pdir / "validation_curves.png", dpi=180); plt.close(fig)


def make_ablation_variants(base: ExperimentConfig) -> List[Tuple[str, ExperimentConfig]]:
    """Pre-registered-style ablations covering the mechanism of every proposed method.

    Every family includes an explicit ``full`` row so the ablation table is self-contained.
    Variants are executed under the exact same split/selection/locked-test pipeline.
    """
    variants: List[Tuple[str, ExperimentConfig]] = []

    def add(tag: str, method: int, mutate=lambda c: None):
        c = copy.deepcopy(base)
        c.enable_mf_baseline = False
        c.enabled_baselines = ()
        c.enabled_methods = (method,)
        c.run_tag = tag
        mutate(c)
        variants.append((tag, c))

    # Method 1
    add("m1_full", 1)
    add("m1_raw", 1, lambda c: setattr(c.method1, "radius_mode", "raw_norm"))
    add("m1_bounded", 1, lambda c: setattr(c.method1, "radius_mode", "bounded_tanh"))
    add("m1_no_mask", 1, lambda c: setattr(c.method1, "input_mode", "values_only"))

    # Method 2
    add("m2_full", 2)
    add("m2_no_consistency", 2, lambda c: setattr(c.method2, "consistency_weight", 0.0))
    add("m2_derived_consistency", 2, lambda c: setattr(
        c.method2, "consistency_weight",
        c.method2.fusion_weight_user_view * (1.0 - c.method2.fusion_weight_user_view),
    ))
    def m2bounded(c):
        c.method2.radius_mode = "bounded_tanh"
        c.method2.counterpart_radius_mode = "bounded_sigmoid"
    add("m2_bounded", 2, m2bounded)

    # Method 3
    add("m3_full", 3)
    add("m3_no_residual", 3, lambda c: setattr(c.method3, "max_abs_log_scale", 0.0))
    add("m3_local_prior", 3, lambda c: setattr(c.method3, "prior_mode", "legacy_local_mean_std"))
    add("m3_delta_regularized", 3, lambda c: setattr(c.method3, "delta_l2", 1e-3))
    add("m3_averaged_ae", 3, lambda c: setattr(c.method3, "ae_loss_mode", "averaged"))

    # Method 4
    add("m4_full", 4)
    add("m4_no_residual", 4, lambda c: setattr(c.method4, "max_abs_logit_shift", 0.0))
    add("m4_local_prior", 4, lambda c: setattr(c.method4, "prior_mode", "legacy_local_mean_std"))
    add("m4_no_mask", 4, lambda c: setattr(c.method4, "input_mode", "values_only"))
    add("m4_delta_regularized", 4, lambda c: setattr(c.method4, "delta_l2", 1e-3))

    # Method 5: faithful raw decoder-coupled dual view
    add("m5_full", 5)
    add("m5_user_branch", 5, lambda c: setattr(c.method5, "branch_mode", "user"))
    add("m5_item_branch", 5, lambda c: setattr(c.method5, "branch_mode", "item"))
    def m5norec(c):
        c.method5.user_reconstruction_weight = 0.0
        c.method5.item_reconstruction_weight = 0.0
    add("m5_no_reconstruction", 5, m5norec)
    add("m5_no_mask", 5, lambda c: setattr(c.method5, "input_mode", "values_only"))

    # Method 6: range-safe decoder-coupled dual view
    add("m6_full", 6)
    add("m6_user_branch", 6, lambda c: setattr(c.method6, "branch_mode", "user"))
    add("m6_item_branch", 6, lambda c: setattr(c.method6, "branch_mode", "item"))
    add("m6_tau_fixed1", 6, lambda c: setattr(c.method6, "radius_temperature", 1.0))
    add("m6_fixed_radius", 6, lambda c: setattr(c.method6, "radius_mode", "fixed"))
    def m6norec(c):
        c.method6.user_reconstruction_weight = 0.0
        c.method6.item_reconstruction_weight = 0.0
    add("m6_no_reconstruction", 6, m6norec)

    # Method 7: retain the original UC-IBMF ablations, under the common v7 protocol.
    add("m7_full", 7)
    add("m7_fixed_radius", 7, lambda c: setattr(c.method7, "radius_mode", "fixed"))
    def m7nocontext(c):
        c.method7.use_context = False
        c.method7.consistency_weight = 0.0
    add("m7_no_context", 7, m7nocontext)
    add("m7_no_shrinkage", 7, lambda c: setattr(c.method7, "use_shrinkage", False))
    add("m7_no_bias_expert", 7, lambda c: setattr(c.method7, "use_bias_expert", False))
    add("m7_homoscedastic", 7, lambda c: setattr(c.method7, "heteroscedastic", False))
    def m7mse(c):
        c.method7.loss_mode = "mse"
        c.method7.consistency_weight = 0.0
    add("m7_mse_only", 7, m7mse)
    return variants


def run_ablation_suite(cfg: ExperimentConfig) -> pd.DataFrame:
    root = Path(cfg.output_dir) / "ablations"
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    for tag, acfg in make_ablation_variants(cfg):
        acfg.output_dir = str(root / tag)
        print(f"\n######## ABLATION {tag} ########")
        _, locked = run_all_experiments(acfg)
        if not locked.empty:
            f = locked.copy()
            f.insert(0, "ablation", tag)
            frames.append(f)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        return combined

    combined.to_csv(root / "ablation_locked_test_runs.csv", index=False)
    metric_cols = [c for c in combined.columns if c.startswith("test_") and pd.api.types.is_numeric_dtype(combined[c])]
    summary = combined.groupby(["ablation", "model"])[metric_cols].agg(["mean", "std", "count"])
    summary.to_csv(root / "ablation_summary.csv")

    # Effect table relative to each family's full model. Positive delta_RMSE is worse.
    means = combined.groupby(["ablation", "model"], as_index=False)[metric_cols].mean()
    effects = []
    for _, row in means.iterrows():
        tag = str(row.ablation)
        family = tag.split("_", 1)[0]
        full_tag = family + "_full"
        base = means[(means.ablation == full_tag) & (means.model == row.model)]
        if base.empty:
            continue
        rec = {"ablation": tag, "model": row.model, "reference": full_tag}
        for metric in metric_cols:
            rec[f"delta_{metric}"] = float(row[metric] - base.iloc[0][metric])
        effects.append(rec)
    effects_df = pd.DataFrame(effects)
    if not effects_df.empty:
        effects_df.to_csv(root / "ablation_effects_vs_full.csv", index=False)

    try:
        import matplotlib.pyplot as plt
        if not effects_df.empty and "delta_test_RMSE" in effects_df.columns:
            q = effects_df[effects_df.ablation != effects_df.reference].sort_values("delta_test_RMSE")
            if not q.empty:
                fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(q))))
                ax.barh(q.ablation, q.delta_test_RMSE)
                ax.axvline(0.0, linewidth=1.0)
                ax.set_xlabel("Delta test RMSE vs family full model (positive = worse)")
                ax.set_ylabel("Ablation")
                ax.set_title("Ablation effects")
                fig.tight_layout()
                fig.savefig(root / "ablation_delta_rmse.png", dpi=180)
                plt.close(fig)
    except Exception as exc:
        print(f"[ablation-plots] skipped: {exc}")
    return combined


# -----------------------------------------------------------------------------
# V7 end-to-end experiment runner
# -----------------------------------------------------------------------------


def run_all_experiments(cfg: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """V7 strict experiment pipeline.

    Phase A trains every candidate and records VALIDATION metrics only.  Checkpoints are
    staged on disk.  Configuration selection is performed from mean validation RMSE over
    model seeds.  Phase B unlocks test labels exactly for the selected checkpoints and
    computes point, ranking, uncertainty, bound, and paired-user outputs.  Thus test is not
    even evaluated for losing hyperparameter configurations.
    """
    if not cfg.strict_locked_test:
        raise ValueError("V7 publication pipeline requires strict_locked_test=True")
    device = get_device(); print(f"[device] {device}")
    geometry_self_test(device)
    heldout_label_invariance_test_v7(device)
    ratings = load_movielens(cfg.data.data_dir)
    manifest = dataset_manifest(cfg.data.data_dir, ratings)
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    staging = out / "_candidate_checkpoints"
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    split_seeds = cfg.split_seeds[:1] if cfg.fast_debug else cfg.split_seeds
    model_seeds = cfg.model_seeds[:1] if cfg.fast_debug else cfg.model_seeds
    latent_dims = cfg.latent_dims[:1] if cfg.fast_debug else cfg.latent_dims
    rows: List[Dict[str, object]] = []; histories: List[Dict[str, object]] = []
    split_frames: Dict[int, pd.DataFrame] = {}
    candidate_counter = 0

    def collect(row, ss, k):
        nonlocal candidate_counter
        hist = row.pop("_history", [])
        row["split_seed"] = int(ss); row["latent_dim"] = int(k); row["run_tag"] = cfg.run_tag
        row["candidate_id"] = candidate_counter
        histories.append({"candidate_id": candidate_counter, "model": row["model"], "seed": row["seed"],
                          "split_seed": int(ss), "latent_dim": int(k), "history": hist})
        candidate_counter += 1; rows.append(row)

    for ss in split_seeds:
        dcfg = copy.deepcopy(cfg.data); dcfg.split_seed = int(ss)
        split = make_safe_split(ratings, dcfg); split_frames[int(ss)] = split.copy()
        data = build_matrix_data(split, dcfg, device)
        for k in latent_dims:
            # Configuration copies keep K symmetric across all method families.
            pc = copy.deepcopy(cfg.paper); pc.latent_dim = int(k)
            mfcfg = copy.deepcopy(cfg.mf); mfcfg.latent_dim = int(k)
            mcfgs = {
                1: copy.deepcopy(cfg.method1), 2: copy.deepcopy(cfg.method2), 3: copy.deepcopy(cfg.method3),
                4: copy.deepcopy(cfg.method4), 5: copy.deepcopy(cfg.method5), 6: copy.deepcopy(cfg.method6),
                7: copy.deepcopy(cfg.method7),
            }
            for mc in mcfgs.values(): mc.latent_dim = int(k)
            if cfg.fast_debug:
                pc.max_epochs=min(pc.max_epochs,4); pc.eval_every=1
                mfcfg.max_epochs=min(mfcfg.max_epochs,4); mfcfg.eval_every=1; mfcfg.val_patience=3
                for mc in mcfgs.values(): mc.max_epochs=min(mc.max_epochs,4); mc.eval_every=1; mc.val_patience=3
            if cfg.enable_mf_baseline:
                for seed in model_seeds:
                    ckpt = staging / f"c{candidate_counter:06d}.pt"
                    print(f"\n=== MF-Bias K={k} split={ss} seed={seed} [VAL ONLY] ===")
                    collect(_train_candidate("mf", data, mfcfg, int(seed), device, ckpt), ss, k)
            for variant in cfg.enabled_baselines:
                modes = pc.init_modes if variant in pc.init_ablation_variants else (pc.default_init,)
                for mode in modes:
                    for seed in model_seeds:
                        ckpt = staging / f"c{candidate_counter:06d}.pt"
                        print(f"\n=== {variant} K={k} init={mode} split={ss} seed={seed} [VAL ONLY] ===")
                        collect(_train_candidate("paper", data, pc, int(seed), device, ckpt, variant=variant, init_mode=mode), ss, k)
            for method in cfg.enabled_methods:
                kind=f"m{method}"; mc=mcfgs[int(method)]
                for seed in model_seeds:
                    ckpt = staging / f"c{candidate_counter:06d}.pt"
                    print(f"\n=== Method {method} K={k} split={ss} seed={seed} [VAL ONLY] ===")
                    collect(_train_candidate(kind, data, mc, int(seed), device, ckpt), ss, k)

    candidates = pd.DataFrame(rows)
    leaked_test_columns = [c for c in candidates.columns if c.startswith("test_") or c.startswith("testdiag_")]
    if leaked_test_columns:
        raise AssertionError(f"candidate table contains forbidden test outputs: {leaked_test_columns}")
    selected = validation_selected_runs(candidates)
    # Test PERFORMANCE is first computed here, for selected checkpoints only.
    locked, per_user = _locked_test_selected(selected, split_frames, cfg, device)
    run_summary = summarise_selected_test(locked)
    split_means, summary = summarise_selected_test_by_split(locked)

    candidate_export = candidates.drop(columns=["checkpoint_path"], errors="ignore")
    selected_export = selected.drop(columns=["checkpoint_path"], errors="ignore")
    candidate_export.to_csv(out / "candidate_validation_runs.csv", index=False)
    selected_export.to_csv(out / "selected_by_validation_candidates.csv", index=False)
    locked.to_csv(out / "locked_test_runs.csv", index=False)
    run_summary.to_csv(out / "locked_test_run_summary.csv")
    split_means.to_csv(out / "locked_test_split_means.csv", index=False)
    summary.to_csv(out / "locked_test_summary.csv")
    if cfg.evaluation.save_per_user_metrics and not per_user.empty:
        per_user.to_csv(out / "per_user_test_metrics.csv", index=False)
        boot = paired_bootstrap_vs_reference(per_user, cfg.evaluation.bootstrap_reference_model,
                                             cfg.evaluation.bootstrap_samples, cfg.evaluation.bootstrap_seed)
        boot.to_csv(out / "paired_user_bootstrap.csv", index=False)
        hboot = hierarchical_paired_bootstrap_vs_reference(
            per_user, cfg.evaluation.bootstrap_reference_model,
            cfg.evaluation.bootstrap_samples, cfg.evaluation.bootstrap_seed,
        )
        hboot.to_csv(out / "hierarchical_paired_bootstrap.csv", index=False)
    with open(out / "training_histories.json", "w", encoding="utf-8") as f: json.dump(histories, f, ensure_ascii=False, indent=2)
    with open(out / "config.json", "w", encoding="utf-8") as f: json.dump(_asdict(cfg), f, ensure_ascii=False, indent=2)
    with open(out / "dataset_manifest.json", "w", encoding="utf-8") as f: json.dump(manifest, f, ensure_ascii=False, indent=2)
    env = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
           "pandas": pd.__version__, "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
           "device": str(device), "pipeline_revision": "v7"}
    with open(out / "environment.json", "w", encoding="utf-8") as f: json.dump(env, f, ensure_ascii=False, indent=2)
    split_export=[]
    for ss,sf in split_frames.items():
        q=sf[["userId","movieId","rating","timestamp","split"]].copy(); q.insert(0,"split_seed",ss); split_export.append(q)
    if split_export: pd.concat(split_export,ignore_index=True).to_csv(out / "split_assignments.csv", index=False)
    if cfg.evaluation.generate_plots: generate_publication_plots(locked, histories, out)
    if not cfg.evaluation.keep_candidate_checkpoints: shutil.rmtree(staging, ignore_errors=True)
    print(f"\n[v7] saved strict locked-test study to {out.resolve()}")
    return candidate_export, locked


# -----------------------------------------------------------------------------
# V7 self-tests
# -----------------------------------------------------------------------------


def heldout_label_invariance_test_v7(device: torch.device) -> None:
    """Regression test for both construction-time and forward-time held-out leakage."""
    data = synthetic_data(device)
    altered = copy.deepcopy(data)
    altered.val_ratings_raw = data.rating_min + data.rating_max - data.val_ratings_raw
    altered.test_ratings_raw = data.rating_min + data.rating_max - data.test_ratings_raw

    builders = [
        lambda d: ThesisMethod1(d, Method1Config(latent_dim=4, hidden_dim=8, max_epochs=1), device),
        lambda d: ThesisMethod2(d, Method2Config(latent_dim=4, max_epochs=1), device),
        lambda d: ProposedMethod3(d, Method3Config(latent_dim=4, max_epochs=1), device),
        lambda d: ProposedMethod4(d, Method4Config(latent_dim=4, max_epochs=1), device),
        lambda d: DecoderCoupledDualViewMBMF(d, Method5Config(latent_dim=4, hidden_dim=8, max_epochs=1), device),
        lambda d: DecoderCoupledDualViewMBMF(d, Method6Config(latent_dim=4, hidden_dim=8, max_epochs=1), device),
        lambda d: ProposedMethod7(d, Method7Config(latent_dim=4, hidden_dim=8, max_epochs=1), device),
    ]
    for idx, build in enumerate(builders, 1):
        seed = 9000 + idx
        set_global_seed(seed)
        model_a = build(data)
        set_global_seed(seed)
        model_b = build(altered)
        model_a.eval(); model_b.eval()
        with torch.no_grad():
            pa = model_a.predict_raw_pairs(data, data.val_users, data.val_items)
            pb = model_b.predict_raw_pairs(altered, altered.val_users, altered.val_items)
        if not torch.equal(pa, pb):
            diff = float((pa - pb).abs().max().item()) if pa.numel() else 0.0
            raise AssertionError(f"Method {idx} construction/forward depends on held-out labels; max diff={diff}")
    print("[self-test] construction + forward held-out label invariance Methods 1-7: PASS")


def synthetic_smoke_test_v7(device: Optional[torch.device] = None) -> None:
    device = device or get_device(); geometry_self_test(device); d = synthetic_data(device)
    configs_models = [
        (Method1Config(latent_dim=4, hidden_dim=8, max_epochs=2), ThesisMethod1, False),
        (Method2Config(latent_dim=4, max_epochs=2, counterpart_radius_mode="scalar_softplus"), ThesisMethod2, False),
        (Method3Config(latent_dim=4, max_epochs=2), ProposedMethod3, False),
        (Method4Config(latent_dim=4, max_epochs=2), ProposedMethod4, True),
        (Method5Config(latent_dim=4, hidden_dim=8, max_epochs=2), DecoderCoupledDualViewMBMF, False),
        (Method6Config(latent_dim=4, hidden_dim=8, max_epochs=2), DecoderCoupledDualViewMBMF, True),
        (Method7Config(latent_dim=4, hidden_dim=8, max_epochs=2), ProposedMethod7, True),
    ]
    for idx,(c,cls,hard_bound) in enumerate(configs_models,1):
        set_global_seed(100+idx); m=cls(d,c,device); loss,_=m.loss(d); loss.backward()
        if not torch.isfinite(loss): raise AssertionError(f"Method {idx} nonfinite loss")
        for p in m.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all(): raise AssertionError(f"Method {idx} nonfinite gradient")
        pred=m.predict_raw_pairs(d,d.test_users,d.test_items)
        if not torch.isfinite(pred).all(): raise AssertionError(f"Method {idx} nonfinite prediction")
        diag=m.diagnostics(d,d.test_users,d.test_items)
        if diag.get("max_structural_bound_violation",0.0)>1e-4: raise AssertionError((idx,diag))
        if hard_bound and (pred.min()<d.rating_min-2e-5 or pred.max()>d.rating_max+2e-5): raise AssertionError(f"Method {idx} hard range bound")
    # Method 4 must reproduce the projected prior exactly at zero-initialised residual.
    m4=ProposedMethod4(d,Method4Config(latent_dim=4),device)
    with torch.no_grad():
        _,_,du,di,rw,rh,*_=m4.encoded(d)
        if du.abs().max()>1e-8 or di.abs().max()>1e-8: raise AssertionError("M4 residual not zero at init")
        if not torch.allclose(rw,m4.rW0,atol=2e-7,rtol=2e-7): raise AssertionError("M4 user prior identity")
        if not torch.allclose(rh,m4.rH0,atol=2e-7,rtol=2e-7): raise AssertionError("M4 item prior identity")
    heldout_label_invariance_test_v7(device)
    print("[smoke-test-v7] PASS: Methods 1-7 finite, leakage-safe; Methods 4/6/7 range-safe")


def _legacy_direct_test_helper_disabled(*args, **kwargs):
    raise RuntimeError(
        "Direct v6 run_*_once helpers are disabled in v7 because they evaluated test during each candidate run. "
        "Use run_all_experiments(), which performs validation-only selection followed by locked-test evaluation."
    )


# Preserve the old names as explicit safety guards rather than silently exposing the v6 test-touching API.
run_paper_once = _legacy_direct_test_helper_disabled
run_method1_once = _legacy_direct_test_helper_disabled
run_method2_once = _legacy_direct_test_helper_disabled
run_method3_once = _legacy_direct_test_helper_disabled


def main() -> None:
    cfg = ExperimentConfig()
    synthetic_smoke_test_v7()
    apply_q1_safe_preset(cfg)
    if cfg.data.rating_min is None or cfg.data.rating_max is None:
        print("[main-v7] synthetic tests passed. Set documented cfg.data.rating_min/rating_max before real experiments.")
        return
    run_all_experiments(cfg)


if __name__ == "__main__":
    main()
