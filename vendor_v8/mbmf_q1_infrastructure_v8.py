"""
Publication/Q1 extension for mbmf_q1_final_v7.py
=================================================

IMPORTANT INVARIANT
-------------------
This module does NOT edit or monkey-patch the v7 scientific core. It imports it and adds:
  * a six-dataset registry and preprocessing layer;
  * a sparse, mathematically equivalent execution backend for the three proposed models;
  * model/checkpoint save-load utilities;
  * richer train/validation/test metrics and prediction archives;
  * publication plots and statistical tests;
  * ablation/protocol helpers;
  * comparison baselines whose formulas are independently documented here;
  * adapters for external/official competitor implementations.

The v7 model equations, angle map, objectives, and primary configurations remain unchanged.
For small/medium matrices one may still run the original v7 dense path verbatim.  The sparse
backend exists so the *same equations* can be evaluated on MovieLens-10M without materialising
an N_user x N_item dense rating matrix.

Supported datasets
------------------
  synthetic, jester, bookcrossing, ml-100k, ml-1m, ml-10m

Competitor-protocol notes encoded below are based on the papers cited in the accompanying
Q1_PROTOCOL_AND_LITERATURE_NOTES_v7.md.  Where an exact official implementation is not
bundled, this module deliberately uses an external-prediction adapter rather than silently
claiming an exact reproduction.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import dataclasses
import gc
import hashlib
import importlib
import io
import json
import math
import os
import platform
import random
import shutil
import statistics
import sys
import threading
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Keep the scientific core physically separate and unmodified.
try:
    import mbmf_q1_final_v7 as core
except ImportError as exc:
    raise ImportError(
        "Place mbmf_q1_final_v7.py in the same directory (or on PYTHONPATH) before importing "
        "mbmf_q1_publication_v7.py. The publication extension intentionally does not copy/edit v6."
    ) from exc


# =============================================================================
# 0. Integrity / provenance
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
V7_CORE_PATH = Path(getattr(core, "__file__", THIS_DIR / "mbmf_q1_final_v7.py")).resolve()


def sha256_file(path: Union[str, Path], chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_scientific_core_v8(path: Union[str, Path]) -> str:
    """Hash the pinned V7 source independent of Windows CRLF checkout.

    Normalise *only* CRLF line endings to LF. This cannot mask changed Python
    code, comments, whitespace, or other source bytes. The original raw hash is
    tracked separately to detect any edits made after importing the module.
    """
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# Scientific identity is always the canonical LF SHA-256 on every operating
# system; raw snapshot is retained as an in-process mutation guard.
V7_CORE_SHA256 = sha256_scientific_core_v8(V7_CORE_PATH) if V7_CORE_PATH.exists() else "unknown"
V7_CORE_RAW_SHA256 = sha256_file(V7_CORE_PATH) if V7_CORE_PATH.exists() else "unknown"


# =============================================================================
# 1. Dataset registry and canonical loaders
# =============================================================================

DatasetName = Literal["synthetic", "jester", "bookcrossing", "ml-100k", "ml-1m", "ml-10m"]


@dataclass(frozen=True)
class DatasetSpec:
    name: DatasetName
    rating_min: float
    rating_max: float
    has_timestamp: bool
    default_relative_path: str
    rating_step: Optional[float]
    notes: str


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "synthetic": DatasetSpec(
        "synthetic", 1.0, 5.0, False, "synthetic/ratings.csv", 0.5,
        "Generated benchmark; rating range is configurable but defaults to 1..5.",
    ),
    "jester": DatasetSpec(
        "jester", -10.0, 10.0, False, "jester", None,
        "Jester 1; 99 denotes missing in the original XLS files and is never treated as a rating.",
    ),
    "bookcrossing": DatasetSpec(
        "bookcrossing", 1.0, 10.0, False, "bookcrossing/BX-Book-Ratings.csv", 1.0,
        "Default publication mode uses explicit ratings only (Book-Rating > 0); zero is implicit feedback.",
    ),
    "ml-100k": DatasetSpec(
        "ml-100k", 1.0, 5.0, True, "ml-100k/u.data", 1.0, "MovieLens 100K official split-independent rating file.",
    ),
    "ml-1m": DatasetSpec(
        "ml-1m", 1.0, 5.0, True, "ml-1m/ratings.dat", 1.0, "MovieLens 1M.",
    ),
    "ml-10m": DatasetSpec(
        "ml-10m", 0.5, 5.0, True, "ml-10M100K/ratings.dat", 0.5, "MovieLens 10M.",
    ),
}


@dataclass
class PublicationDataConfig:
    dataset: DatasetName = "ml-1m"
    data_root: str = "./datasets"
    split_mode: Literal["random_per_user", "temporal_per_user"] = "random_per_user"
    train_fraction: float = 0.80
    val_fraction: float = 0.10
    test_fraction: float = 0.10
    split_seed: int = 2026
    min_train_per_user: int = 1
    enforce_train_support_for_all_items: bool = True
    # Dataset-specific options.
    bookcrossing_include_implicit_zero: bool = False
    bookcrossing_min_user_ratings: int = 1
    bookcrossing_min_item_ratings: int = 1
    jester_files: Tuple[str, ...] = (
        "jester-data-1.xls", "jester-data-2.xls", "jester-data-3.xls",
    )
    # Dense v6 is faster/simpler for small matrices; beyond this threshold use sparse equivalent backend.
    dense_cell_limit: int = 30_000_000
    force_backend: Literal["auto", "dense", "sparse"] = "auto"
    # Synthetic options.
    synthetic_users: int = 400
    synthetic_items: int = 600
    synthetic_rank: int = 10
    synthetic_density: float = 0.08
    synthetic_noise_std: float = 0.20
    synthetic_rating_min: float = 1.0
    synthetic_rating_max: float = 5.0



def _canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    req = {"userId", "movieId", "rating"}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"ratings table missing required columns: {sorted(miss)}")
    keep = ["userId", "movieId", "rating"] + (["timestamp"] if "timestamp" in df.columns else [])
    out = df[keep].copy()
    out = out.dropna(subset=["userId", "movieId", "rating"])
    # Duplicate user-item ratings are resolved deterministically: latest timestamp when available,
    # otherwise final row in file order. This prevents duplicate keys downstream.
    if "timestamp" in out.columns:
        out = out.sort_values(["userId", "movieId", "timestamp"], kind="stable")
    out = out.drop_duplicates(["userId", "movieId"], keep="last")
    out["rating"] = pd.to_numeric(out["rating"], errors="raise").astype(np.float32)
    return out.reset_index(drop=True)



def _reindex_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map arbitrary IDs (e.g. ISBN strings) to stable contiguous int64 IDs.

    Returns canonical ratings and two mapping tables. Mapping is performed before splitting and
    never depends on held-out labels, so it is not leakage.
    """
    out = df.copy()
    uvals = pd.Index(pd.unique(out["userId"])).sort_values()
    ivals = pd.Index(pd.unique(out["movieId"])).sort_values()
    umap = {v: i for i, v in enumerate(uvals)}
    imap = {v: i for i, v in enumerate(ivals)}
    user_map_df = pd.DataFrame({"original_user_id": uvals.astype(str), "userId": np.arange(len(uvals), dtype=np.int64)})
    item_map_df = pd.DataFrame({"original_item_id": ivals.astype(str), "movieId": np.arange(len(ivals), dtype=np.int64)})
    out["userId"] = out["userId"].map(umap).astype(np.int64)
    out["movieId"] = out["movieId"].map(imap).astype(np.int64)
    return out, user_map_df, item_map_df



def _iterative_kcore(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    if min_user <= 1 and min_item <= 1:
        return df
    work = df
    while True:
        n0 = len(work)
        if min_user > 1:
            uc = work.groupby("userId", sort=False).size()
            work = work[work.userId.isin(uc[uc >= min_user].index)]
        if min_item > 1:
            ic = work.groupby("movieId", sort=False).size()
            work = work[work.movieId.isin(ic[ic >= min_item].index)]
        if len(work) == n0:
            break
    return work.reset_index(drop=True)



def _load_jester(root: Path, cfg: PublicationDataConfig) -> pd.DataFrame:
    # Support either preprocessed ratings.csv or official XLS files.
    csv_path = root / "jester" / "ratings.csv"
    if csv_path.exists():
        return _canonical_columns(pd.read_csv(csv_path))
    folder = root / "jester"
    files = [folder / f for f in cfg.jester_files if (folder / f).exists()]
    if not files:
        # Downloader may preserve arbitrary official names.
        files = sorted(folder.glob("*.xls")) + sorted(folder.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No Jester XLS files found under {folder}")
    chunks: List[pd.DataFrame] = []
    user_offset = 0
    for p in files:
        try:
            raw = pd.read_excel(p, header=None)
        except ImportError as exc:
            raise ImportError("Reading official Jester .xls files requires `pip install xlrd`.") from exc
        # First column is the number of jokes rated; remaining columns are ratings, 99 means missing.
        arr = raw.iloc[:, 1:].to_numpy(dtype=np.float32)
        rr, cc = np.where(arr != 99.0)
        vals = arr[rr, cc]
        chunks.append(pd.DataFrame({
            "userId": rr.astype(np.int64) + user_offset,
            "movieId": cc.astype(np.int64),
            "rating": vals.astype(np.float32),
        }))
        user_offset += arr.shape[0]
    return _canonical_columns(pd.concat(chunks, ignore_index=True))



def _load_bookcrossing(root: Path, cfg: PublicationDataConfig) -> pd.DataFrame:
    candidates = [
        root / "bookcrossing" / "BX-Book-Ratings.csv",
        root / "bookcrossing" / "BX-Book-Ratings.csv.txt",
        root / "bookcrossing" / "Ratings.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("BX-Book-Ratings.csv not found under datasets/bookcrossing")
    df = pd.read_csv(path, sep=";", encoding="latin-1", engine="python", on_bad_lines="skip")
    rename = {"User-ID": "userId", "ISBN": "movieId", "Book-Rating": "rating"}
    df = df.rename(columns=rename)
    if not cfg.bookcrossing_include_implicit_zero:
        df = df[pd.to_numeric(df["rating"], errors="coerce") > 0]
    df = _canonical_columns(df)
    df = _iterative_kcore(df, cfg.bookcrossing_min_user_ratings, cfg.bookcrossing_min_item_ratings)
    return df



def _load_movielens_named(root: Path, name: str) -> pd.DataFrame:
    if name == "ml-100k":
        p = root / "ml-100k" / "u.data"
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p, sep="\t", names=["userId", "movieId", "rating", "timestamp"], engine="c")
    elif name == "ml-1m":
        p = root / "ml-1m" / "ratings.dat"
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p, sep="::", names=["userId", "movieId", "rating", "timestamp"], engine="python")
    elif name == "ml-10m":
        candidates = [root / "ml-10M100K" / "ratings.dat", root / "ml-10m" / "ratings.dat"]
        p = next((x for x in candidates if x.exists()), None)
        if p is None:
            raise FileNotFoundError(str(candidates))
        df = pd.read_csv(p, sep="::", names=["userId", "movieId", "rating", "timestamp"], engine="python")
    else:
        raise ValueError(name)
    return _canonical_columns(df)



def _make_synthetic_df(cfg: PublicationDataConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.split_seed)
    n, m, k = cfg.synthetic_users, cfg.synthetic_items, cfg.synthetic_rank
    U = rng.normal(size=(n, k)).astype(np.float32)
    V = rng.normal(size=(m, k)).astype(np.float32)
    score = (U @ V.T) / math.sqrt(k)
    # Smoothly map to documented range, then add noise before quantisation to half-star steps.
    lo, hi = cfg.synthetic_rating_min, cfg.synthetic_rating_max
    raw = lo + (hi - lo) / (1.0 + np.exp(-score))
    raw += rng.normal(0.0, cfg.synthetic_noise_std, size=raw.shape)
    raw = np.clip(np.round(raw * 2.0) / 2.0, lo, hi)
    mask = rng.random((n, m)) < cfg.synthetic_density
    # Ensure every user and item has at least one observation.
    mask[np.arange(n), rng.integers(0, m, size=n)] = True
    mask[rng.integers(0, n, size=m), np.arange(m)] = True
    u, i = np.where(mask)
    return pd.DataFrame({"userId": u.astype(np.int64), "movieId": i.astype(np.int64), "rating": raw[u, i].astype(np.float32)})



def load_publication_dataset(cfg: PublicationDataConfig, *, save_mappings_to: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, DatasetSpec]:
    root = Path(cfg.data_root)
    if cfg.dataset not in DATASET_SPECS:
        raise ValueError(f"unknown dataset {cfg.dataset}")
    spec = DATASET_SPECS[cfg.dataset]
    if cfg.dataset == "synthetic":
        df = _make_synthetic_df(cfg)
        spec = DatasetSpec("synthetic", cfg.synthetic_rating_min, cfg.synthetic_rating_max, False, "synthetic/generated", 0.5, spec.notes)
    elif cfg.dataset == "jester":
        df = _load_jester(root, cfg)
    elif cfg.dataset == "bookcrossing":
        df = _load_bookcrossing(root, cfg)
        if cfg.bookcrossing_include_implicit_zero:
            spec = DatasetSpec("bookcrossing", 0.0, 10.0, False, spec.default_relative_path, 1.0,
                               spec.notes + " This run explicitly includes implicit zero feedback.")
    else:
        df = _load_movielens_named(root, cfg.dataset)

    observed_min, observed_max = float(df.rating.min()), float(df.rating.max())
    if observed_min < spec.rating_min - 1e-8 or observed_max > spec.rating_max + 1e-8:
        raise ValueError(f"observed range [{observed_min},{observed_max}] outside documented [{spec.rating_min},{spec.rating_max}]")
    df, umap, imap = _reindex_ids(df)
    if save_mappings_to is not None:
        p = Path(save_mappings_to); p.mkdir(parents=True, exist_ok=True)
        umap.to_csv(p / f"{cfg.dataset}_user_id_map.csv", index=False)
        imap.to_csv(p / f"{cfg.dataset}_item_id_map.csv", index=False)
    return df, spec



def make_core_split(df: pd.DataFrame, spec: DatasetSpec, pcfg: PublicationDataConfig) -> pd.DataFrame:
    if pcfg.split_mode == "temporal_per_user" and "timestamp" not in df.columns:
        raise ValueError(f"{pcfg.dataset} has no timestamp; temporal split is unavailable")
    cfg = core.DataConfig(
        data_dir=pcfg.data_root,
        rating_min=spec.rating_min,
        rating_max=spec.rating_max,
        require_explicit_rating_scale=True,
        split_mode=pcfg.split_mode,
        train_fraction=pcfg.train_fraction,
        val_fraction=pcfg.val_fraction,
        test_fraction=pcfg.test_fraction,
        split_seed=pcfg.split_seed,
        min_train_per_user=pcfg.min_train_per_user,
        enforce_train_support_for_all_items=pcfg.enforce_train_support_for_all_items,
    )
    return core.make_safe_split(df, cfg)


# =============================================================================
# 2. Sparse MatrixData equivalent for ML-10M-scale proposed-model execution
# =============================================================================


@dataclass
class SparseMatrixData:
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
    dataset_name: str
    rating_step: Optional[float]
    # Compatibility placeholders. PaperMBMF only uses train_mask.device.
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
    # CPU arrays for reproducibility / metrics.
    train_users_np: np.ndarray
    train_items_np: np.ndarray
    train_ratings_np: np.ndarray
    val_users_np: np.ndarray
    val_items_np: np.ndarray
    val_ratings_np: np.ndarray
    test_users_np: np.ndarray
    test_items_np: np.ndarray
    test_ratings_np: np.ndarray
    _sparse_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict, repr=False)


MatrixLike = Union[core.MatrixData, SparseMatrixData]



def choose_backend(num_users: int, num_items: int, cfg: PublicationDataConfig) -> Literal["dense", "sparse"]:
    if cfg.force_backend != "auto":
        return cfg.force_backend
    return "dense" if num_users * num_items <= cfg.dense_cell_limit else "sparse"



def build_sparse_matrix_data(split: pd.DataFrame, spec: DatasetSpec, device: torch.device) -> SparseMatrixData:
    users = np.sort(split.userId.unique()).astype(np.int64)
    items = np.sort(split.movieId.unique()).astype(np.int64)
    u2i = {int(v): i for i, v in enumerate(users)}
    i2i = {int(v): i for i, v in enumerate(items)}
    w = split.copy()
    w["u"] = w.userId.map(u2i).astype(np.int64)
    w["i"] = w.movieId.map(i2i).astype(np.int64)
    if w.duplicated(["u", "i"]).any():
        raise RuntimeError("duplicate user-item pairs after canonical preprocessing")
    rmin, rmax = float(spec.rating_min), float(spec.rating_max)
    mid, half = (rmin + rmax) / 2.0, (rmax - rmin) / 2.0

    def take(name: str):
        p = w[w.split == name]
        un = p.u.to_numpy(np.int64, copy=True)
        it = p.i.to_numpy(np.int64, copy=True)
        yy = p.rating.to_numpy(np.float32, copy=True)
        return (
            torch.as_tensor(un, dtype=torch.long, device=device),
            torch.as_tensor(it, dtype=torch.long, device=device),
            torch.as_tensor(yy, dtype=torch.float32, device=device),
            un, it, yy,
        )

    tu, ti, ty, tun, tin, tyn = take("train")
    vu, vi, vy, vun, vin, vyn = take("val")
    eu, ei, ey, eun, ein, eyn = take("test")
    n, m = len(users), len(items)
    sums = torch.zeros(n, dtype=torch.float32, device=device)
    cnt = torch.zeros(n, dtype=torch.float32, device=device)
    sums.scatter_add_(0, tu, ty)
    cnt.scatter_add_(0, tu, torch.ones_like(ty))
    gm = ty.mean()
    means = torch.full((n,), float(gm.item()), device=device)
    ok = cnt > 0
    means[ok] = sums[ok] / cnt[ok]
    # One-element placeholders preserve core helper compatibility without materialising NxM.
    ph = torch.empty(0, dtype=torch.float32, device=device)
    return SparseMatrixData(
        n, m, u2i, i2i, users, items, rmin, rmax, mid, half, spec.name, spec.rating_step,
        ph, ph,
        tu, ti, vu, vi, eu, ei, ty, vy, ey, means,
        tun, tin, tyn, vun, vin, vyn, eun, ein, eyn,
    )



def build_publication_matrix_data(split: pd.DataFrame, spec: DatasetSpec, pcfg: PublicationDataConfig, device: torch.device) -> Tuple[MatrixLike, str]:
    n, m = split.userId.nunique(), split.movieId.nunique()
    backend = choose_backend(n, m, pcfg)
    if backend == "dense":
        dcfg = core.DataConfig(
            rating_min=spec.rating_min, rating_max=spec.rating_max, require_explicit_rating_scale=True,
            split_mode=pcfg.split_mode, train_fraction=pcfg.train_fraction, val_fraction=pcfg.val_fraction,
            test_fraction=pcfg.test_fraction, split_seed=pcfg.split_seed,
            min_train_per_user=pcfg.min_train_per_user,
            enforce_train_support_for_all_items=pcfg.enforce_train_support_for_all_items,
        )
        d = core.build_matrix_data(split, dcfg, device)
        # Publication metadata only; no v7 scientific state or model logic is modified.
        setattr(d, "dataset_name", spec.name)
        setattr(d, "rating_step", spec.rating_step)
        return d, backend
    return build_sparse_matrix_data(split, spec, device), backend



def _csr_from_pairs(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
    order = np.lexsort((cols, rows))
    rs = rows[order].astype(np.int64, copy=False)
    cs = cols[order].astype(np.int64, copy=False)
    vs = vals[order].astype(np.float32, copy=False)
    counts = np.bincount(rs, minlength=shape[0]).astype(np.int64)
    crow = np.empty(shape[0] + 1, dtype=np.int64)
    crow[0] = 0
    np.cumsum(counts, out=crow[1:])
    return torch.sparse_csr_tensor(
        torch.from_numpy(crow).to(device), torch.from_numpy(cs.copy()).to(device),
        torch.from_numpy(vs.copy()).to(device), size=shape, dtype=torch.float32, device=device,
    )



def sparse_encoder_matrices(data: SparseMatrixData, space: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return user-values,user-mask,item-values,item-mask CSR matrices.

    Values match core.build_encoder_inputs exactly: centered_raw is divided by half_range for encoder
    input only; legacy_unit maps to [0,1].
    """
    key_u = (space, "user")
    key_i = (space, "item")
    if key_u in data._sparse_cache and key_i in data._sparse_cache:
        uv, um = data._sparse_cache[key_u]
        iv, im = data._sparse_cache[key_i]
        return uv, um, iv, im
    y = data.train_ratings_np.astype(np.float32)
    if space == "centered_raw":
        ev = ((y - data.midpoint) / data.half_range).astype(np.float32)
    elif space == "legacy_unit":
        ev = ((y - data.rating_min) / (data.rating_max - data.rating_min)).astype(np.float32)
    else:
        raise ValueError(space)
    ones = np.ones_like(ev, dtype=np.float32)
    dev = data.train_users.device
    uv = _csr_from_pairs(data.train_users_np, data.train_items_np, ev, (data.num_users, data.num_items), dev)
    um = _csr_from_pairs(data.train_users_np, data.train_items_np, ones, (data.num_users, data.num_items), dev)
    iv = _csr_from_pairs(data.train_items_np, data.train_users_np, ev, (data.num_items, data.num_users), dev)
    im = _csr_from_pairs(data.train_items_np, data.train_users_np, ones, (data.num_items, data.num_users), dev)
    data._sparse_cache[key_u] = (uv, um)
    data._sparse_cache[key_i] = (iv, im)
    return uv, um, iv, im



def _sparse_first_linear(linear: nn.Linear, values: torch.Tensor, mask: torch.Tensor, base_dim: int, input_mode: str) -> torch.Tensor:
    w = linear.weight
    out = torch.sparse.mm(values, w[:, :base_dim].T)
    if input_mode == "values_plus_mask":
        out = out + torch.sparse.mm(mask, w[:, base_dim:].T)
    elif input_mode != "values_only":
        raise ValueError(input_mode)
    if linear.bias is not None:
        out = out + linear.bias
    return out



def _sparse_ae_encode(enc: core.AEEncoder, values: torch.Tensor, mask: torch.Tensor, base_dim: int, input_mode: str) -> torch.Tensor:
    return enc.fc2(torch.relu(_sparse_first_linear(enc.fc1, values, mask, base_dim, input_mode)))



def _sparse_two_stage(ex: core.TwoStageFeatureExtractor, values: torch.Tensor, mask: torch.Tensor, base_dim: int, input_mode: str) -> torch.Tensor:
    h = torch.relu(_sparse_first_linear(ex.encoder1, values, mask, base_dim, input_mode))
    q = ex.encoder2(h)
    return ex.decoder1(q)



def _sparse_residual_encode(branch: core.ResidualRadiusAEBranch, values: torch.Tensor, mask: torch.Tensor, base_dim: int, input_mode: str) -> torch.Tensor:
    h = torch.relu(_sparse_first_linear(branch.encoder1, values, mask, base_dim, input_mode))
    return branch.encoder2(h)


class ScalableThesisMethod1(nn.Module):
    """Sparse-input execution of core.ThesisMethod1 with the same modules/equations/state layout."""
    def __init__(self, data: SparseMatrixData, cfg: core.Method1Config, device: torch.device):
        super().__init__(); self.cfg = cfg; self.half_range = data.half_range
        uin, iin = core.input_dims(data, cfg.input_mode)
        self.encoder_user = core.AEEncoder(uin, cfg.hidden_dim, cfg.latent_dim)
        self.decoder_user = core.PairDecoder(cfg.latent_dim, cfg.hidden_dim, data.num_items)
        self.encoder_item = core.AEEncoder(iin, cfg.hidden_dim, cfg.latent_dim)
        self.decoder_item = core.PairDecoder(cfg.latent_dim, cfg.hidden_dim, data.num_users)
        self.Phi = nn.Parameter(core.initialise_angles(data.num_users, cfg.latent_dim, cfg.angle_init, device))
        self.Theta = nn.Parameter(core.initialise_angles(data.num_items, cfg.latent_dim, cfg.angle_init, device))
        self.to(device)

    def encoded(self, data: SparseMatrixData):
        uv, um, iv, im = sparse_encoder_matrices(data, self.cfg.model_space)
        zu = _sparse_ae_encode(self.encoder_user, uv, um, data.num_items, self.cfg.input_mode)
        zi = _sparse_ae_encode(self.encoder_item, iv, im, data.num_users, self.cfg.input_mode)
        bs = core.model_pair_bound_scale(data, self.cfg.model_space)
        rW = core.latent_radius(zu, self.cfg.radius_mode, bs, self.cfg.radius_eps)
        rH = core.latent_radius(zi, self.cfg.radius_mode, bs, self.cfg.radius_eps)
        W = core.spherical_to_cartesian(self.Phi, rW); H = core.spherical_to_cartesian(self.Theta, rH)
        return zu, zi, rW, rH, W, H

    def predict_raw_pairs(self, data: SparseMatrixData, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        *_, W, H = self.encoded(data)
        return core.model_space_to_raw(data, torch.sum(W[users] * H[items], dim=1), self.cfg.model_space)

    @torch.no_grad()
    def diagnostics(self, data: SparseMatrixData, users: torch.Tensor, items: torch.Tensor) -> Dict[str, float]:
        _, _, rW, rH, W, H = self.encoded(data)
        p = torch.sum(W[users] * H[items], dim=1); b = rW[users,0]*rH[items,0]
        return {"max_structural_bound_violation": float(torch.relu(torch.abs(p)-b).max().item()) if b.numel() else 0.0,
                "mean_rW": float(rW.mean().item()), "mean_rH": float(rH.mean().item()),
                "max_pair_bound_eval": float(b.max().item()) if b.numel() else float("nan")}


class ScalableThesisMethod2(nn.Module):
    """Sparse-input execution of core.ThesisMethod2 with unchanged equations."""
    def __init__(self, data: SparseMatrixData, cfg: core.Method2Config, device: torch.device):
        super().__init__(); self.cfg = cfg; k = cfg.latent_dim
        uin, iin = core.input_dims(data, cfg.input_mode)
        self.user_features = core.TwoStageFeatureExtractor(uin, k)
        self.item_features = core.TwoStageFeatureExtractor(iin, k)
        scale = core.model_pair_bound_scale(data, cfg.model_space)
        self.item_counterpart = core.PositiveCounterpartRadii(data.num_items,k,cfg.counterpart_radius_mode,scale,cfg.radius_eps,cfg.bounded_counterpart_init_fraction)
        self.user_counterpart = core.PositiveCounterpartRadii(data.num_users,k,cfg.counterpart_radius_mode,scale,cfg.radius_eps,cfg.bounded_counterpart_init_fraction)
        self.Phi_user = nn.Parameter(core.initialise_angles(data.num_users,k,cfg.angle_init,device))
        self.Theta_user = nn.Parameter(core.initialise_angles(data.num_items,k,cfg.angle_init,device))
        self.Phi_item = nn.Parameter(core.initialise_angles(data.num_items,k,cfg.angle_init,device))
        self.Theta_item = nn.Parameter(core.initialise_angles(data.num_users,k,cfg.angle_init,device))
        self.to(device)

    def factors(self, data: SparseMatrixData):
        uv, um, iv, im = sparse_encoder_matrices(data, self.cfg.model_space)
        hu = _sparse_two_stage(self.user_features,uv,um,data.num_items,self.cfg.input_mode)
        hi = _sparse_two_stage(self.item_features,iv,im,data.num_users,self.cfg.input_mode)
        scale=core.model_pair_bound_scale(data,self.cfg.model_space)
        a=core.latent_radius(hu,self.cfg.radius_mode,scale,self.cfg.radius_eps); b=self.item_counterpart()
        c=core.latent_radius(hi,self.cfg.radius_mode,scale,self.cfg.radius_eps); d=self.user_counterpart()
        WU=core.spherical_to_cartesian(self.Phi_user,a); HU=core.spherical_to_cartesian(self.Theta_user,b)
        WI=core.spherical_to_cartesian(self.Phi_item,c); HI=core.spherical_to_cartesian(self.Theta_item,d)
        return a,b,c,d,WU,HU,WI,HI

    def predict_views(self,data:SparseMatrixData,u:torch.Tensor,i:torch.Tensor):
        a,b,c,d,WU,HU,WI,HI=self.factors(data)
        pU=torch.sum(WU[u]*HU[i],dim=1); pI=torch.sum(WI[i]*HI[u],dim=1)
        g=float(self.cfg.fusion_weight_user_view)
        if not 0<=g<=1: raise ValueError("fusion_weight_user_view must be in [0,1]")
        return pU,pI,g*pU+(1-g)*pI,(a,b,c,d)

    def predict_raw_pairs(self,data:SparseMatrixData,u:torch.Tensor,i:torch.Tensor)->torch.Tensor:
        return core.model_space_to_raw(data,self.predict_views(data,u,i)[2],self.cfg.model_space)

    @torch.no_grad()
    def diagnostics(self,data:SparseMatrixData,u:torch.Tensor,i:torch.Tensor)->Dict[str,float]:
        pU,pI,p,(a,b,c,d)=self.predict_views(data,u,i); g=float(self.cfg.fusion_weight_user_view)
        bU=a[u,0]*b[i,0]; bI=c[i,0]*d[u,0]; bf=g*bU+(1-g)*bI
        return {"max_user_view_bound_violation":float(torch.relu(torch.abs(pU)-bU).max().item()) if bU.numel() else 0.0,
                "max_item_view_bound_violation":float(torch.relu(torch.abs(pI)-bI).max().item()) if bI.numel() else 0.0,
                "max_structural_bound_violation":float(torch.relu(torch.abs(p)-bf).max().item()) if bf.numel() else 0.0,
                "max_pair_bound_eval":float(bf.max().item()) if bf.numel() else float("nan")}


class ScalableProposedMethod3(nn.Module):
    """Sparse-input execution of core.ProposedMethod3 with unchanged equations."""
    def __init__(self,data:SparseMatrixData,cfg:core.Method3Config,device:torch.device):
        super().__init__(); self.cfg=cfg
        uin,iin=core.input_dims(data,cfg.input_mode); rW0,rH0,pinfo=core.method3_initial_radii(data,cfg)
        self.register_buffer("rW0",rW0.to(device)); self.register_buffer("rH0",rH0.to(device)); self.prior_info=pinfo
        self.user_branch=core.ResidualRadiusAEBranch(uin,data.num_items,cfg.latent_dim)
        self.item_branch=core.ResidualRadiusAEBranch(iin,data.num_users,cfg.latent_dim)
        self.Phi=nn.Parameter(core.initialise_angles(data.num_users,cfg.latent_dim,cfg.angle_init,device))
        self.Theta=nn.Parameter(core.initialise_angles(data.num_items,cfg.latent_dim,cfg.angle_init,device)); self.to(device)

    def encoded(self,data:SparseMatrixData):
        uv,um,iv,im=sparse_encoder_matrices(data,self.cfg.model_space)
        zu=_sparse_residual_encode(self.user_branch,uv,um,data.num_items,self.cfg.input_mode)
        zi=_sparse_residual_encode(self.item_branch,iv,im,data.num_users,self.cfg.input_mode)
        du=core._bounded_log_correction(self.user_branch.raw_log_correction(zu),self.cfg)
        di=core._bounded_log_correction(self.item_branch.raw_log_correction(zi),self.cfg)
        rW=self.rW0*torch.exp(du); rH=self.rH0*torch.exp(di)
        W=core.spherical_to_cartesian(self.Phi,rW); H=core.spherical_to_cartesian(self.Theta,rH)
        return zu,zi,du,di,rW,rH,W,H

    def predict_raw_pairs(self,data:SparseMatrixData,u:torch.Tensor,i:torch.Tensor)->torch.Tensor:
        *_,W,H=self.encoded(data)
        return core.model_space_to_raw(data,torch.sum(W[u]*H[i],dim=1),self.cfg.model_space)

    @torch.no_grad()
    def diagnostics(self,data:SparseMatrixData,u:torch.Tensor,i:torch.Tensor)->Dict[str,float]:
        _,_,du,di,rW,rH,W,H=self.encoded(data); p=torch.sum(W[u]*H[i],dim=1); b=rW[u,0]*rH[i,0]
        return {"max_structural_bound_violation":float(torch.relu(torch.abs(p)-b).max().item()) if b.numel() else 0.0,
                "mean_prior_rW":float(self.rW0.mean().item()),"mean_prior_rH":float(self.rH0.mean().item()),
                "mean_rW":float(rW.mean().item()),"mean_rH":float(rH.mean().item()),
                "min_log_correction_user":float(du.min().item()),"max_log_correction_user":float(du.max().item()),
                "min_log_correction_item":float(di.min().item()),"max_log_correction_item":float(di.max().item()),
                "max_pair_bound_eval":float(b.max().item()) if b.numel() else float("nan"),**self.prior_info}


# =============================================================================
# 3. Chunked exact-objective training for sparse proposed models and paper baselines
# =============================================================================


def _chunks(n:int,chunk:int)->Iterator[slice]:
    for s in range(0,n,chunk): yield slice(s,min(n,s+chunk))


def _selected_pair_decoder_from_hidden(dec: core.PairDecoder, hidden: torch.Tensor, sample: torch.Tensor, output: torch.Tensor)->torch.Tensor:
    h=hidden[sample]; w=dec.fc2.weight[output]; b=dec.fc2.bias[output]
    return torch.sum(h*w,dim=1)+b


def train_scalable_method1(model:ScalableThesisMethod1,data:SparseMatrixData,cfg:core.Method1Config,chunk_size:int=250_000)->Dict[str,Any]:
    opt=torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay); best=float("inf"); best_state=copy.deepcopy(model.state_dict()); best_epoch=0; wait=0; hist=[]; t0=time.perf_counter(); N=data.train_users.numel()
    for ep in range(1,cfg.max_epochs+1):
        model.train(); opt.zero_grad(set_to_none=True)
        zu,zi,_,_,W,H=model.encoded(data); hu=model.decoder_user.hidden(zu); hi=model.decoder_item.hidden(zi)
        target_all=core.model_space_targets(data,data.train_ratings_raw,cfg.model_space)
        # Sum chunk means with N denominator -> exactly the full-batch MSE objective.
        sums={"u":0.0,"i":0.0,"m":0.0}
        slices=list(_chunks(N,chunk_size))
        for q,sl in enumerate(slices):
            u=data.train_users[sl]; i=data.train_items[sl]; y=target_all[sl]
            ru=_selected_pair_decoder_from_hidden(model.decoder_user,hu,u,i); ri=_selected_pair_decoder_from_hidden(model.decoder_item,hi,i,u); p=torch.sum(W[u]*H[i],dim=1)
            su=torch.sum((ru-y)**2); si=torch.sum((ri-y)**2); sm=torch.sum((p-y)**2)
            loss=(cfg.alpha_ae*(su+si)+cfg.beta_mbmf*sm)/N
            loss.backward(retain_graph=q < len(slices)-1)
            sums["u"]+=float(su.detach()); sums["i"]+=float(si.detach()); sums["m"]+=float(sm.detach())
        if cfg.grad_clip and cfg.grad_clip>0: torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
        opt.step()
        if ep==1 or ep%cfg.eval_every==0:
            val=evaluate_publication_model(model,data,"val",chunk_size=chunk_size)
            hist.append({"epoch":ep,"loss":cfg.alpha_ae*(sums["u"]+sums["i"])/N+cfg.beta_mbmf*sums["m"]/N,"val_RMSE":val["RMSE"]})
            if val["RMSE"]<best-1e-8: best=val["RMSE"]; best_state=copy.deepcopy(model.state_dict()); best_epoch=ep; wait=0
            else:
                wait+=1
                if wait>=cfg.val_patience: break
    model.load_state_dict(best_state)
    return {"best_epoch":best_epoch,"best_val_RMSE":best,"elapsed_sec":time.perf_counter()-t0,"history":hist}



def train_scalable_method2(model:ScalableThesisMethod2,data:SparseMatrixData,cfg:core.Method2Config,chunk_size:int=500_000)->Dict[str,Any]:
    opt=torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay); best=float("inf"); best_state=copy.deepcopy(model.state_dict()); best_epoch=0; wait=0; hist=[]; t0=time.perf_counter(); N=data.train_users.numel()
    for ep in range(1,cfg.max_epochs+1):
        model.train(); opt.zero_grad(set_to_none=True); a,b,c,d,WU,HU,WI,HI=model.factors(data); yall=core.model_space_targets(data,data.train_ratings_raw,cfg.model_space); g=float(cfg.fusion_weight_user_view); sp=sc=0.0; slices=list(_chunks(N,chunk_size))
        for q,sl in enumerate(slices):
            u=data.train_users[sl]; i=data.train_items[sl]; y=yall[sl]
            pU=torch.sum(WU[u]*HU[i],dim=1); pI=torch.sum(WI[i]*HI[u],dim=1); p=g*pU+(1-g)*pI
            s1=torch.sum((p-y)**2); s2=torch.sum((pU-pI)**2); loss=(s1+cfg.consistency_weight*s2)/N
            loss.backward(retain_graph=q<len(slices)-1); sp+=float(s1.detach()); sc+=float(s2.detach())
        if cfg.grad_clip and cfg.grad_clip>0: torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
        opt.step()
        if ep==1 or ep%cfg.eval_every==0:
            val=evaluate_publication_model(model,data,"val",chunk_size=chunk_size); hist.append({"epoch":ep,"loss":sp/N+cfg.consistency_weight*sc/N,"val_RMSE":val["RMSE"]})
            if val["RMSE"]<best-1e-8: best=val["RMSE"]; best_state=copy.deepcopy(model.state_dict()); best_epoch=ep; wait=0
            else:
                wait+=1
                if wait>=cfg.val_patience: break
    model.load_state_dict(best_state); return {"best_epoch":best_epoch,"best_val_RMSE":best,"elapsed_sec":time.perf_counter()-t0,"history":hist}



def _method3_mse_epoch_backward(model:ScalableProposedMethod3,data:SparseMatrixData,cfg:core.Method3Config,chunk_size:int)->Tuple[float,Dict[str,float]]:
    zu,zi,du,di,_,_,W,H=model.encoded(data); hu=torch.relu(model.user_branch.decoder_hidden(zu)); hi=torch.relu(model.item_branch.decoder_hidden(zi)); N=data.train_users.numel(); yall=core.model_space_targets(data,data.train_ratings_raw,cfg.model_space); su=si=sm=0.0; slices=list(_chunks(N,chunk_size))
    for q,sl in enumerate(slices):
        u=data.train_users[sl]; i=data.train_items[sl]; y=yall[sl]
        ru=torch.sum(hu[u]*model.user_branch.decoder_output.weight[i],dim=1)+model.user_branch.decoder_output.bias[i]
        ri=torch.sum(hi[i]*model.item_branch.decoder_output.weight[u],dim=1)+model.item_branch.decoder_output.bias[u]
        eu=(ru-y)**2; ei=(ri-y)**2
        if cfg.ae_loss_mode=="separate": ae=torch.sum(eu)+torch.sum(ei)
        elif cfg.ae_loss_mode=="averaged": ae=torch.sum((0.5*(ru+ri)-y)**2)
        else: raise ValueError(cfg.ae_loss_mode)
        pm=torch.sum(W[u]*H[i],dim=1); ms=torch.sum((pm-y)**2)
        loss=(cfg.alpha_ae*ae+cfg.beta_mbmf*ms)/N
        if q==0 and cfg.delta_l2:
            loss=loss+cfg.delta_l2*(torch.mean(du**2)+torch.mean(di**2))
        loss.backward(retain_graph=q<len(slices)-1); su+=float(torch.sum(eu).detach()); si+=float(torch.sum(ei).detach()); sm+=float(ms.detach())
    lae=(su+si)/N if cfg.ae_loss_mode=="separate" else float("nan")
    total=cfg.alpha_ae*lae+cfg.beta_mbmf*sm/N+cfg.delta_l2*float((torch.mean(du**2)+torch.mean(di**2)).detach())
    return total,{"L_user_AE":su/N,"L_item_AE":si/N,"L_MBM":sm/N}



def train_scalable_method3(model:ScalableProposedMethod3,data:SparseMatrixData,cfg:core.Method3Config,chunk_size:int=250_000)->Dict[str,Any]:
    if cfg.loss_form!="mse":
        # Legacy Frobenius reconstruction remains available through the original dense v6 path.
        # Avoid pretending a differently normalised large-data loss is identical.
        raise NotImplementedError("Sparse Method-3 backend currently supports loss_form='mse'; use dense v6 for Frobenius reproduction.")
    opt=torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay); best=float("inf"); best_state=copy.deepcopy(model.state_dict()); best_epoch=0; wait=0; hist=[]; t0=time.perf_counter()
    for ep in range(1,cfg.max_epochs+1):
        model.train(); opt.zero_grad(set_to_none=True); loss,parts=_method3_mse_epoch_backward(model,data,cfg,chunk_size)
        if cfg.grad_clip and cfg.grad_clip>0: torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
        opt.step()
        if ep==1 or ep%cfg.eval_every==0:
            val=evaluate_publication_model(model,data,"val",chunk_size=chunk_size); hist.append({"epoch":ep,"loss":loss,"val_RMSE":val["RMSE"],**parts})
            if val["RMSE"]<best-1e-8: best=val["RMSE"]; best_state=copy.deepcopy(model.state_dict()); best_epoch=ep; wait=0
            else:
                wait+=1
                if wait>=cfg.val_patience: break
    model.load_state_dict(best_state); return {"best_epoch":best_epoch,"best_val_RMSE":best,"elapsed_sec":time.perf_counter()-t0,"history":hist}



def train_paper_mbmf_chunked(model:core.PaperMBMF,data:SparseMatrixData,cfg:core.PaperBaselineConfig,chunk_size:int=500_000)->Dict[str,Any]:
    lr=float(cfg.initial_lr); best_state=copy.deepcopy(model.state_dict()); history=[]; prev_obj=None; small=0; t0=time.perf_counter(); N=data.train_users.numel()
    for ep in range(1,cfg.max_epochs+1):
        old_state=copy.deepcopy(model.state_dict()); model.zero_grad(set_to_none=True); W,H=model.factors(); slices=list(_chunks(N,chunk_size)); train_obj=0.0
        for q,sl in enumerate(slices):
            u=data.train_users[sl]; i=data.train_items[sl]; y=model.native_targets(data.train_ratings_raw[sl],u,i); e=torch.sum(W[u]*H[i],dim=1)-y; s=torch.sum(e**2); train_obj+=float(s.detach()); gl=0.5*s if cfg.objective_reduction=="sum" else 0.5*s/N; gl.backward(retain_graph=q<len(slices)-1)
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None: p.add_(p.grad,alpha=-lr)
        # Evaluate new objective exactly in chunks.
        with torch.no_grad():
            W2,H2=model.factors(); new_obj=0.0
            for sl in _chunks(N,chunk_size):
                u=data.train_users[sl]; i=data.train_items[sl]; y=model.native_targets(data.train_ratings_raw[sl],u,i); e=torch.sum(W2[u]*H2[i],dim=1)-y; new_obj+=float(torch.sum(e**2).item())
        if cfg.objective_reduction=="mean":
            new_obj/=N
            train_obj/=N
        accepted=math.isfinite(new_obj) and new_obj <= train_obj
        if accepted:
            best_state=copy.deepcopy(model.state_dict()); lr*=cfg.lr_increase
            if prev_obj is not None and abs(prev_obj-new_obj)<cfg.objective_tol: small+=1
            else: small=0
            prev_obj=new_obj
        else:
            model.load_state_dict(old_state); lr=max(lr*cfg.lr_decrease,cfg.min_lr); small=0
        if ep==1 or ep%cfg.eval_every==0: history.append({"epoch":ep,"objective":float(prev_obj if prev_obj is not None else new_obj),"lr":lr,"accepted":accepted})
        if small>=cfg.objective_patience or lr<=cfg.min_lr: break
    model.load_state_dict(best_state)
    return {"best_epoch":history[-1]["epoch"] if history else ep,"elapsed_sec":time.perf_counter()-t0,"history":history,"final_lr":lr}


# =============================================================================
# 4. Evaluation metrics: rating accuracy, bounds, stability, ranking, calibration
# =============================================================================


@dataclass
class MetricConfig:
    ranking_ks: Tuple[int,...]=(1,5,10,20,50)
    # AEMC-NE explicit-rating protocol: rank the held-out ratings for each user with graded relevance.
    compute_testset_ndcg: bool=True
    compute_binary_ranking: bool=True
    relevance_threshold_mode: Literal["train_user_mean","fixed"]="train_user_mean"
    fixed_relevance_threshold: Optional[float]=None
    compute_paper_f1_reproduction: bool=False
    prediction_chunk_size:int=500_000
    save_test_predictions:bool=True
    compute_pearson_spearman:bool=True
    bootstrap_samples:int=0



def _split_tensors(data:MatrixLike,split:str):
    if split=="train": return data.train_users,data.train_items,data.train_ratings_raw
    if split=="val": return data.val_users,data.val_items,data.val_ratings_raw
    if split=="test": return data.test_users,data.test_items,data.test_ratings_raw
    raise ValueError(split)



def _predict_raw_chunked(model:nn.Module,data:MatrixLike,u:torch.Tensor,i:torch.Tensor,chunk_size:int)->torch.Tensor:
    # Proposed models recompute encodings/factors per call. To avoid that per chunk, call once for reasonable
    # held-out vectors; for very large train split, subclasses below are still memory-safe because result is 1-D.
    with torch.no_grad():
        if u.numel() <= chunk_size:
            return model.predict_raw_pairs(u, i) if isinstance(model, core.PaperMBMF) else model.predict_raw_pairs(data, u, i)
        pieces = []
        for sl in _chunks(int(u.numel()), chunk_size):
            pieces.append(model.predict_raw_pairs(u[sl], i[sl]) if isinstance(model, core.PaperMBMF)
                          else model.predict_raw_pairs(data, u[sl], i[sl]))
        return torch.cat(pieces)



def _rankdata_average(x:np.ndarray)->np.ndarray:
    order=np.argsort(x,kind="mergesort"); ranks=np.empty(len(x),dtype=float); sx=x[order]; k=0
    while k<len(x):
        j=k+1
        while j<len(x) and sx[j]==sx[k]: j+=1
        ranks[order[k:j]]=(k+j-1)/2+1; k=j
    return ranks



def _corr_metrics(y:np.ndarray,p:np.ndarray)->Dict[str,float]:
    out={"Pearson":float("nan"),"Spearman":float("nan")}
    if len(y)<2 or np.std(y)==0 or np.std(p)==0: return out
    out["Pearson"]=float(np.corrcoef(y,p)[0,1]); out["Spearman"]=float(np.corrcoef(_rankdata_average(y),_rankdata_average(p))[0,1]); return out



def _ordinal_uniform_nmae_denominator(values:np.ndarray)->Optional[float]:
    uniq=np.unique(values)
    if len(uniq)<2: return None
    # Deep-MMMF's reported denominator corresponds to E|X-Y| for uniform integer rating levels 1..R.
    # Generalise to equally spaced observed/documented levels: mean pairwise absolute distance under uniform levels.
    dif=np.abs(uniq[:,None]-uniq[None,:]); return float(dif.mean())



def regression_metrics_extended(pred:np.ndarray,true:np.ndarray,rmin:float,rmax:float,*,ordinal_levels:Optional[np.ndarray]=None)->Dict[str,float]:
    e=pred-true; ae=np.abs(e); se=e*e; below=np.maximum(rmin-pred,0.0); above=np.maximum(pred-rmax,0.0); viol=below+above; width=rmax-rmin
    out={
        "MSE":float(se.mean()),"RMSE":float(np.sqrt(se.mean())),"MAE":float(ae.mean()),
        "MedianAE":float(np.median(ae)),"P90AE":float(np.quantile(ae,.90)),"P95AE":float(np.quantile(ae,.95)),"P99AE":float(np.quantile(ae,.99)),
        "NMAE_range":float(ae.mean()/width),"NRMSE_range":float(np.sqrt(se.mean())/width),
        "Global_OOB_Rate":float(np.mean(viol>0)),"Mean_OOB_Violation":float(viol.mean()),"Max_OOB_Violation":float(viol.max(initial=0.0)),
        "Bias_MeanError":float(e.mean()),"Prediction_Mean":float(pred.mean()),"Prediction_Std":float(pred.std(ddof=0)),
    }
    if ordinal_levels is not None:
        den=_ordinal_uniform_nmae_denominator(np.asarray(ordinal_levels,dtype=float)); out["NMAE_ordinal_uniform"] = float(ae.mean()/den) if den and den>0 else float("nan")
    out.update(_corr_metrics(true,pred)); return out



def _dcg(rels:np.ndarray,k:int)->float:
    r=np.asarray(rels,dtype=float)[:k]
    if r.size==0:return float("nan")
    return float(np.sum(r/np.log2(np.arange(2,2+r.size))))



def testset_ranking_metrics(users:np.ndarray,true:np.ndarray,pred:np.ndarray,ks:Sequence[int],train_user_mean:np.ndarray,*,rating_min:float,fixed_threshold:Optional[float]=None)->Dict[str,float]:
    """Metrics on each user's held-out explicit ratings only.

    NDCG uses the explicit rating shifted by the documented dataset minimum when needed,
    preserving global graded-relevance differences while keeping relevance nonnegative.
    Binary Precision/Recall/F1/MAP/MRR use train-user-mean relevance unless fixed_threshold is given.
    This protocol does NOT pretend unobserved catalog entries are negatives.
    """
    acc:Dict[str,List[float]]=defaultdict(list)
    # A stable grouping is O(N log N), instead of rescanning all N interactions
    # for each user (quadratic work on MovieLens-10M). Ranking ties remain stable.
    sorted_idx = np.argsort(users, kind="stable")
    sorted_users = users[sorted_idx]
    unique_users, starts = np.unique(sorted_users, return_index=True)
    ends = np.r_[starts[1:], len(sorted_idx)]
    for u, start, end in zip(unique_users, starts, ends):
        idx = sorted_idx[start:end]
        yt=true[idx]; pp=pred[idx]; order=np.argsort(-pp,kind="stable"); ideal=np.argsort(-yt,kind="stable")
        rel_grade=yt-min(float(rating_min),0.0)
        th=float(train_user_mean[int(u)] if fixed_threshold is None else fixed_threshold); rel_bin=(yt>th).astype(np.int8)
        for k in ks:
            kk=min(k,len(idx)); dcg=_dcg(rel_grade[order],kk); idcg=_dcg(rel_grade[ideal],kk); acc[f"NDCG_testset@{k}"].append(0.0 if not idcg or not np.isfinite(idcg) else dcg/idcg)
            top=order[:kk]; tp=int(rel_bin[top].sum()); denom_rel=int(rel_bin.sum()); prec=tp/kk if kk else 0.0; rec=tp/denom_rel if denom_rel else 0.0
            acc[f"Precision_testset@{k}"].append(prec); acc[f"Recall_testset@{k}"].append(rec); acc[f"F1_testset@{k}"].append(2*prec*rec/(prec+rec) if prec+rec else 0.0)
            # Average precision among this ranked held-out list.
            hits=rel_bin[order[:kk]]; c=np.cumsum(hits); ap=float(np.sum((c/np.arange(1,kk+1))*hits)/min(max(denom_rel,1),kk)) if kk else 0.0; acc[f"MAP_testset@{k}"].append(ap)
            pos=np.flatnonzero(rel_bin[order]>0); acc[f"MRR_testset@{k}"].append(1.0/(pos[0]+1) if pos.size and pos[0]<k else 0.0)
    return {k:float(np.mean(v)) if v else float("nan") for k,v in acc.items()}



def mbmf_paper_f1_reproduction(users:np.ndarray,true:np.ndarray,pred:np.ndarray)->float:
    """Literal-ish reproduction metric from MBMF text: user average true rating as threshold.

    This uses held-out truths to define the threshold and is therefore *not* a deployment-safe primary metric.
    It is deliberately named reproduction-only.
    """
    vals=[]
    for u in np.unique(users):
        idx=np.flatnonzero(users==u); yt=true[idx]; pp=pred[idx]
        if len(idx)==0:continue
        th=float(yt.mean()); a=yt>th; b=pp>th; tp=np.sum(a&b); fp=np.sum(~a&b); fn=np.sum(a&~b); pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0; vals.append(2*pr*rc/(pr+rc) if pr+rc else 0.0)
    return float(np.mean(vals)) if vals else float("nan")



def individual_bound_arrays(model:nn.Module,data:MatrixLike,u:torch.Tensor,i:torch.Tensor)->Optional[Tuple[np.ndarray,np.ndarray]]:
    """Return raw-rating-space lower/upper structural bounds for MBMF/Ours, else None."""
    with torch.no_grad():
        if isinstance(model,core.PaperMBMF):
            b=(model.rW[u,0]*model.rH[i,0]); variant=model.variant
            if variant=="MBMF-F": lo=data.midpoint-b; hi=data.midpoint+b
            elif variant=="MBMF-C": cmin=float(model.data_meta[2]); lo=torch.full_like(b,cmin); hi=cmin+2*b
            elif variant=="MBMF-N": lo=torch.zeros_like(b); hi=b
            else:return None
        elif isinstance(model,(core.ThesisMethod1,ScalableThesisMethod1)):
            _,_,rw,rh,_,_=model.encoded(data); b=rw[u,0]*rh[i,0]
            if model.cfg.model_space=="centered_raw": lo=data.midpoint-b; hi=data.midpoint+b
            else:
                w=data.rating_max-data.rating_min; lo=data.rating_min-w*b; hi=data.rating_min+w*b
        elif isinstance(model,(core.ThesisMethod2,ScalableThesisMethod2)):
            if isinstance(model,core.ThesisMethod2): pU,pI,p,(a,b,c,d)=model.predict_centered_views(data,u,i)
            else: pU,pI,p,(a,b,c,d)=model.predict_views(data,u,i)
            g=float(model.cfg.fusion_weight_user_view); bb=g*a[u,0]*b[i,0]+(1-g)*c[i,0]*d[u,0]
            if model.cfg.model_space=="centered_raw": lo=data.midpoint-bb; hi=data.midpoint+bb
            else:
                w=data.rating_max-data.rating_min; lo=data.rating_min-w*bb; hi=data.rating_min+w*bb
        elif isinstance(model,(core.ProposedMethod3,ScalableProposedMethod3)):
            _,_,_,_,rw,rh,_,_=model.encoded(data); b=rw[u,0]*rh[i,0]
            if model.cfg.model_space=="centered_raw": lo=data.midpoint-b; hi=data.midpoint+b
            else:
                w=data.rating_max-data.rating_min; lo=data.rating_min-w*b; hi=data.rating_min+w*b
        else:return None
    return lo.detach().cpu().numpy(),hi.detach().cpu().numpy()



def bound_metrics(pred:np.ndarray,bounds:Optional[Tuple[np.ndarray,np.ndarray]])->Dict[str,float]:
    if bounds is None:return {"IndividualBoundViolationRate":float("nan"),"MeanIndividualBoundViolation":float("nan"),"MaxIndividualBoundViolation":float("nan"),"MeanBoundWidth":float("nan")}
    lo,hi=bounds; v=np.maximum(lo-pred,0)+np.maximum(pred-hi,0)
    return {"IndividualBoundViolationRate":float(np.mean(v>1e-7)),"MeanIndividualBoundViolation":float(v.mean()),"MaxIndividualBoundViolation":float(v.max(initial=0.0)),"MeanBoundWidth":float(np.mean(hi-lo))}



def train_threshold_f1_fast_v8(users: np.ndarray, true: np.ndarray, pred: np.ndarray,
                               train_user_mean: np.ndarray) -> Dict[str, float]:
    """Same train-derived macro/micro F1 definition as the frozen V7 metric,
    but linear in held-out edge count instead of O(edges * active_users)."""
    u = np.asarray(users, dtype=np.int64)
    yt = np.asarray(true) > train_user_mean[u]
    yp = np.asarray(pred) > train_user_mean[u]
    tp = np.bincount(u, weights=(yt & yp).astype(np.int32), minlength=len(train_user_mean))
    fp = np.bincount(u, weights=((~yt) & yp).astype(np.int32), minlength=len(train_user_mean))
    fn = np.bincount(u, weights=(yt & (~yp)).astype(np.int32), minlength=len(train_user_mean))
    active = np.unique(u)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    macro = np.mean(2 * precision[active] * recall[active] / np.maximum(precision[active] + recall[active], 1e-12)) if len(active) else np.nan
    tpm, fpm, fnm = tp.sum(), fp.sum(), fn.sum()
    pm = tpm / max(tpm + fpm, 1)
    rm = tpm / max(tpm + fnm, 1)
    micro = 2 * pm * rm / max(pm + rm, 1e-12)
    return {"F1_trainThreshold_macro_user": float(macro), "F1_trainThreshold_micro": float(micro)}


def evaluate_publication_model(model:nn.Module,data:MatrixLike,split:Literal["train","val","test"],*,metric_cfg:Optional[MetricConfig]=None,chunk_size:Optional[int]=None,return_predictions:bool=False)->Union[Dict[str,float],Tuple[Dict[str,float],np.ndarray]]:
    mc=metric_cfg or MetricConfig(); u,i,y=_split_tensors(data,split); cs=chunk_size or mc.prediction_chunk_size
    model.eval(); preds=[]
    # Recompute global factors/encodings once where possible by calling model on the full pair vector.
    # 1-D output is much smaller than NxM and remains practical for ML-10M.
    with torch.no_grad():
        p=_predict_raw_chunked(model,data,u,i,cs)
    pred=p.detach().cpu().numpy().astype(np.float64); true=y.detach().cpu().numpy().astype(np.float64); users=u.detach().cpu().numpy()
    step=getattr(data,"rating_step",None)
    levels=(np.arange(data.rating_min,data.rating_max+step/2.0,step,dtype=float) if step is not None else None)
    metrics=regression_metrics_extended(pred,true,data.rating_min,data.rating_max,ordinal_levels=levels)
    # Preserve v6's leakage-safe F1 under the names already returned by v6.
    f1=train_threshold_f1_fast_v8(users,true,pred,data.user_train_mean_raw.detach().cpu().numpy()); metrics.update(f1)
    if mc.compute_testset_ndcg:
        metrics.update(testset_ranking_metrics(users,true,pred,mc.ranking_ks,data.user_train_mean_raw.detach().cpu().numpy(),rating_min=data.rating_min,fixed_threshold=mc.fixed_relevance_threshold if mc.relevance_threshold_mode=="fixed" else None))
    if mc.compute_paper_f1_reproduction:
        metrics["F1_MBMFPaper_reproduction_only"]=mbmf_paper_f1_reproduction(users,true,pred)
    metrics.update(bound_metrics(pred,individual_bound_arrays(model,data,u,i)))
    if isinstance(model, core.ProposedMethod7):
        # Eval mode disables dropout; identical point predictions with extra raw variance.
        with torch.no_grad():
            uncertainty = []
            for sl in _chunks(int(u.numel()), cs):
                pu, var = model.predict_raw_with_uncertainty(data, u[sl], i[sl])
                uncertainty.append(var.detach().cpu().numpy())
            variance = np.concatenate(uncertainty).astype(np.float64)
        metrics.update(gaussian_uncertainty_metrics_v8(pred, true, variance, data.rating_max - data.rating_min))
    return (metrics,pred) if return_predictions else metrics



def gaussian_uncertainty_metrics_v8(pred: np.ndarray, truth: np.ndarray,
                                    variance: np.ndarray, rating_width: float) -> Dict[str, float]:
    """Observed-label calibration diagnostics; Gaussian predictive intervals, unclipped.

    Variance is in the *raw rating* space, as emitted by frozen Method 7.
    These are descriptive test metrics, never training/selection objectives.
    """
    pp = np.asarray(pred, dtype=np.float64)
    yy = np.asarray(truth, dtype=np.float64)
    vv = np.asarray(variance, dtype=np.float64)
    if pp.shape != yy.shape or vv.shape != yy.shape or not np.all(np.isfinite(vv)) or np.any(vv <= 0):
        raise ValueError("invalid Method7 predictive raw variance")
    nll = 0.5 * (np.log(2.0 * np.pi * vv) + (yy - pp)**2 / vv)
    zscores = ((50, 0.6744897501960817), (80, 1.2815515655446004),
               (90, 1.6448536269514722), (95, 1.959963984540054))
    out = {"Gaussian_NLL": float(nll.mean())}
    for percent, z in zscores:
        half = z * np.sqrt(vv)
        covered = (yy >= pp - half) & (yy <= pp + half)
        out[f"PICP_{percent}"] = float(covered.mean())
        out[f"CoverageGap_{percent}"] = float(covered.mean() - percent/100.0)
        out[f"MPIW_{percent}"] = float(np.mean(2.0 * half))
        out[f"NormalizedMPIW_{percent}"] = float(np.mean(2.0 * half) / rating_width)
    return out


def prediction_stability(prediction_files:Sequence[Union[str,Path]])->Dict[str,float]:
    arr=[]; ids0=None
    for f in prediction_files:
        z=np.load(f); ids=z["pair_id"]; p=z["prediction"]
        if ids0 is None: ids0=ids
        elif not np.array_equal(ids0,ids): raise ValueError("prediction archives are not aligned")
        arr.append(p.astype(np.float64))
    if len(arr)<2:return {"AvgPredictionStd":float("nan"),"MaxPredictionStd":float("nan")}
    sd=np.std(np.stack(arr,axis=0),axis=0,ddof=1); return {"AvgPredictionStd":float(sd.mean()),"MaxPredictionStd":float(sd.max(initial=0.0))}


# =============================================================================
# 5. Memory/time instrumentation
# =============================================================================


class PeakMemoryMonitor:
    def __init__(self,device:torch.device,interval:float=0.05): self.device=device; self.interval=interval; self._stop=threading.Event(); self._thr=None; self.base=0; self.peak=0
    def _rss(self)->int:
        try:
            import psutil
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception as exc:
            # `resource` is a Unix-only Python module. Import it only when
            # psutil (the cross-platform dependency in requirements.txt) fails.
            # On Windows, an actionable error is preferable to an import error.
            if os.name == "nt":
                raise RuntimeError(
                    "Peak memory tracking on Windows requires a working psutil installation "
                    "(run: python -m pip install psutil)."
                ) from exc
            import resource
            # Linux ru_maxrss is KiB; macOS bytes. This fallback is a peak,
            # rather than a snapshot, exactly as in the original V8 code.
            v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(v * 1024 if sys.platform != "darwin" else v)
    def _loop(self):
        while not self._stop.is_set(): self.peak=max(self.peak,self._rss()); time.sleep(self.interval)
    def __enter__(self):
        gc.collect(); self.base=self._rss(); self.peak=self.base
        if self.device.type=="cuda": torch.cuda.synchronize(self.device); torch.cuda.reset_peak_memory_stats(self.device)
        self._thr=threading.Thread(target=self._loop,daemon=True); self._thr.start(); return self
    def __exit__(self,*exc):
        self._stop.set();
        if self._thr:self._thr.join(timeout=1.0)
        self.peak=max(self.peak,self._rss())
        if self.device.type=="cuda": torch.cuda.synchronize(self.device)
    def metrics(self)->Dict[str,float]:
        out={"PeakRSS_MB":self.peak/2**20,"PeakRSS_Delta_MB":max(0,self.peak-self.base)/2**20}
        if self.device.type=="cuda": out.update({"PeakGPUAllocated_MB":torch.cuda.max_memory_allocated(self.device)/2**20,"PeakGPUReserved_MB":torch.cuda.max_memory_reserved(self.device)/2**20})
        else:out.update({"PeakGPUAllocated_MB":0.0,"PeakGPUReserved_MB":0.0})
        return out


# =============================================================================
# 6. Checkpoint save/load (core + sparse + comparison baselines)
# =============================================================================


def _config_dict(obj:Any)->Dict[str,Any]: return asdict(obj) if dataclasses.is_dataclass(obj) else dict(obj)


def checkpoint_family(model:nn.Module)->str:
    pairs=[(core.PaperMBMF,"PaperMBMF"),(core.ThesisMethod1,"ThesisMethod1"),(ScalableThesisMethod1,"ScalableThesisMethod1"),(core.ThesisMethod2,"ThesisMethod2"),(ScalableThesisMethod2,"ScalableThesisMethod2"),(core.ProposedMethod3,"ProposedMethod3"),(ScalableProposedMethod3,"ScalableProposedMethod3"),(ItemAutoRec,"AutoRec"),(AEMCNE,"AEMCNE")]
    for cls,name in pairs:
        if isinstance(model,cls): return name
    return model.__class__.__name__



def data_signature(data:MatrixLike)->Dict[str,Any]:
    h=hashlib.sha256()
    for a in (data.train_users.detach().cpu().numpy(),data.train_items.detach().cpu().numpy(),data.train_ratings_raw.detach().cpu().numpy()): h.update(np.ascontiguousarray(a).view(np.uint8))
    return {"num_users":data.num_users,"num_items":data.num_items,"rating_min":data.rating_min,"rating_max":data.rating_max,"train_sha256":h.hexdigest()}



def save_model_checkpoint(path:Union[str,Path],model:nn.Module,*,config:Any,data:MatrixLike,train_info:Optional[Mapping[str,Any]]=None,extra:Optional[Mapping[str,Any]]=None)->Path:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    payload={"format":"MBMF-Q1-v8","v7_core_sha256":V7_CORE_SHA256,"family":checkpoint_family(model),"config":_config_dict(config),"data_signature":data_signature(data),"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"train_info":dict(train_info or {}),"extra":dict(extra or {})}
    if isinstance(model,core.PaperMBMF): payload.update({"variant":model.variant,"init_mode":payload["extra"].get("init_mode","isotropic_gaussian")})
    torch.save(payload,p); return p



def load_model_checkpoint(path:Union[str,Path],data:MatrixLike,device:torch.device,*,strict_data:bool=True)->Tuple[nn.Module,Dict[str,Any]]:
    ck=torch.load(path,map_location="cpu",weights_only=False)
    if ck.get("format")!="MBMF-Q1-v8": raise ValueError("unsupported checkpoint format")
    if strict_data:
        sig=data_signature(data); old=ck["data_signature"]
        for k in ("num_users","num_items","rating_min","rating_max","train_sha256"):
            if str(sig[k])!=str(old[k]): raise ValueError(f"checkpoint/data mismatch for {k}: {old[k]} vs {sig[k]}")
    fam=ck["family"]; c=ck["config"]
    if fam=="PaperMBMF": cfg=core.PaperBaselineConfig(**c); model=core.PaperMBMF(data,cfg,ck["variant"],ck.get("init_mode","isotropic_gaussian"),device)
    elif fam in ("ThesisMethod1","ScalableThesisMethod1"):
        cfg=core.Method1Config(**c); model=(ScalableThesisMethod1 if fam.startswith("Scalable") else core.ThesisMethod1)(data,cfg,device)
    elif fam in ("ThesisMethod2","ScalableThesisMethod2"):
        cfg=core.Method2Config(**c); model=(ScalableThesisMethod2 if fam.startswith("Scalable") else core.ThesisMethod2)(data,cfg,device)
    elif fam in ("ProposedMethod3","ScalableProposedMethod3"):
        cfg=core.Method3Config(**c); model=(ScalableProposedMethod3 if fam.startswith("Scalable") else core.ProposedMethod3)(data,cfg,device)
    elif fam=="BiasMF": cfg=BiasMFConfig(**c); model=BiasMF(data.num_users,data.num_items,cfg,device)
    elif fam=="AutoRec": cfg=AutoRecConfig(**c); model=ItemAutoRec(data,cfg,device)
    elif fam=="AEMCNE": cfg=AEMCNEConfig(**c); model=AEMCNE(data,cfg,device)
    else: raise ValueError(f"unknown checkpoint family {fam}")
    model.load_state_dict(ck["state_dict"]); model.to(device); model.eval(); return model,ck


# =============================================================================
# 7. Comparison baselines that are safe to implement directly
# =============================================================================


@dataclass
class BiasMFConfig:
    latent_dim:int=20; use_bias:bool=True; lr:float=0.01; weight_decay:float=1e-4; max_epochs:int=200; batch_size:int=262_144; eval_every:int=5; val_patience:int=10


class BiasMF(nn.Module):
    def __init__(self,n:int,m:int,cfg:BiasMFConfig,device:torch.device):
        super().__init__(); self.cfg=cfg; self.U=nn.Embedding(n,cfg.latent_dim); self.V=nn.Embedding(m,cfg.latent_dim); self.bu=nn.Embedding(n,1); self.bi=nn.Embedding(m,1); self.mu=nn.Parameter(torch.zeros(()),requires_grad=cfg.use_bias); nn.init.normal_(self.U.weight,std=.05); nn.init.normal_(self.V.weight,std=.05); nn.init.zeros_(self.bu.weight); nn.init.zeros_(self.bi.weight); self.to(device)
    def forward_pairs(self,u,i):
        p=(self.U(u)*self.V(i)).sum(1)
        if self.cfg.use_bias:p=p+self.mu+self.bu(u)[:,0]+self.bi(i)[:,0]
        return p
    def predict_raw_pairs(self,data,u,i): return self.forward_pairs(u,i)



def train_biasmf(model:BiasMF,data:MatrixLike,cfg:BiasMFConfig)->Dict[str,Any]:
    with torch.no_grad(): model.mu.fill_(float(data.train_ratings_raw.mean()))
    opt=torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay); N=data.train_users.numel(); best=float("inf"); state=copy.deepcopy(model.state_dict()); wait=0; be=0; hist=[]; t0=time.perf_counter()
    for ep in range(1,cfg.max_epochs+1):
        model.train(); perm=torch.randperm(N,device=data.train_users.device); total=0.0
        for sl in _chunks(N,cfg.batch_size):
            ix=perm[sl]; u=data.train_users[ix]; i=data.train_items[ix]; y=data.train_ratings_raw[ix]; opt.zero_grad(set_to_none=True); loss=F.mse_loss(model.forward_pairs(u,i),y); loss.backward(); opt.step(); total+=float(loss.detach())*len(ix)
        if ep==1 or ep%cfg.eval_every==0:
            val=evaluate_publication_model(model,data,"val"); hist.append({"epoch":ep,"loss":total/N,"val_RMSE":val["RMSE"]})
            if val["RMSE"]<best-1e-8:best=val["RMSE"];state=copy.deepcopy(model.state_dict());wait=0;be=ep
            else:
                wait+=1
                if wait>=cfg.val_patience:break
    model.load_state_dict(state); return {"best_epoch":be,"best_val_RMSE":best,"elapsed_sec":time.perf_counter()-t0,"history":hist}


def raw_item_csr(data:SparseMatrixData)->torch.Tensor:
    """Cached item x user raw-rating CSR used by item-based neural baselines."""
    key=("raw","item_values")
    hit=data._sparse_cache.get(key)
    if hit is not None:
        return hit[0]
    raw=_csr_from_pairs(data.train_items_np,data.train_users_np,data.train_ratings_np,(data.num_items,data.num_users),data.train_users.device)
    data._sparse_cache[key]=(raw,raw)
    return raw


@dataclass
class AutoRecConfig:
    hidden_dim:int=500; lambda_l2:float=1e-3; max_epochs:int=200; eval_every:int=5; val_patience:int=10; optimizer:Literal["rprop","adam"]="rprop"; lr:float=1e-3


class ItemAutoRec(nn.Module):
    """I-AutoRec: identity output, sigmoid hidden, observed-only reconstruction objective."""
    def __init__(self,data:MatrixLike,cfg:AutoRecConfig,device:torch.device):
        super().__init__(); self.cfg=cfg; self.encoder=nn.Linear(data.num_users,cfg.hidden_dim); self.decoder=nn.Linear(cfg.hidden_dim,data.num_users); self.to(device)
    def item_hidden(self,data:MatrixLike)->torch.Tensor:
        if isinstance(data,SparseMatrixData):
            uv,um,iv,im=sparse_encoder_matrices(data,"legacy_unit")
            # AutoRec operates in raw rating units with zero-filled missing entries; rebuild values CSR in raw space.
            raw=raw_item_csr(data)
            h=torch.sparse.mm(raw,self.encoder.weight.T)+self.encoder.bias
        else:
            raw=data.train_values_centered.T+data.midpoint*data.train_mask.T
            h=self.encoder(raw)
        return torch.sigmoid(h)
    def predict_raw_pairs(self,data,u,i):
        h=self.item_hidden(data); hh=h[i]; w=self.decoder.weight[u]; b=self.decoder.bias[u]; return torch.sum(hh*w,dim=1)+b



def train_autorec(model:ItemAutoRec,data:MatrixLike,cfg:AutoRecConfig,chunk_size:int=250_000)->Dict[str,Any]:
    opt=torch.optim.Rprop(model.parameters(),lr=0.01) if cfg.optimizer=="rprop" else torch.optim.Adam(model.parameters(),lr=cfg.lr); N=data.train_users.numel(); best=float("inf"); state=copy.deepcopy(model.state_dict()); wait=0;be=0;hist=[];t0=time.perf_counter()
    for ep in range(1,cfg.max_epochs+1):
        model.train(); opt.zero_grad(set_to_none=True); h=model.item_hidden(data); slices=list(_chunks(N,chunk_size)); sse=0.0
        for q,sl in enumerate(slices):
            u=data.train_users[sl];i=data.train_items[sl];y=data.train_ratings_raw[sl];p=torch.sum(h[i]*model.decoder.weight[u],dim=1)+model.decoder.bias[u]; ss=torch.sum((p-y)**2); loss=ss/N
            # λ/2 (||W||² + ||V||²), as in AutoRec, added once.
            if q==0: loss=loss+(0.5*cfg.lambda_l2/N)*(torch.sum(model.encoder.weight**2)+torch.sum(model.decoder.weight**2))
            loss.backward(retain_graph=q<len(slices)-1); sse+=float(ss.detach())
        opt.step()
        if ep==1 or ep%cfg.eval_every==0:
            val=evaluate_publication_model(model,data,"val");hist.append({"epoch":ep,"loss":sse/N,"val_RMSE":val["RMSE"]})
            if val["RMSE"]<best-1e-8:best=val["RMSE"];state=copy.deepcopy(model.state_dict());wait=0;be=ep
            else:
                wait+=1
                if wait>=cfg.val_patience:break
    model.load_state_dict(state);return {"best_epoch":be,"best_val_RMSE":best,"elapsed_sec":time.perf_counter()-t0,"history":hist}


@dataclass
class AEMCNEConfig:
    main_hidden_dim:int=900; element_hidden_dim:int=200; lambda_main:float=0.1; lambda_element:float=0.1; lr:float=1e-3; max_epochs:int=200; eval_every:int=5; val_patience:int=10


class AEMCNE(nn.Module):
    """Equation-faithful one-hidden-layer AEMC-NE reimplementation (ICLR 2024 Eq.5-6).

    The main matrix autoencoder uses a ReLU hidden layer and linear pre-output g_W.
    A shared scalar MLP h_Theta is applied element-wise to learn the output activation.
    Hyperparameter sizes are dataset configurable; use 500/900/1000 main hidden for ML100K/1M/10M.
    """
    def __init__(self,data:MatrixLike,cfg:AEMCNEConfig,device:torch.device):
        super().__init__();self.cfg=cfg;self.encoder=nn.Linear(data.num_users,cfg.main_hidden_dim);self.decoder=nn.Linear(cfg.main_hidden_dim,data.num_users);self.element1=nn.Linear(1,cfg.element_hidden_dim);self.element2=nn.Linear(cfg.element_hidden_dim,1);self.to(device)
    def item_hidden(self,data):
        if isinstance(data,SparseMatrixData):
            raw=raw_item_csr(data);return torch.relu(torch.sparse.mm(raw,self.encoder.weight.T)+self.encoder.bias)
        raw=data.train_values_centered.T+data.midpoint*data.train_mask.T;return torch.relu(self.encoder(raw))
    def _h(self,z):return self.element2(torch.relu(self.element1(z[:,None])))[:,0]
    def predict_raw_pairs(self,data,u,i):
        h=self.item_hidden(data);z=torch.sum(h[i]*self.decoder.weight[u],dim=1)+self.decoder.bias[u];return self._h(z)



def train_aemcne(model:AEMCNE,data:MatrixLike,cfg:AEMCNEConfig,chunk_size:int=200_000)->Dict[str,Any]:
    opt=torch.optim.Adam(model.parameters(),lr=cfg.lr);N=data.train_users.numel();best=float("inf");state=copy.deepcopy(model.state_dict());wait=0;be=0;hist=[];t0=time.perf_counter()
    for ep in range(1,cfg.max_epochs+1):
        model.train();opt.zero_grad(set_to_none=True);h=model.item_hidden(data);sse=0.0;slices=list(_chunks(N,chunk_size))
        for q,sl in enumerate(slices):
            u=data.train_users[sl];i=data.train_items[sl];y=data.train_ratings_raw[sl];z=torch.sum(h[i]*model.decoder.weight[u],dim=1)+model.decoder.bias[u];p=model._h(z);ss=torch.sum((p-y)**2);loss=ss/N
            if q==0:
                main=torch.sum(model.encoder.weight**2)+torch.sum(model.decoder.weight**2);elem=torch.sum(model.element1.weight**2)+torch.sum(model.element2.weight**2);loss=loss+(cfg.lambda_main*main+cfg.lambda_element*elem)/N
            loss.backward(retain_graph=q<len(slices)-1);sse+=float(ss.detach())
        opt.step()
        if ep==1 or ep%cfg.eval_every==0:
            val=evaluate_publication_model(model,data,"val");hist.append({"epoch":ep,"loss":sse/N,"val_RMSE":val["RMSE"]})
            if val["RMSE"]<best-1e-8:best=val["RMSE"];state=copy.deepcopy(model.state_dict());wait=0;be=ep
            else:
                wait+=1
                if wait>=cfg.val_patience:break
    model.load_state_dict(state);return {"best_epoch":be,"best_val_RMSE":best,"elapsed_sec":time.perf_counter()-t0,"history":hist}


# BMC-ADMM, Deep-MMMF and DRPRA have materially different solvers/protocols.  Rather than ship a
# superficially similar algorithm under their names, v7 uses an external prediction adapter for exact
# official/author implementations, while recording their literature protocols. This is safer for a Q1 paper.


class ExternalPredictionAdapter(nn.Module):
    """Integrate predictions from an official/external implementation on exactly our split.

    Expected CSV columns: split,userId,movieId,prediction. IDs are canonical integer IDs saved by v7.
    """
    def __init__(self,csv_path:Union[str,Path],data:MatrixLike,device:torch.device):
        super().__init__()
        df = pd.read_csv(csv_path)
        req = {"split", "userId", "movieId", "prediction"}
        if not req.issubset(df.columns):
            raise ValueError(f"external prediction CSV needs {sorted(req)}")
        if df[list(req)].isna().any().any():
            raise ValueError("external prediction CSV contains missing keys/predictions")
        if not set(df["split"].astype(str)).issubset({"train", "val", "test"}):
            raise ValueError("external CSV contains unknown split labels")
        if not np.isfinite(pd.to_numeric(df["prediction"], errors="raise")).all():
            raise ValueError("external CSV contains nonfinite predictions")
        for c in ("userId", "movieId"):
            vals = pd.to_numeric(df[c], errors="raise")
            if not np.all(vals == vals.astype(np.int64)):
                raise ValueError(f"external CSV {c} must be canonical integer IDs")
            df[c] = vals.astype(np.int64)
        if df.duplicated(["userId", "movieId"]).any():
            raise ValueError("external predictions have duplicated user/item pairs")
        self.tables = {}
        for sp in ("train", "val", "test"):
            got = df.loc[df.split == sp]
            expected_u, expected_i, _ = _split_tensors(data, sp)
            expected = set(zip(expected_u.detach().cpu().tolist(), expected_i.detach().cpu().tolist()))
            actual = set(zip(got.userId.tolist(), got.movieId.tolist()))
            if expected != actual:
                raise ValueError(f"external predictions do not match exact {sp} split: "
                                 f"missing={len(expected-actual)}, extra={len(actual-expected)}")
            self.tables[sp] = {(int(r.userId), int(r.movieId)): float(r.prediction)
                               for r in got.itertuples()}
        self.dummy=nn.Parameter(torch.zeros((),device=device),requires_grad=False)
    def predict_raw_pairs(self,data,u,i,split:Optional[str]=None):
        # Search all split tables if split not supplied; pair keys are disjoint by construction.
        uu=u.detach().cpu().numpy();ii=i.detach().cpu().numpy();out=[]
        for a,b in zip(uu,ii):
            key=(int(a),int(b));vals=[t[key] for t in self.tables.values() if key in t]
            if len(vals)!=1:raise KeyError(f"pair {key} not uniquely found in external predictions")
            out.append(vals[0])
        return torch.tensor(out,dtype=torch.float32,device=u.device)


# =============================================================================
# 8. Literature protocol registry / recommended experiments
# =============================================================================


LITERATURE_PROTOCOLS:Dict[str,Dict[str,Any]]={
    "MBMF_2020":{
        "metrics":["RMSE","MAE","F1_user_average_threshold","prediction_std","OOB/range diagnostics","runtime"],
        "ablations":["K in {10,20,50}","matrix density/sparsity","prediction fluctuation","runtime/scalability"],
        "notes":"Historical/present split; 5-fold present evaluation; paper max 500 iterations and objective-change early stop.",
    },
    "AutoRec_2015":{
        "datasets":["ML-1M","ML-10M","Netflix"],"metrics":["RMSE"],
        "protocol":"90/10 random train/test; 10% of train for tuning; 5 repeated splits; 95% CI <= ~0.003.",
        "ablations":["I- vs U-AutoRec","activation functions","hidden units k","deep 500-250-500 extension"],
        "search":{"lambda":[.001,.01,.1,1,100,1000],"k":[10,20,40,80,100,200,300,400,500]},
    },
    "BMC_ADMM_2017":{
        "datasets":["ML-100K","ML-10M","Flixster","Jester"],"metrics":["test RMSE","objective convergence","runtime/scalability"],
        "ranks":[5,10,30],"notes":"Best regularisation selected on random validation; rho1=rho2=1 in reported comparisons.",
    },
    "AEMC_NE_2024":{
        "datasets":["ML-100K","ML-1M","ML-10M","Douban","Flixster"],
        "metrics":["RMSE","NDCG@1","NDCG@5","NDCG@10","NDCG@50","runtime"],
        "protocol":"90/10 known-rating train/test; 5% of train for tuning; 20 random MovieLens splits.",
        "hidden":{"ML-100K":500,"ML-1M":900,"ML-10M":1000,"elementwise":200},
        "lambda_grid":[.01,.1,1,10,50,100,200,500],
        "synthetic":["MCAR/MNAR","missing rates 0.2..0.7","network-width and weight-decay ablations"],
    },
    "Deep_MMMF_2026":{
        "datasets":["ML-1M","EachMovie","Prime Pantry"],"metrics":["MAE","NMAE"],
        "protocol":"weak + strong generalization; randomized hyperparameter grid; each configuration 3 seeds; validation selection; final retrain/test.",
        "statistics":["Friedman alpha=0.05","post-hoc Nemenyi","critical-difference diagram"],
        "ablations":["dropout","regularization","embedding dimension","layers","layer sizes","learning rate"],
    },
    "DRPRA_2022":{
        "datasets":["ML-100K","MovieTweetings-10K","FilmTrust"],"metrics":["RMSE","MAE"],
        "train_ratios":[.90,.80,.70],"repeats":5,"notes":"Parameter impact experiments accompany the accuracy tables.",
    },
}


# =============================================================================
# 9. Publication experiment runner
# =============================================================================


# Default dataset-method matrix follows the comparison plan supplied for the paper.
# It controls only which experiments are scheduled; it does not alter any model equation.
METHOD_DATASET_MATRIX: Dict[str, Tuple[str, ...]] = {
    "BiasMF": ("synthetic","jester","bookcrossing","ml-100k","ml-1m","ml-10m"),
    "MF": ("synthetic","jester","bookcrossing","ml-100k","ml-1m","ml-10m"),
    "MBMF-F": ("synthetic","jester","bookcrossing","ml-100k","ml-1m","ml-10m"),
    "MBMF-C": ("synthetic","jester","bookcrossing","ml-100k","ml-1m","ml-10m"),
    "MBMF-N": ("synthetic","jester","bookcrossing","ml-100k","ml-1m","ml-10m"),
    "AutoRec": ("ml-100k","ml-1m","ml-10m"),
    "AEMC-NE": ("ml-100k","ml-1m","ml-10m"),
    "Method1": ("synthetic","jester","bookcrossing","ml-100k","ml-1m","ml-10m"),
    "Method2": ("synthetic","jester","bookcrossing","ml-100k","ml-1m","ml-10m"),
    "Method3": ("synthetic","jester","bookcrossing","ml-100k","ml-1m","ml-10m"),
}


def method_supported_on_dataset(method: str, dataset: str) -> bool:
    return dataset in METHOD_DATASET_MATRIX.get(method, (dataset,))


@dataclass
class PublicationExperimentConfig:
    data:PublicationDataConfig=field(default_factory=PublicationDataConfig)
    core_config:core.ExperimentConfig=field(default_factory=core.ExperimentConfig)
    metrics:MetricConfig=field(default_factory=MetricConfig)
    methods:Tuple[str,...] = ("BiasMF","MF","MBMF-F","MBMF-C","MBMF-N","AutoRec","AEMC-NE","Method1","Method2","Method3")
    output_dir:str="./q1_publication_results_v7"
    save_checkpoints:bool=True
    save_predictions:bool=True
    scalable_chunk_size:int=250_000
    respect_dataset_method_matrix:bool=True
    auto_generate_plots:bool=True
    auto_run_statistics:bool=True
    # Comparison baseline configs.
    biasmf:BiasMFConfig=field(default_factory=BiasMFConfig)
    autorec:AutoRecConfig=field(default_factory=AutoRecConfig)
    aemcne:AEMCNEConfig=field(default_factory=AEMCNEConfig)



def instantiate_core_model(method:str,data:MatrixLike,cfg:PublicationExperimentConfig,latent_dim:int,model_seed:int,device:torch.device,init_mode:Optional[str]=None)->Tuple[nn.Module,Any]:
    core.set_global_seed(model_seed)
    sparse=isinstance(data,SparseMatrixData)
    if method in ("MBMF-F","MBMF-C","MBMF-N"):
        c=copy.deepcopy(cfg.core_config.paper);c.latent_dim=latent_dim;im=init_mode or c.default_init;return core.PaperMBMF(data,c,method,im,device),c
    if method=="Method1":
        c=copy.deepcopy(cfg.core_config.method1);c.latent_dim=latent_dim;return (ScalableThesisMethod1 if sparse else core.ThesisMethod1)(data,c,device),c
    if method=="Method2":
        c=copy.deepcopy(cfg.core_config.method2);c.latent_dim=latent_dim;return (ScalableThesisMethod2 if sparse else core.ThesisMethod2)(data,c,device),c
    if method=="Method3":
        c=copy.deepcopy(cfg.core_config.method3);c.latent_dim=latent_dim;return (ScalableProposedMethod3 if sparse else core.ProposedMethod3)(data,c,device),c
    if method=="BiasMF":
        c=copy.deepcopy(cfg.biasmf);c.latent_dim=latent_dim;return BiasMF(data.num_users,data.num_items,c,device),c
    if method=="MF":
        c=copy.deepcopy(cfg.biasmf);c.use_bias=False;c.latent_dim=latent_dim;return BiasMF(data.num_users,data.num_items,c,device),c
    if method=="AutoRec":return ItemAutoRec(data,cfg.autorec,device),copy.deepcopy(cfg.autorec)
    if method=="AEMC-NE":
        c=copy.deepcopy(cfg.aemcne)
        if cfg.data.dataset=="ml-100k":c.main_hidden_dim=500
        elif cfg.data.dataset=="ml-1m":c.main_hidden_dim=900
        elif cfg.data.dataset=="ml-10m":c.main_hidden_dim=1000
        return AEMCNE(data,c,device),c
    raise ValueError(method)



def train_instantiated_model(method:str,model:nn.Module,data:MatrixLike,mcfg:Any,pcfg:PublicationExperimentConfig)->Dict[str,Any]:
    if method.startswith("MBMF-"):
        return train_paper_mbmf_chunked(model,data,mcfg,pcfg.scalable_chunk_size) if isinstance(data,SparseMatrixData) else core.train_paper_mbmf(model,data,mcfg)
    if method=="Method1":
        return train_scalable_method1(model,data,mcfg,pcfg.scalable_chunk_size) if isinstance(data,SparseMatrixData) else core.train_thesis_model(model,data,max_epochs=mcfg.max_epochs,lr=mcfg.lr,weight_decay=mcfg.weight_decay,eval_every=mcfg.eval_every,val_patience=mcfg.val_patience,grad_clip=mcfg.grad_clip)
    if method=="Method2":
        return train_scalable_method2(model,data,mcfg,pcfg.scalable_chunk_size) if isinstance(data,SparseMatrixData) else core.train_thesis_model(model,data,max_epochs=mcfg.max_epochs,lr=mcfg.lr,weight_decay=mcfg.weight_decay,eval_every=mcfg.eval_every,val_patience=mcfg.val_patience,grad_clip=mcfg.grad_clip)
    if method=="Method3":
        return train_scalable_method3(model,data,mcfg,pcfg.scalable_chunk_size) if isinstance(data,SparseMatrixData) else core.train_thesis_model(model,data,max_epochs=mcfg.max_epochs,lr=mcfg.lr,weight_decay=mcfg.weight_decay,eval_every=mcfg.eval_every,val_patience=mcfg.val_patience,grad_clip=mcfg.grad_clip)
    if method in ("BiasMF","MF"):return train_biasmf(model,data,mcfg)
    if method=="AutoRec":return train_autorec(model,data,mcfg,pcfg.scalable_chunk_size)
    if method=="AEMC-NE":return train_aemcne(model,data,mcfg,pcfg.scalable_chunk_size)
    raise ValueError(method)



def _prediction_archive_path(out:Path,dataset:str,method:str,split_seed:int,model_seed:int,k:int,init_mode:str)->Path:
    safe=method.replace("/","_");return out/"predictions"/f"{dataset}__{safe}__split{split_seed}__seed{model_seed}__K{k}__{init_mode}.npz"



def save_prediction_archive(path:Path,data:MatrixLike,pred:np.ndarray,split:str="test")->None:
    u,i,y=_split_tensors(data,split);pair=(u.detach().cpu().numpy().astype(np.int64)*data.num_items+i.detach().cpu().numpy().astype(np.int64));path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,pair_id=pair,prediction=pred.astype(np.float32),truth=y.detach().cpu().numpy().astype(np.float32))



def run_publication_core_suite(cfg:PublicationExperimentConfig,device:Optional[torch.device]=None)->Tuple[pd.DataFrame,pd.DataFrame]:
    """Run internally implemented methods on a common split protocol.

    Exact external BMC-ADMM / Deep-MMMF / DRPRA predictions can be merged later through
    `merge_external_prediction_csv`; their paper-specific protocols are stored separately.
    """
    device=device or core.get_device();out=Path(cfg.output_dir);out.mkdir(parents=True,exist_ok=True);(out/"checkpoints").mkdir(exist_ok=True)
    rows=[];histories={}
    # Dataset loaded once; each split_seed rebuilds the split.
    raw,spec=load_publication_dataset(cfg.data,save_mappings_to=out/"id_maps")
    # Manifest independent of split labels.
    manifest={"dataset":cfg.data.dataset,"documented_range":[spec.rating_min,spec.rating_max],"rating_step":spec.rating_step,"n_interactions":int(len(raw)),"n_users":int(raw.userId.nunique()),"n_items":int(raw.movieId.nunique()),"v7_core_sha256":V7_CORE_SHA256,"v7_sha256":sha256_file(__file__) if "__file__" in globals() else None}
    with open(out/"dataset_manifest.json","w") as f:json.dump(manifest,f,indent=2)
    with open(out/"literature_protocols.json","w") as f:json.dump(LITERATURE_PROTOCOLS,f,indent=2)

    for split_seed in cfg.core_config.split_seeds:
        pc=copy.deepcopy(cfg.data);pc.split_seed=int(split_seed);split=make_core_split(raw,spec,pc);split.to_csv(out/f"split_assignments_{split_seed}.csv",index=False)
        data,backend=build_publication_matrix_data(split,spec,pc,device)
        first_k=int(cfg.core_config.latent_dims[0])
        for k in cfg.core_config.latent_dims:
            for method in cfg.methods:
                if cfg.respect_dataset_method_matrix and not method_supported_on_dataset(method,cfg.data.dataset):
                    continue
                # AutoRec and AEMC-NE have their own hidden-width hyperparameters rather than an MBMF/MF K.
                # Run them once per split/seed, not redundantly for every latent K.
                if method in ("AutoRec","AEMC-NE") and int(k)!=first_k:
                    continue
                if method in ("MBMF-F","MBMF-C","MBMF-N"):
                    modes=cfg.core_config.paper.init_modes
                else:modes=("NA",)
                for init_mode in modes:
                    for model_seed in cfg.core_config.model_seeds:
                        model,mcfg=instantiate_core_model(method,data,cfg,int(k),int(model_seed),device,None if init_mode=="NA" else init_mode)
                        with PeakMemoryMonitor(device) as mm:
                            t0=time.perf_counter();info=train_instantiated_model(method,model,data,mcfg,cfg);train_wall=time.perf_counter()-t0
                        row={"dataset":cfg.data.dataset,"method":method,"backend":backend,"split_seed":split_seed,"model_seed":model_seed,"K":(np.nan if method in ("AutoRec","AEMC-NE") else k),"init_mode":init_mode,"train_wall_sec":train_wall,"parameter_count":core.parameter_count(model),**mm.metrics(),"best_epoch":info.get("best_epoch",np.nan)}
                        if method=="AutoRec": row["hidden_dim"]=int(mcfg.hidden_dim)
                        if method=="AEMC-NE": row.update({"main_hidden_dim":int(mcfg.main_hidden_dim),"element_hidden_dim":int(mcfg.element_hidden_dim)})
                        for sp in ("train","val","test"):
                            t1=time.perf_counter();met,pred=evaluate_publication_model(model,data,sp,metric_cfg=cfg.metrics,return_predictions=True);infer=time.perf_counter()-t1
                            row.update({f"{sp}_{a}":b for a,b in met.items()});row[f"{sp}_inference_sec"]=infer
                            if sp=="test" and cfg.save_predictions:
                                pp=_prediction_archive_path(out,cfg.data.dataset,method,split_seed,model_seed,k,init_mode);save_prediction_archive(pp,data,pred,"test");row["test_prediction_file"]=str(pp)
                        if hasattr(model,"diagnostics"):
                            try: row.update({f"diag_{a}":b for a,b in model.diagnostics(data,data.test_users,data.test_items).items()})
                            except TypeError: row.update({f"diag_{a}":b for a,b in model.diagnostics(data.test_users,data.test_items).items()})
                        if cfg.save_checkpoints:
                            cp=out/"checkpoints"/f"{cfg.data.dataset}__{method}__split{split_seed}__seed{model_seed}__K{k}__{init_mode}.pt";save_model_checkpoint(cp,model,config=mcfg,data=data,train_info=info,extra={"init_mode":init_mode,"dataset":cfg.data.dataset});row["checkpoint"]=str(cp)
                        key=f"{cfg.data.dataset}|{method}|{split_seed}|{model_seed}|{k}|{init_mode}";histories[key]=info.get("history",[]);rows.append(row);pd.DataFrame(rows).to_csv(out/"runs.csv",index=False)
                        del model;gc.collect();
                        if device.type=="cuda":torch.cuda.empty_cache()
    runs=pd.DataFrame(rows);summary=summarise_publication_runs(runs);summary.to_csv(out/"summary.csv",index=False)
    hist_path=out/"training_histories.json"
    with open(hist_path,"w") as f:json.dump(histories,f,indent=2)
    with open(out/"config.json","w") as f:json.dump(_jsonable(asdict(cfg)),f,indent=2)
    if not runs.empty:
        selected=validation_select_publication_runs(runs)
        selected.to_csv(out/"selected_by_validation_runs.csv",index=False)
        selected_summary=summarise_publication_runs(selected)
        selected_summary.to_csv(out/"selected_by_validation_summary.csv",index=False)
        stability=attach_prediction_stability(selected) if cfg.save_predictions else pd.DataFrame()
        if not stability.empty: stability.to_csv(out/"prediction_stability_across_seeds.csv",index=False)
        variation_k=attach_prediction_variation_across_k(selected) if cfg.save_predictions else pd.DataFrame()
        if not variation_k.empty: variation_k.to_csv(out/"prediction_stability_across_K.csv",index=False)
        if cfg.auto_run_statistics:
            try: pairwise_wilcoxon_holm(selected).to_csv(out/"wilcoxon_holm.csv",index=False)
            except Exception as e: warnings.warn(f"Wilcoxon-Holm skipped: {e}")
            try:
                ranks,st=friedman_nemenyi_summary(selected);ranks.to_csv(out/"friedman_average_ranks.csv",index=False);pd.DataFrame([st]).to_csv(out/"friedman_nemenyi_stats.csv",index=False)
            except Exception as e: warnings.warn(f"Friedman/Nemenyi skipped: {e}")
        if cfg.auto_generate_plots:
            try: generate_all_publication_plots(selected,out,hist_path)
            except Exception as e: warnings.warn(f"automatic publication plots skipped: {e}")
    return runs,summary



def run_dataset_benchmark_matrix(base_cfg:PublicationExperimentConfig,datasets:Sequence[str] = ("synthetic","jester","bookcrossing","ml-100k","ml-1m","ml-10m"),*,output_root:Optional[Union[str,Path]]=None,device:Optional[torch.device]=None,continue_on_error:bool=False)->Tuple[pd.DataFrame,pd.DataFrame]:
    """Run the publication suite across the requested dataset matrix with identical core equations."""
    root=Path(output_root or base_cfg.output_dir);root.mkdir(parents=True,exist_ok=True);all_runs=[];all_summ=[];errors=[]
    for ds in datasets:
        cfg=copy.deepcopy(base_cfg);cfg.data.dataset=ds;cfg.output_dir=str(root/ds)
        try:
            r,su=run_publication_core_suite(cfg,device=device);all_runs.append(r);all_summ.append(su)
        except Exception as e:
            errors.append({"dataset":ds,"error":repr(e)})
            if not continue_on_error:raise
    runs=pd.concat(all_runs,ignore_index=True,sort=False) if all_runs else pd.DataFrame();summ=pd.concat(all_summ,ignore_index=True,sort=False) if all_summ else pd.DataFrame()
    if not runs.empty:runs.to_csv(root/"ALL_DATASETS_runs.csv",index=False)
    if not summ.empty:summ.to_csv(root/"ALL_DATASETS_summary.csv",index=False)
    if errors:pd.DataFrame(errors).to_csv(root/"ALL_DATASETS_errors.csv",index=False)
    return runs,summ


def _jsonable(x):
    if isinstance(x,dict):return {str(k):_jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [_jsonable(v) for v in x]
    if isinstance(x,(np.integer,np.floating)):return x.item()
    return x



def summarise_publication_runs(runs:pd.DataFrame)->pd.DataFrame:
    # Never pool distinct ablations/scans into one mean (scientific reporting bug).
    keys=[c for c in ["ablation","scan","dataset","method","backend","K","init_mode"] if c in runs]
    nums=[c for c in runs.select_dtypes(include=np.number).columns if c not in ("split_seed","model_seed","K")]
    groups = runs.groupby(keys, dropna=False, sort=True)[nums]
    # pd.DataFrame.groupby(...).agg([mean,std,count]).reset_index() on hundreds
    # of evaluation metrics can create a heavily fragmented DataFrame. Build
    # consolidated column blocks; values and aggregation semantics are identical.
    summaries = [groups.mean().add_suffix("_mean"),
                 groups.std().add_suffix("_std"),
                 groups.count().add_suffix("_count")]
    return pd.concat(summaries, axis=1).copy().reset_index()



def export_split_for_external(data:MatrixLike,out_path:Union[str,Path], *,
                              id_map_dir: Optional[Union[str, Path]] = None,
                              dataset_name: Optional[str] = None)->Path:
    """Export the exact common split for author/official external baselines.

    The `userId`/`movieId` columns are the contiguous internal IDs expected by
    ExternalPredictionAdapter. Original IDs are included for convenience.
    """
    # data.index_to_user/item are canonicalized indices, NOT original dataset IDs.
    # Resolve real original IDs from the saved mapping tables where available.
    user_original = item_original = None
    if id_map_dir is not None and dataset_name is not None:
        user_map = pd.read_csv(Path(id_map_dir) / f"{dataset_name}_user_id_map.csv", dtype={"original_user_id": str})
        item_map = pd.read_csv(Path(id_map_dir) / f"{dataset_name}_item_id_map.csv", dtype={"original_item_id": str})
        user_lookup = dict(zip(user_map.userId.astype(int), user_map.original_user_id))
        item_lookup = dict(zip(item_map.movieId.astype(int), item_map.original_item_id))
        user_original = np.array([user_lookup[int(x)] for x in data.index_to_user], dtype=object)
        item_original = np.array([item_lookup[int(x)] for x in data.index_to_item], dtype=object)
    rows=[]
    for sp in ("train","val","test"):
        u,i,y=_split_tensors(data,sp)
        un=u.detach().cpu().numpy().astype(np.int64); it=i.detach().cpu().numpy().astype(np.int64)
        frame = pd.DataFrame({"split":sp,"userId":un,"movieId":it,
                              "rating":y.detach().cpu().numpy().astype(np.float32)})
        if user_original is not None:
            frame["original_user_id"] = user_original[un]
            frame["original_item_id"] = item_original[it]
        rows.append(frame)
    out=Path(out_path);out.parent.mkdir(parents=True,exist_ok=True);pd.concat(rows,ignore_index=True).to_csv(out,index=False);return out


def merge_external_prediction_csv(runs:pd.DataFrame,data:MatrixLike,csv_path:Union[str,Path],method_name:str,metric_cfg:Optional[MetricConfig]=None)->pd.DataFrame:
    """Append author/official baseline predictions using the identical metric implementation.

    The external method is not given any structural-bound credit; individual-bound columns are NaN.
    """
    device=data.train_users.device;model=ExternalPredictionAdapter(csv_path,data,device);row={"method":method_name,"source":"external_official_adapter"}
    for sp in ("train","val","test"):
        met=evaluate_publication_model(model,data,sp,metric_cfg=metric_cfg or MetricConfig())
        row.update({f"{sp}_{k}":v for k,v in met.items()})
    return pd.concat([runs,pd.DataFrame([row])],ignore_index=True,sort=False)


# =============================================================================
# 10. Statistical analysis: confidence intervals, Friedman/Nemenyi, Wilcoxon-Holm
# =============================================================================


def mean_ci95(x:Sequence[float])->Tuple[float,float,float]:
    a=np.asarray(x,dtype=float);a=a[np.isfinite(a)]
    if len(a)==0:return float("nan"),float("nan"),float("nan")
    m=float(a.mean())
    if len(a)<2:return m,float("nan"),float("nan")
    try:
        from scipy.stats import t
        h=float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/math.sqrt(len(a)))
    except Exception:h=1.96*float(a.std(ddof=1))/math.sqrt(len(a))
    return m,m-h,m+h



def holm_adjust(pvals:Sequence[float])->np.ndarray:
    p=np.asarray(pvals,float);order=np.argsort(p);out=np.empty_like(p);running=0.0;m=len(p)
    for j,idx in enumerate(order):running=max(running,(m-j)*p[idx]);out[idx]=min(running,1.0)
    return out



def pairwise_wilcoxon_holm(runs:pd.DataFrame,metric:str="test_RMSE",group_cols:Sequence[str] = ("dataset","split_seed","K"))->pd.DataFrame:
    try:from scipy.stats import wilcoxon
    except Exception as e:raise ImportError("scipy required for Wilcoxon tests") from e
    methods=sorted(runs.method.dropna().unique());records=[]
    # Average model seeds/init modes inside a block to form matched blocks.
    tab=runs.groupby(list(group_cols)+["method"])[metric].mean().unstack("method")
    for aidx,a in enumerate(methods):
        for b in methods[aidx+1:]:
            if a not in tab or b not in tab:continue
            q=tab[[a,b]].dropna();
            if len(q)<2:continue
            stat,p=wilcoxon(q[a],q[b],zero_method="wilcox",alternative="two-sided");records.append({"A":a,"B":b,"n_blocks":len(q),"stat":stat,"p_raw":p,"mean_diff_A_minus_B":float((q[a]-q[b]).mean())})
    if records:
        adj=holm_adjust([r["p_raw"] for r in records])
        for r,p in zip(records,adj):r["p_holm"]=p
    return pd.DataFrame(records)



def friedman_nemenyi_summary(runs:pd.DataFrame,metric:str="test_RMSE",block_cols:Sequence[str] = ("dataset","split_seed"),alpha:float=.05)->Tuple[pd.DataFrame,Dict[str,float]]:
    try:from scipy.stats import friedmanchisquare,studentized_range
    except Exception as e:raise ImportError("scipy required for Friedman/Nemenyi") from e
    tab = runs.groupby(list(block_cols) + ["method"])[metric].mean().unstack("method")
    # Only methods observed on EVERY block are comparable in Friedman.
    # AutoRec/AEMC-NE are not scheduled on every dataset by design; their
    # missing cells must not erase all blocks and suppress the common-method test.
    common = list(tab.columns[tab.notna().all(axis=0)])
    tab = tab[common].dropna(axis=0, how="any")
    methods=list(tab.columns);N=len(tab);K=len(methods)
    if N<2 or K<3:
        return pd.DataFrame(),{"N":N,"K":K,"friedman_stat":float("nan"),"friedman_p":float("nan"),"CD":float("nan"),"methods_included": ",".join(methods)}
    stat,p=friedmanchisquare(*[tab[m].to_numpy() for m in methods])
    ranks=tab.rank(axis=1,method="average",ascending=True).mean(axis=0)
    q=float(studentized_range.ppf(1-alpha,K,np.inf)/math.sqrt(2))
    cd=q*math.sqrt(K*(K+1)/(6*N))
    return pd.DataFrame({"method":methods,"average_rank":[ranks[m] for m in methods]}).sort_values("average_rank"),{"N":N,"K":K,"friedman_stat":float(stat),"friedman_p":float(p),"CD":float(cd),"alpha":alpha,"methods_included": ",".join(methods)}


# =============================================================================
# 11. Publication figures (Matplotlib only; PNG + PDF)
# =============================================================================


def _mpl():
    import matplotlib.pyplot as plt
    return plt



def _save_fig(fig,path_base:Path):
    path_base.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path_base.with_suffix(".png"),dpi=220,bbox_inches="tight");fig.savefig(path_base.with_suffix(".pdf"),bbox_inches="tight");_mpl().close(fig)



def plot_metric_bars(summary_runs:pd.DataFrame,metric:str,out_dir:Union[str,Path])->None:
    plt=_mpl();d=summary_runs.groupby(["dataset","method"])[metric].agg(["mean","std"]).reset_index();datasets=list(d.dataset.unique());methods=list(d.method.unique());x=np.arange(len(datasets));width=.8/max(1,len(methods));fig,ax=plt.subplots(figsize=(max(7,len(datasets)*1.4),5))
    for j,m in enumerate(methods):
        g=d[d.method==m].set_index("dataset").reindex(datasets);ax.bar(x+(j-(len(methods)-1)/2)*width,g["mean"].to_numpy(),width,yerr=g["std"].fillna(0).to_numpy(),label=m,capsize=2)
    ax.set_xticks(x,datasets,rotation=20);ax.set_ylabel(metric);ax.legend(fontsize=8,ncol=max(1,min(3,len(methods))));_save_fig(fig,Path(out_dir)/f"bar_{metric}")



def plot_training_curves(history_json:Union[str,Path],out_dir:Union[str,Path],max_curves:int=20)->None:
    with open(history_json) as f:h=json.load(f)
    plt=_mpl();fig,ax=plt.subplots(figsize=(8,5));n=0
    for key,rows in h.items():
        if not rows or "val_RMSE" not in rows[0]:continue
        ax.plot([r["epoch"] for r in rows],[r["val_RMSE"] for r in rows],label=key.split("|")[1]);n+=1
        if n>=max_curves:break
    ax.set_xlabel("Epoch");ax.set_ylabel("Validation RMSE");
    if n<=10:ax.legend(fontsize=7)
    _save_fig(fig,Path(out_dir)/"training_validation_rmse")



def plot_ablation_line(ablation:pd.DataFrame,xcol:str,metric:str,out_dir:Union[str,Path],group_col:str="method")->None:
    plt=_mpl();fig,ax=plt.subplots(figsize=(7,5))
    for name,g in ablation.groupby(group_col):
        s=g.groupby(xcol)[metric].agg(["mean","std"]).reset_index().sort_values(xcol);ax.errorbar(s[xcol],s["mean"],yerr=s["std"].fillna(0),marker="o",label=name)
    ax.set_xlabel(xcol);ax.set_ylabel(metric);ax.legend(fontsize=8);_save_fig(fig,Path(out_dir)/f"ablation_{xcol}_{metric}")



def plot_oob_and_bounds(runs:pd.DataFrame,out_dir:Union[str,Path])->None:
    for metric in ("test_Global_OOB_Rate","test_Mean_OOB_Violation","test_Max_OOB_Violation","test_IndividualBoundViolationRate","test_MeanBoundWidth"):
        if metric in runs.columns:plot_metric_bars(runs,metric,out_dir)



def plot_runtime_memory(runs:pd.DataFrame,out_dir:Union[str,Path])->None:
    for metric in ("train_wall_sec","PeakRSS_Delta_MB","PeakGPUAllocated_MB"):
        if metric in runs.columns:plot_metric_bars(runs,metric,out_dir)



def plot_ndcg_curves(runs:pd.DataFrame,out_dir:Union[str,Path])->None:
    cols=[c for c in runs.columns if c.startswith("test_NDCG_testset@")]
    if not cols:return
    ks=sorted((int(c.split("@")[-1]),c) for c in cols);plt=_mpl();fig,ax=plt.subplots(figsize=(7,5))
    for method,g in runs.groupby("method"):
        vals=[g[c].mean() for _,c in ks];ax.plot([k for k,_ in ks],vals,marker="o",label=method)
    ax.set_xlabel("k");ax.set_ylabel("NDCG@k (held-out explicit ratings)");ax.legend(fontsize=8);_save_fig(fig,Path(out_dir)/"ndcg_curve")



def plot_critical_difference(rank_df:pd.DataFrame,stats:Mapping[str,float],out_dir:Union[str,Path])->None:
    if rank_df.empty:return
    plt=_mpl();fig,ax=plt.subplots(figsize=(8,max(3,.4*len(rank_df)+1)));y=np.arange(len(rank_df));ax.scatter(rank_df.average_rank,y);ax.set_yticks(y,rank_df.method);ax.invert_yaxis();ax.set_xlabel("Average rank (lower is better)");ax.set_title(f"Friedman p={stats.get('friedman_p',float('nan')):.3g}; Nemenyi CD={stats.get('CD',float('nan')):.3g}");_save_fig(fig,Path(out_dir)/"critical_difference_ranks")



def plot_train_test_gap(runs:pd.DataFrame,metric:str,out_dir:Union[str,Path])->None:
    tr=f"train_{metric}";te=f"test_{metric}"
    if tr not in runs.columns or te not in runs.columns:return
    plt=_mpl();d=runs.groupby("method")[[tr,te]].mean().sort_values(te);x=np.arange(len(d));w=.38;fig,ax=plt.subplots(figsize=(max(8,.7*len(d)),5));ax.bar(x-w/2,d[tr],w,label="Train");ax.bar(x+w/2,d[te],w,label="Test");ax.set_xticks(x,d.index,rotation=30,ha="right");ax.set_ylabel(metric);ax.legend();_save_fig(fig,Path(out_dir)/f"train_test_{metric}")


def plot_prediction_stability_from_runs(runs:pd.DataFrame,out_dir:Union[str,Path])->None:
    if "test_prediction_file" not in runs.columns:return
    st=attach_prediction_stability(runs)
    if st.empty:return
    plt=_mpl();fig,ax=plt.subplots(figsize=(max(8,.7*len(st)),5));labels=[];avg=[];mx=[]
    for _,r in st.iterrows():
        labels.append(str(r.get("method",""))+(f" K={r.get('K')}" if pd.notna(r.get("K",np.nan)) else ""));avg.append(r["AvgPredictionStd"]);mx.append(r["MaxPredictionStd"])
    x=np.arange(len(labels));w=.38;ax.bar(x-w/2,avg,w,label="Average prediction std");ax.bar(x+w/2,mx,w,label="Max prediction std");ax.set_xticks(x,labels,rotation=35,ha="right");ax.set_ylabel("Prediction standard deviation");ax.legend();_save_fig(fig,Path(out_dir)/"prediction_stability")


def plot_prediction_variation_across_k(runs:pd.DataFrame,out_dir:Union[str,Path])->None:
    vk=attach_prediction_variation_across_k(runs)
    if vk.empty or vk["AvgPredictionStd_acrossK"].notna().sum()==0:return
    g=vk.groupby("method")[["AvgPredictionStd_acrossK","MaxPredictionStd_acrossK"]].mean().dropna(how="all")
    if g.empty:return
    plt=_mpl();x=np.arange(len(g));w=.38;fig,ax=plt.subplots(figsize=(max(7,.7*len(g)),5));ax.bar(x-w/2,g["AvgPredictionStd_acrossK"],w,label="Average std across K");ax.bar(x+w/2,g["MaxPredictionStd_acrossK"],w,label="Max std across K");ax.set_xticks(x,g.index,rotation=35,ha="right");ax.set_ylabel("Prediction std across latent dimensions");ax.legend();_save_fig(fig,Path(out_dir)/"prediction_variation_across_K")


def plot_accuracy_efficiency(runs:pd.DataFrame,out_dir:Union[str,Path])->None:
    if not {"test_RMSE","train_wall_sec"}.issubset(runs.columns):return
    plt=_mpl();d=runs.groupby("method")[["test_RMSE","train_wall_sec"]].mean().dropna();fig,ax=plt.subplots(figsize=(7,5));ax.scatter(d["train_wall_sec"],d["test_RMSE"]);
    for m,r in d.iterrows():ax.annotate(str(m),(r["train_wall_sec"],r["test_RMSE"]),fontsize=8)
    ax.set_xlabel("Training wall time (s)");ax.set_ylabel("Test RMSE");ax.set_xscale("log" if (d["train_wall_sec"]>0).all() and d["train_wall_sec"].max()/max(d["train_wall_sec"].min(),1e-12)>20 else "linear");_save_fig(fig,Path(out_dir)/"accuracy_efficiency")


def plot_metric_heatmap(runs:pd.DataFrame,metric:str,out_dir:Union[str,Path])->None:
    if metric not in runs.columns:return
    plt=_mpl();tab=runs.groupby(["dataset","method"])[metric].mean().unstack("method")
    if tab.empty:return
    arr=tab.to_numpy(float);fig,ax=plt.subplots(figsize=(max(7,.9*tab.shape[1]),max(3,.6*tab.shape[0]+2)));im=ax.imshow(arr,aspect="auto");fig.colorbar(im,ax=ax,label=metric);ax.set_xticks(np.arange(tab.shape[1]),tab.columns,rotation=35,ha="right");ax.set_yticks(np.arange(tab.shape[0]),tab.index)
    for a in range(tab.shape[0]):
        for b in range(tab.shape[1]):
            if np.isfinite(arr[a,b]):ax.text(b,a,f"{arr[a,b]:.3g}",ha="center",va="center",fontsize=7)
    _save_fig(fig,Path(out_dir)/f"heatmap_{metric}")


def generate_all_publication_plots(runs:pd.DataFrame,output_dir:Union[str,Path],history_json:Optional[Union[str,Path]]=None)->None:
    out=Path(output_dir);fig=out/"figures";fig.mkdir(parents=True,exist_ok=True)
    for metric in ("test_RMSE","test_MAE","test_NMAE_ordinal_uniform","test_NMAE_range","test_Pearson","test_P95AE","test_Bias_MeanError"):
        if metric in runs.columns:plot_metric_bars(runs,metric,fig)
    plot_train_test_gap(runs,"RMSE",fig);plot_train_test_gap(runs,"MAE",fig)
    plot_oob_and_bounds(runs,fig);plot_runtime_memory(runs,fig);plot_ndcg_curves(runs,fig);plot_prediction_stability_from_runs(runs,fig);plot_prediction_variation_across_k(runs,fig);plot_accuracy_efficiency(runs,fig);plot_metric_heatmap(runs,"test_RMSE",fig)
    if "K" in runs:plot_ablation_line(runs,"K","test_RMSE",fig)
    if history_json and Path(history_json).exists():plot_training_curves(history_json,fig)
    try:
        ranks,st=friedman_nemenyi_summary(runs);ranks.to_csv(out/"friedman_average_ranks.csv",index=False);pd.DataFrame([st]).to_csv(out/"friedman_nemenyi_stats.csv",index=False);plot_critical_difference(ranks,st,fig)
    except Exception as e:warnings.warn(f"Friedman/Nemenyi plot skipped: {e}")


# =============================================================================
# 12. Ablation scan plans (validation-selected; no model-logic mutation)
# =============================================================================


@dataclass
class AblationPlan:
    latent_dims:Tuple[int,...]=(5,10,20,50)
    mbmf_initializations:Tuple[str,...] = ("legacy_uniform_2pi","canonical_uniform","isotropic_gaussian")
    method1_alpha:Tuple[float,...]=(0.25,0.5,0.7,1.0)
    method1_beta:Tuple[float,...]=(0.15,0.35,0.5,1.0)
    method2_gamma:Tuple[float,...]=(0.25,0.5,0.75)
    method2_consistency:Tuple[float,...]=(0.0,0.01,0.1)
    method3_log_scale:Tuple[float,...]=(0.5,1.0,2.0,4.0)
    method3_delta_l2:Tuple[float,...]=(0.0,1e-5,1e-4,1e-3)
    train_fractions:Tuple[float,...]=(0.70,0.80,0.90)  # DRPRA-inspired density sensitivity.
    synthetic_missing_rates:Tuple[float,...]=(0.2,0.3,0.4,0.5,0.6,0.7) # AEMC-NE-inspired MCAR density sweep.
    autorec_hidden:Tuple[int,...]=(100,200,300,500)
    autorec_lambda:Tuple[float,...]=(0.001,0.01,0.1,1.0)
    aemc_element_hidden:Tuple[int,...]=(50,100,200,400)
    aemc_lambda:Tuple[float,...]=(0.01,0.1,1.0,10.0)



def essential_ablation_configs(base:PublicationExperimentConfig,plan:Optional[AblationPlan]=None)->List[Tuple[str,PublicationExperimentConfig]]:
    """Independent one-factor scans. Test labels are never used to choose a setting."""
    p=plan or AblationPlan();jobs=[]
    for k in p.latent_dims:
        c=copy.deepcopy(base);c.core_config.latent_dims=(k,);c.methods=tuple(m for m in c.methods if m not in ("AutoRec","AEMC-NE"));jobs.append((f"K={k}",c))
    # All three paper angle initializations are evaluated in one paper-only job so their predictions align.
    c=copy.deepcopy(base);c.methods=("MBMF-F","MBMF-C","MBMF-N");c.core_config.paper.init_modes=p.mbmf_initializations;jobs.append(("MBMF_angle_initialization",c))
    for a in p.method1_alpha:
        c=copy.deepcopy(base);c.methods=("Method1",);c.core_config.method1.alpha_ae=a;jobs.append((f"M1_alpha={a}",c))
    for b in p.method1_beta:
        c=copy.deepcopy(base);c.methods=("Method1",);c.core_config.method1.beta_mbmf=b;jobs.append((f"M1_beta={b}",c))
    for g in p.method2_gamma:
        c=copy.deepcopy(base);c.methods=("Method2",);c.core_config.method2.fusion_weight_user_view=g;jobs.append((f"M2_gamma={g}",c))
    for w in p.method2_consistency:
        c=copy.deepcopy(base);c.methods=("Method2",);c.core_config.method2.consistency_weight=w;jobs.append((f"M2_cons={w}",c))
    for sc in p.method3_log_scale:
        c=copy.deepcopy(base);c.methods=("Method3",);c.core_config.method3.max_abs_log_scale=sc;jobs.append((f"M3_logscale={sc}",c))
    for d in p.method3_delta_l2:
        c=copy.deepcopy(base);c.methods=("Method3",);c.core_config.method3.delta_l2=d;jobs.append((f"M3_deltaL2={d}",c))
    for h in p.autorec_hidden:
        c=copy.deepcopy(base);c.methods=("AutoRec",);c.autorec.hidden_dim=h;jobs.append((f"AutoRec_hidden={h}",c))
    for lam in p.autorec_lambda:
        c=copy.deepcopy(base);c.methods=("AutoRec",);c.autorec.lambda_l2=lam;jobs.append((f"AutoRec_lambda={lam}",c))
    for h in p.aemc_element_hidden:
        c=copy.deepcopy(base);c.methods=("AEMC-NE",);c.aemcne.element_hidden_dim=h;jobs.append((f"AEMC_elementHidden={h}",c))
    for lam in p.aemc_lambda:
        c=copy.deepcopy(base);c.methods=("AEMC-NE",);c.aemcne.lambda_main=lam;c.aemcne.lambda_element=lam;jobs.append((f"AEMC_lambda={lam}",c))
    for f in p.train_fractions:
        c=copy.deepcopy(base);c.data.train_fraction=f;c.data.val_fraction=min(.1,1-f);c.data.test_fraction=1-c.data.train_fraction-c.data.val_fraction;jobs.append((f"train_fraction={f}",c))
    if base.data.dataset=="synthetic":
        for miss in p.synthetic_missing_rates:
            c=copy.deepcopy(base);c.data.synthetic_density=1.0-miss;jobs.append((f"synthetic_missing={miss}",c))
    return jobs


def run_essential_ablation_scan(base:PublicationExperimentConfig,plan:Optional[AblationPlan]=None,output_root:Optional[Union[str,Path]]=None,device:Optional[torch.device]=None)->pd.DataFrame:
    """Execute named independent scans and save a combined validation/test table.

    This is intentionally serial and resumable at the job-directory level. For expensive ML-10M
    studies, run the returned configs as separate cluster jobs instead.
    """
    root=Path(output_root or (Path(base.output_dir)/"ablations"));root.mkdir(parents=True,exist_ok=True);all_rows=[]
    for name,cfg in essential_ablation_configs(base,plan):
        safe="".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
        cfg.output_dir=str(root/safe)
        runs,_=run_publication_core_suite(cfg,device=device)
        runs=runs.copy();runs["ablation_name"]=name;all_rows.append(runs)
    out=pd.concat(all_rows,ignore_index=True,sort=False) if all_rows else pd.DataFrame()
    if not out.empty:
        out.to_csv(root/"all_ablation_runs.csv",index=False)
        # A generic long-format summary is easy to use for one-factor plots.
        summarise_publication_runs(out).to_csv(root/"all_ablation_summary.csv",index=False)
    return out



# =============================================================================
# 13. Validation-only model/config selection and stability aggregation
# =============================================================================



def validation_select_publication_runs(runs:pd.DataFrame,metric:str="val_RMSE")->pd.DataFrame:
    """Select hyperparameters/init without touching test metrics; preserve independent model seeds."""
    # A configuration is identified by method,K,init_mode and any caller-added ablation columns.
    candidates=[c for c in ["dataset","method","split_seed","K","init_mode"] if c in runs]
    cfg_cols=[c for c in runs.columns if c.startswith("ablation_")]
    block=[c for c in ["dataset","method","split_seed"] if c in runs]
    group=candidates+cfg_cols
    score=runs.groupby(group,dropna=False)[metric].mean().reset_index()
    idx=score.groupby(block,dropna=False)[metric].idxmin();chosen=score.loc[idx,group]
    return runs.merge(chosen,on=group,how="inner")



def attach_prediction_stability(selected_runs:pd.DataFrame)->pd.DataFrame:
    rows=[]
    group=[c for c in ["dataset","method","split_seed","K","init_mode"] if c in selected_runs]
    for key,g in selected_runs.groupby(group,dropna=False):
        files=[x for x in g.get("test_prediction_file",pd.Series(dtype=str)).dropna().tolist() if Path(x).exists()];st=prediction_stability(files);row=dict(zip(group,key if isinstance(key,tuple) else (key,)));row.update(st);rows.append(row)
    return pd.DataFrame(rows)


def attach_prediction_variation_across_k(selected_runs:pd.DataFrame)->pd.DataFrame:
    """MBMF-paper-style prediction variation when latent dimension K changes.

    K is intentionally removed from the grouping key. Prediction files must refer to the same
    held-out pairs (same dataset/split seed/model seed/init mode).
    """
    if "test_prediction_file" not in selected_runs.columns:return pd.DataFrame()
    rows=[];group=[c for c in ["dataset","method","split_seed","model_seed","init_mode"] if c in selected_runs]
    for key,g in selected_runs.groupby(group,dropna=False):
        gg=g[g["K"].notna()] if "K" in g else g
        files=[x for x in gg.get("test_prediction_file",pd.Series(dtype=str)).dropna().tolist() if Path(x).exists()]
        ks=sorted(pd.unique(gg["K"].dropna()).tolist()) if "K" in gg else []
        st=prediction_stability(files) if len(set(ks))>=2 else {"AvgPredictionStd":float("nan"),"MaxPredictionStd":float("nan")}
        row=dict(zip(group,key if isinstance(key,tuple) else (key,)));row.update({"K_values":",".join(map(str,ks)),"AvgPredictionStd_acrossK":st["AvgPredictionStd"],"MaxPredictionStd_acrossK":st["MaxPredictionStd"]});rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 14. Self-tests: sparse/dense equivalence, checkpoint round-trip, metrics
# =============================================================================



def _matrixdata_to_sparse(d:core.MatrixData)->SparseMatrixData:
    # Build a split DataFrame from v7 synthetic data, preserving exact pairs/labels.
    rows=[]
    for sp,u,i,y in (("train",d.train_users,d.train_items,d.train_ratings_raw),("val",d.val_users,d.val_items,d.val_ratings_raw),("test",d.test_users,d.test_items,d.test_ratings_raw)):
        rows.append(pd.DataFrame({"userId":u.cpu().numpy(),"movieId":i.cpu().numpy(),"rating":y.cpu().numpy(),"split":sp}))
    df=pd.concat(rows,ignore_index=True);spec=DatasetSpec("synthetic",d.rating_min,d.rating_max,False,"",0.5,"");return build_sparse_matrix_data(df,spec,d.train_users.device)



def sparse_dense_equivalence_test(device:Optional[torch.device]=None)->None:
    device=device or core.get_device();d=core.synthetic_data(device);s=_matrixdata_to_sparse(d)
    # Method1
    for num,(dense_cls,sparse_cls,cfg) in enumerate([
        (core.ThesisMethod1,ScalableThesisMethod1,core.Method1Config(latent_dim=4,hidden_dim=8,max_epochs=1)),
        (core.ThesisMethod2,ScalableThesisMethod2,core.Method2Config(latent_dim=4,max_epochs=1,counterpart_radius_mode="vector_norm")),
        (core.ProposedMethod3,ScalableProposedMethod3,core.Method3Config(latent_dim=4,max_epochs=1)),
    ],1):
        core.set_global_seed(123);a=dense_cls(d,cfg,device);core.set_global_seed(123);b=sparse_cls(s,cfg,device);b.load_state_dict(a.state_dict())
        with torch.no_grad():
            pa=a.predict_raw_pairs(d,d.test_users,d.test_items);pb=b.predict_raw_pairs(s,s.test_users,s.test_items)
        err=float(torch.max(torch.abs(pa-pb)).item());
        if err>2e-5:raise AssertionError(f"dense/sparse Method{num} prediction mismatch {err}")
    print("[v7-test] dense/sparse proposed-model prediction equivalence: PASS")



def checkpoint_roundtrip_test(device:Optional[torch.device]=None,tmp:Optional[Union[str,Path]]=None)->None:
    device=device or core.get_device();d=core.synthetic_data(device);cfg=core.Method1Config(latent_dim=4,hidden_dim=8,max_epochs=1);core.set_global_seed(19);m=core.ThesisMethod1(d,cfg,device);p0=m.predict_raw_pairs(d,d.test_users,d.test_items).detach().cpu();path=Path(tmp or THIS_DIR/"_checkpoint_test.pt");save_model_checkpoint(path,m,config=cfg,data=d);m2,_=load_model_checkpoint(path,d,device);p1=m2.predict_raw_pairs(d,d.test_users,d.test_items).detach().cpu();
    if not torch.allclose(p0,p1,atol=0,rtol=0):raise AssertionError("checkpoint round-trip changed predictions")
    with contextlib.suppress(Exception):path.unlink()
    print("[v7-test] checkpoint round-trip: PASS")



def comparison_checkpoint_roundtrip_test(device:Optional[torch.device]=None)->None:
    device=device or core.get_device();d=core.synthetic_data(device)
    cases=[
        (BiasMF(d.num_users,d.num_items,BiasMFConfig(latent_dim=3,max_epochs=1),device),BiasMFConfig(latent_dim=3,max_epochs=1)),
        (ItemAutoRec(d,AutoRecConfig(hidden_dim=4,max_epochs=1),device),AutoRecConfig(hidden_dim=4,max_epochs=1)),
        (AEMCNE(d,AEMCNEConfig(main_hidden_dim=4,element_hidden_dim=3,max_epochs=1),device),AEMCNEConfig(main_hidden_dim=4,element_hidden_dim=3,max_epochs=1)),
    ]
    for j,(m,cfg) in enumerate(cases):
        with torch.no_grad():p0=m.predict_raw_pairs(d,d.test_users,d.test_items).detach().cpu()
        path=THIS_DIR/f"_checkpoint_comp_test_{j}.pt";save_model_checkpoint(path,m,config=cfg,data=d);m2,_=load_model_checkpoint(path,d,device)
        with torch.no_grad():p1=m2.predict_raw_pairs(d,d.test_users,d.test_items).detach().cpu()
        if not torch.allclose(p0,p1,atol=0,rtol=0):raise AssertionError(f"comparison checkpoint round-trip failed for {checkpoint_family(m)}")
        with contextlib.suppress(Exception):path.unlink()
    print("[v7-test] comparison checkpoint round-trip: PASS")


def publication_extension_self_test(device:Optional[torch.device]=None)->None:
    core.synthetic_smoke_test(device);sparse_dense_equivalence_test(device);checkpoint_roundtrip_test(device);comparison_checkpoint_roundtrip_test(device);print("[v7-test] publication extension: PASS")


# =============================================================================
# 15. Complete experiment instructions embedded in the .py file, as requested
# =============================================================================

EXPERIMENT_INSTRUCTIONS = r"""
Q1 EXPERIMENT EXECUTION CHECKLIST
=================================

A. DO NOT EDIT THE SCIENTIFIC CORE
----------------------------------
Keep `mbmf_q1_final_v7.py` and `mbmf_q1_publication_v7.py` together. The latter imports the
former. Record the SHA-256 hashes in every experiment archive.

B. DOWNLOAD DATASETS
--------------------
Run `download_datasets_q1_v7.ipynb`. It only downloads/extracts raw sources. All parsing,
filtering, ID mapping, missing-value handling and splits happen in this module.

C. PRIMARY COMMON-PROTOCOL EXPERIMENT
-------------------------------------
1. Choose one dataset from:
   synthetic, jester, bookcrossing, ml-100k, ml-1m, ml-10m.
2. Use `random_per_user` for the main common comparison unless you preregister temporal splits.
3. Use several independent split seeds AND model seeds.
4. Use the same split assignments for every method.
5. Select K/init/hyperparameters using validation metrics only.
6. Evaluate test once after selection. Never select on test.
7. Preserve raw predictions; never clamp predictions before metrics.
8. Save model checkpoints, configs, environment, split assignments and prediction archives.

Example:
    from mbmf_q1_publication_v7 import *
    import mbmf_q1_final_v7 as core
    cfg = PublicationExperimentConfig()
    cfg.data.dataset = 'ml-1m'
    cfg.data.data_root = './datasets'
    core.apply_q1_safe_preset(cfg.core_config)
    cfg.core_config.split_seeds = (2026, 2027, 2028, 2029, 2030)
    cfg.core_config.model_seeds = (11, 22, 33, 44, 55)
    cfg.core_config.latent_dims = (10, 20, 50)
    cfg.output_dir = './results/ml1m'
    publication_extension_self_test()
    runs, summary = run_publication_core_suite(cfg)
    selected = validation_select_publication_runs(runs)
    selected.to_csv(Path(cfg.output_dir)/'selected_by_validation_runs.csv', index=False)
    stab = attach_prediction_stability(selected)
    stab.to_csv(Path(cfg.output_dir)/'prediction_stability.csv', index=False)
    generate_all_publication_plots(selected, cfg.output_dir, Path(cfg.output_dir)/'training_histories.json')

D. DATASET-SPECIFIC NOTES
-------------------------
Synthetic: report density/noise/rank generator settings; add MCAR/MNAR missing-rate sweeps.
Jester: official missing marker 99 is removed; documented range [-10,10].
BookCrossing: primary rating-prediction mode uses explicit ratings 1..10; report whether zero
              implicit events are excluded (default: excluded).
ML-100K: [1,5].
ML-1M: [1,5].
ML-10M: [0.5,5]. Sparse backend is selected automatically to avoid NxM dense matrices.

E. BASELINES / COMPETITORS
--------------------------
Internally available:
  MF / BiasMF, MBMF-F/C/N, AutoRec, AEMC-NE reimplementation, Methods 1/2/3.
For BMC-ADMM, Deep-MMMF and DRPRA, use author/official code where available and export
`split,userId,movieId,prediction` CSVs for exactly the v7 split. Then ingest them through
ExternalPredictionAdapter. This avoids falsely labelling an approximate solver as the published method.

F. METRICS
----------
Always report on TRAIN and TEST (also retain validation): RMSE, MAE, MSE, median/P90/P95/P99
absolute error, range-normalised errors, Pearson/Spearman, raw OOB rate/mean/max violation,
runtime and peak memory. For MBMF/Ours additionally report structural individual-bound violation
and mean bound width. Across random initialisations report AvgPredictionStd and MaxPredictionStd.
For explicit-rating ranking add test-set NDCG@1/5/10/50 and binary ranking metrics; clearly state
candidate protocol. `F1_MBMFPaper_reproduction_only` is intentionally separate from the primary
leakage-safe F1. NMAE_ordinal_uniform follows the uniform-rating baseline normalization used by
Deep-MMMF when rating levels are equally spaced.

G. ABLATIONS
------------
Run independent validation-selected scans rather than one huge Cartesian grid:
  K; angle initialization; Method1 alpha/beta; Method2 fusion gamma/consistency; Method3 residual
  log-scale/delta regularizer; train density 70/80/90; synthetic missing rate 0.2..0.7.
Use `essential_ablation_configs` as a starting plan. Report mean +/- std and 95% CI.

H. STATISTICS
-------------
Use matched blocks (same dataset/split) and report Wilcoxon-Holm pairwise comparisons where
appropriate. Across multiple datasets/methods also report Friedman followed by Nemenyi average
ranks and a critical-difference figure. Do not run significance tests on arbitrarily unmatched runs.

I. FIGURES
----------
Generate: accuracy bars, OOB/bound plots, runtime/memory plots, NDCG curves, K ablation, training
curves, hyperparameter/density ablations, prediction-stability plots and critical-difference ranks.
Archive both PNG and vector PDF figures and the CSV used to produce each table/figure.

J. REPORTING
------------
Never mix literature-reported numbers with results from your common split in the same statistical
test. Put literature numbers in a separate 'reported results' table for sanity/context only.

K. EXACT COMMAND SEQUENCE
-------------------------
1. Install requirements:
       pip install -r requirements_q1_v7.txt
2. Keep these files together:
       mbmf_q1_final_v7.py
       mbmf_q1_publication_v7.py
3. Download raw datasets by running:
       download_datasets_q1_v7.ipynb
4. Run integrity tests:
       python -c "import mbmf_q1_publication_v7 as p; p.publication_extension_self_test()"
5. Run one small pilot with one split seed, one model seed and one K.
6. Run the full common protocol only after the pilot has completed and archived outputs correctly.

L. RUN ALL SIX DATASETS
-----------------------
    cfg = PublicationExperimentConfig()
    cfg.data.data_root = './datasets'
    core.apply_q1_safe_preset(cfg.core_config)
    cfg.core_config.split_seeds = (2026,2027,2028,2029,2030)
    cfg.core_config.model_seeds = (11,22,33,44,55)
    cfg.core_config.latent_dims = (10,20,50)
    cfg.output_dir = './results_q1_v7'
    all_runs, all_summary = run_dataset_benchmark_matrix(cfg)

The requested default dataset-method schedule is encoded in METHOD_DATASET_MATRIX.
BiasMF/MF, MBMF-F/C/N and Methods1/2/3 run on all six datasets. AutoRec and AEMC-NE
run on ML-100K/1M/10M by default, matching the supplied comparison plan.

M. SAVE / LOAD
---------------
All internally trainable families support save_model_checkpoint/load_model_checkpoint.
Loading verifies the training-triple signature by default. Do not use strict_data=False for
publication evaluation unless you are intentionally transferring a model and document it.

N. EXTERNAL AUTHOR IMPLEMENTATIONS
----------------------------------
For BMC-ADMM, Deep-MMMF and DRPRA:
  1. Build a v7 MatrixData object for the desired split.
  2. export_split_for_external(data, 'common_split.csv').
  3. Run author/official code using exactly those training/validation/test rows.
  4. Export split,userId,movieId,prediction.
  5. Use merge_external_prediction_csv so all metrics come from the same evaluator.
Do not substitute literature-reported numbers for common-split predictions.

O. ABLATION EXECUTION
---------------------
    plan = AblationPlan()
    jobs = essential_ablation_configs(cfg, plan)
    # inspect jobs first
    ablation_runs = run_essential_ablation_scan(cfg, plan, './results_q1_v7/ablations')

The default plan contains K, MBMF initialization, Method1 alpha/beta, Method2 fusion/consistency,
Method3 residual scale/regularization, 70/80/90 train fraction, synthetic missing rate, AutoRec
hidden width/regularization and AEMC-NE element width/regularization. Each scan is independent.

P. OUTPUTS TO ARCHIVE FOR THE PAPER
-----------------------------------
For every dataset archive: runs.csv, summary.csv, selected_by_validation_runs.csv,
selected_by_validation_summary.csv, prediction_stability_across_seeds.csv,
prediction_stability_across_K.csv, split assignments, ID maps, checkpoints, raw NPZ predictions,
training histories, config JSON, dataset manifest, literature protocol JSON, Wilcoxon-Holm table,
Friedman/Nemenyi tables, and both PNG and PDF figures.
"""


# =============================================================================
# V8 INFRASTRUCTURE-ONLY OVERRIDES
# =============================================================================
# IMPORTANT: the frozen scientific model file `mbmf_q1_final_v7.py` is imported
# above and is never edited or monkey-patched here.  V8 changes only data I/O,
# scalable execution, evaluation/reporting, model persistence, statistics,
# plotting, ablation orchestration and command-line/notebook ergonomics.

V8_REVISION = "8.0.1-infrastructure-fixes"
FROZEN_V7_CORE_SHA256 = V7_CORE_SHA256
FROZEN_V7_CORE_RAW_SHA256 = V7_CORE_RAW_SHA256


# -----------------------------------------------------------------------------
# V8 dataset construction: all-or-none timestamp semantics
# -----------------------------------------------------------------------------

def _attach_train_timestamps_v8(data: MatrixLike, split: pd.DataFrame) -> MatrixLike:
    """Attach timestamp features only when *every* training observation has one.

    V7 filled partially missing timestamps with zero, which can create a fake
    ancient timestamp. V8 disables time features for a split unless the complete
    training subset has finite timestamps. This changes no model equation; it only
    prevents malformed metadata from entering Method 7's optional history features.
    """
    tr = split[split["split"] == "train"]
    usable = (
        "timestamp" in tr.columns
        and len(tr) > 0
        and tr["timestamp"].notna().all()
        and np.isfinite(pd.to_numeric(tr["timestamp"], errors="coerce").to_numpy(dtype=float)).all()
    )
    if usable:
        ts = torch.as_tensor(
            pd.to_numeric(tr["timestamp"], errors="raise").to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=data.train_users.device,
        )
        data.train_timestamps = ts
        data.train_timestamp_status = "complete"
    else:
        data.train_timestamps = None
        if "timestamp" not in tr.columns:
            status = "absent"
        elif tr["timestamp"].isna().all():
            status = "absent"
        else:
            status = "partial_disabled"
        data.train_timestamp_status = status
    return data


def build_publication_matrix_data_v8(
    split: pd.DataFrame,
    spec: DatasetSpec,
    pcfg: PublicationDataConfig,
    device: torch.device,
) -> Tuple[MatrixLike, str]:
    """Build a dense or sparse MatrixData object without changing scientific logic."""
    n, m = split.userId.nunique(), split.movieId.nunique()
    backend = choose_backend(n, m, pcfg)
    if backend == "dense":
        dcfg = core.DataConfig(
            rating_min=spec.rating_min,
            rating_max=spec.rating_max,
            require_explicit_rating_scale=True,
            split_mode=pcfg.split_mode,
            train_fraction=pcfg.train_fraction,
            val_fraction=pcfg.val_fraction,
            test_fraction=pcfg.test_fraction,
            split_seed=pcfg.split_seed,
            min_train_per_user=pcfg.min_train_per_user,
            enforce_train_support_for_all_items=pcfg.enforce_train_support_for_all_items,
        )
        # Deliberately use the frozen v7/v6-core builder and attach timestamps
        # separately so partial missing timestamps cannot be zero-filled.
        builder = getattr(core, "_build_matrix_data_v6", core.build_matrix_data_v6_core)
        d = builder(split, dcfg, device)
        setattr(d, "dataset_name", spec.name)
        setattr(d, "rating_step", spec.rating_step)
        return _attach_train_timestamps_v8(d, split), backend
    d = build_sparse_matrix_data(split, spec, device)
    return _attach_train_timestamps_v8(d, split), backend

# Make all pre-existing infrastructure helpers resolve the safe V8 builder at call time.
build_publication_matrix_data = build_publication_matrix_data_v8


# -----------------------------------------------------------------------------
# Sparse Method 4 — same modules/equations as frozen core, sparse first layer only
# -----------------------------------------------------------------------------

class ScalableProposedMethod4(nn.Module):
    """Sparse-input execution of core.ProposedMethod4 with unchanged equations/state roles."""
    def __init__(self, data: SparseMatrixData, cfg: core.Method4Config, device: torch.device):
        super().__init__()
        if cfg.model_space != "centered_raw":
            raise ValueError("Method 4 requires centered_raw")
        if not 0.0 < cfg.prior_floor_fraction < 0.5:
            raise ValueError("prior_floor_fraction must be in (0,0.5)")
        self.cfg = cfg
        uin, iin = core.input_dims(data, cfg.input_mode)
        self.user_branch = core.ResidualRadiusAEBranch(uin, data.num_items, cfg.latent_dim)
        self.item_branch = core.ResidualRadiusAEBranch(iin, data.num_users, cfg.latent_dim)
        rW_raw, rH_raw, pinfo = core.method3_initial_radii(
            data,
            core.Method3Config(
                latent_dim=cfg.latent_dim,
                input_mode=cfg.input_mode,
                model_space=cfg.model_space,
                angle_init=cfg.angle_init,
                prior_mode=cfg.prior_mode,
                rho=cfg.rho,
            ),
        )
        cap = math.sqrt(data.half_range)
        eps = cap * cfg.prior_floor_fraction
        rW0 = rW_raw.clamp(eps, cap - eps)
        rH0 = rH_raw.clamp(eps, cap - eps)
        qW0 = (rW0 / cap).clamp(cfg.prior_floor_fraction, 1.0 - cfg.prior_floor_fraction)
        qH0 = (rH0 / cap).clamp(cfg.prior_floor_fraction, 1.0 - cfg.prior_floor_fraction)
        self.register_buffer("rW0_raw", rW_raw.to(device))
        self.register_buffer("rH0_raw", rH_raw.to(device))
        self.register_buffer("rW0", rW0.to(device))
        self.register_buffer("rH0", rH0.to(device))
        self.register_buffer("logitW0", (torch.log(qW0) - torch.log1p(-qW0)).to(device))
        self.register_buffer("logitH0", (torch.log(qH0) - torch.log1p(-qH0)).to(device))
        self.radius_cap = float(cap)
        self.prior_info = {
            **pinfo,
            "prior_projection_rate_user": float((rW_raw != rW0).float().mean().item()),
            "prior_projection_rate_item": float((rH_raw != rH0).float().mean().item()),
            "radius_cap": float(cap),
        }
        self.Phi = nn.Parameter(core.initialise_angles(data.num_users, cfg.latent_dim, cfg.angle_init, device))
        self.Theta = nn.Parameter(core.initialise_angles(data.num_items, cfg.latent_dim, cfg.angle_init, device))
        self.to(device)

    def encoded(self, data: SparseMatrixData):
        uv, um, iv, im = sparse_encoder_matrices(data, self.cfg.model_space)
        zu = _sparse_residual_encode(self.user_branch, uv, um, data.num_items, self.cfg.input_mode)
        zi = _sparse_residual_encode(self.item_branch, iv, im, data.num_users, self.cfg.input_mode)
        du = core._bounded_shift(
            self.user_branch.raw_log_correction(zu),
            self.cfg.max_abs_logit_shift,
            self.cfg.residual_transform,
        )
        di = core._bounded_shift(
            self.item_branch.raw_log_correction(zi),
            self.cfg.max_abs_logit_shift,
            self.cfg.residual_transform,
        )
        rW = self.radius_cap * torch.sigmoid(self.logitW0 + du)
        rH = self.radius_cap * torch.sigmoid(self.logitH0 + di)
        W = core.spherical_to_cartesian(self.Phi, rW)
        H = core.spherical_to_cartesian(self.Theta, rH)
        return zu, zi, du, di, rW, rH, W, H

    def predict_raw_pairs(self, data: SparseMatrixData, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        *_, W, H = self.encoded(data)
        return torch.sum(W[users] * H[items], dim=1) + data.midpoint

    @torch.no_grad()
    def diagnostics(self, data: SparseMatrixData, users: torch.Tensor, items: torch.Tensor) -> Dict[str, float]:
        _, _, du, di, rW, rH, W, H = self.encoded(data)
        p = torch.sum(W[users] * H[items], dim=1)
        pair_bound = rW[users, 0] * rH[items, 0]
        raw = p + data.midpoint
        rv = torch.maximum(torch.relu(data.rating_min - raw), torch.relu(raw - data.rating_max))
        return {
            "max_structural_bound_violation": float(torch.relu(torch.abs(p) - pair_bound).max().item()) if p.numel() else 0.0,
            "max_rating_bound_violation": float(rv.max().item()) if rv.numel() else 0.0,
            "max_pair_bound_eval": float(pair_bound.max().item()) if pair_bound.numel() else float("nan"),
            "mean_rW": float(rW.mean().item()),
            "mean_rH": float(rH.mean().item()),
            "min_logit_shift_user": float(du.min().item()),
            "max_logit_shift_user": float(du.max().item()),
            "min_logit_shift_item": float(di.min().item()),
            "max_logit_shift_item": float(di.max().item()),
            **self.prior_info,
        }


def train_scalable_method4(
    model: ScalableProposedMethod4,
    data: SparseMatrixData,
    cfg: core.Method4Config,
    chunk_size: int = 250_000,
) -> Dict[str, Any]:
    """Exact MSE-form Method-4 objective in edge chunks; equations are unchanged."""
    if cfg.loss_form != "mse":
        raise NotImplementedError("Sparse Method 4 supports loss_form='mse'; use dense backend for Frobenius reproduction.")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = float("inf"); best_state = copy.deepcopy(model.state_dict()); best_epoch = 0; wait = 0
    hist: List[Dict[str, Any]] = []; t0 = time.perf_counter(); N = int(data.train_users.numel())
    for ep in range(1, cfg.max_epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        zu, zi, du, di, _, _, W, H = model.encoded(data)
        hu = torch.relu(model.user_branch.decoder_hidden(zu))
        hi = torch.relu(model.item_branch.decoder_hidden(zi))
        target_all = data.train_ratings_raw - data.midpoint
        su = si = sm = 0.0
        slices = list(_chunks(N, chunk_size))
        for q, sl in enumerate(slices):
            u = data.train_users[sl]; i = data.train_items[sl]; y = target_all[sl]
            ru = torch.sum(hu[u] * model.user_branch.decoder_output.weight[i], dim=1) + model.user_branch.decoder_output.bias[i]
            ri = torch.sum(hi[i] * model.item_branch.decoder_output.weight[u], dim=1) + model.item_branch.decoder_output.bias[u]
            eu = torch.sum((ru - y) ** 2); ei = torch.sum((ri - y) ** 2)
            if cfg.ae_loss_mode == "separate":
                ae = eu + ei
            elif cfg.ae_loss_mode == "averaged":
                ae = torch.sum((0.5 * (ru + ri) - y) ** 2)
            else:
                raise ValueError(cfg.ae_loss_mode)
            pm = torch.sum(W[u] * H[i], dim=1)
            ms = torch.sum((pm - y) ** 2)
            loss = (cfg.alpha_ae * ae + cfg.beta_mbmf * ms) / N
            if q == 0 and cfg.delta_l2:
                loss = loss + cfg.delta_l2 * (torch.mean(du.square()) + torch.mean(di.square()))
            loss.backward(retain_graph=q < len(slices) - 1)
            su += float(eu.detach()); si += float(ei.detach()); sm += float(ms.detach())
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if ep == 1 or ep % cfg.eval_every == 0:
            val = evaluate_publication_model(model, data, "val")
            total = cfg.alpha_ae * (su + si) / N + cfg.beta_mbmf * sm / N + cfg.delta_l2 * float((torch.mean(du.square()) + torch.mean(di.square())).detach())
            hist.append({"epoch": ep, "loss": total, "val_RMSE": val["RMSE"]})
            if val["RMSE"] < best - 1e-8:
                best = val["RMSE"]; best_state = copy.deepcopy(model.state_dict()); best_epoch = ep; wait = 0
            else:
                wait += 1
                if wait >= cfg.val_patience:
                    break
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_val_RMSE": best, "elapsed_sec": time.perf_counter() - t0, "history": hist}


def train_scalable_method56(
    model: core.DecoderCoupledDualViewMBMF,
    data: MatrixLike,
    cfg: Union[core.Method5Config, core.Method6Config],
    chunk_size: int = 250_000,
) -> Dict[str, Any]:
    """Chunked edge objective exactly matching frozen Method 5/6 loss terms."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = float("inf"); best_state = copy.deepcopy(model.state_dict()); best_epoch = 0; wait = 0
    hist: List[Dict[str, Any]] = []; t0 = time.perf_counter(); N = int(data.train_users.numel())
    width = float(data.rating_max - data.rating_min)
    for ep in range(1, cfg.max_epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        state = model._all_state(data)
        zu, zi, du, di, ru_u, ru_i, ri_i, ri_u, qu, pu, qi, pi = state
        l2 = 0.25 * (zu.square().mean() + zi.square().mean() + du.square().mean() + di.square().mean())
        sums = {"pred": 0.0, "ru": 0.0, "ri": 0.0, "cons": 0.0}
        slices = list(_chunks(N, chunk_size))
        for q, sl in enumerate(slices):
            u = data.train_users[sl]; i = data.train_items[sl]; y = data.train_ratings_raw[sl]
            uv_cos = torch.sum(qu[u] * pu[i], dim=1)
            iv_cos = torch.sum(qi[u] * pi[i], dim=1)
            uv = data.midpoint + ru_u[u, 0] * ru_i[i, 0] * uv_cos
            iv = data.midpoint + ri_u[u, 0] * ri_i[i, 0] * iv_cos
            if cfg.branch_mode == "user":
                pred = uv
            elif cfg.branch_mode == "item":
                pred = iv
            else:
                g = float(cfg.fusion_weight_user_view)
                pred = g * uv + (1.0 - g) * iv
            target_unit = (y - data.rating_min) / width
            pred_unit = (pred - data.rating_min) / width
            rec_u = torch.sum(zu[u] * du[i], dim=1) + model.user_view_item_bias(i).squeeze(-1)
            rec_i = torch.sum(zi[i] * di[u], dim=1) + model.item_view_user_bias(u).squeeze(-1)
            sp = torch.sum((pred_unit - target_unit) ** 2)
            sru = torch.sum((rec_u - target_unit) ** 2)
            sri = torch.sum((rec_i - target_unit) ** 2)
            sc = torch.sum(((uv - iv) / width) ** 2)
            loss = (
                cfg.prediction_weight * sp
                + cfg.user_reconstruction_weight * sru
                + cfg.item_reconstruction_weight * sri
                + cfg.branch_consistency_weight * sc
            ) / N
            if q == 0 and cfg.l2_code_weight:
                loss = loss + cfg.l2_code_weight * l2
            loss.backward(retain_graph=q < len(slices) - 1)
            sums["pred"] += float(sp.detach()); sums["ru"] += float(sru.detach()); sums["ri"] += float(sri.detach()); sums["cons"] += float(sc.detach())
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if ep == 1 or ep % cfg.eval_every == 0:
            val = evaluate_publication_model(model, data, "val")
            total = (
                cfg.prediction_weight * sums["pred"]
                + cfg.user_reconstruction_weight * sums["ru"]
                + cfg.item_reconstruction_weight * sums["ri"]
                + cfg.branch_consistency_weight * sums["cons"]
            ) / N + cfg.l2_code_weight * float(l2.detach())
            hist.append({"epoch": ep, "loss": total, "val_RMSE": val["RMSE"]})
            if val["RMSE"] < best - 1e-8:
                best = val["RMSE"]; best_state = copy.deepcopy(model.state_dict()); best_epoch = ep; wait = 0
            else:
                wait += 1
                if wait >= cfg.val_patience:
                    break
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_val_RMSE": best, "elapsed_sec": time.perf_counter() - t0, "history": hist}


# -----------------------------------------------------------------------------
# Extended structural bounds for Methods 4-7
# -----------------------------------------------------------------------------

def individual_bound_arrays_v8(model: nn.Module, data: MatrixLike, u: torch.Tensor, i: torch.Tensor) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    with torch.no_grad():
        if isinstance(model, core.PaperMBMF):
            b = model.rW[u, 0] * model.rH[i, 0]
            shift = float(model.rating_origin_shift.item()) if isinstance(model, OriginShiftedPaperMBMF) else 0.0
            if model.variant == "MBMF-F":
                lo, hi = data.midpoint - b, data.midpoint + b
            elif model.variant == "MBMF-C":
                cmin = float(model.data_meta[2]) - shift; lo, hi = torch.full_like(b, cmin), cmin + 2 * b
            elif model.variant == "MBMF-N":
                lo, hi = torch.full_like(b, -shift), b - shift
            else:
                return None
        elif isinstance(model, (core.ThesisMethod1, ScalableThesisMethod1)):
            _, _, rw, rh, _, _ = model.encoded(data); b = rw[u, 0] * rh[i, 0]
            if model.cfg.model_space == "centered_raw": lo, hi = data.midpoint - b, data.midpoint + b
            else:
                w = data.rating_max - data.rating_min; lo, hi = data.rating_min - w * b, data.rating_min + w * b
        elif isinstance(model, (core.ThesisMethod2, ScalableThesisMethod2)):
            if isinstance(model, core.ThesisMethod2):
                pU, pI, p, (a, b, c, d) = model.predict_centered_views(data, u, i)
            else:
                pU, pI, p, (a, b, c, d) = model.predict_views(data, u, i)
            g = float(model.cfg.fusion_weight_user_view); bb = g * a[u, 0] * b[i, 0] + (1 - g) * c[i, 0] * d[u, 0]
            if model.cfg.model_space == "centered_raw": lo, hi = data.midpoint - bb, data.midpoint + bb
            else:
                w = data.rating_max - data.rating_min; lo, hi = data.rating_min - w * bb, data.rating_min + w * bb
        elif isinstance(model, (core.ProposedMethod3, ScalableProposedMethod3)):
            _, _, _, _, rw, rh, _, _ = model.encoded(data); b = rw[u, 0] * rh[i, 0]
            if model.cfg.model_space == "centered_raw": lo, hi = data.midpoint - b, data.midpoint + b
            else:
                w = data.rating_max - data.rating_min; lo, hi = data.rating_min - w * b, data.rating_min + w * b
        elif isinstance(model, (core.ProposedMethod4, ScalableProposedMethod4)):
            _, _, _, _, rw, rh, _, _ = model.encoded(data); b = rw[u, 0] * rh[i, 0]
            lo, hi = data.midpoint - b, data.midpoint + b
        elif isinstance(model, core.DecoderCoupledDualViewMBMF):
            _, _, pred, state = model.predict_views(data, u, i)
            _, _, _, _, ru_u, ru_i, ri_i, ri_u, *_ = state
            bu = ru_u[u, 0] * ru_i[i, 0]; bi = ri_u[u, 0] * ri_i[i, 0]
            if model.cfg.branch_mode == "user": bb = bu
            elif model.cfg.branch_mode == "item": bb = bi
            else:
                g = float(model.cfg.fusion_weight_user_view); bb = g * bu + (1.0 - g) * bi
            lo, hi = data.midpoint - bb, data.midpoint + bb
        elif isinstance(model, core.ProposedMethod7):
            # Method 7 has a hard global interval theorem but not a tighter per-pair
            # magnitude envelope. Report the exact guaranteed rating interval.
            lo = torch.full_like(u, float(data.rating_min), dtype=torch.float32)
            hi = torch.full_like(u, float(data.rating_max), dtype=torch.float32)
        else:
            return None
    return lo.detach().cpu().numpy().astype(float), hi.detach().cpu().numpy().astype(float)

# Override lookup used by evaluate_publication_model at call time.
individual_bound_arrays = individual_bound_arrays_v8


# -----------------------------------------------------------------------------
# V8 checkpointing: persistent selected models for every internal family
# -----------------------------------------------------------------------------

def checkpoint_family_v8(model: nn.Module) -> str:
    pairs = [
        (core.PaperMBMF, "PaperMBMF"),
        (core.ThesisMethod1, "ThesisMethod1"),
        (ScalableThesisMethod1, "ScalableThesisMethod1"),
        (core.ThesisMethod2, "ThesisMethod2"),
        (ScalableThesisMethod2, "ScalableThesisMethod2"),
        (core.ProposedMethod3, "ProposedMethod3"),
        (ScalableProposedMethod3, "ScalableProposedMethod3"),
        (core.ProposedMethod4, "ProposedMethod4"),
        (ScalableProposedMethod4, "ScalableProposedMethod4"),
        (core.ProposedMethod7, "ProposedMethod7"),
        (core.DecoderCoupledDualViewMBMF, "DecoderCoupledDualViewMBMF"),
        (ItemAutoRec, "AutoRec"),
        (AEMCNE, "AEMCNE"),
        (BiasMF, "BiasMF"),
    ]
    for cls, name in pairs:
        if isinstance(model, cls):
            return name
    return model.__class__.__name__


def save_model_checkpoint_v8(
    path: Union[str, Path],
    model: nn.Module,
    *,
    config: Any,
    data: MatrixLike,
    train_info: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    ex = dict(extra or {})
    payload = {
        "format": "MBMF-Q1-v8",
        "frozen_v7_core_sha256": FROZEN_V7_CORE_SHA256,
        "family": checkpoint_family_v8(model),
        "config": _config_dict(config),
        "data_signature": data_signature(data),
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "train_info": dict(train_info or {}),
        "extra": ex,
    }
    if isinstance(model, core.PaperMBMF):
        payload.update({"variant": model.variant, "init_mode": ex.get("init_mode", "isotropic_gaussian")})
    if isinstance(model, core.DecoderCoupledDualViewMBMF):
        payload["dual_view_config_class"] = model.cfg.__class__.__name__
    torch.save(payload, p)
    return p


def load_model_checkpoint_v8(path: Union[str, Path], data: MatrixLike, device: torch.device, *, strict_data: bool = True) -> Tuple[nn.Module, Dict[str, Any]]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != "MBMF-Q1-v8":
        raise ValueError("unsupported checkpoint format")
    if ck.get("frozen_v7_core_sha256") != FROZEN_V7_CORE_SHA256:
        raise ValueError("checkpoint scientific-core SHA256 differs from this package")
    if strict_data:
        sig = data_signature(data); old = ck["data_signature"]
        for k in ("num_users", "num_items", "rating_min", "rating_max", "train_sha256"):
            if str(sig[k]) != str(old[k]):
                raise ValueError(f"checkpoint/data mismatch for {k}: {old[k]} vs {sig[k]}")
    fam, c = ck["family"], ck["config"]
    if fam == "PaperMBMF":
        cfg = core.PaperBaselineConfig(**c); model = _make_paper_baseline_v8(data, cfg, ck["variant"], ck.get("init_mode", "isotropic_gaussian"), device)
    elif fam in ("ThesisMethod1", "ScalableThesisMethod1"):
        cfg = core.Method1Config(**c); model = (ScalableThesisMethod1 if fam.startswith("Scalable") else core.ThesisMethod1)(data, cfg, device)
    elif fam in ("ThesisMethod2", "ScalableThesisMethod2"):
        cfg = core.Method2Config(**c); model = (ScalableThesisMethod2 if fam.startswith("Scalable") else core.ThesisMethod2)(data, cfg, device)
    elif fam in ("ProposedMethod3", "ScalableProposedMethod3"):
        cfg = core.Method3Config(**c); model = (ScalableProposedMethod3 if fam.startswith("Scalable") else core.ProposedMethod3)(data, cfg, device)
    elif fam in ("ProposedMethod4", "ScalableProposedMethod4"):
        cfg = core.Method4Config(**c); model = (ScalableProposedMethod4 if fam.startswith("Scalable") else core.ProposedMethod4)(data, cfg, device)
    elif fam == "DecoderCoupledDualViewMBMF":
        clsname = ck.get("dual_view_config_class", "Method5Config")
        cfg = core.Method6Config(**c) if clsname == "Method6Config" or c.get("radius_mode") == "range_safe" else core.Method5Config(**c)
        model = core.DecoderCoupledDualViewMBMF(data, cfg, device)
    elif fam == "ProposedMethod7":
        cfg = core.Method7Config(**c); model = core.ProposedMethod7(data, cfg, device)
    elif fam == "BiasMF":
        cfg = BiasMFConfig(**c); model = BiasMF(data.num_users, data.num_items, cfg, device)
    elif fam == "AutoRec":
        cfg = AutoRecConfig(**c); model = ItemAutoRec(data, cfg, device)
    elif fam == "AEMCNE":
        cfg = AEMCNEConfig(**c); model = AEMCNE(data, cfg, device)
    else:
        raise ValueError(f"unknown checkpoint family {fam}")
    model.load_state_dict(ck["state_dict"], strict=True); model.to(device); model.eval()
    return model, ck


# -----------------------------------------------------------------------------
# V8 method registry and configs
# -----------------------------------------------------------------------------

METHOD_DATASET_MATRIX_V8: Dict[str, Tuple[str, ...]] = {
    "BiasMF": ("synthetic", "jester", "bookcrossing", "ml-100k", "ml-1m", "ml-10m"),
    "MF": ("synthetic", "jester", "bookcrossing", "ml-100k", "ml-1m", "ml-10m"),
    "MBMF-F": ("synthetic", "jester", "bookcrossing", "ml-100k", "ml-1m", "ml-10m"),
    "MBMF-C": ("synthetic", "jester", "bookcrossing", "ml-100k", "ml-1m", "ml-10m"),
    "MBMF-N": ("synthetic", "jester", "bookcrossing", "ml-100k", "ml-1m", "ml-10m"),
    "AutoRec": ("ml-100k", "ml-1m", "ml-10m"),
    "AEMC-NE": ("ml-100k", "ml-1m", "ml-10m"),
    **{f"Method{i}": ("synthetic", "jester", "bookcrossing", "ml-100k", "ml-1m", "ml-10m") for i in range(1, 8)},
}
EXTERNAL_COMPETITORS_V8 = ("BMC-ADMM", "Deep-MMMF", "DRPRA")


def _default_core_config_v8() -> core.ExperimentConfig:
    c = core.ExperimentConfig()
    core.apply_q1_safe_preset(c)
    c.latent_dims = (10, 20, 50)
    return c


@dataclass
class PublicationExperimentConfigV8:
    data: PublicationDataConfig = field(default_factory=PublicationDataConfig)
    core_config: core.ExperimentConfig = field(default_factory=_default_core_config_v8)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    methods: Tuple[str, ...] = (
        "BiasMF", "MF", "MBMF-F", "MBMF-C", "MBMF-N", "AutoRec", "AEMC-NE",
        "Method1", "Method2", "Method3", "Method4", "Method5", "Method6", "Method7",
    )
    output_dir: str = "./q1_publication_results_v8"
    save_candidate_checkpoints: bool = False
    save_selected_checkpoints: bool = True
    save_predictions: bool = True
    scalable_chunk_size: int = 250_000
    respect_dataset_method_matrix: bool = True
    auto_generate_plots: bool = True
    auto_run_statistics: bool = True
    strict_locked_test: bool = True
    validation_only: bool = False  # infrastructure guard for hyperparameter scans; no test evaluation
    biasmf: BiasMFConfig = field(default_factory=BiasMFConfig)
    autorec: AutoRecConfig = field(default_factory=AutoRecConfig)
    aemcne: AEMCNEConfig = field(default_factory=AEMCNEConfig)


def method_supported_on_dataset_v8(method: str, dataset: str) -> bool:
    return dataset in METHOD_DATASET_MATRIX_V8.get(method, (dataset,))


# -----------------------------------------------------------------------------
# V8 instantiate/train dispatch; no scientific formula is modified
# -----------------------------------------------------------------------------

class OriginShiftedPaperMBMF(core.PaperMBMF):
    """Signed-rating adapter for the *unchanged* MBMF-C/N equations.

    Eq.42 assumes nonnegative historical observations. For a scale such as
    Jester [-10,10], fit C/N on the fixed, documented origin shift y'=y-rmin
    and subtract the same constant on prediction. The shift is not fitted
    from validation/test labels. All core objectives and updates stay intact.
    This is an explicitly labelled dataset/protocol adaptation, not the
    unmodified original MBMF-C/N benchmark on negative-valued observations.
    """
    def __init__(self, data: MatrixLike, cfg: core.PaperBaselineConfig, variant: str,
                 init_mode: str, device: torch.device):
        if variant not in ("MBMF-C", "MBMF-N") or data.rating_min >= 0:
            raise ValueError("OriginShiftedPaperMBMF is for signed C/N data only")
        shift = -float(data.rating_min)
        shifted = copy.copy(data)
        shifted.rating_min = float(data.rating_min + shift)
        shifted.rating_max = float(data.rating_max + shift)
        shifted.midpoint = float(data.midpoint + shift)
        shifted.train_ratings_raw = data.train_ratings_raw + shift
        shifted.val_ratings_raw = data.val_ratings_raw + shift
        shifted.test_ratings_raw = data.test_ratings_raw + shift
        shifted.user_train_mean_raw = data.user_train_mean_raw + shift
        super().__init__(shifted, cfg, variant, init_mode, device)
        self.register_buffer("rating_origin_shift", torch.tensor(shift, dtype=torch.float32, device=device))

    def native_targets(self, raw: torch.Tensor, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return super().native_targets(raw + self.rating_origin_shift, users, items)

    def predict_raw_pairs(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return super().predict_raw_pairs(users, items) - self.rating_origin_shift


def _make_paper_baseline_v8(data: MatrixLike, cfg: core.PaperBaselineConfig,
                             variant: str, init_mode: str, device: torch.device) -> core.PaperMBMF:
    if variant in ("MBMF-C", "MBMF-N") and data.rating_min < 0:
        return OriginShiftedPaperMBMF(data, cfg, variant, init_mode, device)
    return core.PaperMBMF(data, cfg, variant, init_mode, device)


def instantiate_model_v8(method: str, data: MatrixLike, cfg: PublicationExperimentConfigV8, latent_dim: int, model_seed: int, device: torch.device, init_mode: Optional[str] = None) -> Tuple[nn.Module, Any]:
    core.set_global_seed(model_seed)
    sparse = isinstance(data, SparseMatrixData)
    cc = cfg.core_config
    if method in ("MBMF-F", "MBMF-C", "MBMF-N"):
        c = copy.deepcopy(cc.paper); c.latent_dim = latent_dim; im = init_mode or c.default_init
        return _make_paper_baseline_v8(data, c, method, im, device), c
    if method == "Method1":
        c = copy.deepcopy(cc.method1); c.latent_dim = latent_dim
        return (ScalableThesisMethod1 if sparse else core.ThesisMethod1)(data, c, device), c
    if method == "Method2":
        c = copy.deepcopy(cc.method2); c.latent_dim = latent_dim
        return (ScalableThesisMethod2 if sparse else core.ThesisMethod2)(data, c, device), c
    if method == "Method3":
        c = copy.deepcopy(cc.method3); c.latent_dim = latent_dim
        return (ScalableProposedMethod3 if sparse else core.ProposedMethod3)(data, c, device), c
    if method == "Method4":
        c = copy.deepcopy(cc.method4); c.latent_dim = latent_dim
        return (ScalableProposedMethod4 if sparse else core.ProposedMethod4)(data, c, device), c
    if method == "Method5":
        c = copy.deepcopy(cc.method5); c.latent_dim = latent_dim
        return core.DecoderCoupledDualViewMBMF(data, c, device), c
    if method == "Method6":
        c = copy.deepcopy(cc.method6); c.latent_dim = latent_dim
        return core.DecoderCoupledDualViewMBMF(data, c, device), c
    if method == "Method7":
        c = copy.deepcopy(cc.method7); c.latent_dim = latent_dim
        return core.ProposedMethod7(data, c, device), c
    if method == "BiasMF":
        c = copy.deepcopy(cfg.biasmf); c.latent_dim = latent_dim
        return BiasMF(data.num_users, data.num_items, c, device), c
    if method == "MF":
        c = copy.deepcopy(cfg.biasmf); c.use_bias = False; c.latent_dim = latent_dim
        return BiasMF(data.num_users, data.num_items, c, device), c
    if method == "AutoRec":
        c = copy.deepcopy(cfg.autorec)
        return ItemAutoRec(data, c, device), c
    if method == "AEMC-NE":
        c = copy.deepcopy(cfg.aemcne)
        if cfg.data.dataset == "ml-100k": c.main_hidden_dim = 500
        elif cfg.data.dataset == "ml-1m": c.main_hidden_dim = 900
        elif cfg.data.dataset == "ml-10m": c.main_hidden_dim = 1000
        return AEMCNE(data, c, device), c
    raise ValueError(method)


def train_model_v8(method: str, model: nn.Module, data: MatrixLike, mcfg: Any, pcfg: PublicationExperimentConfigV8) -> Dict[str, Any]:
    sparse = isinstance(data, SparseMatrixData)
    if method.startswith("MBMF-"):
        return train_paper_mbmf_chunked(model, data, mcfg, pcfg.scalable_chunk_size) if sparse else core.train_paper_mbmf(model, data, mcfg)
    if method == "Method1":
        return train_scalable_method1(model, data, mcfg, pcfg.scalable_chunk_size) if sparse else core.train_thesis_model(model, data, max_epochs=mcfg.max_epochs, lr=mcfg.lr, weight_decay=mcfg.weight_decay, eval_every=mcfg.eval_every, val_patience=mcfg.val_patience, grad_clip=mcfg.grad_clip)
    if method == "Method2":
        return train_scalable_method2(model, data, mcfg, pcfg.scalable_chunk_size) if sparse else core.train_thesis_model(model, data, max_epochs=mcfg.max_epochs, lr=mcfg.lr, weight_decay=mcfg.weight_decay, eval_every=mcfg.eval_every, val_patience=mcfg.val_patience, grad_clip=mcfg.grad_clip)
    if method == "Method3":
        return train_scalable_method3(model, data, mcfg, pcfg.scalable_chunk_size) if sparse else core.train_thesis_model(model, data, max_epochs=mcfg.max_epochs, lr=mcfg.lr, weight_decay=mcfg.weight_decay, eval_every=mcfg.eval_every, val_patience=mcfg.val_patience, grad_clip=mcfg.grad_clip)
    if method == "Method4":
        return train_scalable_method4(model, data, mcfg, pcfg.scalable_chunk_size) if sparse else core.train_thesis_model(model, data, max_epochs=mcfg.max_epochs, lr=mcfg.lr, weight_decay=mcfg.weight_decay, eval_every=mcfg.eval_every, val_patience=mcfg.val_patience, grad_clip=mcfg.grad_clip)
    if method in ("Method5", "Method6"):
        return train_scalable_method56(model, data, mcfg, pcfg.scalable_chunk_size) if sparse else core.train_thesis_model(model, data, max_epochs=mcfg.max_epochs, lr=mcfg.lr, weight_decay=mcfg.weight_decay, eval_every=mcfg.eval_every, val_patience=mcfg.val_patience, grad_clip=mcfg.grad_clip)
    if method == "Method7":
        return core.train_method7_minibatch(model, data, mcfg)
    if method in ("BiasMF", "MF"):
        return train_biasmf(model, data, mcfg)
    if method == "AutoRec":
        return train_autorec(model, data, mcfg, pcfg.scalable_chunk_size)
    if method == "AEMC-NE":
        return train_aemcne(model, data, mcfg, pcfg.scalable_chunk_size)
    raise ValueError(method)


# -----------------------------------------------------------------------------
# V8 strict validation-selection / locked-test runner
# -----------------------------------------------------------------------------

def _candidate_config_key_v8(method: str, mcfg: Any, init_mode: str) -> str:
    return json.dumps({"method": method, "config": _jsonable(_config_dict(mcfg)), "init_mode": init_mode}, sort_keys=True, default=str)


def _selected_config_rows_v8(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    if any(c.startswith("test_") for c in candidates.columns):
        raise AssertionError("candidate table must not contain test metrics")
    parts = []
    for (dataset, split_seed, method), g in candidates.groupby(["dataset", "split_seed", "method"], dropna=False):
        if not np.isfinite(g["val_RMSE"].to_numpy(dtype=float)).all():
            raise FloatingPointError(f"nonfinite validation RMSE for {dataset}/{split_seed}/{method}")
        means = g.groupby("config_key", as_index=False)["val_RMSE"].mean().sort_values(["val_RMSE", "config_key"], kind="mergesort")
        key = means.iloc[0]["config_key"]
        q = g[g.config_key == key].copy(); q["selection_val_RMSE_mean"] = float(means.iloc[0].val_RMSE); parts.append(q)
    return pd.concat(parts, ignore_index=True) if parts else candidates.iloc[:0].copy()


def _prediction_archive_path_v8(out: Path, dataset: str, method: str, split_seed: int, model_seed: int, k: Union[int, float], init_mode: str, split: str) -> Path:
    safe = method.replace("/", "_")
    kval = "NA" if pd.isna(k) else str(int(k))
    return out / "predictions" / f"{dataset}__{safe}__split{split_seed}__seed{model_seed}__K{kval}__{init_mode}__{split}.npz"


def save_prediction_archive_v8(path: Path, data: MatrixLike, pred: np.ndarray, split: str) -> None:
    u, i, y = _split_tensors(data, split)
    pair = u.detach().cpu().numpy().astype(np.int64) * data.num_items + i.detach().cpu().numpy().astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pair_id=pair, prediction=pred.astype(np.float32), truth=y.detach().cpu().numpy().astype(np.float32))


def _model_diagnostics_v8(model: nn.Module, data: MatrixLike, split: str) -> Dict[str, float]:
    u, i, _ = _split_tensors(data, split)
    if not hasattr(model, "diagnostics"):
        return {}
    try:
        return model.diagnostics(data, u, i)
    except TypeError:
        return model.diagnostics(u, i)


def run_publication_study_v8(cfg: PublicationExperimentConfigV8, device: Optional[torch.device] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Publication runner with strict validation-only selection and locked test.

    Train and validation metrics are computed for every candidate. Test metrics are
    first computed only after one configuration has been selected by mean validation
    RMSE within each dataset/split/model block. Selected checkpoints are persisted.
    """
    if not cfg.strict_locked_test:
        raise ValueError("V8 publication runner requires strict_locked_test=True")
    scientific_core_immutability_test_v8()
    device = device or core.get_device()
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    cand_dir = out / "_candidate_checkpoints"; cand_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = out / "selected_models"; selected_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []; histories: Dict[str, Any] = {}; split_frames: Dict[int, pd.DataFrame] = {}

    raw, spec = load_publication_dataset(cfg.data, save_mappings_to=out / "id_maps")
    manifest = {
        "dataset": cfg.data.dataset,
        "documented_range": [spec.rating_min, spec.rating_max],
        "rating_step": spec.rating_step,
        "n_interactions": int(len(raw)),
        "n_users": int(raw.userId.nunique()),
        "n_items": int(raw.movieId.nunique()),
        "frozen_v7_core_sha256": FROZEN_V7_CORE_SHA256,
        "infrastructure_revision": V8_REVISION,
    }
    with open(out / "dataset_manifest.json", "w", encoding="utf-8") as f: json.dump(manifest, f, indent=2)

    split_seeds = tuple(cfg.core_config.split_seeds)
    model_seeds = tuple(cfg.core_config.model_seeds)
    latent_dims = tuple(cfg.core_config.latent_dims)
    first_k = int(latent_dims[0])

    for split_seed in split_seeds:
        dc = copy.deepcopy(cfg.data); dc.split_seed = int(split_seed)
        split = make_core_split(raw, spec, dc)
        if any((split.split == sp).sum() == 0 for sp in ("train", "val", "test")):
            raise ValueError(f"split {split_seed} has empty train/validation/test partition; "
                             "cannot report publication metrics")
        split_frames[int(split_seed)] = split.copy()
        split.to_csv(out / f"split_assignments_{split_seed}.csv", index=False)
        data, backend = build_publication_matrix_data_v8(split, spec, dc, device)
        export_split_for_external(data, out / "external_splits" / f"{cfg.data.dataset}__split{split_seed}.csv",
                                  id_map_dir=out / "id_maps", dataset_name=cfg.data.dataset)
        for k in latent_dims:
            for method in cfg.methods:
                if cfg.respect_dataset_method_matrix and not method_supported_on_dataset_v8(method, cfg.data.dataset):
                    continue
                if method in ("AutoRec", "AEMC-NE") and int(k) != first_k:
                    continue
                modes = cfg.core_config.paper.init_modes if method in ("MBMF-F", "MBMF-C", "MBMF-N") else ("NA",)
                for init_mode in modes:
                    for model_seed in model_seeds:
                        model, mcfg = instantiate_model_v8(method, data, cfg, int(k), int(model_seed), device, None if init_mode == "NA" else init_mode)
                        with PeakMemoryMonitor(device) as mm:
                            t0 = time.perf_counter(); info = train_model_v8(method, model, data, mcfg, cfg); train_wall = time.perf_counter() - t0
                        train_met = evaluate_publication_model(model, data, "train", metric_cfg=cfg.metrics)
                        if cfg.save_predictions:
                            val_met, val_pred = evaluate_publication_model(model, data, "val", metric_cfg=cfg.metrics, return_predictions=True)
                        else:
                            val_met = evaluate_publication_model(model, data, "val", metric_cfg=cfg.metrics)
                            val_pred = None
                        diag_val = _model_diagnostics_v8(model, data, "val")
                        cp = cand_dir / f"{cfg.data.dataset}__{method}__split{split_seed}__seed{model_seed}__K{k}__{init_mode}.pt"
                        save_model_checkpoint_v8(cp, model, config=mcfg, data=data, train_info=info, extra={"dataset": cfg.data.dataset, "init_mode": init_mode, "method": method, "backend": backend})
                        row = {
                            "dataset": cfg.data.dataset, "method": method, "backend": backend,
                            "split_seed": int(split_seed), "model_seed": int(model_seed),
                            "K": (np.nan if method in ("AutoRec", "AEMC-NE") else int(k)),
                            "init_mode": init_mode, "train_wall_sec": train_wall,
                            "parameter_count": core.parameter_count(model), "checkpoint_path": str(cp),
                            "config_key": _candidate_config_key_v8(method, mcfg, init_mode),
                            "best_epoch": info.get("best_epoch", info.get("final_epoch", np.nan)),
                            "timestamp_status": getattr(data, "train_timestamp_status", "unknown"),
                            **mm.metrics(),
                            **{f"train_{a}": b for a, b in train_met.items()},
                            **{f"val_{a}": b for a, b in val_met.items()},
                            **{f"valdiag_{a}": b for a, b in diag_val.items()},
                        }
                        if cfg.save_predictions and val_pred is not None:
                            vpp = _prediction_archive_path_v8(out, cfg.data.dataset, method, int(split_seed), int(model_seed), (np.nan if method in ("AutoRec", "AEMC-NE") else int(k)), init_mode, "val")
                            save_prediction_archive_v8(vpp, data, val_pred, "val")
                            row["val_prediction_file"] = str(vpp)
                        rows.append(row)
                        histories[f"{method}|split={split_seed}|seed={model_seed}|K={k}|init={init_mode}"] = info.get("history", [])
                        pd.DataFrame(rows).drop(columns=["checkpoint_path"], errors="ignore").to_csv(out / "candidate_validation_runs.csv", index=False)
                        del model; gc.collect()
                        if device.type == "cuda": torch.cuda.empty_cache()

    candidates = pd.DataFrame(rows)
    if cfg.save_predictions and not candidates.empty:
        kstab = prediction_variation_across_k_v8(candidates, split="val")
        if not kstab.empty:
            kstab.to_csv(out / "prediction_stability_across_K_validation.csv", index=False)
            if cfg.auto_generate_plots:
                plot_k_stability_v8(kstab, out / "figures")
    forbidden = [c for c in candidates.columns if c.startswith("test_")]
    if forbidden:
        raise AssertionError(f"test leakage into candidate table: {forbidden}")
    selected = _selected_config_rows_v8(candidates)
    selected.drop(columns=["checkpoint_path"], errors="ignore").to_csv(out / "selected_by_validation_candidates.csv", index=False)
    if cfg.validation_only:
        # A hyperparameter *search* must not expose test results before the grid
        # has been frozen. Keep every candidate and its validation metrics.
        with open(out / "training_histories.json", "w", encoding="utf-8") as f:
            json.dump(_jsonable(histories), f, indent=2)
        with open(out / "config.json", "w", encoding="utf-8") as f:
            json.dump(_jsonable(asdict(cfg)), f, indent=2)
        if not cfg.save_candidate_checkpoints:
            shutil.rmtree(cand_dir, ignore_errors=True)
        return candidates.drop(columns=["checkpoint_path"], errors="ignore"), pd.DataFrame()

    locked_rows: List[Dict[str, Any]] = []
    data_cache: Dict[int, MatrixLike] = {}
    for _, row in selected.iterrows():
        ss = int(row.split_seed)
        if ss not in data_cache:
            dc = copy.deepcopy(cfg.data); dc.split_seed = ss
            data_cache[ss], _ = build_publication_matrix_data_v8(split_frames[ss], spec, dc, device)
        data = data_cache[ss]
        model, ck = load_model_checkpoint_v8(row.checkpoint_path, data, device)
        outrow = row.drop(labels=["checkpoint_path"]).to_dict()
        for sp in ("train", "val", "test"):
            t0 = time.perf_counter(); met, pred = evaluate_publication_model(model, data, sp, metric_cfg=cfg.metrics, return_predictions=True); infer = time.perf_counter() - t0
            outrow.update({f"{sp}_{a}": b for a, b in met.items()}); outrow[f"{sp}_inference_sec"] = infer
            outrow.update({f"{sp}diag_{a}": b for a, b in _model_diagnostics_v8(model, data, sp).items()})
            if cfg.save_predictions:
                pp = _prediction_archive_path_v8(out, cfg.data.dataset, str(row.method), ss, int(row.model_seed), row.K, str(row.init_mode), sp)
                save_prediction_archive_v8(pp, data, pred, sp); outrow[f"{sp}_prediction_file"] = str(pp)
        if cfg.save_selected_checkpoints:
            selp = selected_dir / f"{cfg.data.dataset}__{row.method}__split{ss}__seed{int(row.model_seed)}__K{('NA' if pd.isna(row.K) else int(row.K))}__{row.init_mode}.pt"
            shutil.copy2(row.checkpoint_path, selp); outrow["selected_checkpoint"] = str(selp)
        locked_rows.append(outrow)
        del model; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

    locked = pd.DataFrame(locked_rows)
    locked.to_csv(out / "locked_test_runs.csv", index=False)
    summary = summarise_publication_runs(locked); summary.to_csv(out / "locked_test_summary.csv", index=False)

    # Primary statistical unit: outer split. Average non-convex seed replicates first.
    numeric = [c for c in locked.select_dtypes(include=np.number).columns if c not in ("model_seed",)]
    split_means = locked.groupby(["dataset", "method", "split_seed"], as_index=False)[numeric].mean(numeric_only=True)
    split_means.to_csv(out / "locked_test_split_means.csv", index=False)

    # Stability across independent model seeds at the selected configuration.
    if cfg.save_predictions:
        st = attach_prediction_stability(locked.rename(columns={"test_prediction_file": "test_prediction_file"}))
        if not st.empty: st.to_csv(out / "prediction_stability_across_seeds.csv", index=False)

    with open(out / "training_histories.json", "w", encoding="utf-8") as f: json.dump(_jsonable(histories), f, indent=2)
    with open(out / "config.json", "w", encoding="utf-8") as f: json.dump(_jsonable(asdict(cfg)), f, indent=2)
    env = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "device": str(device), "v8": V8_REVISION, "frozen_v7_core_sha256": FROZEN_V7_CORE_SHA256}
    with open(out / "environment.json", "w", encoding="utf-8") as f: json.dump(env, f, indent=2)

    # Candidate checkpoints are infrastructure staging; selected checkpoints persist.
    if not cfg.save_candidate_checkpoints:
        shutil.rmtree(cand_dir, ignore_errors=True)

    if cfg.auto_run_statistics and not locked.empty:
        try:
            pairwise_wilcoxon_holm(split_means, metric="test_RMSE", group_cols=("dataset", "split_seed")).to_csv(out / "wilcoxon_holm.csv", index=False)
        except Exception as exc:
            warnings.warn(f"Wilcoxon-Holm skipped: {exc}")
        try:
            ranks, stat = friedman_nemenyi_summary(split_means, metric="test_RMSE", block_cols=("dataset", "split_seed"))
            ranks.to_csv(out / "friedman_average_ranks.csv", index=False); pd.DataFrame([stat]).to_csv(out / "friedman_nemenyi_stats.csv", index=False)
        except Exception as exc:
            warnings.warn(f"Friedman/Nemenyi skipped: {exc}")
    if cfg.auto_generate_plots and not locked.empty:
        try:
            # Figure rendering is infrastructure-only; the frozen scientific
            # models, metrics and locked-test protocol remain untouched.
            from mbmf_q1_visuals_v8 import generate_publication_figures_v8
            generate_publication_figures_v8(locked, out, out / "training_histories.json")
        except Exception as exc:
            (out / "figure_generation_error.txt").write_text(repr(exc), encoding="utf-8")
            raise RuntimeError(f"Publication figure generation failed: {exc}") from exc

    return candidates.drop(columns=["checkpoint_path"], errors="ignore"), locked


# -----------------------------------------------------------------------------
# Across-K stability diagnostic — validation by default, never used for selection
# -----------------------------------------------------------------------------

def prediction_variation_across_k_v8(runs: pd.DataFrame, split: str = "val") -> pd.DataFrame:
    col = f"{split}_prediction_file"
    if col not in runs.columns:
        return pd.DataFrame()
    rows = []
    group_cols = [c for c in ["dataset", "method", "split_seed", "model_seed", "init_mode"] if c in runs.columns]
    for key, g in runs.groupby(group_cols, dropna=False):
        gg = g[g["K"].notna()].sort_values("K")
        if gg["K"].nunique() < 2:
            continue
        files = [Path(x) for x in gg[col].dropna().tolist() if Path(x).exists()]
        if len(files) < 2:
            continue
        st = prediction_stability(files)
        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        rec.update({"K_values": ",".join(map(str, sorted(gg.K.unique()))), f"AvgPredictionStd_acrossK_{split}": st["AvgPredictionStd"], f"MaxPredictionStd_acrossK_{split}": st["MaxPredictionStd"]})
        rows.append(rec)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# V8 external competitor adapter helpers
# -----------------------------------------------------------------------------

def evaluate_external_predictions_v8(
    data: MatrixLike,
    csv_path: Union[str, Path],
    method_name: str,
    metric_cfg: Optional[MetricConfig] = None,
) -> Dict[str, Any]:
    model = ExternalPredictionAdapter(csv_path, data, data.train_users.device)
    row: Dict[str, Any] = {"method": method_name, "source": "external_official_adapter"}
    for sp in ("train", "val", "test"):
        met = evaluate_publication_model(model, data, sp, metric_cfg=metric_cfg or MetricConfig())
        row.update({f"{sp}_{k}": v for k, v in met.items()})
    return row


# -----------------------------------------------------------------------------
# V8 ablations: remove the two duplicated V7 jobs; add mechanism-revealing scans
# -----------------------------------------------------------------------------

def make_ablation_variants_v8(base: PublicationExperimentConfigV8) -> List[Tuple[str, PublicationExperimentConfigV8]]:
    variants: List[Tuple[str, PublicationExperimentConfigV8]] = []
    def add(tag: str, method: int, mutate=lambda c: None):
        c = copy.deepcopy(base); c.methods = (f"Method{method}",); mutate(c); variants.append((tag, c))

    # M1: full != every ablation under the Q1 preset.
    add("m1_full", 1)
    add("m1_no_ae", 1, lambda c: setattr(c.core_config.method1, "alpha_ae", 0.0))
    add("m1_no_mbmf", 1, lambda c: setattr(c.core_config.method1, "beta_mbmf", 0.0))
    add("m1_bounded_radius", 1, lambda c: setattr(c.core_config.method1, "radius_mode", "bounded_tanh"))
    add("m1_no_mask", 1, lambda c: setattr(c.core_config.method1, "input_mode", "values_only"))

    add("m2_full", 2)
    add("m2_no_consistency", 2, lambda c: setattr(c.core_config.method2, "consistency_weight", 0.0))
    add("m2_user_view_only", 2, lambda c: setattr(c.core_config.method2, "fusion_weight_user_view", 1.0))
    add("m2_item_view_only", 2, lambda c: setattr(c.core_config.method2, "fusion_weight_user_view", 0.0))
    def m2bounded(c):
        c.core_config.method2.radius_mode = "bounded_tanh"; c.core_config.method2.counterpart_radius_mode = "bounded_sigmoid"
    add("m2_bounded", 2, m2bounded)

    add("m3_full", 3)
    add("m3_no_residual", 3, lambda c: setattr(c.core_config.method3, "max_abs_log_scale", 0.0))
    add("m3_local_prior", 3, lambda c: setattr(c.core_config.method3, "prior_mode", "legacy_local_mean_std"))
    add("m3_delta_regularized", 3, lambda c: setattr(c.core_config.method3, "delta_l2", 1e-3))
    add("m3_averaged_ae", 3, lambda c: setattr(c.core_config.method3, "ae_loss_mode", "averaged"))

    add("m4_full", 4)
    add("m4_no_residual", 4, lambda c: setattr(c.core_config.method4, "max_abs_logit_shift", 0.0))
    add("m4_local_prior", 4, lambda c: setattr(c.core_config.method4, "prior_mode", "legacy_local_mean_std"))
    add("m4_no_mask", 4, lambda c: setattr(c.core_config.method4, "input_mode", "values_only"))
    add("m4_delta_regularized", 4, lambda c: setattr(c.core_config.method4, "delta_l2", 1e-3))

    add("m5_full", 5)
    add("m5_user_branch", 5, lambda c: setattr(c.core_config.method5, "branch_mode", "user"))
    add("m5_item_branch", 5, lambda c: setattr(c.core_config.method5, "branch_mode", "item"))
    def m5norec(c):
        c.core_config.method5.user_reconstruction_weight = 0.0; c.core_config.method5.item_reconstruction_weight = 0.0
    add("m5_no_reconstruction", 5, m5norec)
    add("m5_no_mask", 5, lambda c: setattr(c.core_config.method5, "input_mode", "values_only"))

    add("m6_full", 6)
    add("m6_user_branch", 6, lambda c: setattr(c.core_config.method6, "branch_mode", "user"))
    add("m6_item_branch", 6, lambda c: setattr(c.core_config.method6, "branch_mode", "item"))
    add("m6_tau_fixed1", 6, lambda c: setattr(c.core_config.method6, "radius_temperature", 1.0))
    add("m6_fixed_radius", 6, lambda c: setattr(c.core_config.method6, "radius_mode", "fixed"))
    def m6norec(c):
        c.core_config.method6.user_reconstruction_weight = 0.0; c.core_config.method6.item_reconstruction_weight = 0.0
    add("m6_no_reconstruction", 6, m6norec)

    add("m7_full", 7)
    add("m7_fixed_radius", 7, lambda c: setattr(c.core_config.method7, "radius_mode", "fixed"))
    def m7nocontext(c):
        c.core_config.method7.use_context = False; c.core_config.method7.consistency_weight = 0.0
    add("m7_no_context", 7, m7nocontext)
    add("m7_no_shrinkage", 7, lambda c: setattr(c.core_config.method7, "use_shrinkage", False))
    add("m7_no_bias_expert", 7, lambda c: setattr(c.core_config.method7, "use_bias_expert", False))
    add("m7_homoscedastic", 7, lambda c: setattr(c.core_config.method7, "heteroscedastic", False))
    def m7mse(c):
        c.core_config.method7.loss_mode = "mse"; c.core_config.method7.consistency_weight = 0.0
    add("m7_mse_only", 7, m7mse)

    # Guard against future accidental duplicate ablations.
    seen: Dict[str, str] = {}
    for tag, c in variants:
        m = int(tag[1])
        obj = getattr(c.core_config, f"method{m}")
        key = json.dumps(_jsonable(asdict(obj)), sort_keys=True, default=str)
        if key in seen:
            raise AssertionError(f"duplicate ablation configs: {seen[key]} == {tag}")
        seen[key] = tag
    return variants


def run_ablation_suite_v8(base: PublicationExperimentConfigV8, output_root: Optional[Union[str, Path]] = None, device: Optional[torch.device] = None) -> pd.DataFrame:
    root = Path(output_root or (Path(base.output_dir) / "ablations")); root.mkdir(parents=True, exist_ok=True)
    frames = []
    for tag, cfg in make_ablation_variants_v8(base):
        cfg.output_dir = str(root / tag)
        print(f"\n######## V8 ABLATION {tag} ########")
        _, locked = run_publication_study_v8(cfg, device=device)
        if not locked.empty:
            x = locked.copy(); x.insert(0, "ablation", tag); frames.append(x)
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(root / "ablation_locked_test_runs.csv", index=False)
        summarise_publication_runs(combined).to_csv(root / "ablation_summary.csv", index=False)
        # Both raster and vector output.
        try:
            from mbmf_q1_visuals_v8 import generate_ablation_figures_v8
            generate_ablation_figures_v8(combined, root)
        except Exception as exc:
            warnings.warn(f"ablation plot skipped: {exc}")
    return combined


# -----------------------------------------------------------------------------
# V8 all-dataset orchestrator
# -----------------------------------------------------------------------------

def run_all_datasets_v8(
    base: PublicationExperimentConfigV8,
    datasets: Sequence[str] = ("synthetic", "jester", "bookcrossing", "ml-100k", "ml-1m", "ml-10m"),
    *,
    output_root: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    continue_on_error: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(output_root or base.output_dir); root.mkdir(parents=True, exist_ok=True)
    all_runs, errors = [], []
    for ds in datasets:
        c = copy.deepcopy(base); c.data.dataset = ds; c.output_dir = str(root / ds)
        try:
            _, locked = run_publication_study_v8(c, device=device)
            if not locked.empty: all_runs.append(locked)
        except Exception as exc:
            if not continue_on_error: raise
            errors.append({"dataset": ds, "error": repr(exc)})
    runs = pd.concat(all_runs, ignore_index=True, sort=False) if all_runs else pd.DataFrame()
    if not runs.empty:
        runs.to_csv(root / "all_datasets_locked_runs.csv", index=False)
        summarise_publication_runs(runs).to_csv(root / "all_datasets_summary.csv", index=False)
        if base.auto_run_statistics:
            try: pairwise_wilcoxon_holm(runs, metric="test_RMSE", group_cols=("dataset", "split_seed")).to_csv(root / "all_datasets_wilcoxon_holm.csv", index=False)
            except Exception as exc: warnings.warn(f"all-dataset Wilcoxon skipped: {exc}")
            try:
                ranks, stat = friedman_nemenyi_summary(runs, metric="test_RMSE", block_cols=("dataset", "split_seed")); ranks.to_csv(root / "all_datasets_friedman_ranks.csv", index=False); pd.DataFrame([stat]).to_csv(root / "all_datasets_friedman_stats.csv", index=False)
                plot_critical_difference(ranks, stat, root / "figures")
            except Exception as exc: warnings.warn(f"all-dataset Friedman/Nemenyi skipped: {exc}")
        if base.auto_generate_plots:
            from mbmf_q1_visuals_v8 import generate_multi_dataset_figures_v8
            generate_multi_dataset_figures_v8(runs, root)
    if errors: pd.DataFrame(errors).to_csv(root / "dataset_errors.csv", index=False)
    return runs, summarise_publication_runs(runs) if not runs.empty else pd.DataFrame()


# -----------------------------------------------------------------------------
# V8 self-tests / immutability checks
# -----------------------------------------------------------------------------

def _state_max_abs_diff(a: nn.Module, b: nn.Module) -> float:
    sa, sb = a.state_dict(), b.state_dict()
    if sa.keys() != sb.keys():
        return float("inf")
    vals = []
    for k in sa:
        if torch.is_tensor(sa[k]) and sa[k].dtype.is_floating_point:
            vals.append(float(torch.max(torch.abs(sa[k].detach().cpu() - sb[k].detach().cpu())).item()))
        elif not torch.equal(sa[k].detach().cpu(), sb[k].detach().cpu()):
            return float("inf")
    return max(vals, default=0.0)


def scientific_core_immutability_test_v8(expected_sha256: Optional[str] = None) -> None:
    # Pin to the independent tracked manifest. Comparing only against the hash
    # captured at import time used to give a false PASS for pre-edited V7 code.
    manifest_path = Path(__file__).with_name("SCIENTIFIC_CORE_MANIFEST_V8.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"scientific core manifest missing: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    pinned = str(manifest.get("sha256", "")).lower()
    if len(pinned) != 64 or any(c not in "0123456789abcdef" for c in pinned):
        raise ValueError("invalid SHA-256 in scientific core manifest")
    actual = sha256_scientific_core_v8(V7_CORE_PATH)
    actual_raw = sha256_file(V7_CORE_PATH)
    if actual != pinned:
        raise AssertionError(
            f"Scientific core differs from GitHub-pinned V7: "
            f"normalised_sha256={actual}, raw_sha256={actual_raw}, expected={pinned}. "
            "Inspect `git diff -- mbmf_q1_final_v7.py`; do not report Q1 results until resolved."
        )
    if expected_sha256 is not None and actual != expected_sha256:
        raise AssertionError(f"frozen v7 core changed: {actual} != {expected_sha256}")
    if actual != FROZEN_V7_CORE_SHA256 or actual_raw != FROZEN_V7_CORE_RAW_SHA256:
        raise AssertionError("scientific core changed after infrastructure import")
    print(f"[v8-test] scientific core matches pinned manifest: PASS ({actual})")


def sparse_dense_equivalence_test_v8(device: Optional[torch.device] = None) -> None:
    device = device or core.get_device(); d = core.synthetic_data(device); s = _matrixdata_to_sparse(d)
    setattr(s, "train_timestamps", None); setattr(s, "train_timestamp_status", "absent")
    cases = [
        (core.ThesisMethod1, ScalableThesisMethod1, core.Method1Config(latent_dim=4, hidden_dim=8, max_epochs=1)),
        (core.ThesisMethod2, ScalableThesisMethod2, core.Method2Config(latent_dim=4, max_epochs=1, counterpart_radius_mode="vector_norm")),
        (core.ProposedMethod3, ScalableProposedMethod3, core.Method3Config(latent_dim=4, max_epochs=1)),
        (core.ProposedMethod4, ScalableProposedMethod4, core.Method4Config(latent_dim=4, max_epochs=1)),
    ]
    for idx, (dense_cls, sparse_cls, cfg) in enumerate(cases, 1):
        core.set_global_seed(123); a = dense_cls(d, cfg, device)
        core.set_global_seed(123); b = sparse_cls(s, cfg, device); b.load_state_dict(a.state_dict(), strict=True)
        with torch.no_grad():
            pa = a.predict_raw_pairs(d, d.test_users, d.test_items); pb = b.predict_raw_pairs(s, s.test_users, s.test_items)
        err = float(torch.max(torch.abs(pa - pb)).item())
        if err > 3e-5: raise AssertionError(f"dense/sparse Method{idx} mismatch {err}")
    print("[v8-test] dense/sparse Methods 1-4 prediction equivalence: PASS")


def one_step_training_equivalence_test_v8(device: Optional[torch.device] = None) -> None:
    """Regression guard: scalable execution must not change one Adam update.

    Methods 1-3 compare frozen dense V7 classes with the V8 sparse algebraic
    wrappers. Methods 4-6 compare the frozen full-objective trainer with the V8
    edge-chunk implementation using a single chunk.  The scientific core itself
    is never edited; this test guards the infrastructure translation only.
    """
    device = device or core.get_device()
    d = core.synthetic_data(device)
    s = _matrixdata_to_sparse(d)
    setattr(s, "train_timestamps", None)
    setattr(s, "train_timestamp_status", "absent")

    def _train_core(model: nn.Module, data: core.MatrixData, cfg: Any) -> None:
        core.train_thesis_model(
            model, data,
            max_epochs=1, lr=cfg.lr, weight_decay=cfg.weight_decay,
            eval_every=1, val_patience=max(int(cfg.val_patience), 2),
            grad_clip=cfg.grad_clip,
        )

    def _max_parameter_difference(a: nn.Module, b: nn.Module) -> float:
        bp = dict(b.named_parameters())
        diffs: List[float] = []
        if set(dict(a.named_parameters())) != set(bp):
            return float("inf")
        for name, value in a.named_parameters():
            diffs.append(float(torch.max(torch.abs(value.detach() - bp[name].detach())).item()))
        return max(diffs, default=0.0)

    sparse_cases = [
        (
            core.ThesisMethod1,
            ScalableThesisMethod1,
            core.Method1Config(latent_dim=4, hidden_dim=8, max_epochs=1, eval_every=1, val_patience=3),
            train_scalable_method1,
        ),
        (
            core.ThesisMethod2,
            ScalableThesisMethod2,
            core.Method2Config(latent_dim=4, max_epochs=1, eval_every=1, val_patience=3, counterpart_radius_mode="vector_norm"),
            train_scalable_method2,
        ),
        (
            core.ProposedMethod3,
            ScalableProposedMethod3,
            core.Method3Config(latent_dim=4, max_epochs=1, eval_every=1, val_patience=3),
            train_scalable_method3,
        ),
    ]
    n = int(s.train_users.numel()) + 16
    for idx, (dense_cls, sparse_cls, cfg, trainer) in enumerate(sparse_cases, 1):
        core.set_global_seed(778)
        dense = dense_cls(d, cfg, device)
        core.set_global_seed(778)
        sparse = sparse_cls(s, cfg, device)
        sparse.load_state_dict(dense.state_dict(), strict=True)
        _train_core(dense, d, cfg)
        trainer(sparse, s, cfg, chunk_size=n)
        err = _max_parameter_difference(dense, sparse)
        if err > 1e-6:
            raise AssertionError(f"Method {idx} dense/sparse one-step update mismatch: {err}")

    full_cases = [
        (
            core.ProposedMethod4,
            core.Method4Config(latent_dim=4, max_epochs=1, eval_every=1, val_patience=3),
            train_scalable_method4,
        ),
        (
            core.DecoderCoupledDualViewMBMF,
            core.Method5Config(latent_dim=4, hidden_dim=8, max_epochs=1, eval_every=1, val_patience=3),
            train_scalable_method56,
        ),
        (
            core.DecoderCoupledDualViewMBMF,
            core.Method6Config(latent_dim=4, hidden_dim=8, max_epochs=1, eval_every=1, val_patience=3),
            train_scalable_method56,
        ),
    ]
    n_dense = int(d.train_users.numel()) + 16
    for idx, (cls, cfg, trainer) in enumerate(full_cases, 4):
        core.set_global_seed(779)
        reference = cls(d, cfg, device)
        core.set_global_seed(779)
        translated = cls(d, cfg, device)
        translated.load_state_dict(reference.state_dict(), strict=True)
        _train_core(reference, d, cfg)
        trainer(translated, d, cfg, chunk_size=n_dense)
        err = _max_parameter_difference(reference, translated)
        if err > 1e-7:
            raise AssertionError(f"Method {idx} full/chunk one-step update mismatch: {err}")

    print("[v8-test] one-step optimizer/update equivalence Methods 1-6: PASS")


def checkpoint_roundtrip_all_methods_v8(device: Optional[torch.device] = None) -> None:
    device = device or core.get_device(); d = core.synthetic_data(device); tmp = THIS_DIR / "_v8_checkpoint_tests"; tmp.mkdir(exist_ok=True)
    cc = _default_core_config_v8(); cc.latent_dims = (4,)
    pcfg = PublicationExperimentConfigV8(core_config=cc); pcfg.data.dataset = "synthetic"
    methods = ["BiasMF", "MF", "MBMF-F", "Method1", "Method2", "Method3", "Method4", "Method5", "Method6", "Method7"]
    for j, method in enumerate(methods):
        model, mcfg = instantiate_model_v8(method, d, pcfg, 4, 111 + j, device, "isotropic_gaussian" if method.startswith("MBMF-") else None)
        model.eval()
        with torch.no_grad():
            p0 = _predict_raw_chunked(model, d, d.test_users, d.test_items, 10000).detach().cpu()
        path = tmp / f"{j}.pt"; save_model_checkpoint_v8(path, model, config=mcfg, data=d, extra={"init_mode": "isotropic_gaussian"})
        m2, _ = load_model_checkpoint_v8(path, d, device)
        with torch.no_grad():
            p1 = _predict_raw_chunked(m2, d, d.test_users, d.test_items, 10000).detach().cpu()
        if not torch.allclose(p0, p1, atol=0, rtol=0):
            raise AssertionError(f"checkpoint roundtrip failed for {method}")
    shutil.rmtree(tmp, ignore_errors=True)
    print("[v8-test] checkpoint round-trip all internal scientific families: PASS")


def ablation_uniqueness_test_v8() -> None:
    cfg = PublicationExperimentConfigV8(); variants = make_ablation_variants_v8(cfg)
    tags = [t for t, _ in variants]
    if len(tags) != len(set(tags)):
        raise AssertionError("duplicate ablation tag")
    print(f"[v8-test] ablation configurations unique: PASS ({len(tags)} unique jobs)")


def publication_extension_self_test_v8(device: Optional[torch.device] = None) -> None:
    device = device or core.get_device()
    scientific_core_immutability_test_v8()
    core.geometry_self_test(device)
    core.heldout_label_invariance_test_v7(device)
    core.synthetic_smoke_test_v7(device)
    sparse_dense_equivalence_test_v8(device)
    one_step_training_equivalence_test_v8(device)
    checkpoint_roundtrip_all_methods_v8(device)
    ablation_uniqueness_test_v8()
    print("[v8-test] publication infrastructure: PASS")


EXPERIMENT_INSTRUCTIONS_V8 = r"""
V8 QUICK START
==============
1) Keep mbmf_q1_final_v7.py and mbmf_q1_infrastructure_v8.py together.
2) Run: python -c "import mbmf_q1_infrastructure_v8 as p; p.publication_extension_self_test_v8()"
3) Download datasets with download_datasets_q1_v8.ipynb.
4) Use K=(10,20,50) for the main MBMF-comparable study.
5) Candidate configuration selection is validation-only. Test is locked until selection.
6) The selected checkpoint is permanently stored in selected_models/.
7) External author implementations use exported common splits and prediction CSV adapters.
8) Ablations are run through make_ablation_variants_v8/run_ablation_suite_v8; duplicate jobs raise.
"""


if __name__ == "__main__":
    print(EXPERIMENT_INSTRUCTIONS_V8)
    publication_extension_self_test_v8()

# V8 late-bound plotting helper for validation prediction variation across K.
def plot_k_stability_v8(table: pd.DataFrame, out_dir: Union[str, Path]) -> None:
    if table.empty or "AvgPredictionStd_acrossK_val" not in table.columns:
        return
    try:
        plt = _mpl()
        g = table.groupby("method")[["AvgPredictionStd_acrossK_val", "MaxPredictionStd_acrossK_val"]].mean().dropna(how="all")
        if g.empty:
            return
        x = np.arange(len(g)); w = 0.38
        fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(g)), 5))
        ax.bar(x - w/2, g["AvgPredictionStd_acrossK_val"], w, label="Average prediction std")
        ax.bar(x + w/2, g["MaxPredictionStd_acrossK_val"], w, label="Maximum prediction std")
        ax.set_xticks(x, g.index, rotation=35, ha="right")
        ax.set_ylabel("Validation prediction std across K")
        ax.legend(fontsize=8)
        _save_fig(fig, Path(out_dir) / "prediction_stability_across_K_validation")
    except Exception as exc:
        warnings.warn(f"K-stability plot skipped: {exc}")

# -----------------------------------------------------------------------------
# V8 one-factor hyperparameter scans (separate from mechanism ablations)
# -----------------------------------------------------------------------------

@dataclass
class HyperparameterScanPlanV8:
    latent_dims: Tuple[int, ...] = (5, 10, 20, 50)
    method1_alpha: Tuple[float, ...] = (0.25, 0.5, 0.7, 1.0)
    method1_beta: Tuple[float, ...] = (0.15, 0.35, 0.5, 1.0)
    method2_gamma: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    method2_consistency: Tuple[float, ...] = (0.0, 0.01, 0.1, 0.25)
    method3_log_scale: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    method3_delta_l2: Tuple[float, ...] = (0.0, 1e-5, 1e-4, 1e-3)
    method4_logit_shift: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    method5_fusion: Tuple[float, ...] = (0.25, 0.5, 0.75)
    method6_temperature: Tuple[float, ...] = (0.5, 1.0, 2.0, 0.0)
    method7_context_strength: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)
    method7_tau_init: Tuple[float, ...] = (5.0, 10.0, 20.0, 50.0)
    train_fractions: Tuple[float, ...] = (0.70, 0.80, 0.90)
    synthetic_densities: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.60)
    autorec_hidden: Tuple[int, ...] = (100, 200, 300, 500)
    autorec_lambda: Tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)
    aemc_element_hidden: Tuple[int, ...] = (50, 100, 200, 400)
    aemc_lambda: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)


def make_hyperparameter_scan_jobs_v8(base: PublicationExperimentConfigV8, plan: Optional[HyperparameterScanPlanV8] = None) -> List[Tuple[str, PublicationExperimentConfigV8]]:
    p = plan or HyperparameterScanPlanV8(); jobs: List[Tuple[str, PublicationExperimentConfigV8]] = []
    def one(tag, methods, mutate):
        c = copy.deepcopy(base); c.methods = tuple(methods); mutate(c); jobs.append((tag, c))
    for k in p.latent_dims: one(f"K={k}", tuple(f"Method{i}" for i in range(1,8)) + ("MF","BiasMF","MBMF-F","MBMF-C","MBMF-N"), lambda c, k=k: setattr(c.core_config, "latent_dims", (k,)))
    for v in p.method1_alpha: one(f"M1_alpha={v}", ("Method1",), lambda c,v=v: setattr(c.core_config.method1,"alpha_ae",v))
    for v in p.method1_beta: one(f"M1_beta={v}", ("Method1",), lambda c,v=v: setattr(c.core_config.method1,"beta_mbmf",v))
    for v in p.method2_gamma: one(f"M2_gamma={v}", ("Method2",), lambda c,v=v: setattr(c.core_config.method2,"fusion_weight_user_view",v))
    for v in p.method2_consistency: one(f"M2_cons={v}", ("Method2",), lambda c,v=v: setattr(c.core_config.method2,"consistency_weight",v))
    for v in p.method3_log_scale: one(f"M3_logscale={v}", ("Method3",), lambda c,v=v: setattr(c.core_config.method3,"max_abs_log_scale",v))
    for v in p.method3_delta_l2: one(f"M3_deltaL2={v}", ("Method3",), lambda c,v=v: setattr(c.core_config.method3,"delta_l2",v))
    for v in p.method4_logit_shift: one(f"M4_logitShift={v}", ("Method4",), lambda c,v=v: setattr(c.core_config.method4,"max_abs_logit_shift",v))
    for v in p.method5_fusion: one(f"M5_gamma={v}", ("Method5",), lambda c,v=v: setattr(c.core_config.method5,"fusion_weight_user_view",v))
    for v in p.method6_temperature: one(f"M6_temperature={v}", ("Method6",), lambda c,v=v: setattr(c.core_config.method6,"radius_temperature",v))
    for v in p.method7_context_strength: one(f"M7_context={v}", ("Method7",), lambda c,v=v: setattr(c.core_config.method7,"context_strength",v))
    for v in p.method7_tau_init: one(f"M7_tau={v}", ("Method7",), lambda c,v=v: setattr(c.core_config.method7,"tau_init",v))
    for v in p.train_fractions:
        def mut(c,v=v): c.data.train_fraction=v; c.data.val_fraction=min(0.1,1.0-v); c.data.test_fraction=1.0-c.data.train_fraction-c.data.val_fraction
        one(f"train_fraction={v}", base.methods, mut)
    if base.data.dataset == "synthetic":
        for v in p.synthetic_densities: one(f"synthetic_density={v}", base.methods, lambda c,v=v: setattr(c.data,"synthetic_density",v))
    for v in p.autorec_hidden: one(f"AutoRec_hidden={v}", ("AutoRec",), lambda c,v=v: setattr(c.autorec,"hidden_dim",v))
    for v in p.autorec_lambda: one(f"AutoRec_lambda={v}", ("AutoRec",), lambda c,v=v: setattr(c.autorec,"lambda_l2",v))
    for v in p.aemc_element_hidden: one(f"AEMC_elementHidden={v}", ("AEMC-NE",), lambda c,v=v: setattr(c.aemcne,"element_hidden_dim",v))
    for v in p.aemc_lambda:
        def muta(c,v=v): c.aemcne.lambda_main=v; c.aemcne.lambda_element=v
        one(f"AEMC_lambda={v}", ("AEMC-NE",), muta)
    return jobs


def run_hyperparameter_scan_v8(base: PublicationExperimentConfigV8, plan: Optional[HyperparameterScanPlanV8] = None, output_root: Optional[Union[str, Path]] = None, device: Optional[torch.device] = None) -> pd.DataFrame:
    root = Path(output_root or (Path(base.output_dir) / "hyperparameter_scans")); root.mkdir(parents=True, exist_ok=True)
    frames=[]
    for tag,cfg in make_hyperparameter_scan_jobs_v8(base,plan):
        safe="".join(ch if ch.isalnum() or ch in "-_.=" else "_" for ch in tag)
        cfg.output_dir=str(root/safe)
        cfg.validation_only=True
        print(f"\n######## V8 HYPERPARAMETER SCAN {tag} (validation only) ########")
        candidates,_=run_publication_study_v8(cfg,device=device)
        if not candidates.empty:
            x=candidates.copy();x.insert(0,"scan",tag);frames.append(x)
    out=pd.concat(frames,ignore_index=True,sort=False) if frames else pd.DataFrame()
    if not out.empty:
        if any(c.startswith("test_") for c in out.columns):
            raise AssertionError("test metrics cannot enter hyperparameter search")
        out.to_csv(root/"hyperparameter_scan_validation_runs.csv",index=False)
        summarise_publication_runs(out).to_csv(root/"hyperparameter_scan_validation_summary.csv",index=False)
    return out
