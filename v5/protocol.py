"""Deterministic validation roles, common to replacements and baselines."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def digest_tensors(*values):
    digest = hashlib.sha256()
    for value in values:
        if value is None:
            digest.update(b"absent")
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validation_roles(data, settings, split_seed=2026, output=None):
    """Keep original train/test. Split validation by *unique pair*, without Y.

    Selection, probability fit, probability early stopping are disjoint. The
    selection population is the same for ALL methods, including old baselines.
    """
    settings.validate()
    out = copy.copy(data)
    u, i, y = data.val_users, data.val_items, data.val_ratings_raw
    pair = (u*int(data.num_items)+i).detach().cpu().numpy()
    unique, inverse = np.unique(pair, return_inverse=True)
    if len(unique) < 3:
        raise ValueError("V5 requires at least three unique validation pairs")
    rng = np.random.default_rng(np.random.SeedSequence([settings.role_seed, int(split_seed)]))
    order = rng.permutation(len(unique))
    ns = min(max(1, int(len(unique)*settings.validation_selection_fraction)), len(unique)-2)
    nf = min(max(1, int(len(unique)*settings.validation_calibration_fraction)), len(unique)-ns-1)
    role = np.full(len(unique), 2, dtype=np.int64)
    role[order[:ns]] = 0
    role[order[ns:ns+nf]] = 1
    roles = role[inverse]
    if not settings.split_validation_roles:roles=np.zeros_like(roles)
    select = torch.as_tensor(roles == 0, device=u.device)
    for name, value in vars(data).items():
        if name.startswith("val_") and torch.is_tensor(value) and value.ndim and len(value) == len(u):
            setattr(out, name, value[select])
        elif name.startswith("val_") and isinstance(value, np.ndarray) and value.ndim and len(value) == len(u):
            setattr(out, name, value[roles == 0])
    for label, code in (("fit", 1), ("hold", 2)):
        mask = torch.as_tensor(roles == code, device=u.device)
        for part, value in zip(("users", "items", "ratings"), (u, i, y)):
            setattr(out, f"v5_calibration_{label}_{part}", value[mask])
    out.v5_role_signature = hashlib.sha256(np.column_stack([pair, roles]).astype("<i8").tobytes()).hexdigest()
    out.v5_protocol = dict(split_seed=int(split_seed), role_seed=settings.role_seed,
                           validation_original=len(u), selection=int((roles == 0).sum()),
                           calibration_fit=int((roles == 1).sum()), calibration_hold=int((roles == 2).sum()),
                           role_sha256=out.v5_role_signature,
                           train_and_test_preserved=True, assignment_uses_ratings=False,
                           validation_roles_split=settings.split_validation_roles,protocol=settings.protocol)
    out.v5_protocol["data_signature"] = data_signature(out)
    if output is not None:
        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        names = np.asarray(["selection", "probability_fit", "probability_hold"])
        pd.DataFrame(dict(pair_id=pair, validation_role=names[roles])).to_csv(
            root/f"V5_ROLE_ASSIGNMENTS_{split_seed}.csv", index=False)
        (root/f"V5_PROTOCOL_{split_seed}.json").write_text(json.dumps(out.v5_protocol, indent=2))
    return out


def data_signature(data):
    out = {key: getattr(data, key) for key in ("num_users", "num_items", "rating_min", "rating_max")}
    out["rating_step"] = getattr(data, "rating_step", None)
    out["train_sha256"] = digest_tensors(data.train_users, data.train_items, data.train_ratings_raw,
                                         getattr(data, "train_timestamps", None))
    out["selection_sha256"] = digest_tensors(data.val_users, data.val_items, data.val_ratings_raw)
    for role in ("fit", "hold"):
        out[f"calibration_{role}_sha256"] = digest_tensors(*[
            getattr(data, f"v5_calibration_{role}_{part}", None) for part in ("users", "items", "ratings")])
    out["role_sha256"] = getattr(data, "v5_role_signature", None)
    return out
