"""Atomic V6 checkpoints; the legacy loaders remain available for baselines."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import torch

from . import __version__
from .config import ModelConfig, implementation_signature
from .models import instantiate
from .protocol import data_signature


def save(path, model, *, config, data, train_info=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format="MBMF-V6", architecture_version=__version__,
                   implementation_sha256=implementation_signature(),
                   method=model.v6_method, model_seed=model.model_seed, config=asdict(model.cfg),
                   data_signature=data_signature(data),
                   state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                   train_info=dict(train_info or {}), extra=dict(extra or {}))
    tmp = path.with_name(path.name+".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def restore(payload, data, device, *, strict_data=True):
    if payload.get("format") != "MBMF-V6" or payload.get("architecture_version") != __version__:
        raise ValueError("V6 checkpoint format/architecture mismatch")
    if payload.get("implementation_sha256") != implementation_signature():
        raise ValueError("V6 checkpoint implementation SHA mismatch; use its exact source revision")
    if strict_data:
        expected, actual = payload["data_signature"], data_signature(data)
        for key, value in actual.items():
            if expected.get(key) != value:
                raise ValueError(f"V6 checkpoint/data mismatch for {key}")
    config = ModelConfig(**payload["config"]).validate()
    if config.method != payload.get("method"):
        raise ValueError("V6 checkpoint method/config mismatch")
    model = instantiate(data, config, payload["model_seed"], device)
    model.load_state_dict(payload["state_dict"], strict=True)
    # M4 baseline was fixed after its bias warmup, including in restored models.
    if model.v6_method == "Method4":
        (model.inner.bias if hasattr(model, "inner") else model.bias).requires_grad_(False)
    model.eval()
    return model, payload


def load(path, data, device, *, strict_data=True):
    return restore(torch.load(path, map_location="cpu", weights_only=False), data, device,
                   strict_data=strict_data)
