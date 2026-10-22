"""Actual frozen architectures, evaluated under the same V6 data protocol."""
from dataclasses import fields
from importlib import import_module
import json
from pathlib import Path
import torch

HISTORICAL = Path(__file__).resolve().parents[1] / "examples/v6/historical_incumbents.json"


def historical_records():
    return json.loads(HISTORICAL.read_text(encoding="utf8"))


def historical_models():
    from .config import ModelConfig
    allowed = {f.name for f in fields(ModelConfig)}
    out = {}
    for method, record in historical_records().items():
        cfg = ModelConfig(**{k: v for k, v in record["config"].items() if k in allowed})
        cfg.lineage = record["source_namespace"] if method in ("Method3", "Method4", "Method5") else "v6"
        # Encoder regularization has a declared, fixed reference horizon; its
        # strength no longer vanishes when max_epochs is raised.
        if cfg.lineage == "v6":
            cfg.full_history_weight = 0.
            cfg.collaborative_l2 = 100.
            cfg.mechanism_auxiliary_weight = .25
        if method == "Method7":
            cfg.probability_fit = True
            cfg.hidden_dim = 128
            cfg.feature_dropout = .30
            cfg.val_patience = 8
        out[method] = cfg.validate()
    return out


def legacy_config(cfg):
    module = import_module(cfg.lineage + ".config")
    allowed = {f.name for f in fields(module.ModelConfig)}
    return module.ModelConfig(**{k: v for k, v in vars(cfg).items() if k in allowed}).validate()


def make_legacy(data, cfg, seed, device):
    module = import_module(cfg.lineage + ".models")
    return module.instantiate(data, legacy_config(cfg), seed, device)


def train_legacy(pub, adapter, data):
    module = import_module(adapter.cfg.lineage + ".training")
    inner = adapter.inner
    if hasattr(adapter, "_v6_artifact_dir"):
        setattr(inner, "_" + adapter.cfg.lineage + "_artifact_dir", adapter._v6_artifact_dir)
    # Historical trainers for the selected M3/M4/M5 have no probability stage.
    # Use their exact training function and the shared V6 validation population.
    method = getattr(module, "train_" + adapter.cfg.lineage)
    result = method(pub, inner, data, inner.cfg, None)
    adapter.clear_cache()
    result["source_lineage"] = adapter.cfg.lineage
    result["legacy_training_objective_preserved"] = True
    return result
