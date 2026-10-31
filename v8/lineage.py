"""Actual frozen architectures, evaluated under the same V8 data protocol."""
from dataclasses import fields
from importlib import import_module
import json
from pathlib import Path
import torch

HISTORICAL = Path(__file__).resolve().parents[1] / "examples/v8/historical_incumbents.json"


def historical_records():
    return json.loads(HISTORICAL.read_text(encoding="utf8"))


def historical_models():
    from .config import ModelConfig
    allowed = {f.name for f in fields(ModelConfig)}
    out = {}
    for method, record in historical_records().items():
        cfg = ModelConfig(**{k: v for k, v in record["config"].items() if k in allowed})
        # One current implementation per public method. Archived architectures
        # remain internal validation controls, never additional public methods.
        cfg.lineage = 'v8'
        # Encoder regularization has a declared, fixed reference horizon; its
        # strength no longer vanishes when max_epochs is raised.
        if method in ('Method1','Method2','Method6','Method7'):
            cfg.full_history_weight = 0.
            cfg.collaborative_l2 = 100.
            cfg.mechanism_auxiliary_weight = .25
        if method == "Method7":
            cfg.probability_fit = True
            cfg.hidden_dim = 128
            cfg.feature_dropout = .30
            cfg.val_patience = 8
        if method in ('Method1','Method2','Method3','Method5','Method7'):
            cfg.history_signal='gated'
        if method=='Method6':
            cfg.history_signal='raw'
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
    namespace=adapter.cfg.lineage
    module = import_module("v8.legacy_"+namespace+"_training" if namespace in ("v3","v4","v52") else namespace+".training")
    inner = adapter.inner
    if hasattr(adapter, "_v8_artifact_dir"):
        setattr(inner, "_" + adapter.cfg.lineage + "_artifact_dir", adapter._v8_artifact_dir)
    inner._v8_artifact_dir=getattr(adapter,"_v8_artifact_dir",None)
    inner._v8_fused_history=adapter.cfg.fused_history
    inner._v8_resume=getattr(adapter,"_v8_resume",False)
    inner._v8_recovery_interval=getattr(adapter,"_v8_recovery_interval",5)
    # Historical trainers for the selected M3/M4/M5 have no probability stage.
    # Use their exact training function and the shared V8 validation population.
    method = getattr(module, "train_" + adapter.cfg.lineage)
    result = method(pub, inner, data, inner.cfg, None)
    for suffix in ("routing","probability"):
        key=namespace+"_"+suffix
        if key in result:result["v8_"+suffix]=result[key]
    adapter.clear_cache()
    result["source_lineage"] = adapter.cfg.lineage
    result["legacy_training_objective_preserved"] = True
    return result
