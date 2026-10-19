"""Freeze selected configurations and data before any locked-test refit."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path


def prepare_locked_plan(base, root, winners, jobs):
    """Validate archived scan provenance and restore final artifact settings.

    Search disables prediction/checkpoint retention to bound disk usage. Those
    switches must not leak into `all`'s final stage. Rebuilding a named scan job
    must also not silently substitute changed CLI settings or changed data.
    """
    from v9.engine import basename
    if winners.empty or any(str(c).startswith("test_") for c in winners.columns):
        raise ValueError("Locked refit requires nonempty validation-only winners")
    plan, specifications = copy.deepcopy(jobs), {}
    for _, row in winners.iterrows():
        method, tag, seed = str(row.method), str(row.scan_job), int(row.split_seed)
        if str(row.dataset) != base.data.dataset or method not in base.methods or tag not in plan:
            raise ValueError("Winner does not match the requested dataset, methods or scan plan")
        cfg = plan[tag]
        folder = Path(root)/"search"/"jobs"/basename(tag)
        path = folder/"config.json"
        if not path.is_file():
            raise ValueError(f"Missing frozen scan configuration: {path}; rerun scan")
        archived = json.loads(path.read_text(encoding="utf8"))
        current = asdict(cfg)
        comparisons = {"data protocol": (current["data"], archived["data"])}
        for key in ("model_seeds", "split_seeds"):
            comparisons[key] = (current["core_config"][key], archived["core_config"][key])
        for key, value in current["v4"].items():
            if key != "models":
                comparisons["V4 "+key] = (value, archived["v4"].get(key))
        if method in cfg.v4.models:
            comparisons[method+" settings"] = (current["v4"]["models"][method], archived["v4"]["models"][method])
        for label, (now, before) in comparisons.items():
            if json.dumps(now, sort_keys=True) != json.dumps(before, sort_keys=True):
                raise ValueError(f"Frozen scan mismatch for {label}; rerun scan with the requested settings")
        protocol_file = folder/f"V4_PROTOCOL_{seed}.json"
        protocol = json.loads(protocol_file.read_text(encoding="utf8"))
        if "data_signature" not in protocol:
            raise ValueError("Frozen scan lacks a data signature; rerun scan")
        key = method, seed
        if key in specifications:
            raise ValueError("Duplicate locked winner for a method/split")
        specifications[key] = dict(config_key=str(row.config_key), data_signature=protocol["data_signature"])
        cfg.save_predictions = base.save_predictions
        cfg.save_selected_checkpoints = base.save_selected_checkpoints
        cfg.validation_only = False
    return plan, specifications
