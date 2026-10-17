"""V3 CLI, preserving the V2 entry point and original scientific sources."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from .config import StudyConfig
from .runtime import installed


def run(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v3-max-epochs", dest="max_epochs", type=int)
    parser.add_argument("--v3-steps-per-epoch", dest="steps_per_epoch", type=int)
    parser.add_argument("--v3-threads", dest="threads", type=int, default=min(4, torch.get_num_threads()))
    parser.add_argument("--v3-config", type=Path, help="JSON with StudyConfig fields and per-model overrides")
    parser.add_argument("--v3-no-plots", dest="no_plots", action="store_true")
    options, rest = parser.parse_known_args(argv)
    for name in ("max_epochs", "steps_per_epoch", "threads"):
        value = getattr(options, name)
        if value is not None and value < 1:
            parser.error(f"--v3-{name.replace('_', '-')} must be positive")
    study = StudyConfig()
    if options.v3_config is not None:
        payload = json.loads(options.v3_config.read_text())
        for key, value in payload.items():
            if key == "models":
                for method, overrides in value.items():
                    if method not in study.models:
                        raise ValueError(f"Unknown model {method}")
                    for setting, target in overrides.items():
                        if setting not in vars(study.models[method]):
                            raise ValueError(f"Unknown {method} configuration: {setting}")
                        setattr(study.models[method], setting, target)
            elif key in vars(study) and key not in ("architecture_version", "implementation_sha256"):
                setattr(study, key, value)
            else:
                raise ValueError(f"Unknown or protected V3 study field: {key}")
    options.study = study.validate()
    if not any(arg == "--output" or arg.startswith("--output=") for arg in rest):
        rest += ["--output", "./results_v3"]
    from v9 import engine
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(options.threads)
    try:
        with installed(engine, options):
            return engine.main(rest)
    finally:
        torch.set_num_threads(previous_threads)
