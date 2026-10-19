"""V4 CLI, preserving the V2 entry point and original scientific sources."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from .config import StudyConfig
from .runtime import installed
from .execution import ExecutionConfig


def run(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v4-max-epochs", dest="max_epochs", type=int)
    parser.add_argument("--v4-steps-per-epoch", dest="steps_per_epoch", type=int)
    parser.add_argument("--v4-threads", dest="threads", type=int, default=min(4, torch.get_num_threads()))
    parser.add_argument("--v4-config", type=Path, help="JSON with StudyConfig fields and per-model overrides")
    parser.add_argument("--v4-no-plots", dest="no_plots", action="store_true")
    parser.add_argument("--v4-storage",choices=["auto","resident","host","ssd"],default="auto")
    parser.add_argument("--v4-host-budget-mb",type=int,default=1024)
    parser.add_argument("--v4-cache-dir")
    parser.add_argument("--v4-device",choices=["auto","cpu","cuda","cuda:0"],default="auto")
    parser.add_argument("--v4-memory-reserve",type=float,default=.15)
    parser.add_argument("--v4-pin-memory",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--v4-retain-candidates",choices=["all","metrics"],default="all")
    parser.add_argument("--v4-search-jobs-per-method",type=int)
    parser.add_argument("--v4-protocol",choices=["common","aemc-paper","autorec-paper"])
    options, rest = parser.parse_known_args(argv)
    for name in ("max_epochs", "steps_per_epoch", "threads"):
        value = getattr(options, name)
        if value is not None and value < 1:
            parser.error(f"--v4-{name.replace('_', '-')} must be positive")
    study = StudyConfig()
    if options.v4_config is not None:
        payload = json.loads(options.v4_config.read_text())
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
                raise ValueError(f"Unknown or protected V4 study field: {key}")
    if options.v4_search_jobs_per_method is not None:study.search_jobs_per_method=options.v4_search_jobs_per_method
    if options.v4_protocol is not None:study.protocol=options.v4_protocol
    if study.protocol!="common":study.split_validation_roles=False
    options.study = study.validate()
    options.execution=ExecutionConfig(storage=options.v4_storage,host_budget_mb=options.v4_host_budget_mb,cache_dir=options.v4_cache_dir,
        device=options.v4_device,reserve_fraction=options.v4_memory_reserve,pin_memory=options.v4_pin_memory).validate()
    if options.v4_device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but this PyTorch environment has no CUDA device")
    options.retain_candidates=options.v4_retain_candidates
    if not any(arg == "--output" or arg.startswith("--output=") for arg in rest):
        rest += ["--output", "./results_v4"]
    from . import engine
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(options.threads)
    try:
        with installed(engine, options):
            return engine.main(rest)
    finally:
        torch.set_num_threads(previous_threads)
