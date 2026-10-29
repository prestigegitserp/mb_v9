"""V7 CLI, preserving the V2 entry point and original scientific sources."""
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
    parser.add_argument("--v7-max-epochs", dest="max_epochs", type=int)
    parser.add_argument("--v7-steps-per-epoch", dest="steps_per_epoch", type=int)
    parser.add_argument("--v7-baseline-max-epochs", dest="baseline_max_epochs", type=int)
    parser.add_argument("--v7-threads", dest="threads", type=int, default=min(4, torch.get_num_threads()))
    parser.add_argument("--v7-config", type=Path, help="JSON with StudyConfig fields and per-model overrides")
    parser.add_argument("--v7-no-plots", dest="no_plots", action="store_true")
    parser.add_argument("--v7-storage",choices=["auto","resident","host","ssd"],default="auto")
    parser.add_argument("--v7-host-budget-mb",type=int,default=1024)
    parser.add_argument("--v7-cache-dir")
    parser.add_argument("--v7-device",choices=["auto","cpu","cuda","cuda:0"],default="auto")
    parser.add_argument("--v7-memory-reserve",type=float,default=.15)
    parser.add_argument("--v7-pin-memory",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--v7-cpu-fallback",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--v7-ssd-budget-gb",type=float,default=64.)
    parser.add_argument("--v7-retain-candidates",choices=["all","metrics"],default="metrics")
    parser.add_argument("--v7-search-jobs-per-method",type=int)
    parser.add_argument("--v7-protocol",choices=["common","aemc-paper","autorec-paper"])
    # V5 execution/config option spellings are accepted as explicit aliases.
    # They resolve to V7 destinations, never to the V5 implementation.
    for action in list(parser._actions):
        for spelling in list(action.option_strings):
            alias=spelling.replace("--v7-","--v5-").replace("--no-v7-","--no-v5-")
            if alias!=spelling and alias not in parser._option_string_actions:
                action.option_strings.append(alias);parser._option_string_actions[alias]=action
    parser.add_argument("--v7-isolate-jobs",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--v7-recovery-interval",type=int,default=5)
    parser.add_argument("--v7-deterministic",action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--manual-seed",type=int)
    options, rest = parser.parse_known_args(argv)
    if options.v7_recovery_interval<1: parser.error("Recovery interval must be positive")
    if options.manual_seed is not None:
        if options.manual_seed<0: parser.error("Manual seed must be nonnegative")
        if any(x=="--model-seeds" or x.startswith("--model-seeds=") for x in rest): parser.error("Choose --manual-seed OR --model-seeds")
        rest += ["--model-seeds",str(options.manual_seed)]
    for name in ("max_epochs", "baseline_max_epochs", "steps_per_epoch", "threads"):
        value = getattr(options, name)
        if value is not None and value < 1:
            parser.error(f"--v7-{name.replace('_', '-')} must be positive")
    study = StudyConfig()
    if options.v7_config is not None:
        payload = json.loads(options.v7_config.read_text())
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
                raise ValueError(f"Unknown or protected V7 study field: {key}")
    if options.v7_search_jobs_per_method is not None:study.search_jobs_per_method=options.v7_search_jobs_per_method
    if options.v7_protocol is not None:study.protocol=options.v7_protocol
    if study.protocol!="common":study.split_validation_roles=False
    if options.manual_seed is not None: study.search_seed=options.manual_seed
    options.study = study.validate()
    options.execution=ExecutionConfig(storage=options.v7_storage,host_budget_mb=options.v7_host_budget_mb,cache_dir=options.v7_cache_dir,
        device=options.v7_device,reserve_fraction=options.v7_memory_reserve,pin_memory=options.v7_pin_memory,
        cpu_fallback=options.v7_cpu_fallback,ssd_budget_gb=options.v7_ssd_budget_gb).validate()
    if options.v7_device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but this PyTorch environment has no CUDA device")
    options.retain_candidates=options.v7_retain_candidates
    if not any(arg == "--output" or arg.startswith("--output=") for arg in rest):
        rest += ["--output", "./results_v7"]
    from . import engine
    previous_threads = torch.get_num_threads()
    previous_deterministic=torch.are_deterministic_algorithms_enabled()
    if options.v7_deterministic:
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(options.threads)
    try:
        with installed(engine, options):
            return engine.main(rest)
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)
