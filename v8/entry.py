"""V8 CLI, preserving the V2 entry point and original scientific sources."""
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
    parser.add_argument("--v8-max-epochs", dest="max_epochs", type=int)
    parser.add_argument("--v8-steps-per-epoch", dest="steps_per_epoch", type=int)
    parser.add_argument("--v8-baseline-max-epochs", dest="baseline_max_epochs", type=int)
    parser.add_argument("--v8-threads", dest="threads", type=int, default=min(4, torch.get_num_threads()))
    parser.add_argument("--v8-config", type=Path, help="JSON with StudyConfig fields and per-model overrides")
    parser.add_argument("--v8-no-plots", dest="no_plots", action="store_true")
    parser.add_argument("--v8-storage",choices=["auto","resident","host","ssd"],default="auto")
    parser.add_argument("--v8-host-budget-mb",type=int,default=1024)
    parser.add_argument("--v8-cache-dir")
    parser.add_argument("--v8-device",choices=["auto","cpu","cuda","cuda:0"],default="auto")
    parser.add_argument("--v8-memory-reserve",type=float,default=.15)
    parser.add_argument("--v8-pin-memory",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--v8-cpu-fallback",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--v8-ssd-budget-gb",type=float,default=64.)
    parser.add_argument("--v8-retain-candidates",choices=["all","metrics"],default="metrics")
    parser.add_argument("--v8-search-jobs-per-method",type=int)
    parser.add_argument("--v8-protocol",choices=["common","aemc-paper","autorec-paper"])
    # V5 execution/config option spellings are accepted as explicit aliases.
    # They resolve to V8 destinations, never to the V5 implementation.
    for action in list(parser._actions):
        for spelling in list(action.option_strings):
            alias=spelling.replace("--v8-","--v5-").replace("--no-v8-","--no-v5-")
            if alias!=spelling and alias not in parser._option_string_actions:
                action.option_strings.append(alias);parser._option_string_actions[alias]=action
    parser.add_argument("--v8-isolate-jobs",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--v8-recovery-interval",type=int,default=5)
    parser.add_argument("--v8-deterministic",action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--manual-seed",type=int)
    options, rest = parser.parse_known_args(argv)
    if options.v8_recovery_interval<1: parser.error("Recovery interval must be positive")
    if options.manual_seed is not None:
        if options.manual_seed<0: parser.error("Manual seed must be nonnegative")
        if any(x=="--model-seeds" or x.startswith("--model-seeds=") for x in rest): parser.error("Choose --manual-seed OR --model-seeds")
        rest += ["--model-seeds",str(options.manual_seed)]
    for name in ("max_epochs", "baseline_max_epochs", "steps_per_epoch", "threads"):
        value = getattr(options, name)
        if value is not None and value < 1:
            parser.error(f"--v8-{name.replace('_', '-')} must be positive")
    study = StudyConfig()
    if options.v8_config is not None:
        payload = json.loads(options.v8_config.read_text())
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
                raise ValueError(f"Unknown or protected V8 study field: {key}")
    if options.v8_search_jobs_per_method is not None:study.search_jobs_per_method=options.v8_search_jobs_per_method
    if options.v8_protocol is not None:study.protocol=options.v8_protocol
    if study.protocol!="common":study.split_validation_roles=False
    if options.manual_seed is not None: study.search_seed=options.manual_seed
    options.study = study.validate()
    if options.v8_cache_dir is not None:options.v8_cache_dir=str(Path(options.v8_cache_dir).resolve())
    options.execution=ExecutionConfig(storage=options.v8_storage,host_budget_mb=options.v8_host_budget_mb,cache_dir=options.v8_cache_dir,
        device=options.v8_device,reserve_fraction=options.v8_memory_reserve,pin_memory=options.v8_pin_memory,
        cpu_fallback=options.v8_cpu_fallback,ssd_budget_gb=options.v8_ssd_budget_gb).validate()
    if options.v8_device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but this PyTorch environment has no CUDA device")
    options.retain_candidates=options.v8_retain_candidates
    if not any(arg == "--output" or arg.startswith("--output=") for arg in rest):
        rest += ["--output", "./results_v8"]
    from . import engine
    previous_threads = torch.get_num_threads()
    previous_deterministic=torch.are_deterministic_algorithms_enabled()
    if options.v8_deterministic:
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
