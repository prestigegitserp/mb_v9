#!/usr/bin/env python3
"""CLI for the MBMF Q1 v8 infrastructure package.

Scientific core: mbmf_q1_final_v7.py (frozen byte-for-byte).
"""
from __future__ import annotations
import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import mbmf_q1_final_v7 as core
import mbmf_q1_infrastructure_v8 as pub


def _ints(s: str):
    return tuple(int(x.strip()) for x in s.split(',') if x.strip())


def _strings(s: str):
    return tuple(x.strip() for x in s.split(',') if x.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ml-1m', choices=['all','synthetic','jester','bookcrossing','ml-100k','ml-1m','ml-10m'])
    ap.add_argument('--data-root', default='./datasets')
    ap.add_argument('--output', default='./results_q1_v8')
    ap.add_argument('--split-seeds', default='2026,2027,2028,2029,2030')
    ap.add_argument('--model-seeds', default='11,22,33,44,55')
    ap.add_argument('--k', default='10,20,50')
    ap.add_argument('--methods', default=None, help='Comma-separated internal methods')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--ablation', action='store_true')
    ap.add_argument('--hyper-scan', action='store_true')
    ap.add_argument('--self-test-only', action='store_true')
    ap.add_argument('--continue-on-error', action='store_true')
    ap.add_argument('--force-backend', choices=['auto','dense','sparse'], default='auto')
    args = ap.parse_args()

    pub.publication_extension_self_test_v8()
    if args.self_test_only:
        return

    cfg = pub.PublicationExperimentConfigV8()
    cfg.data.data_root = args.data_root
    cfg.data.force_backend = args.force_backend
    cfg.output_dir = args.output
    cfg.core_config.split_seeds = _ints(args.split_seeds)
    cfg.core_config.model_seeds = _ints(args.model_seeds)
    cfg.core_config.latent_dims = _ints(args.k)
    if args.methods:
        cfg.methods = _strings(args.methods)

    if args.quick:
        cfg.core_config.split_seeds = cfg.core_config.split_seeds[:1]
        cfg.core_config.model_seeds = cfg.core_config.model_seeds[:1]
        cfg.core_config.latent_dims = cfg.core_config.latent_dims[:1]
        cfg.core_config.paper.init_modes = ('isotropic_gaussian',)
        cfg.core_config.paper.max_epochs = min(cfg.core_config.paper.max_epochs, 3)
        cfg.core_config.paper.eval_every = 1
        for n in range(1, 8):
            c = getattr(cfg.core_config, f'method{n}')
            c.max_epochs = min(c.max_epochs, 3)
            c.eval_every = 1
            c.val_patience = min(c.val_patience, 2)
        cfg.biasmf.max_epochs = min(cfg.biasmf.max_epochs, 3)
        cfg.biasmf.eval_every = 1
        cfg.autorec.max_epochs = min(cfg.autorec.max_epochs, 3)
        cfg.autorec.eval_every = 1
        cfg.aemcne.max_epochs = min(cfg.aemcne.max_epochs, 3)
        cfg.aemcne.eval_every = 1

    status_path = Path(args.output) / "run_status_v8.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = {"status": "RUNNING", "dataset": args.dataset, "quick": bool(args.quick),
              "ablation": bool(args.ablation), "hyper_scan": bool(args.hyper_scan),
              "started_utc": datetime.now(timezone.utc).isoformat()}
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"[v8] RUNNING | dataset={args.dataset} | output={args.output}", flush=True)
    try:
        if args.dataset == 'all':
            if args.ablation or args.hyper_scan:
                raise ValueError('Run ablations/hyperparameter scans one dataset at a time.')
            pub.run_all_datasets_v8(
                cfg, output_root=args.output,
                continue_on_error=args.continue_on_error,
            )
        else:
            cfg.data.dataset = args.dataset
            if args.ablation:
                pub.run_ablation_suite_v8(cfg, Path(args.output) / 'ablations')
            elif args.hyper_scan:
                pub.run_hyperparameter_scan_v8(cfg, output_root=Path(args.output) / 'hyperparameter_scans')
            else:
                pub.run_publication_study_v8(cfg)
    except BaseException as exc:
        status.update(status="FAILED", error=repr(exc), elapsed_seconds=round(time.perf_counter()-started,3),
                      traceback=traceback.format_exc(), finished_utc=datetime.now(timezone.utc).isoformat())
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"[v8] FAILED | {type(exc).__name__}: {exc}", flush=True)
        raise
    else:
        status.update(status="COMPLETED", elapsed_seconds=round(time.perf_counter()-started,3),
                      finished_utc=datetime.now(timezone.utc).isoformat())
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"[v8] COMPLETED | {status['elapsed_seconds']} s | output={args.output}", flush=True)


if __name__ == '__main__':
    main()
