#!/usr/bin/env python3
"""Post-run completeness gate. Does not train models or change experiment choices.

Run after the FULL study and optional ablation/hyperparameter jobs. An execution
smoke test is not evidence of a completed publication study.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import mbmf_q1_infrastructure_v8 as pub


def audit_study(root: Path) -> dict:
    findings = {"root": str(root), "errors": [], "warnings": [], "checks": {}}
    err, warn, checks = findings["errors"], findings["warnings"], findings["checks"]
    required = ("config.json", "dataset_manifest.json", "candidate_validation_runs.csv",
                "selected_by_validation_candidates.csv", "locked_test_runs.csv",
                "locked_test_summary.csv", "environment.json")
    for name in required:
        if not (root/name).is_file():
            err.append(f"missing {name}")
    if err:
        return findings
    cfg = json.loads((root/'config.json').read_text())
    dataset = cfg['data']['dataset']
    core_cfg = cfg['core_config']
    splits = [int(x) for x in core_cfg['split_seeds']]
    seeds = [int(x) for x in core_cfg['model_seeds']]
    dims = [int(x) for x in core_cfg['latent_dims']]
    candidates = pd.read_csv(root/'candidate_validation_runs.csv')
    selected = pd.read_csv(root/'selected_by_validation_candidates.csv')
    locked = pd.read_csv(root/'locked_test_runs.csv')
    checks.update(dataset=dataset, split_seeds=splits, model_seeds=seeds, latent_dims=dims,
                  n_candidates=len(candidates), n_selected=len(selected), n_locked=len(locked))
    if any(col.startswith('test_') for col in candidates.columns):
        err.append('candidate table contains locked-test metrics')
    if candidates.empty or locked.empty:
        err.append('missing candidate or final selected results')
        return findings
    expected_methods = [m for m in cfg['methods'] if not cfg.get('respect_dataset_method_matrix', True)
                        or pub.method_supported_on_dataset_v8(m, dataset)]
    expected_candidates = {}
    for m in expected_methods:
        n_k = 1 if m in ('AutoRec','AEMC-NE') else len(dims)
        n_inits = len(core_cfg['paper']['init_modes']) if m.startswith('MBMF-') else 1
        expected_candidates[m] = len(splits)*len(seeds)*n_k*n_inits
    actual_counts = candidates.method.value_counts().to_dict()
    for m, exp in expected_candidates.items():
        actual = int(actual_counts.get(m, 0))
        if actual != exp:
            err.append(f'{m}: candidate count {actual}; expected {exp}')
    expected_locked = len(splits)*len(seeds)
    for m in expected_methods:
        actual = int((locked.method == m).sum())
        if actual != expected_locked:
            err.append(f'{m}: locked count {actual}; expected {expected_locked}')
    for label, frame in [('candidate', candidates), ('selected', selected), ('locked', locked)]:
        key_cols = ['dataset','method','split_seed','model_seed','init_mode']
        if label == 'candidate':
            key_cols.append('config_key')
        if frame.duplicated(key_cols).any():
            err.append(f'{label}: duplicate identification keys')
    if not np.isfinite(candidates['val_RMSE'].to_numpy(dtype=float)).all():
        err.append('nonfinite candidate validation RMSE')
    for metric in ('train_RMSE','val_RMSE','test_RMSE','test_MAE','test_MSE'):
        if metric not in locked or not np.isfinite(locked[metric].to_numpy(dtype=float)).all():
            err.append(f'missing or nonfinite locked {metric}')
    if cfg.get('save_selected_checkpoints', True):
        if 'selected_checkpoint' not in locked:
            err.append('selected model paths missing')
        else:
            missing = locked.selected_checkpoint.isna() | ~locked.selected_checkpoint.map(lambda x: bool(isinstance(x,str) and Path(x).is_file()))
            if missing.any(): err.append(f'{int(missing.sum())} selected model checkpoints missing')
    for name in ('Method4', 'Method6', 'Method7'):
        rows = locked[locked.method == name]
        if not rows.empty and (rows.test_Max_OOB_Violation.to_numpy(dtype=float) > 1e-4).any():
            err.append(f'{name} violated analytical global rating bounds by >1e-4')
    rows = locked[locked.method == 'Method7']
    if not rows.empty:
        for suffix in ('Gaussian_NLL',*(f'PICP_{x}' for x in (50,80,90,95)),
                       *(f'MPIW_{x}' for x in (50,80,90,95))):
            col = f'test_{suffix}'
            if col not in rows or not np.isfinite(rows[col].to_numpy(dtype=float)).all():
                err.append(f'Method7 uncertainty metric missing/nonfinite: {col}')
    for ss in splits:
        if not (root/f'split_assignments_{ss}.csv').exists():
            err.append(f'split assignments {ss} missing')
    if len(splits) < 3 or len(seeds) < 3:
        warn.append('fewer than 3 split seeds or 3 model seeds: likely a pilot, not final Q1 evidence')
    if cfg.get('auto_run_statistics'):
        for file in ('wilcoxon_holm.csv','friedman_nemenyi_stats.csv'):
            if not (root/file).is_file():
                warn.append(f'{file} missing')
    if cfg.get('save_predictions') and 'test_prediction_file' in locked:
        missing = locked.test_prediction_file.isna() | ~locked.test_prediction_file.map(lambda x: bool(isinstance(x,str) and Path(x).is_file()))
        if missing.any():
            err.append(f'{int(missing.sum())} test prediction archives missing')
    if any(m in cfg['methods'] for m in pub.EXTERNAL_COMPETITORS_V8):
        warn.append('external baseline reproducibility must be verified separately (author code not packaged)')
    checks['methods_expected'] = expected_methods
    checks['expected_candidate_counts'] = expected_candidates
    checks['n_errors'] = len(err)
    checks['n_warnings'] = len(warn)
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--ablation-root', type=Path)
    ap.add_argument('--scan-root', type=Path)
    ap.add_argument('--report', type=Path)
    args = ap.parse_args()
    result = {'main': audit_study(args.study)}
    errors = len(result['main']['errors'])
    if args.ablation_root:
        tags = [x for x in args.ablation_root.iterdir() if x.is_dir() and (x/'config.json').exists()]
        expected = len(pub.make_ablation_variants_v8(pub.PublicationExperimentConfigV8()))
        result['ablations'] = {'expected': expected, 'found': len(tags),
                               'tags': [x.name for x in tags]}
        if len(tags) != expected:
            result['ablations']['error'] = 'incomplete ablation matrix'
            errors += 1
    if args.scan_root:
        file = args.scan_root/'hyperparameter_scan_validation_runs.csv'
        if not file.exists():
            result['hyperparameter_scan'] = {'error': 'missing validation-only scan file'}
            errors += 1
        else:
            df = pd.read_csv(file, nrows=1)
            leaked = [c for c in df if c.startswith('test_')]
            result['hyperparameter_scan'] = {'file': str(file), 'test_columns': leaked}
            if leaked: errors += 1
    result['status'] = 'FAIL' if errors else 'PASS_WITH_WARNINGS' if result['main']['warnings'] else 'PASS'
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding='utf-8')
    print(payload)
    raise SystemExit(int(errors > 0))

if __name__ == '__main__':
    main()
