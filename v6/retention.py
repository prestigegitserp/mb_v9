"""Bound search artifacts by current validation leaders, preserving every score."""
import json
from pathlib import Path
import shutil
import pandas as pd
from v9.selection import select_validation_winners


def retain_leaders(root, table, base):
    root = Path(root).resolve()
    if base.v6.artifact_policy != "selected":
        return
    winners, _ = select_validation_winners(table, len(base.core_config.split_seeds),
                                          len(base.core_config.model_seeds))
    keep = set(winners.scan_job.astype(str))
    keys=["dataset","method","split_seed","scan_job","config_key"]
    leaders=table.merge(winners[keys],on=keys,validate="many_to_one")
    removed = []
    # Do not touch future jobs during a resumed scan.
    for tag, frame in table.groupby("scan_job"):
        folder = Path(str(frame.artifact_root.iloc[0])).resolve()
        if folder.parent != root / "jobs":
            raise ValueError("Refusing cleanup outside this search/jobs directory")
        if str(tag) in keep:
            selected=leaders[leaders.scan_job==tag]
            allowed={Path(p).resolve() for p in selected.get("val_prediction_file",[])
                     if isinstance(p,str)}
            if allowed:
                for path in (folder/"predictions").rglob("*.npz"):
                    if path.resolve() not in allowed:
                        path.unlink();removed.append(str(path.relative_to(root)))
            mechanisms=folder/"mechanisms"
            if mechanisms.exists():
                import fnmatch
                patterns=[]
                for _,row in selected.iterrows():
                    k="*" if pd.isna(row.K) else str(int(row.K))
                    init="NA" if pd.isna(row.init_mode) else str(row.init_mode)
                    patterns.append(f"{row.method}__split{int(row.split_seed)}__seed{int(row.model_seed)}__K{k}__{init}")
                for path in mechanisms.iterdir():
                    if path.is_symlink():
                        raise ValueError("Refusing cleanup through a symbolic link")
                    if path.is_dir() and not any(fnmatch.fnmatchcase(path.name,p) for p in patterns):
                        shutil.rmtree(path);removed.append(str(path.relative_to(root)))
            continue
        for name in ("predictions", "mechanisms", "publication", "figures"):
            path = folder / name
            if path.is_symlink():
                raise ValueError("Refusing cleanup through a symbolic link")
            if path.is_dir():
                shutil.rmtree(path)
                removed.append(str(path.relative_to(root)))
    manifest = dict(policy="selected", kept_jobs=sorted(keep), removed_this_pass=removed,
                    all_candidate_scores_preserved=True, train_prediction_archives_during_scan=False,
                    final_artifacts_untouched=True)
    (root / "V6_ARTIFACT_RETENTION.json").write_text(json.dumps(manifest, indent=2))


def skip_trial_report(cfg, validation_only, extra):
    return (validation_only and cfg.v6.artifact_policy == "selected"
            and (extra or {}).get("type") == "hyper")
