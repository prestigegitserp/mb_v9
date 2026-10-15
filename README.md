# MBMF V9 — Validation-first publication experiments

An auditable orchestration layer for the frozen `prestigegitserp/mb_v8` fixed-branch science. No scientific-model changes. Candidate-level progress, joint and one-factor validation search, per-method fixed winners, locked test evaluation, 38 ablations, fixed-K train/val/test studies, train-only sparsity sensitivity and convergence snapshots.

**Status:** research beta. This repository already includes the pinned V8 scientific runtime in `vendor_v8/`. Seven local protocol unit tests passed; full real-dataset end-to-end training has not yet been verified.

Detailed Persian installation/protocol: [README_FA.md](README_FA.md).

```powershell
python -m pip install -r requirements.txt
python run_v9.py self-test
python run_v9.py plan --datasets ml-100k
python run_v9.py main --datasets ml-100k --quick
.\RUN_EVERYTHING_WINDOWS.ps1 -Datasets ml-100k -Resume
python run_v9.py audit --datasets ml-100k
```

Original fixed scientific core: SHA-256 `6da726cb155ab6ad291f5f608febc37f42d1c25ac3924910f56917908c1f124c`.
No modifications are made to the existing V8 repository.
