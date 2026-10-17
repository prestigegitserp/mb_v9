# MBMF V3 — research implementation

**V3 branch:** `v3`; **development branch:** `v3-development`, based on V2 commit `61097d772620e97fff7ec64a9f21b74a6eb9f913`. Seven independent replacement models, target-excluded history training, OOF routing, constrained geometry and mean-preserving probability laws. V1/main and V2 branches are unchanged. The original `v2/`, `v9/`, `vendor_v8/` implementations and `run_v2.py` are preserved.

`run_v9.py` and `run_v3.py` now execute V3 on this branch. `run_v2.py` explicitly executes V2. See [README_V3_FA.md](README_V3_FA.md) for usage and limitations and [V3_DESIGN.md](V3_DESIGN.md) for the exact equations and approximations. Historical V2 documentation remains in [README_V2_FA.md](README_V2_FA.md).

Run `python -m pytest -q tests` before experiments. Smoke tests verify execution, isolation and numerical constraints; they do not establish accuracy improvement over V2 or research novelty.
