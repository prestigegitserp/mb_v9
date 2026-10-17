"""Read-only check that all protected V1/V2 source blobs remain identical.

No git mutation, network request or model training is performed. The optional
base check is for reviewing the deliberately uncommitted development handoff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def verify(root=None, require_uncommitted_base=False):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    manifest = json.loads((root/"V3_BASELINE_MANIFEST.json").read_text())
    mismatches = []
    for name, record in manifest["protected_files"].items():
        path = root/name
        current = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if current != record["sha256"]:
            mismatches.append(name)
    if mismatches:
        raise ValueError("Protected original files changed: "+", ".join(mismatches))
    if require_uncommitted_base:
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
        if git("rev-parse", "HEAD") != manifest["base_commit"]:
            raise ValueError("HEAD differs from the original V2 base")
        if git("branch", "--show-current") != "v3-development":
            raise ValueError("Expected the temporary v3-development branch")
        if git("diff", "--cached", "--name-only"):
            raise ValueError("Expected all handoff changes to remain unstaged")
    return dict(protected_files_verified=len(manifest["protected_files"]),
                base_commit=manifest["base_commit"], preservation="PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-uncommitted-base", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(require_uncommitted_base=args.require_uncommitted_base), indent=2))
