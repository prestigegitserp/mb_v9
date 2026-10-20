"""Verify all parent files and new V5.2 Git blob identities without writing code."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def git_blob(content):
    return hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()


def verify(root=None,require_uncommitted_base=False):
    root=Path(root) if root else Path(__file__).resolve().parent
    manifest=json.loads((root/"V52_SOURCE_MANIFEST.json").read_text())
    baseline=json.loads((root/"V52_BASELINE_MANIFEST.json").read_text())
    bad=[]
    for name,expected in {**baseline["protected_parent_blobs"],**manifest["new_blobs"]}.items():
        file=root/name
        if not file.is_file():bad.append(name+" (missing)");continue
        content=file.read_bytes()
        if git_blob(content)!=expected:
            is_text=file.suffix in (".py",".json",".md",".txt",".csv",".yml") or file.name=="VERSION"
            if not is_text or git_blob(content.replace(b"\r\n",b"\n"))!=expected:bad.append(name)
    if bad:raise ValueError("Integrity mismatch: "+", ".join(bad))
    if require_uncommitted_base:
        git=lambda *args:subprocess.check_output(["git",*args],cwd=root,text=True).strip()
        if git("rev-parse","HEAD")!=baseline["commit_base"]:raise ValueError("Expected unchanged base commit")
        if git("branch","--show-current")!="v5.2-development":raise ValueError("Expected v5.2-development")
        if git("diff","--cached","--name-only"):raise ValueError("Expected uncommitted, unstaged changes")
    return dict(status="PASS",protected_parent_files=len(baseline["protected_parent_blobs"]),
        new_files_verified=len(manifest["new_blobs"]),python_runtime_tests_executed=manifest["python_runtime_tests_executed"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-uncommitted-base",action="store_true");args=parser.parse_args()
    print(json.dumps(verify(require_uncommitted_base=args.require_uncommitted_base),indent=2))
