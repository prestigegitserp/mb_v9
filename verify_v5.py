"""Read-only file-integrity verification for V5 and protected older versions."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def verify(root=None,require_uncommitted_base=False):
    root=Path(root) if root else Path(__file__).resolve().parent
    manifest=json.loads((root/'V5_BASELINE_MANIFEST.json').read_text())
    source=json.loads((root/'V5_SOURCE_MANIFEST.json').read_text())
    bad=[]
    for name,expected in {**manifest['protected_files'],**source['files']}.items():
        file=root/name
        if not file.is_file():bad.append(name+' (missing)');continue
        content=file.read_bytes();actual=hashlib.sha256(content).hexdigest()
        if actual!=expected:
            text=file.suffix in ('.py','.md','.json','.txt','.csv','.yml') or name=='VERSION'
            if not text or hashlib.sha256(content.replace(b'\r\n',b'\n')).hexdigest()!=expected:bad.append(name)
    if bad:raise ValueError('File integrity mismatch: '+', '.join(bad))
    if require_uncommitted_base:
        def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
        if git('rev-parse','HEAD')!=manifest['base_commit']:raise ValueError('Expected unchanged V4 base HEAD; V5 must remain uncommitted')
        if git('branch','--show-current')!='v5-development':raise ValueError('Expected v5-development branch')
        if git('diff','--cached','--name-only'):raise ValueError('Expected unstaged working-tree changes')
    return dict(status='PASS',protected_files_verified=len(manifest['protected_files']),
        v5_files_verified=len(source['files']),base_commit=manifest['base_commit'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-uncommitted-base',action='store_true');args=parser.parse_args()
    print(json.dumps(verify(require_uncommitted_base=args.require_uncommitted_base),indent=2))
