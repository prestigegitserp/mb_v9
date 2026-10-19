"""Read-only verification of legacy file hashes and an uncommitted V4 handoff."""
import argparse,hashlib,json,subprocess
from pathlib import Path


def verify(root=None,require_uncommitted_base=False):
    root=Path(root) if root else Path(__file__).resolve().parent
    manifest=json.loads((root/'V4_BASELINE_MANIFEST.json').read_text())
    bad=[]
    for name,sha in manifest['protected_files'].items():
        path=root/name
        if not path.is_file():bad.append(name);continue
        content=path.read_bytes()
        if sha not in (hashlib.sha256(content).hexdigest(),hashlib.sha256(content.replace(b'\r\n',b'\n')).hexdigest()):bad.append(name)
    if bad:raise ValueError('Protected legacy files changed: '+', '.join(bad))
    if require_uncommitted_base:
        git=lambda *args:subprocess.check_output(['git',*args],cwd=root,text=True).strip()
        if git('rev-parse','HEAD')!=manifest['base_commit']:raise ValueError('Expected original V3.0.1 HEAD, no new commit')
        if git('branch','--show-current')!='v4-development':raise ValueError('Expected v4-development branch')
        if git('diff','--cached','--name-only'):raise ValueError('Changes must remain unstaged')
    return dict(protected_files_verified=len(manifest['protected_files']),base_commit=manifest['base_commit'],preservation='PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--require-uncommitted-base',action='store_true');a=p.parse_args()
    print(json.dumps(verify(require_uncommitted_base=a.require_uncommitted_base),indent=2))
