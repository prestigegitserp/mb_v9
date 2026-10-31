"""Verify every frozen V7-tree file and the independent V8 source manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def blob(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def verify(root=None,parent_only=False):
    root=Path(root) if root else Path(__file__).resolve().parent
    parent=json.loads((root/'V8_BASELINE_MANIFEST.json').read_text())
    protected=parent['protected_parent_blobs']
    new={} if parent_only else json.loads((root/'V8_SOURCE_MANIFEST.json').read_text())['new_blobs']
    bad=[]
    for name,expected in {**protected,**new}.items():
        file=root/name
        if not file.is_file():bad.append(name+' (missing)');continue
        data=file.read_bytes()
        if blob(data)!=expected and blob(data.replace(b'\r\n',b'\n'))!=expected:bad.append(name)
    if bad:raise ValueError('Integrity mismatch: '+', '.join(bad))
    return dict(status='PASS',parent_commit=parent['parent_commit'],
        unchanged_v1_v7_tree_files=len(protected),new_files_verified=len(new),
        previous_release_files_modified=False,explicit_entrypoint='run_v8.py',
        branch='v8',development_branch='v8_development',
        public_methods=[f'Method{i}' for i in range(1,9)])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-only',action='store_true')
    print(json.dumps(verify(parent_only=parser.parse_args().parent_only),indent=2))
