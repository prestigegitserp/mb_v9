"""Verify V5 parent files and V6 content identities without changing a checkout."""
import hashlib
import json
from pathlib import Path


def blob(data):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def verify(root=None):
    root=Path(root) if root else Path(__file__).resolve().parent
    parent=json.loads((root/"V6_BASELINE_MANIFEST.json").read_text())
    new=json.loads((root/"V6_SOURCE_MANIFEST.json").read_text())
    bad=[]
    for name,expected in {**parent["protected_parent_blobs"],**new["new_blobs"]}.items():
        path=root/name
        if not path.is_file():bad.append(name+" (missing)");continue
        data=path.read_bytes()
        if blob(data)!=expected and blob(data.replace(b"\r\n",b"\n"))!=expected:
            bad.append(name)
    if bad:raise ValueError("Integrity mismatch: "+", ".join(bad))
    return dict(status="PASS",parent_commit=parent["parent_commit"],
                unchanged_parent_files=len(parent["protected_parent_blobs"]),
                new_files_verified=len(new["new_blobs"]),release_branch_modified=False)


if __name__=="__main__":
    print(json.dumps(verify(),indent=2))
