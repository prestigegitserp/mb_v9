#!/usr/bin/env python3
"""Vendor the audited, immutable V8 implementation without writing to its repository."""
from __future__ import annotations
import argparse, hashlib, json, pathlib, shutil, subprocess, sys

PINNED_V7_SHA256 = '6da726cb155ab6ad291f5f608febc37f42d1c25ac3924910f56917908c1f124c'
PINNED_V8_FIX_COMMIT = 'f58ec7b95ddc802561cbc1399bcc17be226b10de'
PINNED_V8_INFRA_GIT_BLOB_SHA1 = '6319902dc48385661c959472fd5edee3daf51f51'
FILES = [
 'mbmf_q1_final_v7.py','mbmf_q1_infrastructure_v8.py','mbmf_q1_visuals_v8.py',
 'SCIENTIFIC_CORE_MANIFEST_V8.json','PROTOCOL_IMMUTABILITY_REPORT_8.0.1.json',
 'method_dataset_matrix_v8.csv','metric_dictionary_v8.csv','requirements.txt',
 'download_datasets_q1_v8.ipynb','audit_results_v8.py','run_v8.py',
]

def normalized_sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()

def bootstrap(source, destination='vendor_v8', strict_commit=True):
    src = pathlib.Path(source).expanduser().resolve()
    dst = pathlib.Path(destination).expanduser().resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError('Source and destination must be separate directories, never nested.')
    missing=[name for name in FILES if not (src/name).is_file()]
    if missing: raise FileNotFoundError(f'V8 source missing files: {missing}')
    if normalized_sha256(src/FILES[0]) != PINNED_V7_SHA256:
        raise RuntimeError('Scientific core is NOT the approved V7 version; bootstrap aborted')
    infra_bytes=(src/'mbmf_q1_infrastructure_v8.py').read_bytes().replace(b'\r\n',b'\n')
    infra_blob=hashlib.sha1(b'blob '+str(len(infra_bytes)).encode()+b'\0'+infra_bytes).hexdigest()
    if strict_commit and infra_blob != PINNED_V8_INFRA_GIT_BLOB_SHA1:
        raise RuntimeError(f'V8 infrastructure differs from pinned fix blob: {infra_blob}')
    try:
        commit=subprocess.run(['git','-C',str(src),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    except (OSError,subprocess.CalledProcessError): commit='unknown (e.g. archive)'
    if strict_commit and commit != PINNED_V8_FIX_COMMIT:
        raise RuntimeError(f'Source is not pinned FIX commit {PINNED_V8_FIX_COMMIT}; found {commit}. Use --allow-unpinned only after manual audit.')
    dst.mkdir(parents=True,exist_ok=True)
    for name in FILES:
        data=(src/name).read_bytes().replace(b'\r\n',b'\n')
        (dst/name).write_bytes(data)
    record={'parent_v8_fix_commit':commit,'scientific_core_normalized_sha256':PINNED_V7_SHA256,
            'copied_files':{name:normalized_sha256(dst/name) for name in FILES}}
    (dst/'V9_VENDOR_PROVENANCE.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(f'V8 read-only copy to {dst}; original repository NOT modified. source={commit}')
    return record

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--from-v8',required=True);ap.add_argument('--dest',default='vendor_v8');ap.add_argument('--allow-unpinned',action='store_true');a=ap.parse_args()
    bootstrap(a.from_v8,a.dest,strict_commit=not a.allow_unpinned)
