"""Prevent simultaneous writers from corrupting study recovery files."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import time


@contextmanager
def study_lock(folder):
    path=Path(folder)/'.v7_writer.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    token=dict(pid=os.getpid(),created=time.time())
    for attempt in range(2):
        try:
            fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'w') as handle:json.dump(token,handle)
            break
        except FileExistsError:
            old=json.loads(path.read_text())
            try:os.kill(int(old['pid']),0)
            except ProcessLookupError:path.unlink();continue
            except PermissionError:pass
            raise RuntimeError('Another process owns this study: '+str(old))
    else: raise RuntimeError('Cannot acquire study lock')
    try:yield
    finally:
        if path.exists() and json.loads(path.read_text())==token:path.unlink()
