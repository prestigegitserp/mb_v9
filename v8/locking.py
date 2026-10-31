"""OS-backed single-writer lock, automatically released on process death."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import socket
import time
import uuid
from filelock import FileLock, Timeout


@contextmanager
def study_lock(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    owner=folder/'V8_WRITER_OWNER.json'
    token=dict(pid=os.getpid(),host=socket.gethostname(),created=time.time(),token=uuid.uuid4().hex)
    lock=FileLock(str(folder/'.v8_writer.lock'),timeout=0)
    try:lock.acquire()
    except Timeout as exc:
        info=owner.read_text(encoding='utf8') if owner.is_file() else 'owner metadata pending'
        raise RuntimeError('Another process owns this study: '+info) from exc
    try:
        from .engine import json_write
        json_write(owner,token)
        yield
    finally:
        try:
            if owner.is_file() and json.loads(owner.read_text(encoding='utf8'))==token:owner.unlink()
        finally:lock.release()
