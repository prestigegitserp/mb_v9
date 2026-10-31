"""Release references, then the allocator cache; empty_cache alone cannot do this."""
import gc
import json
from pathlib import Path
import time
import torch


def memory_snapshot():
    if not torch.cuda.is_initialized(): return dict(cuda_initialized=False)
    free, total = torch.cuda.mem_get_info()
    return dict(cuda_initialized=True, allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(), device_free_bytes=free, device_total_bytes=total)


def release_model(model, folder=None):
    before = memory_snapshot()
    if model is not None and "_parameters" in model.__dict__:
        model.zero_grad(set_to_none=True)
        if hasattr(model, 'clear_cache'): model.clear_cache()
        # Called only after prediction archives/checkpoints have been saved.
        model.__dict__.clear()
    gc.collect()
    if torch.cuda.is_initialized(): torch.cuda.empty_cache()
    record = dict(time_unix=time.time(), before=before, after=memory_snapshot())
    if folder:
        path=Path(folder)/'V8_RESOURCE_LIFECYCLE.jsonl'; path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a',encoding='utf8') as handle: handle.write(json.dumps(record)+'\n')
    return record
