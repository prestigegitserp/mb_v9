"""Sequential subprocess per study job: exit releases the complete CUDA context."""
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
import torch


def hydrate(template, payload):
    if is_dataclass(template):
        return replace(template, **{f.name:hydrate(getattr(template,f.name),payload[f.name])
                                   for f in fields(template) if f.name in payload})
    if isinstance(template,dict):
        return {key:hydrate(template[key],value) if key in template else value for key,value in payload.items()}
    if isinstance(template,tuple): return tuple(payload)
    return payload


def run(request):
    from .config import StudyConfig
    from .execution import ExecutionConfig
    from .runtime import installed
    from . import engine
    payload=json.loads(Path(request).read_text(encoding='utf8'))
    opts=SimpleNamespace(**payload['options'])
    opts.study=hydrate(StudyConfig(),payload['study'])
    opts.execution=ExecutionConfig(**payload['execution'])
    opts._v8_worker=True
    args=SimpleNamespace(**payload['args'])
    torch.set_num_threads(opts.threads)
    if getattr(opts,'v8_deterministic',False):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.use_deterministic_algorithms(True)
    with installed(engine,opts) as (core,pub):
        cfg=hydrate(engine.make_base(pub,args,payload['config']['data']['dataset']),payload['config'])
        cfg._v8_locked_specs={(entry['method'],entry['split_seed']):entry['spec'] for entry in payload['locked_specs']}
        engine.run_job(pub,core,cfg,payload['folder'],args=args,
                       validation_only=payload['validation_only'],extra=payload['extra'])


if __name__=='__main__': run(sys.argv[1])
