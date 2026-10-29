"""Explicit device placement and transactional GPU-to-CPU candidate replay.

Storage relocation preserves dtype, values and optimizer schedule. Cross-device
floating point execution is not claimed bitwise identical. Recoverable failures
restart the candidate, including RNG; a partial optimizer update is never kept.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path
import torch
from .execution import capture_rng, restore_rng, run_training, allocation_failure


def resolve_device(requested="auto"):
    if requested=="cpu":return torch.device("cpu"),"explicit CPU"
    if not torch.cuda.is_available():
        if requested!="auto":raise RuntimeError("CUDA requested but unavailable in "+sys.executable)
        return torch.device("cpu"),"CUDA unavailable in this Python interpreter"
    try:
        device=torch.device("cuda:0")
        torch.empty(1,device=device);torch.cuda.synchronize(device)
        return device,"CUDA initialized successfully"
    except RuntimeError as exc:
        if requested!="auto":raise
        return torch.device("cpu"),"CUDA initialization failed: "+str(exc)


def move_data_(data,device):
    """Move all tensor fields, including calibration roles, without recasting."""
    device=torch.device(device)
    if getattr(data,"_sparse_cache",None):data._sparse_cache.clear()
    for name,value in list(vars(data).items()):
        if torch.is_tensor(value):setattr(data,name,value.to(device=device))
    return data


def placement(model,data,requested):
    parameters=list(model.parameters());buffers=list(model.buffers())
    devices=sorted({str(t.device) for t in parameters+buffers})
    if len(devices)!=1:raise RuntimeError("Mixed-device model: "+str(devices))
    actual=devices[0]
    if torch.device(actual)!=data.train_users.device:raise RuntimeError("Model/data device mismatch")
    return dict(requested_device=requested,actual_device=actual,
        python_executable=sys.executable,torch_version=torch.__version__,
        cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available(),
        parameter_bytes=sum(p.numel()*p.element_size() for p in parameters),
        data_tensor_bytes=sum(t.numel()*t.element_size() for t in vars(data).values() if torch.is_tensor(t)))


def execute_candidate(model,data,callback,rebuild,config,on_replay=None):
    started=time.perf_counter();device=next(model.parameters()).device
    rng=capture_rng(device)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()} if device.type=='cuda' else None
    record=placement(model,data,config.device);fallback=None
    record['initialization_fallback']=getattr(model,'_v7_initialization_fallback',None)
    try:
        result=run_training(model,callback,lambda:rebuild(),config,on_replay=on_replay)
    except (torch.cuda.OutOfMemoryError,FloatingPointError,MemoryError) as exc:
        if device.type!="cuda" or not config.cpu_fallback:raise
        if isinstance(exc,MemoryError) and not allocation_failure(exc):raise
        fallback=dict(from_device=str(device),to_device="cpu",reason=type(exc).__name__+": "+str(exc),
                      replay="entire_candidate_from_initial_state",cross_device_bitwise_equivalence=False)
        exc.__traceback__=None
        model.zero_grad(set_to_none=True)
        if hasattr(model,"clear_cache"):model.clear_cache()
        # Release GPU parameters and all graph caches before moving data.
        model.__dict__.clear();gc.collect();torch.cuda.empty_cache()
        move_data_(data,"cpu")
        fresh=rebuild(torch.device("cpu"));fresh.load_state_dict(initial)
        model.__dict__.update(fresh.__dict__);del fresh
        restore_rng(rng)
        if on_replay:on_replay()
        print("[V7 DEVICE] replaying failed CUDA candidate on CPU; dtype and training configuration unchanged",flush=True)
        result=run_training(model,callback,lambda:rebuild(torch.device("cpu")),config,on_replay=on_replay)
    record.update(actual_device=str(next(model.parameters()).device),fallback=fallback,
                  total_candidate_seconds=time.perf_counter()-started)
    result.setdefault("v7_execution",{}).update(placement=record)
    folder=getattr(model,"_v7_artifact_dir",None)
    if folder:
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        (folder/"DEVICE_AND_STORAGE.json").write_text(json.dumps(result["v7_execution"],indent=2,default=str))
    return result
