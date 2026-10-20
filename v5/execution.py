"""Lossless autograd storage and transactional candidate replay for one CUDA."""
from contextlib import nullcontext
from dataclasses import dataclass
import gc
import random
from pathlib import Path
import tempfile
import threading
import time
import weakref
import shutil
import psutil
import numpy as np
import torch


@dataclass
class ExecutionConfig:
    storage:str="auto"
    host_budget_mb:int=1024
    reserve_fraction:float=.15
    cache_dir:str|None=None
    device:str="auto"
    pin_memory:bool=True
    memory_poll_interval:int=32
    cpu_fallback:bool=True
    ssd_budget_gb:float=64.0
    ram_reserve_fraction:float=.15
    def validate(self):
        if self.storage not in ("auto","resident","host","ssd"):raise ValueError("Invalid tensor storage")
        if isinstance(self.host_budget_mb,bool) or not isinstance(self.host_budget_mb,int) or self.host_budget_mb<0:raise ValueError("Invalid RAM budget")
        if not 0<self.reserve_fraction<1:raise ValueError("Invalid reserve fraction")
        if self.device not in ("auto","cpu","cuda","cuda:0"):raise ValueError("Use one device: auto, cpu or cuda:0")
        if not isinstance(self.pin_memory,bool):raise ValueError("pin_memory must be boolean")
        if not isinstance(self.cpu_fallback,bool):raise ValueError("cpu_fallback must be boolean")
        if not np.isfinite(self.ssd_budget_gb) or self.ssd_budget_gb<=0:raise ValueError("Invalid SSD budget")
        if not 0<self.ram_reserve_fraction<1:raise ValueError("Invalid RAM reserve")
        if isinstance(self.memory_poll_interval,bool) or not isinstance(self.memory_poll_interval,int) or self.memory_poll_interval<1:raise ValueError("Invalid memory polling interval")
        return self


class _Handle:
    def __init__(self,device,value=None,path=None,shape=None,dtype=None,stride=None):
        self.device,self.value,self.path=device,value,path
        self.shape,self.dtype=shape,dtype
        self.stride=stride


class SavedTensorStore:
    def __init__(self,config,policy=None,persistent=()):
        self.config=config.validate();self.policy=policy or config.storage
        self.host_bytes=self.disk_bytes=self.host_peak_bytes=self.disk_peak_bytes=0
        self.host_current=self.disk_current=0;self.counter=0;self.temp=None
        self.persistent={self._storage_key(t) for t in persistent if torch.is_tensor(t) and t.layout==torch.strided}
        self.packed=weakref.WeakValueDictionary();self.pack_calls=self.cache_hits=self.persistent_hits=0
        self.restores=0;self.restored_bytes=0;self._memory={}
    @staticmethod
    def _storage_key(tensor):
        return (str(tensor.device),tensor.untyped_storage().data_ptr())
    def __enter__(self):
        parent=self.config.cache_dir
        if parent:Path(parent).mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix="v5_graph_",dir=parent)
        self.hooks=torch.autograd.graph.saved_tensors_hooks(self.pack,self.unpack)
        self.hooks.__enter__();return self
    def _release(self,kind,size,path=None):
        if kind=="host":self.host_current-=size
        else:
            self.disk_current-=size
            if path:Path(path).unlink(missing_ok=True)
    def pack(self,tensor):
        if tensor.layout!=torch.strided:return tensor.detach()
        self.pack_calls+=1
        storage_key=self._storage_key(tensor)
        if storage_key in self.persistent:
            self.persistent_hits+=1
            return tensor.detach()
        policy=self.policy
        if policy=="auto":
            if tensor.device.type!="cuda":
                if 'ram' not in self._memory or self.pack_calls%self.config.memory_poll_interval==0:
                    memory=psutil.virtual_memory();self._memory['ram']=(memory.available,memory.total)
                available,total=self._memory['ram']
                if available>self.config.ram_reserve_fraction*total+tensor.numel()*tensor.element_size():return tensor.detach()
                policy='ssd'
            else:
                if str(tensor.device) not in self._memory or self.pack_calls%self.config.memory_poll_interval==0:
                    self._memory[str(tensor.device)]=torch.cuda.mem_get_info(tensor.device)
                free,total=self._memory[str(tensor.device)]
                if free>self.config.reserve_fraction*total+tensor.numel()*tensor.element_size():return tensor.detach()
                policy="host"
        if policy=="resident":return tensor.detach()
        # Only reuse an identical view at an identical version. A weak cache
        # avoids retaining dead graphs or corrupting later tensors at reused pointers.
        try:version=tensor._version
        except RuntimeError:version=None
        key=(storage_key,version,tensor.dtype,tuple(tensor.shape),tuple(tensor.stride()),tensor.storage_offset())
        hit=self.packed.get(key) if version is not None else None
        if hit is not None and hit.source() is tensor:
            self.cache_hits+=1
            return hit
        size=tensor.numel()*tensor.element_size()
        memory=psutil.virtual_memory()
        available=max(0,memory.available-self.config.ram_reserve_fraction*memory.total)
        host=policy=="host" and self.host_current+size<=self.config.host_budget_mb*1024**2 and size<=available
        if not host:
            parent=Path(self.temp.name)
            free=shutil.disk_usage(parent).free
            if self.disk_current+size>self.config.ssd_budget_gb*1024**3 or size+64*1024**2>free:
                raise MemoryError("V5 SSD cache budget or free disk exhausted; candidate aborted without changing precision/batch")
            path=parent/f"tensor_{self.counter}";self.counter+=1
            if tensor.is_contiguous():
                path=path.with_suffix('.bin');flat=tensor.detach().reshape(-1)
                block=max(1,(8*1024**2)//tensor.element_size())
                with path.open("wb") as stream:
                    for start in range(0,flat.numel(),block):
                        raw=flat[start:start+block].to("cpu",copy=True).contiguous().view(torch.uint8)
                        stream.write(raw.numpy().tobytes())
                handle=_Handle(tensor.device,path=path,shape=tuple(tensor.shape),dtype=tensor.dtype,stride=tuple(tensor.stride()))
            else:
                # Preserve transpose/layout metadata: merely restoring the same
                # values contiguously can select a different BLAS reduction and
                # change optimizer updates. mmap retains torch.save's strides.
                path=path.with_suffix('.pt')
                torch.save(tensor.detach().to('cpu',copy=True),path)
                handle=_Handle(tensor.device,path=path)
            actual_size=path.stat().st_size
            if self.disk_current+actual_size>self.config.ssd_budget_gb*1024**3:
                path.unlink();raise MemoryError('V5 SSD cache budget exhausted by storage metadata')
            self.disk_bytes+=size;self.disk_current+=actual_size
            self.disk_peak_bytes=max(self.disk_peak_bytes,self.disk_current)
            weakref.finalize(handle,self._release,"ssd",actual_size,path)
            if version is not None:
                handle.source=weakref.ref(tensor);self.packed[key]=handle
            return handle
        if host and self.config.pin_memory and tensor.device.type=="cuda":
            copied=torch.empty_like(tensor,device="cpu",pin_memory=True)
            copied.copy_(tensor.detach(),non_blocking=False)
        else:copied=tensor.detach().to("cpu",copy=True)
        if host:
            handle=_Handle(tensor.device,value=copied)
            self.host_bytes+=size;self.host_current+=size
            self.host_peak_bytes=max(self.host_peak_bytes,self.host_current)
            weakref.finalize(handle,self._release,"host",size)
        else:
            path=Path(self.temp.name)/f"tensor_{self.counter}.pt";self.counter+=1
            torch.save(copied,path);handle=_Handle(tensor.device,path=path)
            self.disk_bytes+=size;self.disk_current+=size
            self.disk_peak_bytes=max(self.disk_peak_bytes,self.disk_current)
            weakref.finalize(handle,self._release,"ssd",size,path)
        if version is not None:
            handle.source=weakref.ref(tensor)
            self.packed[key]=handle
        return handle
    def unpack(self,handle):
        if torch.is_tensor(handle):return handle
        if handle.path is None:value=handle.value
        elif handle.shape is not None:
            import math
            value=torch.from_file(str(handle.path),shared=False,size=math.prod(handle.shape),dtype=handle.dtype).as_strided(handle.shape,handle.stride)
        else:value=torch.load(handle.path,map_location="cpu",weights_only=True,mmap=True)
        self.restores+=1;self.restored_bytes+=value.numel()*value.element_size()
        return value.to(handle.device,non_blocking=bool(value.is_pinned() and handle.device.type=="cuda"))
    def __exit__(self,*exc):
        try:return self.hooks.__exit__(*exc)
        finally:self.temp.cleanup()


def capture_rng(device=None):
    cuda=torch.device(device) if device is not None and torch.device(device).type=="cuda" else None
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(cuda) if cuda is not None else None,device=cuda)


def restore_rng(state):
    random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:torch.cuda.set_rng_state(state["cuda"],state["device"])


def storage_policies(config,cuda):
    return ["auto","host","ssd"] if config.storage=="auto" and cuda else [config.storage]


class DeviceSampler:
    def __init__(self,device):
        self.device=device;self.samples=[];self.error=None;self.stop=threading.Event();self.thread=None
    def __enter__(self):
        self.start=time.perf_counter()
        if self.device.type=="cuda":
            self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start()
        else:self.error="CUDA/NVML unavailable on CPU"
        return self
    def sample(self):
        while not self.stop.is_set():
            try:self.samples.append(dict(seconds=time.perf_counter()-self.start,
                device_kernel_active_percent=torch.cuda.utilization(self.device),
                device_memory_bus_active_percent=torch.cuda.memory_usage(self.device)))
            except Exception as exc:self.error=str(exc);break
            self.stop.wait(.5)
    def __exit__(self,*exc):
        self.stop.set()
        if self.thread:self.thread.join(timeout=1)


def run_training(model,callback,rebuild,config,on_replay=None):
    device=next(model.parameters()).device
    with DeviceSampler(device) as sampler:
        result=_run_training(model,callback,rebuild,config,on_replay)
    result["v5_execution"]["device_sampling"]=dict(samples=sampler.samples,
        unavailable_reason=sampler.error,scope="device-wide NVML, not process-only or achieved FLOPs")
    return result


def _run_training(model,callback,rebuild,config,on_replay):
    config.validate();device=next(model.parameters()).device;cuda=device.type=="cuda"
    policies=storage_policies(config,cuda);rng=capture_rng(device)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()} if len(policies)>1 else None
    start=time.perf_counter();attempts=[]
    if cuda:
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
    for number,policy in enumerate(policies):
        persistent=list(model.parameters())+list(model.buffers())
        data=getattr(model,"_data",None)
        if data is not None:persistent.extend(value for value in vars(data).values() if torch.is_tensor(value))
        storage=SavedTensorStore(config,policy,persistent);del persistent;failed=False
        try:
            with (nullcontext() if policy=="resident" else storage):result=callback()
        except torch.cuda.OutOfMemoryError as exc:
            if number==len(policies)-1:raise
            attempts.append(dict(policy=policy,status="OOM_REPLAY_FROM_INITIAL_STATE"))
            exc.__traceback__=None;failed=True
        if failed:
            model.zero_grad(set_to_none=True)
            if hasattr(model,"clear_cache"):model.clear_cache()
            model.__dict__.clear();gc.collect()
            if cuda:torch.cuda.empty_cache()
            fresh=rebuild();fresh.load_state_dict(initial);model.__dict__.update(fresh.__dict__);del fresh
            restore_rng(rng)
            if on_replay:on_replay()
            continue
        attempts.append(dict(policy=policy,status="COMPLETED",host_bytes=storage.host_bytes,
            ssd_bytes=storage.disk_bytes,host_peak_bytes=storage.host_peak_bytes,ssd_peak_bytes=storage.disk_peak_bytes,
            pack_calls=storage.pack_calls,duplicate_save_hits=storage.cache_hits,persistent_storage_hits=storage.persistent_hits,
            restored_bytes=storage.restored_bytes,restore_calls=storage.restores))
        details=dict(storage_attempts=attempts,wall_seconds=time.perf_counter()-start,
            dtype_policy="unchanged",logical_batch_and_updates="unchanged",cuda_available=cuda,
            cuda_bitwise_parity_verified=False,numerical_overflow_policy="fail; do not silently change precision or learning rate")
        details["storage_contract"]="same dtype, tensor values, masks, optimizer updates and logical batch; CUDA parity requires host hardware verification"
        if cuda:
            end.record();torch.cuda.synchronize(device);free,total=torch.cuda.mem_get_info(device)
            details.update(device=torch.cuda.get_device_name(device),device_total_bytes=total,device_free_bytes=free,
                cuda_stream_elapsed_ms=begin.elapsed_time(end),peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        result["v5_execution"]=details;return result
    raise RuntimeError("Unreachable storage policy state")
