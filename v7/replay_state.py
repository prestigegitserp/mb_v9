"""One lossless initial state shared by storage replay and CPU fallback."""
from dataclasses import replace
from pathlib import Path
import gc
import shutil
import tempfile
import torch
from .execution import host_memory


def tensor_bytes(tensor):
    if tensor.layout==torch.sparse_coo:
        return tensor._values().numel()*tensor.element_size()+tensor._indices().numel()*tensor._indices().element_size()
    return tensor.numel()*tensor.element_size()


class ReplayState:
    def __init__(self,model,config,enabled=True):
        self.model=model;self.config=config;self.enabled=enabled
        self.state=None;self.temp=None;self.path=None;self.disk_bytes=0
        self.record=dict(policy='disabled',state_bytes=0,temporary_disk_bytes=0,shared_by_storage_and_device_replay=True)

    def __enter__(self):
        if not self.enabled:return self
        state=self.model.state_dict()
        size=sum(tensor_bytes(value) for value in state.values())
        available,total=host_memory()
        capacity=min(self.config.host_budget_mb*1024**2,
                     max(0,available-self.config.ram_reserve_fraction*total))
        self.record['state_bytes']=size
        if size<=capacity:
            self.state={key:value.detach().to('cpu',copy=True) for key,value in state.items()}
            self.record['policy']='host'
        else:
            parent=self.config.cache_dir
            if parent:Path(parent).mkdir(parents=True,exist_ok=True)
            self.temp=tempfile.TemporaryDirectory(prefix='v7_initial_',dir=parent)
            self.path=Path(self.temp.name)/'initial.pt'
            budget=self.config.ssd_budget_gb*1024**3
            try:
                if size+65536>budget or size+64*1024**2>shutil.disk_usage(self.temp.name).free:
                    raise MemoryError('V7 SSD replay-state budget/free disk exhausted')
                # torch.save streams state storages; do not build a second full CPU clone.
                torch.save(state,self.path)
                self.disk_bytes=self.path.stat().st_size
                if self.disk_bytes+65536>budget:raise MemoryError('V7 SSD replay-state budget exhausted by metadata')
            except BaseException:
                self.temp.cleanup();raise
            self.record.update(policy='ssd',temporary_disk_bytes=self.disk_bytes)
        del state
        return self

    def execution_config(self):
        # Snapshot and saved tensors consume one combined temporary disk quota.
        return replace(self.config,ssd_budget_gb=self.config.ssd_budget_gb-self.disk_bytes/1024**3,
                       host_budget_mb=max(0,self.config.host_budget_mb-
                           ((self.record['state_bytes']+1024**2-1)//1024**2 if self.state is not None else 0)))

    def restore_into(self,model):
        if not self.enabled:raise RuntimeError('Replay state was not enabled')
        if self.state is not None:model.load_state_dict(self.state)
        else:
            state=torch.load(self.path,map_location='cpu',weights_only=True,mmap=True)
            try:model.load_state_dict(state)
            finally:del state;gc.collect()

    def __exit__(self,*exc):
        self.state=None;self.model=None
        if self.temp:self.temp.cleanup()
