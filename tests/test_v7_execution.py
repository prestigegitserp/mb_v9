"""CPU storage parity and transaction replay; these do not assert CUDA parity."""
import copy
from contextlib import nullcontext
from types import SimpleNamespace
import pytest
import torch
from v7.execution import ExecutionConfig,SavedTensorStore,run_training,host_memory,allocation_failure,storage_policies


def test_container_ram_limit_is_respected(monkeypatch):
    from pathlib import Path
    import v7.execution as module
    gb=1024**3
    monkeypatch.setattr(module.psutil,'virtual_memory',lambda:SimpleNamespace(available=32*gb,total=64*gb))
    original=Path.read_text
    def read(path,*args,**kwargs):
        if str(path)=='/sys/fs/cgroup/memory.max':return str(8*gb)
        if str(path)=='/sys/fs/cgroup/memory.current':return str(6*gb)
        if str(path).startswith('/sys/fs/cgroup/memory/'):raise FileNotFoundError
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    assert host_memory()==(2*gb,8*gb)


def test_cpu_auto_has_lossless_ssd_fallback_and_does_not_hide_other_errors():
    assert storage_policies(ExecutionConfig(storage='auto'),False)==['auto','ssd']
    assert allocation_failure(RuntimeError("DefaultCPUAllocator: can't allocate memory: error code 12"))
    assert not allocation_failure(RuntimeError('matrix shapes cannot be multiplied'))
    assert not allocation_failure(MemoryError('SSD cache budget exhausted'))


def test_ssd_storage_preserves_transposes_values_gradients_and_updates(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(819)
    template=torch.nn.Sequential(torch.nn.Linear(6,5),torch.nn.Tanh(),torch.nn.Linear(5,1)).double()
    x=torch.randn(6,9,dtype=torch.double).T
    y=torch.randn(9,1,dtype=torch.double)
    def train(policy):
        model=copy.deepcopy(template);opt=torch.optim.Adam(model.parameters(),lr=.002)
        store=SavedTensorStore(ExecutionConfig(storage=policy,cache_dir=str(tmp_path)),persistent=model.parameters())
        with store if policy=='ssd' else nullcontext():
            for step in range(3):
                opt.zero_grad(set_to_none=True)
                loss=(model(x)-y).square().mean()+.001*sum(p.square().sum() for p in model.parameters())
                loss.backward();opt.step()
        return model,store
    reference,_=train('resident');actual,store=train('ssd')
    assert store.disk_bytes>0 and store.restores>0
    for key,value in reference.state_dict().items():torch.testing.assert_close(value,actual.state_dict()[key],rtol=0,atol=0)


def test_cpu_allocation_failure_replays_from_original_state(tmp_path):
    torch.manual_seed(41);model=torch.nn.Linear(2,1).double()
    reference=copy.deepcopy(model)
    x=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.double);y=torch.tensor([[1.],[2.]],dtype=torch.double)
    def step(m):
        opt=torch.optim.SGD(m.parameters(),lr=.01);opt.zero_grad(set_to_none=True)
        (m(x)-y).square().mean().backward();opt.step()
    step(reference)
    calls=0
    def train():
        nonlocal calls
        calls+=1
        if calls==1:
            with torch.no_grad():model.weight.add_(100.)
            raise RuntimeError("DefaultCPUAllocator: can't allocate memory: error code 12")
        step(model)
        return dict(history=[])
    result=run_training(model,train,lambda:torch.nn.Linear(2,1).double(),
                        ExecutionConfig(storage='auto',cache_dir=str(tmp_path)))
    assert calls==2 and result['v7_execution']['storage_attempts'][0]['status']=='OOM_REPLAY_FROM_INITIAL_STATE'
    for key,value in reference.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
