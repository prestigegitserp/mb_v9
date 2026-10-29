"""Real first-order equivalence and bounded replay/backward storage."""
import copy
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from v7.aggregation import sparse_value_mm
from v7.execution import ExecutionConfig,run_training
from v7.replay_state import ReplayState
from v7.resources import execute_candidate


@pytest.mark.parametrize('chunk',[1,7,64])
def test_sparse_value_and_weight_gradients_include_duplicates(chunk):
    torch.manual_seed(18);torch.set_num_threads(1)
    u=torch.tensor([0,0,1,2,2,2,3]);i=torch.tensor([0,0,2,3,3,1,4])
    v=torch.randn(7,dtype=torch.double,requires_grad=True)
    w=torch.randn(5,6,dtype=torch.double,requires_grad=True)
    a=torch.sparse_coo_tensor(torch.stack((u,i)),v,(4,5)).coalesce()
    expected=torch.sparse.mm(a,w);actual=sparse_value_mm(u,i,v,w,4,chunk)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    g=torch.randn_like(expected)
    reference=torch.autograd.grad((expected*g).sum(),(v,w),retain_graph=True)
    bounded=torch.autograd.grad((actual*g).sum(),(v,w))
    for x,y in zip(reference,bounded):torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-12)
    assert torch.autograd.gradcheck(lambda values,weights:sparse_value_mm(u,i,values,weights,4,chunk),(v,w))


def test_sparse_gate_backward_does_not_materialize_owner_by_counterpart():
    torch.manual_seed(3);torch.set_num_threads(1)
    u=torch.randint(100,(400,));i=torch.randint(200,(400,))
    v=torch.randn(400,requires_grad=True);w=torch.randn(200,8,requires_grad=True)
    with torch.autograd.profiler.profile(record_shapes=True) as profile:
        sparse_value_mm(u,i,v,w,100,17).square().sum().backward()
    assert not any(event.name=='aten::mm' and event.input_shapes[:2]==[[100,8],[8,200]]
                   for event in profile.function_events)


@pytest.mark.parametrize('host_budget',[0,8])
def test_replay_snapshot_is_lossless_and_releases_temporary_files(tmp_path,host_budget):
    model=torch.nn.Linear(8,5).double();model.register_buffer('sparse_history',
        torch.sparse_coo_tensor(torch.tensor([[0,1],[1,0]]),torch.tensor([1.,2.]),(2,2)))
    expected=copy.deepcopy(model.state_dict())
    cfg=ExecutionConfig(host_budget_mb=host_budget,cache_dir=str(tmp_path),ssd_budget_gb=.01)
    with ReplayState(model,cfg) as initial:
        assert initial.record['policy']==('ssd' if host_budget==0 else 'host')
        assert initial.execution_config().ssd_budget_gb<=cfg.ssd_budget_gb
        with torch.no_grad():model.weight.add_(100.)
        initial.restore_into(model)
        for key,value in expected.items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
    assert not list(tmp_path.iterdir())


def test_device_and_storage_replay_share_exactly_one_initial_snapshot(tmp_path,monkeypatch):
    import v7.replay_state as module
    created=0;original=module.ReplayState.__enter__
    def enter(self):
        nonlocal created
        created+=1;return original(self)
    monkeypatch.setattr(module.ReplayState,'__enter__',enter)
    model=torch.nn.Linear(2,1).double();reference=copy.deepcopy(model)
    data=type('Data',(),dict(train_users=torch.tensor([0,1])))()
    x=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.double)
    def step(m):
        opt=torch.optim.SGD(m.parameters(),lr=.01);opt.zero_grad(set_to_none=True)
        m(x).square().mean().backward();opt.step()
    step(reference);calls=0
    def callback():
        nonlocal calls
        calls+=1
        if calls==1:
            with torch.no_grad():model.weight.add_(100.)
            raise RuntimeError("DefaultCPUAllocator: can't allocate memory: error code 12")
        step(model);return dict(history=[])
    result=execute_candidate(model,data,callback,lambda *args:torch.nn.Linear(2,1).double(),
        ExecutionConfig(cache_dir=str(tmp_path),host_budget_mb=0))
    assert created==1 and calls==2
    assert result['v7_execution']['initial_state']['policy']=='ssd'
    for key,value in reference.state_dict().items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Real CUDA required for value-backward verification')
def test_cuda_sparse_value_backward_matches_native():
    u=torch.tensor([0,0,1,2,2],device='cuda');i=torch.tensor([0,0,3,1,2],device='cuda')
    v=torch.randn(5,device='cuda',dtype=torch.double,requires_grad=True)
    w=torch.randn(4,7,device='cuda',dtype=torch.double,requires_grad=True)
    expected=torch.sparse.mm(torch.sparse_coo_tensor(torch.stack((u,i)),v,(3,4)).coalesce(),w)
    actual=sparse_value_mm(u,i,v,w,3,2)
    for x,y in zip(torch.autograd.grad(expected.square().sum(),(v,w),retain_graph=True),
                   torch.autograd.grad(actual.square().sum(),(v,w))):torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-11)
