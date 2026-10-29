"""Value and gradient parity, including duplicate edges and uneven chunks."""
import copy
import pytest
import torch
from v7.aggregation import pair_dot,sparse_encoder_forward
from v7.history import HistoryEncoder


@pytest.mark.parametrize("namespace",["v3","v4","v7"])
def test_fused_history_matches_explicit_edge_tensor_outputs_and_gradients(namespace):
    from importlib import import_module
    torch.manual_seed(81);torch.set_num_threads(1)
    if namespace=="v7":
        reference=HistoryEncoder(7,9,4,degree_gate=True,residual_skip=True).double()
        reference._v7_sparse_history=False
    else:
        reference=import_module(namespace+".history").HistoryEncoder(7,9,4).double()
    fast=copy.deepcopy(reference)
    u=torch.tensor([0,1,1,2,1,3,0,1,2,2,3])
    i=torch.tensor([1,2,2,3,0,6,2,4,3,4,1])
    y=torch.linspace(-1.,1.,len(u),dtype=torch.double)
    count=torch.bincount(u,minlength=5).double()
    expected=reference(u,i,y,count,5)
    actual=sparse_encoder_forward(fast,u,i,y,count,5)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    expected.square().sum().backward();actual.square().sum().backward()
    for a,b in zip(reference.parameters(),fast.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-10,atol=1e-12)


@pytest.mark.parametrize("chunk",[1,3,5,32])
def test_pair_dot_preserves_values_and_gradients(chunk):
    torch.manual_seed(19)
    a=torch.randn(5,7,dtype=torch.double,requires_grad=True)
    b=torch.randn(6,7,dtype=torch.double,requires_grad=True)
    u=torch.tensor([0,1,1,4,3,2,4,1,0,0,3])
    i=torch.tensor([0,2,4,2,1,3,5,3,2,2,1])
    expected=(a[u]*b[i]).sum(-1)
    actual=pair_dot(a,b,u,i,chunk)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    weights=torch.linspace(-1.,1.,len(u),dtype=torch.double)
    direct=torch.autograd.grad((expected*weights).sum(),(a,b))
    bounded=torch.autograd.grad((actual*weights).sum(),(a,b))
    for x,y in zip(direct,bounded):torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)


def test_chunked_dot_autograd_finite_difference():
    torch.manual_seed(9)
    a=torch.randn(3,4,dtype=torch.double,requires_grad=True)
    b=torch.randn(4,4,dtype=torch.double,requires_grad=True)
    u=torch.tensor([0,1,2,0,2]);i=torch.tensor([2,3,1,1,3])
    assert torch.autograd.gradcheck(lambda x,y:pair_dot(x,y,u,i,2),(a,b))
