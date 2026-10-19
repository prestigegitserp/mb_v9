"""Independent optimality checks for the mean/dispersion metric prototype."""
import pytest
import torch
from experiments.v4_moment_metric_candidate import metric_projection
from v4.numerics import signed_moment_projection


@pytest.mark.parametrize('step',[None,1.])
def test_unit_metric_reduces_to_original_polygon_projection(step):
    torch.manual_seed(73)
    base=torch.linspace(1.,5.,101,dtype=torch.double)
    points=torch.randn(101,2,dtype=torch.double)*3
    a=metric_projection(points,base,1.,5.,step,1.)
    b=signed_moment_projection(points,base,1.,5.,step)
    torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize('gamma',[.02,.1,.5])
def test_projection_minimizes_weighted_distance_over_polygon_edges(gamma):
    torch.manual_seed(12)
    base=torch.linspace(1.05,4.95,31,dtype=torch.double)
    point=torch.randn(31,2,dtype=torch.double)*2
    out=metric_projection(point,base,1.,5.,1.,gamma)
    zero=torch.zeros_like(base)
    lower,upper=base.floor(),base.ceil()
    vertices=torch.stack([torch.stack([zero,base-1],-1),torch.stack([5-base,zero],-1),
                          torch.stack([upper-base,zero],-1),torch.stack([zero,base-lower],-1)],-2)
    edge=vertices.roll(-1,-2)-vertices
    grid=torch.linspace(0.,1.,1001,dtype=torch.double)
    candidates=vertices[:,:,None,:]+grid[None,None,:,None]*edge[:,:,None,:]
    distance=lambda x:(x[...,0]-x[...,1]).square()+gamma*(x[...,0]+x[...,1]).square()
    best_grid=distance(candidates-point[:,None,None,:]).flatten(1).min(-1).values
    actual=distance(out-point)
    assert (actual<=best_grid+1e-10).all()
    # Every returned point is feasible according to the independent original
    # Euclidean projection, not merely our own inside predicate.
    torch.testing.assert_close(signed_moment_projection(out,base,1.,5.,1.),out,atol=1e-9,rtol=1e-9)
    torch.testing.assert_close(metric_projection(out,base,1.,5.,1.,gamma),out,atol=0,rtol=0)


def test_mean_metric_keeps_finite_gradients_away_from_polygon_corners():
    point=torch.tensor([[3.2,-.4],[-.3,2.7],[.4,.5]],dtype=torch.double,requires_grad=True)
    base=torch.tensor([2.3,3.4,2.8],dtype=torch.double)
    fn=lambda x:metric_projection(x,base,1.,5.,1.,.1)
    assert torch.autograd.gradcheck(fn,(point,),eps=1e-6,atol=1e-5)


@pytest.mark.parametrize('gamma',[.02,.1,.5,1.])
def test_production_operator_matches_independent_prototype(gamma):
    torch.manual_seed(901)
    base=torch.linspace(1.,5.,301)
    points=torch.randn(301,2)*3
    prototype=metric_projection(points,base,1.,5.,1.,gamma)
    production=signed_moment_projection(points,base,1.,5.,1.,gamma)
    torch.testing.assert_close(prototype,production,atol=2e-6,rtol=1e-6)


def test_release_default_couples_metric_and_loss_and_restores_checkpoint(tmp_path):
    from dataclasses import replace
    from test_v4_research import config
    from test_v3_research import tiny_data,training_pub
    from v4.config import StudyConfig
    from v4.models import instantiate
    from v4.protocol import validation_roles
    from v4.training import train_v4
    from v4 import checkpoints
    default=StudyConfig().models['Method4']
    assert default.moment_projection_weight==default.moment_dispersion_weight==.1
    assert default.moment_projection_mode=='mean-priority'
    data=validation_roles(tiny_data(),StudyConfig());data.test_ratings_raw[:]=torch.nan
    cfg=replace(config('Method4'),moment_projection_weight=.1,moment_projection_mode='mean-priority')
    model=instantiate(data,cfg,19)
    info=train_v4(training_pub(),model,data,cfg,None)
    path=tmp_path/'metric.pt';checkpoints.save(path,model,config=cfg,data=data,train_info=info)
    loaded,_=checkpoints.load(path,data,torch.device('cpu'))
    with torch.no_grad():
        a=model.predict_raw_pairs(data,data.val_users,data.val_items)
        b=loaded.predict_raw_pairs(data,data.val_users,data.val_items)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert a.min()>=1-1e-5 and a.max()<=5+1e-5


@pytest.mark.parametrize('step',[None,1.])
def test_mean_priority_has_optimal_mean_and_independently_feasible_moments(step):
    from v4.numerics import signed_moment_priority_projection
    from experiments.v4_moment_metric_candidate import priority_projection
    torch.manual_seed(19)
    base=torch.linspace(1.,5.,301,dtype=torch.double)
    points=torch.randn(301,2,dtype=torch.double)*3
    out=signed_moment_priority_projection(points,base,1.,5.,step)
    torch.testing.assert_close(out,priority_projection(points,base,1.,5.,step),atol=0,rtol=0)
    torch.testing.assert_close(signed_moment_projection(out,base,1.,5.,step),out,atol=1e-9,rtol=1e-9)
    expected=(points[:,0]-points[:,1]).clamp(min=1-base,max=5-base)
    torch.testing.assert_close(out[:,0]-out[:,1],expected,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(signed_moment_priority_projection(out,base,1.,5.,step),out,atol=1e-9,rtol=1e-9)


def test_mean_priority_point_gradient_ignores_dispersion_proposal():
    from v4.numerics import signed_moment_priority_projection
    mean=torch.tensor([-.5,.8],dtype=torch.double,requires_grad=True)
    dispersion=torch.tensor([.001,10.],dtype=torch.double,requires_grad=True)
    points=torch.stack([.5*(dispersion+mean),.5*(dispersion-mean)],-1)
    result=signed_moment_priority_projection(points,torch.tensor([3.2,3.2],dtype=torch.double),1.,5.,1.)
    prediction=result[:,0]-result[:,1]
    gm,ga=torch.autograd.grad(prediction.sum(),(mean,dispersion))
    torch.testing.assert_close(gm,torch.ones_like(gm),atol=1e-12,rtol=0)
    torch.testing.assert_close(ga,torch.zeros_like(ga),atol=0,rtol=0)


@pytest.mark.parametrize('gamma',[0.,-1.,2.])
def test_invalid_projection_metric_fails(gamma):
    with pytest.raises(ValueError):
        metric_projection(torch.ones(1,2),torch.tensor([3.]),1.,5.,1.,gamma)
