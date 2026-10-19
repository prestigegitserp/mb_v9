"""V4 scientific invariants, lossless execution and version-preserving workflow."""
from dataclasses import replace
import copy,json,random,weakref,gc
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from test_v3_research import tiny_data, training_pub, config as old_config
from v4.config import ModelConfig,StudyConfig
from v4.models import instantiate,RiskRouter
from v4.protocol import validation_roles
from v4.training import train_v4,optimizer_groups,fit_probability
from v4.history import MaskSampler,masked_targets
from v4.numerics import preconditioned_cg
from v4.execution import ExecutionConfig,SavedTensorStore,run_training
from v4 import checkpoints
from v4.bounds import structural_bounds

METHODS=[f'Method{i}' for i in range(1,8)]


def config(method):
    return ModelConfig(**{**vars(old_config(method)),'posterior_cg_steps':64,'posterior_solver_tolerance':1e-4})


@pytest.mark.parametrize('method',METHODS)
def test_all_seven_training_poisoned_test_checkpoint_and_bounds(method,tmp_path):
    data=validation_roles(tiny_data(),StudyConfig());data.test_ratings_raw[:]=torch.nan
    model=instantiate(data,config(method),17);model._v4_artifact_dir=tmp_path/'mechanism'
    info=train_v4(training_pub(),model,data,model.cfg,None)
    assert np.isfinite(info['best_val_RMSE']) and info['best_epoch']>=1
    with torch.no_grad():p=model.predict_raw_pairs(data,data.val_users,data.val_items)
    lo,hi=structural_bounds(model,data,data.val_users,data.val_items)
    assert np.all(p.numpy()>=lo-2e-5) and np.all(p.numpy()<=hi+2e-5)
    path=tmp_path/'model.pt';checkpoints.save(path,model,config=model.cfg,data=data,train_info=info)
    loaded,_=checkpoints.load(path,data,torch.device('cpu'))
    with torch.no_grad():q=loaded.predict_raw_pairs(data,data.val_users,data.val_items)
    torch.testing.assert_close(p,q,atol=2e-6,rtol=1e-6)
    catalog=loaded.score_catalog(data,torch.tensor([2]),chunk_size=5)
    torch.testing.assert_close(catalog[0],loaded.predict_raw_pairs(data,torch.full((16,),2),torch.arange(16)))
    trace=pd.read_csv(tmp_path/'mechanism'/'training_steps.csv')
    assert len(trace)==info['mean_optimizer_steps'] and np.isfinite(trace.gradient_norm_before_clip).all()
    if method=='Method2':
        assert info['v4_routing']['oof_pairs']==128 and info['v4_routing']['router_history']
        assert (tmp_path/'mechanism'/'out_of_fold_risk_targets.csv.gz').is_file()
        assert 'L_consistency' in trace
    if method=='Method5':assert model.posterior_cg_residual<=model.cfg.posterior_solver_tolerance
    if method=='Method7':
        dist=loaded.distribution(data,data.val_users,data.val_items)
        torch.testing.assert_close(dist['probabilities']@loaded.levels,q,atol=2e-5,rtol=1e-6)
        assert info['v4_probability']['mean_parameters_unchanged'] and info['v4_probability']['shape_history']
        assert 'best_calibration_hold_nll' not in info['v4_probability']


@pytest.mark.parametrize('method',METHODS)
def test_cavity_excludes_query_rating_and_timestamp(method):
    data=tiny_data();m=instantiate(data,config(method),3).eval()
    keep=torch.arange(128)%3!=0;h=m.history(keep);s=m.prepare(h)
    u,i=data.train_users[~keep],data.train_items[~keep]
    with torch.no_grad():p=m.predict(h,s,u,i)[0]
    data.train_ratings_raw[~keep]=1000;data.train_timestamps[~keep]=-1e12
    h2=m.history(keep);s2=m.prepare(h2)
    with torch.no_grad():q=m.predict(h2,s2,u,i)[0]
    torch.testing.assert_close(p,q,atol=2e-6,rtol=1e-6)


def _same(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:_same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):_same(x,y)
    else:assert a==b


@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('storage',['host','ssd'])
def test_lossless_saved_tensors_preserve_adam_updates(method,storage,tmp_path):
    data=tiny_data();cfg=replace(config(method),feature_dropout=0.,gradient_surgery=False)
    keep,query=masked_targets(data,.25,torch.Generator().manual_seed(19))
    outputs=[]
    for policy in ('resident',storage):
        m=instantiate(data,cfg,6);groups,_=optimizer_groups(m,2);opt=torch.optim.AdamW(groups)
        with SavedTensorStore(ExecutionConfig(storage=policy,cache_dir=str(tmp_path))):
            for _ in range(2):
                m.train();h=m.history(keep);state=m.prepare(h)
                a,b,_=m.loss_for_targets(h,state,data.train_users[query],data.train_items[query],data.train_ratings_raw[query])
                opt.zero_grad(set_to_none=True);(a+b).backward();opt.step();m.clear_cache()
        outputs.append((copy.deepcopy(m.state_dict()),copy.deepcopy(opt.state_dict())))
    _same(*outputs)
    assert not list(tmp_path.glob('v4_graph_*'))


def test_hooks_multiple_backward_and_cleanup_on_failure(tmp_path):
    with pytest.raises(RuntimeError,match='forced'):
        with SavedTensorStore(ExecutionConfig(storage='ssd',cache_dir=str(tmp_path))) as storage:
            x=torch.randn(10,requires_grad=True);loss=(x.sin()*x).sum()
            a=torch.autograd.grad(loss,x,retain_graph=True)[0];b=torch.autograd.grad(loss,x)[0]
            torch.testing.assert_close(a,b,atol=0,rtol=0);assert storage.disk_peak_bytes>0
            raise RuntimeError('forced')
    assert not list(tmp_path.glob('v4_graph_*'))


def test_host_budget_spills_without_dtype_conversion(tmp_path):
    with SavedTensorStore(ExecutionConfig(storage='host',host_budget_mb=0,cache_dir=str(tmp_path))) as store:
        x=torch.randn(9,dtype=torch.float64,requires_grad=True);loss=x.square().sum();loss.backward()
        assert store.disk_bytes>0 and store.host_bytes==0
        torch.testing.assert_close(x.grad,2*x,atol=0,rtol=0)


def test_oom_replays_whole_candidate_after_releasing_failed_parameters(tmp_path,monkeypatch):
    import v4.execution as execution
    monkeypatch.setattr(execution,'storage_policies',lambda config,cuda:['resident','host'])
    seed=129
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=torch.nn.Linear(3,2);initial=copy.deepcopy(model.state_dict());seen=[];weak=[];rollbacks=[]
    def rebuild():
        gc.collect();assert all(p() is None for p in weak)
        return torch.nn.Linear(3,2)
    def callback():
        opt=torch.optim.Adam(model.parameters(),lr=.1)
        draws=(random.random(),float(np.random.rand()),torch.rand(3))
        x=torch.randn(4,3);opt.zero_grad();model(x).square().mean().backward();opt.step()
        seen.append((draws,copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict())))
        if len(seen)==1:
            weak.extend(weakref.ref(p) for p in model.parameters())
            raise torch.cuda.OutOfMemoryError('simulated after optimizer update')
        return {'ok':True}
    result=run_training(model,callback,rebuild,ExecutionConfig(storage='auto',cache_dir=str(tmp_path)),lambda:rollbacks.append(True))
    _same(seen[0],seen[1]);assert rollbacks==[True]
    assert result['v4_execution']['storage_attempts'][0]['status'].startswith('OOM_REPLAY')
    assert any(not torch.equal(initial[k],model.state_dict()[k]) for k in initial)


def test_pcg_certificate_is_actual_residual_on_ill_conditioned_spd():
    torch.manual_seed(10);q,_=torch.linalg.qr(torch.randn(18,18,dtype=torch.double))
    h=q@torch.diag(torch.logspace(-3,3,18,dtype=torch.double))@q.T;b=torch.randn(18,dtype=torch.double)
    solve,residual,iterations=preconditioned_cg(lambda x:h@x,b,lambda x:torch.linalg.solve(h,x),steps=4,tolerance=1e-8)
    assert iterations==1 and residual<1e-8
    actual=float(torch.linalg.vector_norm(b-h@solve)/torch.linalg.vector_norm(b))
    assert residual==pytest.approx(actual,rel=1e-10)
    zero,r,n=preconditioned_cg(lambda x:h@x,torch.zeros_like(b),lambda x:x)
    assert not zero.any() and r==0 and n==0
    with pytest.raises(FloatingPointError):preconditioned_cg(lambda x:-x,b,lambda x:x)


def test_router_coarse_curvature_ignores_pair_information():
    r=RiskRouter(config('Method2'));x=torch.randn(15,11);y=x.clone();y[:,8:]+=100
    torch.testing.assert_close(r.curvatures(x),r.curvatures(y),atol=0,rtol=0)
    assert (r.curvatures(x)>0).all()
    with torch.no_grad():
        r.lambda0.fill_(.5);r.strength.fill_(1.);r.noise_floor.zero_()
        for h in r.heads:h[-1].bias.fill_(.2)
    diff=torch.linspace(-2,2,15);gate,_=r(x,diff)
    benefit=r.moments(x).mean(0);curv=diff.square()+r.cfg.router_context_penalty*r.curvatures(x).mean(0)+r.cfg.router_ridge
    eta=gate-.5
    for shift in (-.01,.01):
        alt=(eta+shift).clamp(-.5,.5)
        assert (curv*eta.square()-2*benefit*eta<=curv*alt.square()-2*benefit*alt+1e-6).all()


def test_m3_context_center_has_no_stale_trainable_delta():
    m=instantiate(tiny_data(),config('Method3'),6);h=m.history();s=m.prepare(h)
    assert 'delta' not in dict(m.named_parameters()) and 'center' in s
    assert s['center'].abs().max()<=m.cfg.log_radius_bound
    loss=s['center'].square().mean();loss.backward()
    assert m.user_log_prior.weight.grad is not None and torch.isfinite(m.user_log_prior.weight.grad).all()
    legacy=instantiate(tiny_data(),replace(config('Method3'),context_radius=False),6)
    assert 'delta' in dict(legacy.named_parameters())
    legacy.refresh_radius();assert legacy.last_radius_objective_after<=legacy.last_radius_objective_before+1e-6


def test_m4_unprojected_ablation_uses_wider_valid_support():
    data=tiny_data();m=instantiate(data,replace(config('Method4'),project_moments=False),4).eval()
    lo,hi=structural_bounds(m,data,data.val_users,data.val_items)
    assert np.all(lo==0.) and np.all(hi==6.)
    with torch.no_grad():p=m.predict_raw_pairs(data,data.val_users,data.val_items).numpy()
    assert np.all(p>=lo-1e-6) and np.all(p<=hi+1e-6)


def test_m5_refinement_descends_fixed_ridge_objective():
    data=tiny_data();cfg=config('Method5');m=instantiate(data,cfg,6);h=m.history();z0,_=m.encode(h)
    target=((h.y-m.base(h,h.u,h.i))/m.width).detach();prior=m.user_decoder.weight
    d=m._ridge(z0,prior,h.u,h.i,target,m.ni)
    a=m._ridge(d,z0,h.i,h.u,target,m.nu);z=(1-cfg.posterior_relaxation)*z0+cfg.posterior_relaxation*a
    objective=lambda z,d:((z[h.u]*d[h.i]).sum(-1)-target).square().sum()/(2*cfg.posterior_noise**2)+cfg.posterior_precision*.5*((z-z0).square().sum()+(d-prior).square().sum())
    d2=m._ridge(z,prior,h.u,h.i,target,m.ni)
    assert objective(z,d2)<=objective(z0,d)+1e-5


def test_histogram_hierarchy_matches_scalar_right_ecdf_with_ties():
    from v4.transport import AngularTransport
    t=AngularTransport(4,grid=33,groups=8)
    scores=torch.tensor([-.5,-.5,0.,.5,.5,1.]);groups=torch.tensor([0,0,1,4,4,6]);tau=10.;weight=.7
    t.update_reference(scores,groups,0,weight,tau,True,2)
    iso=t.isotropic_cdf;grid=t.angle_grid
    ecdf=lambda v:torch.stack([(v<=x).float().mean() for x in grid]) if len(v) else torch.zeros_like(grid)
    global_cdf=len(scores)/(len(scores)+tau)*ecdf(scores)+tau/(len(scores)+tau)*iso
    for g in range(8):
        v=scores[groups==g];parent=scores[groups//2==g//2]
        p=len(parent)/(len(parent)+tau)*ecdf(parent)+tau/(len(parent)+tau)*global_cdf
        c=len(v)/(len(v)+tau)*ecdf(v)+tau/(len(v)+tau)*p
        expected=(1-weight)*iso+weight*c;expected[0]=0.;expected[-1]=1.
        torch.testing.assert_close(t.reference_cdf[0,g],expected,atol=1e-7,rtol=1e-6)


@pytest.mark.parametrize('continuous',[False,True])
def test_m7_shape_fibre_and_total_variance(continuous):
    data=validation_roles(tiny_data(continuous=continuous),StudyConfig());m=instantiate(data,config('Method7'),7).eval()
    before={k:v.clone() for k,v in m.state_dict().items() if not k.startswith('shape_heads.') and k!='shape_fitted'}
    info=fit_probability(m,data)
    for k,v in before.items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
    dist=m.distribution(data,data.val_users,data.val_items)
    torch.testing.assert_close(dist['variance'],dist['within_variance']+dist['between_variance'],atol=2e-6,rtol=1e-5)
    if not continuous:
        mean=dist['probabilities']@m.levels;torch.testing.assert_close(mean,dist['member_means'].mean(-1),atol=2e-5,rtol=1e-6)
    assert np.isfinite(info['best_calibration_hold_objective'])


def test_mask_sampler_preserves_original_rng_and_duplicate_pair_grouping():
    d=tiny_data();d.train_users=torch.cat([d.train_users,d.train_users[:4]]);d.train_items=torch.cat([d.train_items,d.train_items[:4]])
    a=torch.Generator().manual_seed(92);b=torch.Generator().manual_seed(92);sampler=MaskSampler(d,.3,a)
    for _ in range(10):
        _same(sampler.sample(),masked_targets(d,.3,b));torch.testing.assert_close(a.get_state(),b.get_state(),atol=0,rtol=0)


@pytest.mark.parametrize('field,value',[('posterior_relaxation',0.),('posterior_solver_tolerance',0.),('posterior_mean_steps',True),('router_context_penalty',-1.),('hard_transport_balance','false'),('probability_rps_weight',float('nan'))])
def test_invalid_new_configuration_fails(field,value):
    with pytest.raises(ValueError):replace(config('Method1'),**{field:value}).validate()


def test_version_dispatch_calls_selected_entry(monkeypatch):
    import run_v9,v2.entry,v3.entry,v4.entry
    seen=[]
    for n,module in [(2,v2.entry),(3,v3.entry),(4,v4.entry)]:monkeypatch.setattr(module,'run',lambda argv,n=n:seen.append((n,argv)))
    run_v9.run(['main']);run_v9.run(['--code-version','3','plan']);run_v9.run(['main','--code-version=2'])
    assert seen==[(4,['main']),(3,['plan']),(2,['main'])]


def test_legacy_sources_identical_to_v3_release():
    from verify_v4 import verify
    assert verify()['preservation']=='PASS'


def test_preservation_accepts_windows_checkout_line_endings_only(tmp_path):
    from verify_v4 import verify
    import hashlib
    content=b'print("legacy")\n'
    (tmp_path/'legacy.py').write_bytes(content.replace(b'\n',b'\r\n'))
    (tmp_path/'V4_BASELINE_MANIFEST.json').write_text(json.dumps(dict(protected_files={'legacy.py':hashlib.sha256(content).hexdigest()},base_commit='base')))
    assert verify(tmp_path)['preservation']=='PASS'
    (tmp_path/'legacy.py').write_bytes(b'print("changed")\r\n')
    with pytest.raises(ValueError,match='changed'):verify(tmp_path)


def test_sensitivity_averages_seeds_within_split_before_sd():
    from v4.figures import _outer_summary
    d=pd.DataFrame(dict(dataset=['x']*4,method=['M']*4,K=[20]*4,split_seed=[1,1,2,2],model_seed=[1,2,1,2],test_RMSE=[0.,2.,2.,4.]))
    outer,summary=_outer_summary(d,'K',['test_RMSE'])
    assert list(outer.test_RMSE)==[1.,3.] and summary.test_RMSE__mean.iloc[0]==2.
    assert summary.test_RMSE__std.iloc[0]==pytest.approx(2**.5)


def test_validation_plot_rejects_test_input(tmp_path):
    from v4.figures import plot_validation_landscape
    path=tmp_path/'rank.csv';pd.DataFrame(dict(method=['M'],test_RMSE=[1.])).to_csv(path,index=False)
    with pytest.raises(ValueError,match='leakage'):plot_validation_landscape(path,tmp_path/'plots')


@pytest.mark.parametrize('method',METHODS)
def test_fixed_masked_objective_learns_with_finite_gradients(method):
    cfg=replace(config(method),lr=.01,direction_lr=.01,bias_lr=.01,feature_dropout=0.,gradient_surgery=False)
    data=tiny_data();m=instantiate(data,cfg,6);groups,_=optimizer_groups(m,20);opt=torch.optim.AdamW(groups)
    include,query=masked_targets(data,.25,torch.Generator().manual_seed(19));losses=[]
    for _ in range(20):
        m.train();h=m.history(include);state=m.prepare(h)
        a,b,_=m.loss_for_targets(h,state,data.train_users[query],data.train_items[query],data.train_ratings_raw[query]);loss=a+b
        losses.append(float(loss.detach()));opt.zero_grad(set_to_none=True);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']],5.);opt.step();m.clear_cache()
    assert min(losses[-3:])<.95*losses[0],(method,losses)


@pytest.mark.parametrize('method',METHODS)
def test_zero_context_new_entities_have_finite_bounded_predictions(method):
    data=tiny_data();data.num_users+=1;data.num_items+=1
    m=instantiate(data,config(method),3).eval();u=torch.tensor([12,0,12]);i=torch.tensor([0,16,16])
    p=m.predict_raw_pairs(data,u,i)
    assert torch.isfinite(p).all() and p.min()>=1-1e-5 and p.max()<=5+1e-5


@pytest.mark.parametrize('method',METHODS)
def test_dense_and_sparse_construction_preserve_predictions(method):
    from v9.engine import load_v8
    _,pub=load_v8();torch.set_num_threads(2);d=tiny_data();parts=[]
    for split in ('train','val','test'):
        parts.append(pd.DataFrame(dict(userId=getattr(d,split+'_users').numpy(),movieId=getattr(d,split+'_items').numpy(),rating=getattr(d,split+'_ratings_raw').numpy(),split=split)))
    frame=pd.concat(parts,ignore_index=True);outputs=[]
    for backend in ('dense','sparse'):
        data,_=pub.build_publication_matrix_data_v8(frame,pub.DATASET_SPECS['ml-100k'],pub.PublicationDataConfig(dataset='ml-100k',force_backend=backend),torch.device('cpu'))
        model=instantiate(data,config(method),37).eval()
        with torch.no_grad():outputs.append(model.predict_raw_pairs(data,data.val_users,data.val_items))
    torch.testing.assert_close(*outputs,atol=3e-6,rtol=1e-6)
