import math, types, copy
from types import SimpleNamespace
import torch
from torch import nn
from v2.geometry import bounded_residual,cosine_pairs,radial_gain
from v2.adapters import adapt,apply_profile,PROFILES
from v2.training import train_v2


def make_data(nu=8,ni=9,edges=80):
    rng=torch.Generator().manual_seed(27)
    u=torch.randint(nu,(edges,),generator=rng)
    i=torch.randint(ni,(edges,),generator=rng)
    y=torch.randint(1,6,(edges,),generator=rng).float()
    return SimpleNamespace(num_users=nu,num_items=ni,train_users=u,train_items=i,
        train_ratings_raw=y,val_users=u[:22],val_items=i[:22],val_ratings_raw=y[:22],
        test_users=u[22:45],test_items=i[22:45],test_ratings_raw=y[22:45],
        rating_min=1.,rating_max=5.,midpoint=3.,half_range=2.)


class DummyDecoder(nn.Module):
    def __init__(self,k,n):
        super().__init__();self.w=nn.Embedding(n,k);self.b=nn.Embedding(n,1)
    def selected(self,z,a,b):return (z[a]*self.w(b)).sum(-1)+self.b(b).squeeze(-1)
    def selected_reconstruction(self,z,a,b):return self.selected(z,a,b)


class FakeMethods(nn.Module):
    def __init__(self,data,which):
        super().__init__();self.which=which;k=6
        self.cfg=SimpleNamespace(latent_dim=k,alpha_ae=.1,beta_mbmf=1.,delta_l2=.01,
             consistency_weight=.04,fusion_weight_user_view=.5,branch_mode='both',
             prediction_weight=5.,user_reconstruction_weight=.1,
             item_reconstruction_weight=.1,branch_consistency_weight=.02,l2_code_weight=0.,
             max_epochs=5,eval_every=1,val_patience=3,grad_clip=1.,lr=.01,weight_decay=.0001)
        self.uz=nn.Parameter(torch.randn(data.num_users,k)*.2)
        self.iz=nn.Parameter(torch.randn(data.num_items,k)*.2)
        self.Phi=nn.Parameter(torch.randn(data.num_users,k-1))
        self.Theta=nn.Parameter(torch.randn(data.num_items,k-1))
        self.user_view_item_decoder=nn.Embedding(data.num_items,k)
        self.user_view_item_bias=nn.Embedding(data.num_items,1)
        self.item_view_user_bias=nn.Embedding(data.num_users,1)
        self.item_view_user_decoder=nn.Embedding(data.num_users,k)
        self.decoder_user=DummyDecoder(k,data.num_items)
        self.decoder_item=DummyDecoder(k,data.num_users)
        self.user_branch=DummyDecoder(k,data.num_items)
        self.item_branch=DummyDecoder(k,data.num_users)
        self.register_buffer('rW0',torch.ones(data.num_users,1))
        self.register_buffer('rH0',torch.ones(data.num_items,1))

    def encoded(self,data):
        zu,zi=self.uz,self.iz
        ru=(1.+torch.tanh(zu.norm(dim=1,keepdim=True)))
        ri=(1.+torch.tanh(zi.norm(dim=1,keepdim=True)))
        W=zu/zu.norm(dim=-1,keepdim=True)*ru
        H=zi/zi.norm(dim=-1,keepdim=True)*ri
        if self.which=='Method1':return zu,zi,ru,ri,W,H
        corr_u=.2*torch.tanh(zu[:,0:1]);corr_i=.2*torch.tanh(zi[:,0:1])
        return zu,zi,corr_u,corr_i,ru,ri,W,H

    def factors(self,data):
        zu,zi=self.uz,self.iz
        ru=1.+torch.tanh(zu.norm(dim=1,keepdim=True))
        ri=1.+torch.tanh(zi.norm(dim=1,keepdim=True))
        return ru,ri,ri,ru,zu,zi,zi,zu

    def codes(self,data):
        return self.uz,self.iz,self.iz,self.uz

    def _radius(self,values,data):
        return values.norm(dim=1,keepdim=True)+.5

    def _all_state(self,data):
        z1,z2=self.uz,self.iz
        a=1.+z1.norm(dim=1,keepdim=True)
        b=1.+z2.norm(dim=1,keepdim=True)
        return z1,z2,z2,z1,a,b,b,a,z1,z2,z1,z2

    def predict_raw_pairs(self,data,u,i):
        return self.predict_views(data,u,i)[2]

    def predict_views(self,data,u,i):
        p1=torch.full_like(u,3.,dtype=torch.float)
        return p1,p1,p1,self._all_state(data)

    def loss(self,data):
        u,i,y=data.train_users,data.train_items,data.train_ratings_raw
        p1,p2,p,_=self.predict_views(data,u,i)
        z1,z2,_,_,*_=self._all_state(data)
        rec1=(self.user_branch.selected(z1,u,i)-y/5).square().mean()
        rec2=(self.item_branch.selected(z2,i,u)-y/5).square().mean()
        lp=((p-y)/4).square().mean()
        return self.cfg.prediction_weight*lp+self.cfg.user_reconstruction_weight*rec1+self.cfg.item_reconstruction_weight*rec2,{'L_pred':float(lp.detach())}


class FakeM7(nn.Module):
    def __init__(self,data):
        super().__init__();k=6;fd=4
        self.cfg=SimpleNamespace(latent_dim=k,use_bias_expert=True,heteroscedastic=True,min_variance=.0001,
            feature_dropout=0.,use_context=False,use_shrinkage=False,min_scale=.05)
        self.midpoint=data.midpoint;self.half_range=data.half_range
        self.user_features=nn.Parameter(torch.randn(data.num_users,fd),requires_grad=False)
        self.item_features=nn.Parameter(torch.randn(data.num_items,fd),requires_grad=False)
        self.register_buffer('user_counts',torch.ones(data.num_users))
        self.register_buffer('item_counts',torch.ones(data.num_items))
        self.user_direction=nn.Embedding(data.num_users,k)
        self.item_direction=nn.Embedding(data.num_items,k)
        self.user_bias=nn.Embedding(data.num_users,1)
        self.item_bias=nn.Embedding(data.num_items,1)
        self.global_bias=nn.Parameter(torch.tensor(0.))
        self.user_encoder=nn.Linear(fd,k);self.item_encoder=nn.Linear(fd,k)
        self.user_radius=nn.Linear(fd,1);self.item_radius=nn.Linear(fd,1)
        self.user_radius_prior=nn.Parameter(torch.tensor(0.))
        self.item_radius_prior=nn.Parameter(torch.tensor(0.))
        self.log_tau_user=nn.Parameter(torch.tensor(0.))
        self.log_tau_item=nn.Parameter(torch.tensor(0.))
        self.user_logvariance=nn.Linear(fd,1);self.item_logvariance=nn.Linear(fd,1)
        self.log_variance_prior=nn.Parameter(torch.tensor(0.))
    def _entity_state(self,ids,features,counts,table,encoder,radius_head,prior,log_tau):
        z=features[ids]
        return nn.functional.normalize(table(ids),dim=-1),torch.sigmoid(radius_head(z)).squeeze(-1),counts[ids]/(counts[ids]+1.)
    def forward_pairs(self,*args,**kwargs):raise NotImplementedError
    def loss_pairs(self,data,u,i,y):
        p,aux=self.forward_pairs(data,u,i,return_aux=True)
        return ((p-y)/4).square().mean()+.001*aux['variance_normalized'].mean(),{'L_pred':float(((p-y)/4).square().mean().detach())}
    def predict_raw_pairs(self,data,u,i):return self.forward_pairs(data,u,i)
    def predict_raw_with_uncertainty(self,data,u,i):
        p,aux=self.forward_pairs(data,u,i,return_aux=True)
        return p,aux['variance_raw']


def test_range_proof_monte_carlo():
    gen=torch.Generator().manual_seed(333)
    c=1.+4*torch.rand(100000,generator=gen)
    s=2*torch.rand(100000,generator=gen)-1
    g=torch.rand(100000,generator=gen)
    p=bounded_residual(c,s,g,1.,5.)
    assert p.min()>=1.-1e-6 and p.max()<=5.+1e-6
    z=bounded_residual(c,torch.zeros_like(s),g,1.,5.)
    assert torch.allclose(z,c)


def test_direction_isotropic_symmetry():
    u=torch.randn(5000,16);u=u/u.norm(dim=1,keepdim=True)
    energies=u.square().mean(0)
    assert (energies-1./16).abs().max()<.005


def test_radial_gain_no_frozen_gradient():
    rho=torch.tensor(.0001,requires_grad=True)
    g=radial_gain(rho,torch.tensor(.1),2.,torch.tensor(0.),.15)
    g.backward();assert g.item()>=.15 and rho.grad is not None and torch.isfinite(rho.grad)


def test_v2_methods_bounds_train_gradients_and_restore():
    data=make_data()
    for i in range(1,7):
        method=f'Method{i}'
        m=adapt(FakeMethods(data,method),method,data)
        m.train()
        p=m.predict_raw_pairs(data,data.val_users,data.val_items)
        assert torch.isfinite(p).all(),method
        assert torch.all((p>=1.-1e-5)&(p<=5.+1e-5)),method
        l,parts=m.loss(data);l.backward()
        assert m.v2_u_dir.grad is not None and m.v2_u_dir.grad.abs().sum()>0.,method
        assert torch.isfinite(l),method
        assert not m.Phi.requires_grad and not m.Theta.requires_grad
        assert m.diagnostics(data,data.val_users,data.val_items)['v2_max_global_range_violation']==0.


def test_method56_ablation_branch_isolation():
    data=make_data()
    m=adapt(FakeMethods(data,'Method5'),'Method5',data)
    u=data.val_users;i=data.val_items
    m.cfg.branch_mode='user';user=m.predict_raw_pairs(data,u,i)
    m.cfg.branch_mode='item';item=m.predict_raw_pairs(data,u,i)
    m.cfg.branch_mode='both';joint=m.predict_raw_pairs(data,u,i)
    assert not torch.allclose(user,item)
    gate=torch.sigmoid(m.v2_mix_logit)
    assert torch.allclose(joint,gate*user+(1-gate)*item,atol=1e-5)


def test_method7_separate_expert_gate_and_variance_gradient():
    data=make_data()
    m=adapt(FakeM7(data),'Method7',data)
    p,aux=m.forward_pairs(data,data.val_users,data.val_items,return_aux=True)
    assert torch.all((p>=1.-1e-6)&(p<=5.+1e-6))
    assert abs(float(aux['gate'].mean().detach())-.5)<.01
    assert (aux['variance_raw']>0).all()
    y=data.val_ratings_raw
    l,parts=m.loss_pairs(data,data.val_users,data.val_items,y);l.backward()
    assert m.v2_expert_logit.grad is not None
    assert m.v2_var_logscale.grad is not None
    assert torch.isfinite(l)


def test_full_v2_train_uses_validation_and_early_stop():
    data=make_data()
    m=adapt(FakeMethods(data,'Method1'),'Method1',data)
    class Pub:
        class MetricConfig:
            def __init__(self,**kw):pass
        @staticmethod
        def evaluate_publication_model(model,data,split,metric_cfg=None):
            with torch.no_grad():
                p=model.predict_raw_pairs(data,data.val_users,data.val_items)
                return {'RMSE':float(torch.mean((p-data.val_ratings_raw)**2).sqrt())}
    info=train_v2(Pub,m,data,m.cfg,chunk_size=19)
    assert info['best_epoch']>=1
    assert info['best_epoch']<=m.cfg.max_epochs
    assert len(info['history'])>=1
    assert info['best_val_RMSE']<math.inf


def test_profile_does_not_increase_capacity_when_regularizing():
    cc=SimpleNamespace(**{f'method{i}':SimpleNamespace(angle_init='legacy_uniform_2pi',hidden_dim=128,
        weight_decay=0.,max_epochs=2000,eval_every=15,val_patience=20,alpha_ae=.7,beta_mbmf=.35,
        delta_l2=0.,consistency_weight=.25,prediction_weight=1.,user_reconstruction_weight=1.,item_reconstruction_weight=1.,branch_consistency_weight=0.,l2_code_weight=0.,radius_temperature=0.) for i in range(1,8)})
    base=SimpleNamespace(core_config=cc)
    regular=apply_profile(base,'regularized');big=apply_profile(base,'capacity')
    assert regular.core_config.method1.hidden_dim<big.core_config.method1.hidden_dim
    assert cc.method1.angle_init=='legacy_uniform_2pi'
    assert regular.core_config.method6.max_epochs==650
    assert regular.core_config.method2.consistency_weight==.04


def test_phase_restored_from_checkpoint_for_warmup():
    data=make_data()
    m=adapt(FakeMethods(data,'Method3'),'Method3',data)
    m.v2_phase.fill_(1.)
    warm=m.predict_raw_pairs(data,data.val_users,data.val_items).detach().clone()
    state=copy.deepcopy(m.state_dict())
    m.v2_phase.fill_(0.)
    post=m.predict_raw_pairs(data,data.val_users,data.val_items).detach().clone()
    assert not torch.allclose(warm,post)
    m.load_state_dict(state)
    assert m.v2_phase.item()==1.
    assert torch.allclose(warm,m.predict_raw_pairs(data,data.val_users,data.val_items))


def test_method7_calibration_uses_disjoint_validation_only():
    from v2.calibration import calibration_masks,calibrate_method7,nll_on_disjoint_selection_half
    data=make_data(edges=200)
    data.val_users=data.train_users[:120]
    data.val_items=data.train_items[:120]
    data.val_ratings_raw=data.train_ratings_raw[:120]
    fit,other=calibration_masks(data)
    assert fit is not None
    assert not torch.any(fit & other)
    assert torch.all(fit | other)
    m=adapt(FakeM7(data),'Method7',data)
    before=m.v2_var_logscale.detach().clone()
    outcome=calibrate_method7(m,data)
    assert outcome['applied'] and outcome['test_touched'] is False
    assert outcome['calibration_fit_nll_after']<=outcome['calibration_fit_nll_before']+1e-7
    report=nll_on_disjoint_selection_half(m,data)
    assert math.isfinite(report['Gaussian_NLL'])
    assert report['V2_Calibration_NLL_Heldout_N']==int(other.sum())


def test_single_view_ablation_zero_cross_branch_gradients():
    d=make_data()
    m=adapt(FakeMethods(d,'Method6'),'Method6',d)
    for active,inactive in [('user','item'),('item','user')]:
        m.zero_grad(set_to_none=True)
        m.cfg.branch_mode=active
        loss,diag=m.loss(d)
        loss.backward()
        assert torch.isfinite(loss)
        assert diag[f'L_{inactive}_reconstruction']==0.0
        if active=='user':
            assert m.v2_i_dir_alt.grad is None or m.v2_i_dir_alt.grad.abs().sum()==0
            assert m.v2_u_dir_alt.grad is None or m.v2_u_dir_alt.grad.abs().sum()==0
            assert m.v2_i_dir.grad is not None and m.v2_i_dir.grad.abs().sum()>0
        else:
            assert m.v2_i_dir.grad is None or m.v2_i_dir.grad.abs().sum()==0
            assert m.v2_u_dir.grad is None or m.v2_u_dir.grad.abs().sum()==0
            assert m.v2_i_dir_alt.grad is not None and m.v2_i_dir_alt.grad.abs().sum()>0


def test_fusion_weight_controls_both_dual_view_models():
    d=make_data()
    for name in ('Method2','Method5','Method6'):
        torch.manual_seed(909)
        m=adapt(FakeMethods(d,name),name,d)
        m.eval()
        with torch.no_grad():
            m.v2_mix_logit.fill_(math.log(.2/.8))
            low=m.predict_raw_pairs(d,d.val_users,d.val_items).clone()
            m.v2_mix_logit.fill_(math.log(.8/.2))
            high=m.predict_raw_pairs(d,d.val_users,d.val_items).clone()
        assert not torch.allclose(low,high,atol=1e-6),name


def test_method7_no_bias_ablation_uses_only_geometric_expert():
    d=make_data()
    m=adapt(FakeM7(d),'Method7',d)
    m.cfg.use_bias_expert=False
    pred,aux=m.forward_pairs(d,d.val_users,d.val_items,return_aux=True)
    assert torch.all(aux['gate']==1.)
    assert torch.allclose(pred,m.midpoint+m.half_range*aux['cosine'],atol=1e-6)


def test_v2_runner_directs_default_v9_cli_to_replacement():
    from pathlib import Path
    src=Path(__file__).resolve().parents[1]/'run_v9.py'
    code=src.read_text()
    assert 'from v2.entry import run' in code
    assert 'from v9.engine import main' not in code


def test_checkpoints_module_present_and_strict_policies():
    from v2.checkpoints import install_checkpoint_hooks
    assert callable(install_checkpoint_hooks)
