"""Prepared V5.2 regressions. Execution status is recorded separately."""
from dataclasses import replace
from types import SimpleNamespace
import copy
import pytest
import torch
from test_v3_research import tiny_data,training_pub,config as old_tiny_config
from v52.config import ModelConfig,StudyConfig,default_models
from v52.models import instantiate
from v52.protocol import validation_roles
from v52.training import train_v52
from v52.checkpoints import save,load
from v52.aemc_baseline import AEMCConfig,AEMCNE,train_aemcne,activate


def enhanced(method):
    return ModelConfig(**{**vars(old_tiny_config(method)),"hidden_dim":12,
        "posterior_cg_steps":64,"posterior_solver_tolerance":1e-4,
        "history_activation":"sigmoid","history_normalization":"none","history_pooling_power":0.,
        "wide_context_readout":True,"readout_mode":"residual","full_history_weight":.25,"probability_fit":False})


def test_method5_selected_zero_penalty_is_not_overwritten():
    assert default_models()["Method5"].collaborative_l2==0.


@pytest.mark.parametrize("method",["Method"+str(i) for i in range(1,8)])
def test_hybrid_wide_model_never_reads_test_and_restores_exactly(method,tmp_path):
    torch.set_num_threads(1)
    data=validation_roles(tiny_data(),StudyConfig(protocol="aemc-paper",split_validation_roles=False))
    data.test_ratings_raw[:]=torch.nan
    model=instantiate(data,enhanced(method),23)
    assert model.wide_user_decoder.embedding_dim==model.cfg.hidden_dim
    assert model.wide_user_decoder.embedding_dim>model.k
    info=train_v52(training_pub(),model,data,model.cfg,None)
    assert info["full_observed_targets_seen"]>0
    model.eval()
    with torch.no_grad():before=model.predict_raw_pairs(data,data.val_users,data.val_items)
    assert torch.isfinite(before).all() and bool((before>=data.rating_min-1e-5).all()) and bool((before<=data.rating_max+1e-5).all())
    if method=="Method7":
        assert not bool(model.shape_fitted)
        assert info["v52_probability"]["status"]=="SKIPPED_POINT_PREDICTION_PROTOCOL"
        assert data.v52_protocol["calibration_fit"]==data.v52_protocol["calibration_hold"]==0
    path=tmp_path/"model.pt";save(path,model,config=model.cfg,data=data,train_info=info)
    restored,_=load(path,data,torch.device("cpu"))
    with torch.no_grad():after=restored.predict_raw_pairs(data,data.val_users,data.val_items)
    torch.testing.assert_close(before,after,atol=0,rtol=0)


@pytest.mark.parametrize("method",["Method"+str(i) for i in range(1,8)])
def test_wide_context_never_contains_query_values_or_query_timestamps(method):
    torch.set_num_threads(1);data=tiny_data();model=instantiate(data,enhanced(method),13).eval()
    keep=torch.arange(len(data.train_users))%3!=0
    def predict():
        with torch.no_grad():
            h=model.history(keep);state=model.prepare(h)
            return model.predict(h,state,data.train_users[~keep],data.train_items[~keep])[0]
    before=predict();data.train_ratings_raw[~keep]=999.;data.train_timestamps[~keep]=-1e12
    torch.testing.assert_close(before,predict(),atol=0,rtol=0)


def test_aemc_identity_transfer_is_exact_before_training():
    model=AEMCNE(tiny_data(),AEMCConfig(main_activation="sigmoid",initialization="identity-transfer"),torch.device("cpu")).double()
    z=torch.linspace(-10.,10.,101,dtype=torch.double)
    torch.testing.assert_close(model._h(z),z,atol=0,rtol=0)


@pytest.mark.parametrize("activation",["relu","sigmoid","tanh"])
def test_aemc_sparse_equation_and_uniform_item_gradient_match_dense_definition(activation):
    torch.set_num_threads(1);torch.manual_seed(29);data=tiny_data();data.train_ratings_raw=data.train_ratings_raw.double()
    cfg=AEMCConfig(main_hidden_dim=7,element_hidden_dim=5,main_activation=activation,lambda_main=.2,lambda_element=.7)
    model=AEMCNE(data,cfg,torch.device("cpu")).double().train()
    dense=torch.zeros(data.num_items,data.num_users,dtype=torch.double)
    dense.index_put_((data.train_items,data.train_users),data.train_ratings_raw,accumulate=True)
    h=activate(model.encoder(dense),activation)
    z=(h[data.train_items]*model.decoder.weight[data.train_users]).sum(-1)+model.decoder.bias[data.train_users]
    objective=((model._h(z)-data.train_ratings_raw).square().sum()+model.penalty())/len(data.train_users)
    expected=torch.autograd.grad(objective,tuple(model.parameters()))
    observed=((model.predict_raw_pairs(data,data.train_users,data.train_items)-data.train_ratings_raw).square().sum()+model.penalty())/len(data.train_users)
    sparse_gradient=torch.autograd.grad(observed,tuple(model.parameters()))
    for a,b in zip(expected,sparse_gradient):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9)
    items=torch.unique(data.train_items);average=[torch.zeros_like(p) for p in model.parameters()]
    for chosen in items.split(5):
        lookup=torch.full((data.num_items,),-1,dtype=torch.long);lookup[chosen]=torch.arange(len(chosen))
        mask=lookup[data.train_items]>=0;u=data.train_users[mask];local=lookup[data.train_items[mask]]
        hidden=model.item_hidden(data,chosen);z=(hidden[local]*model.decoder.weight[u]).sum(-1)+model.decoder.bias[u]
        estimate=(len(items)/len(chosen)*(model._h(z)-data.train_ratings_raw[mask]).square().sum()+model.penalty())/len(data.train_users)
        gradients=torch.autograd.grad(estimate,tuple(model.parameters()))
        for target,value in zip(average,gradients):target.add_(value,alpha=len(chosen)/len(items))
    for a,b in zip(expected,average):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9)


@pytest.mark.parametrize("batch",[0,4])
def test_aemc_control_training_is_validation_only(batch):
    torch.set_num_threads(1);data=tiny_data();data.test_ratings_raw[:]=torch.nan
    cfg=AEMCConfig(main_hidden_dim=8,element_hidden_dim=6,main_activation="sigmoid",initialization="identity-transfer",
        item_batch_size=batch,max_epochs=3,eval_every=2,val_patience=3,lambda_main=.1,lambda_element=.2)
    model=AEMCNE(data,cfg,torch.device("cpu"));info=train_aemcne(training_pub(),model,data,cfg,13)
    assert torch.isfinite(torch.tensor(info["best_val_RMSE"])) and info["optimizer_steps"]>=3


def test_paper_protocol_allows_method7_without_extra_validation_and_restores_native_class():
    from v52 import engine
    from v52.runtime import installed
    from v52.execution import ExecutionConfig
    args=SimpleNamespace(data_root="./datasets",backend="auto",split_seeds="4101",model_seeds="11",k="4",
        methods="AEMC-NE,Method7",eval_every=None,quick=True)
    options=SimpleNamespace(study=StudyConfig(protocol="aemc-paper",split_validation_roles=False),
        execution=ExecutionConfig(),max_epochs=None,baseline_max_epochs=None,steps_per_epoch=None,no_plots=True,retain_candidates="metrics")
    _,original_pub=engine.load_v8();before=original_pub.AEMCNE
    with installed(engine,options) as (_,pub):
        base=engine.make_base(pub,args,"ml-100k")
        assert (base.data.train_fraction,base.data.val_fraction,base.data.test_fraction)==(.855,.045,.1)
        assert not base.v52.models["Method7"].probability_fit and pub.AEMCNE is AEMCNE
        model,cfg=pub.instantiate_model_v8("AEMC-NE",tiny_data(),base,4,11,torch.device("cpu"))
        assert isinstance(model,AEMCNE) and isinstance(cfg,AEMCConfig) and cfg.main_hidden_dim==500
    assert original_pub.AEMCNE is before


@pytest.mark.parametrize("method",["Method1","Method4","Method7"])
def test_decoder_only_readout_is_independent_of_a_nonfinite_geometric_branch(method):
    from v52.readout import context_mean
    data=tiny_data();cfg=replace(enhanced(method),readout_mode="decoder",mechanism_auxiliary_weight=0.,code_prior=0.,direction_prior=0.)
    model=instantiate(data,cfg,19).eval()
    geometric=torch.full_like(data.val_ratings_raw,float("nan"))
    if method=="Method7":geometric=geometric.expand(cfg.ensemble_members,-1)
    with torch.no_grad():
        h=model.history();state=model.prepare(h)
        prediction,_=context_mean(model,h,state,data.val_users,data.val_items,geometric)
    assert prediction.shape==geometric.shape and bool(torch.isfinite(prediction).all())
    with torch.no_grad():
        point=model.predict_raw_pairs(data,data.val_users,data.val_items)
        lp,aux,_=model.loss_for_targets(h,state,data.val_users,data.val_items,data.val_ratings_raw)
    assert point.shape==data.val_ratings_raw.shape and lp.ndim==aux.ndim==0
    if method=="Method7":
        epsilon=cfg.mean_interior_epsilon*(data.rating_max-data.rating_min)
        expected=prediction.clamp(data.rating_min+epsilon,data.rating_max-epsilon).mean(0)
    else:expected=prediction
    torch.testing.assert_close(point,expected,atol=1e-6,rtol=0.)


def test_method7_has_no_fake_auxiliary_only_ablation():
    from v52.search import ablations
    base=SimpleNamespace(methods=("Method7",),v52=StudyConfig())
    tags=[tag for tag,cfg in ablations(base)]
    assert "Method7__no_mechanism_auxiliary" not in tags


def test_method1_decoder_without_auxiliary_priors_can_backpropagate():
    from v52.training import _backward
    data=tiny_data()
    cfg=replace(enhanced("Method1"),readout_mode="decoder",mechanism_auxiliary_weight=0.,
        code_prior=0.,bias_prior=0.,direction_prior=0.,collaborative_l2=0.)
    model=instantiate(data,cfg,31).train();h=model.history();state=model.prepare(h)
    lp,aux,_=model.loss_for_targets(h,state,data.val_users,data.val_items,data.val_ratings_raw)
    assert not aux.requires_grad
    loss,_,_=_backward(model,lp,aux)
    assert loss>=0 and model.wide_user_decoder.weight.grad is not None
