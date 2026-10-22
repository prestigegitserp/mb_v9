"""V6 invariants: historical architectures, final-head selection and honest reports."""
import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_v3_research import tiny_data, training_pub, config as tiny_config
from v6.config import ModelConfig, StudyConfig, default_models
from v6.models import instantiate, LegacyAdapter
from v6.protocol import validation_roles
from v6.training import train_v6, train_mean, fit_router, optimizer_groups
from v6.response import fit_response
from v6.checkpoints import save, load

ROOT=Path(__file__).resolve().parents[1]


def small(method, **overrides):
    values=vars(tiny_config(method)).copy()
    values.update(hidden_dim=12,probability_fit=False)
    values.update(overrides)
    return replace(default_models()[method], **values)


@pytest.mark.parametrize("method",["Method"+str(i) for i in range(1,8)])
def test_all_lineages_ignore_test_and_checkpoint_exactly(method,tmp_path):
    torch.set_num_threads(1)
    data=validation_roles(tiny_data(),StudyConfig(protocol="aemc-paper",split_validation_roles=False))
    data.test_ratings_raw[:]=torch.nan
    model=instantiate(data,small(method),23)
    info=train_v6(training_pub(),model,data,model.cfg,None)
    p=model.predict_raw_pairs(data,data.val_users,data.val_items).detach()
    assert torch.isfinite(p).all()
    correction=info["v6_response"]
    assert correction["corrected_val_RMSE"]<=correction["raw_val_RMSE"]+1e-10
    path=tmp_path/"model.pt";save(path,model,config=model.cfg,data=data,train_info=info)
    restored,_=load(path,data,"cpu")
    torch.testing.assert_close(p,restored.predict_raw_pairs(data,data.val_users,data.val_items),atol=0,rtol=0)
    if method in ("Method3","Method4","Method5"):
        assert isinstance(model,LegacyAdapter)
        assert type(model.inner).__module__.startswith("v3." if method=="Method5" else "v4.")


@pytest.mark.parametrize("method",["Method"+str(i) for i in range(1,8)])
def test_masked_query_is_not_an_encoder_input(method):
    data=tiny_data();model=instantiate(data,small(method),13).eval()
    keep=torch.arange(len(data.train_users))%3!=0
    def prediction():
        with torch.no_grad():
            h=model.history(keep);s=model.prepare(h)
            return model.predict(h,s,data.train_users[~keep],data.train_items[~keep])[0]
    before=prediction()
    data.train_ratings_raw[~keep]=999.;data.train_timestamps[~keep]=-1e12
    torch.testing.assert_close(before,prediction(),atol=0,rtol=0)


def test_router_strength_is_selected_using_its_actual_final_readout():
    torch.set_num_threads(1);data=tiny_data()
    model=instantiate(data,small("Method2",lineage="v6",response_shrinkage=False),23)
    pub=training_pub();train_mean(pub,model,data)
    result=fit_router(pub,model,data)
    with torch.no_grad():
        model.eval();h,s=model.eval_state()
        actual=float((model.predict(h,s,data.val_users,data.val_items)[0]-data.val_ratings_raw).double().square().mean())
    assert result["selection_head"]=="final_readout"
    assert actual==pytest.approx(result["selected_final_val_MSE"],abs=1e-10)


def test_m3_prediction_loss_has_declared_consistent_units():
    data=tiny_data()
    cfg=small("Method3",lineage="v6",response_shrinkage=False,prediction_loss_units="mse")
    model=instantiate(data,cfg,3).eval()
    h=model.history();s=model.prepare(h)
    u,i,y=data.val_users,data.val_items,data.val_ratings_raw
    mse=model.loss_for_targets(h,s,u,i,y)[0]
    expected=((model.predict(h,s,u,i)[0]-y)/model.width).square().mean()
    torch.testing.assert_close(mse,expected)
    model.cfg.prediction_loss_units="gaussian"
    gaussian=model.loss_for_targets(h,s,u,i,y)[0]
    torch.testing.assert_close(gaussian,.5*mse/model.cfg.radius_noise**2)


def test_regularization_does_not_dilute_when_epoch_budget_is_raised():
    data=tiny_data()
    cfg=small("Method1",lineage="v6",encoder_decay_horizon_epochs=20)
    model=instantiate(data,cfg)
    _,short=optimizer_groups(model,100)
    _,long=optimizer_groups(model,1000)
    assert short==long


@pytest.mark.parametrize("namespace",["v6","v52"])
def test_probability_fit_preserves_its_distribution_mean(namespace):
    torch.set_num_threads(1)
    data=validation_roles(tiny_data(),StudyConfig())
    model=instantiate(data,small("Method7",lineage=namespace,probability_fit=True),9)
    info=train_v6(training_pub(),model,data,model.cfg,None)
    model.eval()
    with torch.no_grad():
        p=model.predict_raw_pairs(data,data.val_users,data.val_items)
        dist=model.distribution(data,data.val_users,data.val_items)
    torch.testing.assert_close(dist["probabilities"]@model.levels,p,atol=2e-5,rtol=1e-6)
    assert info["v6_response"]["status"]=="DISABLED_FIXED_PROBABILITY_MEAN"


def test_templates_cover_every_dataset_and_are_valid_for_their_entry():
    from v52.config import StudyConfig as OldStudy
    from v6 import entry
    datasets=("synthetic","ml-100k","ml-1m","ml-10m","jester","bookcrossing")
    for profile in ("development","paper","v5_compatible"):
        for dataset in datasets:
            value=json.loads((ROOT/"examples/v6"/profile/(dataset+".json")).read_text())
            study=OldStudy() if profile=="v5_compatible" else StudyConfig()
            for key,item in value.items():
                if key=="models":
                    for method,settings in item.items():
                        for name,v in settings.items():setattr(study.models[method],name,v)
                else:setattr(study,key,item)
            study.validate()
            assert study.protocol==("aemc-paper" if dataset.startswith("ml-") else "common")


def test_ablation_never_changes_only_a_setting_ignored_by_legacy_architecture():
    from v6.search import ablations
    from v6.lineage import legacy_config
    base=SimpleNamespace(methods=("Method5",),v6=StudyConfig())
    for tag,cfg in ablations(base):
        old,new=base.v6.models["Method5"],cfg.v6.models["Method5"]
        assert tag.endswith("__full") or old.response_shrinkage!=new.response_shrinkage or vars(legacy_config(old))!=vars(legacy_config(new))


def test_report_only_changes_do_not_invalidate_a_scientific_checkpoint(tmp_path,monkeypatch):
    from v6 import config
    root=tmp_path/"v6";root.mkdir()
    science=root/"config.py";science.write_text("first")
    report=root/"export.py";report.write_text("old report")
    ex=tmp_path/"examples/v6";ex.mkdir(parents=True)
    (ex/"historical_incumbents.json").write_text("{}")
    monkeypatch.setattr(config,"__file__",str(science))
    first=config.implementation_signature()
    report.write_text("repaired report")
    assert first==config.implementation_signature()
    science.write_text("changed scientific code")
    assert first!=config.implementation_signature()
