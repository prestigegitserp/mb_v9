"""V8 coupling, exact calibration law, source isolation and real final refits."""
from dataclasses import replace
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import pytest
import torch
from test_v3_research import tiny_data,training_pub,config as tiny_config
from v8.config import StudyConfig,default_models
from v8.models import instantiate
from v8.protocol import validation_roles
from v8.training import train_v8


def small(method,**changes):
    values=vars(tiny_config(method)).copy()
    values.update(lineage='v8',hidden_dim=12,probability_fit=False)
    values.update(changes)
    return replace(default_models()[method],**values)


@pytest.mark.parametrize('method',[f'Method{i}' for i in range(1,8)])
def test_training_ignores_test_and_calibration_preserves_bounds(method,tmp_path):
    torch.set_num_threads(1)
    data=validation_roles(tiny_data(),StudyConfig())
    data.test_ratings_raw.fill_(float('nan'))
    cfg=small(method)
    model=instantiate(data,cfg,23)
    result=train_v8(training_pub(),model,data,cfg,None)
    r=result['v8_response']
    assert r['corrected_val_RMSE']<=r['raw_val_RMSE']+1e-7
    assert not r['test_used']
    p=model.predict_raw_pairs(data,data.val_users,data.val_items).detach()
    assert torch.isfinite(p).all() and (p>=1).all() and (p<=5).all()
    from v8.checkpoints import save,load
    file=tmp_path/'model.pt';save(file,model,config=cfg,data=data)
    restored,_=load(file,data,'cpu')
    torch.testing.assert_close(p,restored.predict_raw_pairs(data,data.val_users,data.val_items),atol=0,rtol=0)


def test_reliability_features_ignore_excluded_targets_and_gate_is_trainable():
    cfg=small('Method6')
    data=tiny_data();model=instantiate(data,cfg,9)
    keep=torch.arange(len(data.train_users))%3!=0
    u,i=data.train_users[~keep],data.train_items[~keep]
    def forward():
        model.clear_cache();h=model.history(keep);s=model.prepare(h)
        return model.predict(h,s,u,i)[0]
    before=forward().detach().clone()
    forward().square().sum().backward()
    assert model.readout_reliability.grad.abs().sum()>0
    data.train_ratings_raw[~keep]=999.
    torch.testing.assert_close(before,forward(),atol=0,rtol=0)


def test_method7_probability_mean_matches_jointly_calibrated_point():
    torch.set_num_threads(1)
    data=validation_roles(tiny_data(),StudyConfig())
    data.test_ratings_raw.fill_(float('nan'))
    cfg=small('Method7',probability_fit=True)
    model=instantiate(data,cfg,31)
    result=train_v8(training_pub(),model,data,cfg,None)
    assert result['v8_probability']['mean_parameters_unchanged']
    with torch.no_grad():
        for q,a in ((.25,.75),(0.,0.),(1.,1.)):
            model.geometry_scale.fill_(q);model.response_scale.fill_(a);model.clear_cache()
            dist=model.distribution(data,data.val_users,data.val_items)
            point=model.predict_raw_pairs(data,data.val_users,data.val_items)
            mean=(dist['probabilities']*model.levels).sum(-1)
            torch.testing.assert_close(mean,point,atol=2e-5,rtol=2e-6)
            torch.testing.assert_close(dist['probabilities'].sum(-1),torch.ones_like(point),atol=2e-6,rtol=0)


def test_method7_active_direction_penalty_is_in_registry():
    from v8.losses import coefficients
    cfg=small('Method7')
    assert 'direction_prior' in coefficients(cfg)
    assert 'mechanism_auxiliary_weight' not in coefficients(cfg)
    assert 'readout_gate_prior' in coefficients(cfg)


def test_fragmented_groupby_export_uses_a_consolidated_frame():
    frame=pd.DataFrame({'dataset':['x']*4,'method':['Method1']*4,'split_seed':[1]*4,
        **{f'test_m{i}':np.arange(4.) for i in range(180)}})
    metrics=[c for c in frame if c.startswith('test_')]
    variation=frame.groupby(['dataset','method','split_seed'])[metrics].agg(['mean','std','count'])
    variation.columns=[m+'__'+s for m,s in variation.columns]
    with warnings.catch_warnings():
        warnings.simplefilter('error',pd.errors.PerformanceWarning)
        exported=variation.copy().reset_index()
    assert exported.shape==(1,543)


def test_canonical_mbmf_key_preserves_optimizer_and_rejects_invalid_init():
    from types import SimpleNamespace
    from v8.configuration import canonical_candidate_key
    original=lambda method,cfg,mode:json.dumps({'method':method,'init_mode':mode,'config':vars(cfg)})
    all_modes=SimpleNamespace(init_modes=('a','b','c'),optimizer='adam',adam_lr=.01)
    one_mode=SimpleNamespace(init_modes=('b',),optimizer='adam',adam_lr=.01)
    a=canonical_candidate_key(original,'MBMF-C',all_modes,'b')
    b=canonical_candidate_key(original,'MBMF-C',one_mode,'b')
    assert a==b
    one_mode.adam_lr=.003
    assert a!=canonical_candidate_key(original,'MBMF-C',one_mode,'b')
    with pytest.raises(ValueError,match='absent'):
        canonical_candidate_key(original,'MBMF-C',one_mode,'a')


def test_host_memory_without_procfs(monkeypatch):
    from v8 import execution
    def unavailable():raise FileNotFoundError('/proc/meminfo')
    monkeypatch.setattr(execution.psutil,'virtual_memory',unavailable)
    available,total=execution.host_memory()
    assert 0<available<=total
