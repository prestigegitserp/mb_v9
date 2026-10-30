"""Executable coefficient grids, tied metrics and gradient-neutral telemetry."""
from dataclasses import replace
import copy
import json
import pytest
import torch
from test_v3_research import tiny_data
from test_v7_rebuild import small
from v7.config import default_models,StudyConfig
from v7.models import instantiate
from v7.losses import coefficients,trial_values,coefficient_overrides
from v7.lineage import legacy_config


@pytest.mark.parametrize('method',['Method'+str(i) for i in range(1,8)])
def test_every_declared_coefficient_trial_is_executable(method):
    for lineage in (default_models()[method].lineage,'v7'):
        cfg=replace(default_models()[method],lineage=lineage)
        for key in coefficients(cfg):
            for value in trial_values(cfg,key):
                variant=replace(cfg,**coefficient_overrides(cfg,{key:value})).validate()
                if lineage!='v7':legacy_config(variant)


def test_tied_moment_metric_and_its_zero_weight_limit():
    cfg=replace(default_models()['Method4'],lineage='v7',moment_projection_mode='metric',
                moment_projection_weight=.1,moment_dispersion_weight=.1,orthogonal_moment_loss=True)
    changed=coefficient_overrides(cfg,{'moment_dispersion_weight':.4})
    assert changed['moment_projection_weight']==.4
    assert coefficient_overrides(cfg,{'moment_dispersion_weight':0.})['moment_projection_mode']=='mean-priority'
    independent=replace(cfg,moment_projection_weight=1.)
    assert 'moment_projection_weight' not in coefficient_overrides(independent,{'moment_dispersion_weight':.4})


def test_recording_prior_components_preserves_loss_and_gradients_exactly():
    from v6.models import BaseModel as OriginalBase
    data=tiny_data();model=instantiate(data,small('Method1',True),7)
    history=model.history();state=model.prepare(history)
    expected=OriginalBase.common_prior(model,state)
    actual=model.common_prior(state)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    parameters=list(model.parameters())
    a=torch.autograd.grad(expected,parameters,allow_unused=True,retain_graph=True)
    b=torch.autograd.grad(actual,parameters,allow_unused=True,retain_graph=True)
    for x,y in zip(a,b):
        if x is None:assert y is None
        else:torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert sum(model._last_prior_terms.values())==pytest.approx(float(actual.detach()),abs=1e-7)
    assert all(isinstance(value,float) for value in model._last_prior_terms.values())


def test_inactive_probability_and_moment_loss_knobs_are_excluded():
    m7=replace(default_models()['Method7'],probability_fit=False,readout_mode='geometric')
    assert not {'probability_rps_weight','code_prior','direction_prior'} & set(coefficients(m7))
    m4=replace(default_models()['Method4'],orthogonal_moment_loss=True)
    assert 'auxiliary_weight' not in coefficients(m4)
    assert 'radius_precision' in coefficients(m4)
