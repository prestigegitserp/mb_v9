"""Regression checks for V4.1 baseline fairness and the density objective."""
from dataclasses import replace
from types import SimpleNamespace
import copy

import pytest
import torch

from test_v3_research import tiny_data, training_pub
from test_v4_research import config
from v4.baselines import paper_config, train_autorec, train_mbmf, restore_mbmf
from v4.config import StudyConfig
from v4.history import MaskSampler
from v4.models import instantiate
from v4.protocol import validation_roles
from v4.training import train_v4
from v9.engine import load_v8


def public_data():
    import pandas as pd
    _, pub = load_v8()
    d = tiny_data()
    frames = [pd.DataFrame(dict(userId=getattr(d, s+'_users').numpy(), movieId=getattr(d, s+'_items').numpy(),
                                rating=getattr(d, s+'_ratings_raw').numpy(), split=s)) for s in ('train','val','test')]
    data, _ = pub.build_publication_matrix_data_v8(pd.concat(frames), pub.DATASET_SPECS['ml-100k'],
               pub.PublicationDataConfig(dataset='ml-100k', force_backend='dense'), torch.device('cpu'))
    data.test_ratings_raw[:] = torch.nan
    return data


def test_rprop_rate_changes_real_update_and_preserves_legacy_effective_default():
    _, pub = load_v8()
    data = public_data()
    deltas = []
    for lr in (.001, .01):
        torch.manual_seed(821)
        cfg = pub.AutoRecConfig(hidden_dim=8, max_epochs=1, lr=lr)
        model = pub.ItemAutoRec(data, cfg, torch.device('cpu'))
        old = model.decoder.bias.detach().clone()
        info = train_autorec(training_pub(), model, data, cfg, chunk_size=13)
        deltas.append((model.decoder.bias.detach()-old).abs())
        assert info['effective_initial_lr'] == lr
        assert info['history'][-1]['epoch'] == 1
    torch.testing.assert_close(deltas[1], 10*deltas[0], atol=2e-7, rtol=1e-5)


def test_mbmf_keeps_paper_update_but_restores_best_validation_checkpoint(tmp_path):
    core, pub = load_v8()
    torch.set_num_threads(2)
    data = public_data()
    cfg = core.PaperBaselineConfig(latent_dim=4, max_epochs=3, eval_every=1, initial_lr=1e-5)
    torch.manual_seed(92)
    model = core.PaperMBMF(data, cfg, 'MBMF-F', 'isotropic_gaussian', torch.device('cpu'))
    states = []
    scores = iter((1., 2., 3.))

    def evaluate(current, data, split, **kwargs):
        assert split == 'val'
        states.append(copy.deepcopy(current.state_dict()))
        return {'RMSE': next(scores)}

    evaluator = SimpleNamespace(MetricConfig=lambda **kwargs: None, evaluate_publication_model=evaluate)
    extended = paper_config(cfg, patience=2)
    info = train_mbmf(evaluator, model, data, extended, chunk_size=17)
    assert info['best_epoch'] == 1 and info['stopping_reason'] == 'validation_patience'
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, states[0][key], atol=0, rtol=0)
    assert all(a['objective'] >= b['objective'] for a, b in zip(info['history'], info['history'][1:]))
    # The first accepted gradient step is the unmodified dense paper algorithm.
    torch.manual_seed(92)
    native_cfg = replace(cfg, max_epochs=1)
    native = core.PaperMBMF(data, native_cfg, 'MBMF-F', 'isotropic_gaussian', torch.device('cpu'))
    core.train_paper_mbmf(native, data, native_cfg)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, native.state_dict()[key], atol=3e-7, rtol=1e-6)
    path = tmp_path/'baseline.pt'
    pub.save_model_checkpoint_v8(path, model, config=extended, data=data, train_info=info,
                                 extra={'init_mode':'isotropic_gaussian'})
    payload = torch.load(path, weights_only=False)
    restored, _ = restore_mbmf(pub, core, payload, data, torch.device('cpu'))
    with torch.no_grad():
        torch.testing.assert_close(restored.predict_raw_pairs(data.val_users, data.val_items),
                                   model.predict_raw_pairs(data.val_users, data.val_items), atol=0, rtol=0)
    data.train_ratings_raw[0] += .25
    with pytest.raises(ValueError, match='train_sha256'):
        restore_mbmf(pub, core, payload, data, torch.device('cpu'))


def test_density_sampler_reproducible_and_preserves_duplicate_exclusion():
    d = tiny_data()
    d.train_users = torch.cat([d.train_users, d.train_users[:4]])
    d.train_items = torch.cat([d.train_items, d.train_items[:4]])
    a = MaskSampler(d, .2, torch.Generator().manual_seed(9), .15)
    b = MaskSampler(d, .2, torch.Generator().manual_seed(9), .15)
    fractions = []
    for _ in range(500):
        keep, query = a.sample()
        keep2, query2 = b.sample()
        assert torch.equal(keep, keep2) and torch.equal(query, query2)
        assert keep.any() and query.any() and not (keep & query).any()
        assert torch.equal(query[:4], query[-4:])
        assert .05 <= a.last_fraction <= .35
        fractions.append(a.last_fraction)
    assert abs(sum(fractions)/len(fractions)-.2) < .01


@pytest.mark.parametrize('method', ['Method'+str(i) for i in range(1,8)])
def test_density_training_all_seven_ignores_poisoned_test(method):
    data = validation_roles(tiny_data(), StudyConfig())
    data.test_ratings_raw[:] = torch.nan
    cfg = replace(config(method), mask_fraction_jitter=.15)
    model = instantiate(data, cfg, 14)
    info = train_v4(training_pub(), model, data, cfg, None)
    assert info['best_epoch'] >= 1 and info['best_val_RMSE'] < float('inf')
    assert data.test_ratings_raw.isnan().all()


@pytest.mark.parametrize('value',[-.01,.2,.5])
def test_density_invalid_probability_bounds_fail(value):
    with pytest.raises(ValueError, match='jitter'):
        replace(config('Method1'), mask_fraction_jitter=value).validate()


def test_all_baselines_scan_final_checkpoint_roundtrip(tmp_path, monkeypatch):
    from v4.entry import run
    _, pub = load_v8()
    monkeypatch.setattr(pub, 'method_supported_on_dataset_v8', lambda method, dataset: True)
    methods = 'MF,BiasMF,MBMF-F,MBMF-C,MBMF-N,AutoRec'
    run(['main','--datasets','synthetic','--quick','--methods',methods,'--k','4',
         '--split-seeds','2026','--model-seeds','1','--v4-threads','2','--v4-no-plots',
         '--output',str(tmp_path/'baselines')])
    import pandas as pd
    table = pd.read_csv(tmp_path/'baselines'/'synthetic'/'main'/'locked_test_runs.csv')
    assert set(table.method) == set(methods.split(','))
    assert table.test_RMSE.notna().all()


def test_mf_budget_is_explicit_and_quick_stays_small():
    from v4 import engine
    from v4.runtime import installed
    from v4.execution import ExecutionConfig
    options=SimpleNamespace(study=StudyConfig(mf_max_epochs=1000),execution=ExecutionConfig(),retain_candidates='all',
                            max_epochs=None,steps_per_epoch=None,no_plots=True)
    args=SimpleNamespace(data_root='datasets',backend='auto',split_seeds='2026',model_seeds='1',k='20',
                         methods='MF',eval_every=None,quick=False)
    with installed(engine,options) as (_,pub):
        assert engine.make_base(pub,args,'ml-100k').biasmf.max_epochs==1000
        assert engine.make_base(pub,args,'ml-100k').core_config.paper.max_epochs==2000
        args.quick=True
        assert engine.make_base(pub,args,'ml-100k').biasmf.max_epochs==3
        assert engine.make_base(pub,args,'ml-100k').core_config.paper.max_epochs==3
