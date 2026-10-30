"""The optional support prior is active, target-excluded and separately ablated."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from test_v3_research import tiny_data
from test_v7_rebuild import small
from v7.models import instantiate


def test_method1_shared_gradient_budget_includes_learned_history_gates():
    from v7.training import _shared_encoder_parameters
    model=instantiate(tiny_data(),small('Method1',True,history_signal='gated'),17)
    shared={id(value) for value in _shared_encoder_parameters(model)}
    assert id(model.history_mix_user) in shared and id(model.history_mix_item) in shared
    assert id(model.bias.user.weight) not in shared


def test_recipe_can_skip_extra_train_evaluations_without_disabling_reports(tmp_path):
    from paper_v7 import parser,build_commands
    args=parser().parse_args(['--stage','develop','--profile','common','--output',str(tmp_path)])
    command=build_commands(args)[0]
    assert '--no-epoch-train-rmse' in command and '--v7-no-plots' not in command
    args.epoch_train_rmse=True
    assert '--epoch-train-rmse' in build_commands(args)[0]


def test_missing_raw_data_does_not_poison_a_resumable_search_plan(tmp_path):
    from v7.entry import run
    with pytest.raises(FileNotFoundError,match='prepare data'):
        run(['scan','--scan-only','--datasets','ml-100k','--methods','Method8','--quick',
             '--data-root',str(tmp_path/'missing'),'--output',str(tmp_path/'results')])
    assert not list((tmp_path/'results').rglob('V9_SEARCH_PLAN.json'))
    assert not list((tmp_path/'results').rglob('V9_JOB_STATUS.json'))


@pytest.mark.parametrize('method',['Method'+str(i) for i in range(1,8)])
def test_support_gate_preserves_initial_model_then_changes_learned_mixture(method):
    data=tiny_data();cfg=small(method,True,history_signal='gated')
    plain=instantiate(data,cfg,41);shrunk=instantiate(data,replace(cfg,history_gate_support_shrinkage=True),41)
    keep=torch.arange(len(data.train_users))%3!=0
    def encode(model):return model.encode(model.history(keep))
    for x,y in zip(encode(plain),encode(shrunk)):torch.testing.assert_close(x,y,atol=0,rtol=0)
    with torch.no_grad():
        for model in (plain,shrunk):model.history_mix_user[0]=4.;model.history_mix_item[0]=-4.
    old=encode(plain);new=encode(shrunk)
    assert any(not torch.allclose(x,y) for x,y in zip(old,new))
    (new[0].square().sum()+new[1].square().sum()).backward()
    assert shrunk.history_mix_user.grad.abs().sum()>0 and shrunk.history_mix_item.grad.abs().sum()>0
    data.train_ratings_raw[~keep]=999.;data.train_timestamps[~keep]=-1e12
    for x,y in zip(new,encode(shrunk)):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_support_gate_is_in_bounded_search_and_has_its_own_ablation(tmp_path):
    from v7 import engine,runtime,search
    from v7.config import StudyConfig
    from v7.execution import ExecutionConfig
    options=SimpleNamespace(study=StudyConfig(search_jobs_per_method=6),execution=ExecutionConfig(),
        max_epochs=None,baseline_max_epochs=None,steps_per_epoch=None,no_plots=True,retain_candidates='metrics',
        v7_recovery_interval=5,v7_isolate_jobs=False)
    args=SimpleNamespace(data_root='.',backend='auto',split_seeds='1',model_seeds='1',k='4',methods='Method1',eval_every=None,quick=False,resume=True)
    with runtime.installed(engine,options) as (_,pub):
        base=engine.make_base(pub,args,'ml-100k');jobs=search.build_jobs(pub,base,(4,))
        cfg=jobs['support_gated_context__Method1']
        assert cfg.v7.models['Method1'].lineage=='v7' and cfg.v7.models['Method1'].history_gate_support_shrinkage
        variants=dict(search.ablations(cfg))
        assert not variants['Method1__no_gate_support_shrinkage'].v7.models['Method1'].history_gate_support_shrinkage
