"""Partial-job reuse, frozen winners, reports and repeatable experiment recipes."""
import json
from pathlib import Path
import pytest
import torch
pytest.importorskip('xlsxwriter',reason='V7 publication integrations require requirements_v7.txt')
from v7 import engine,runtime
from v7.entry import run


@pytest.fixture
def tiny_engine(monkeypatch):
    original=engine.make_base
    def small(pub,args,dataset):
        cfg=original(pub,args,dataset)
        cfg.data.synthetic_users=18;cfg.data.synthetic_items=22
        cfg.data.synthetic_rank=3;cfg.data.synthetic_density=.7
        return cfg
    monkeypatch.setattr(engine,'make_base',small)
    torch.set_num_threads(1)


def arguments(tmp_path,methods='Method1,Method4,Method8,AutoRec,AEMC-NE'):
    cfg=tmp_path/'settings.json'
    cfg.write_text(json.dumps(dict(split_validation_roles=False,models={
        'Method'+str(i):dict(hidden_dim=12,probability_fit=False) for i in range(1,8)},
        method8=dict(hidden_dim=12,decoder_dim=8))))
    return ['--datasets','synthetic','--methods',methods,'--quick','--k','4','--split-seeds','4101',
        '--model-seeds','11','--v7-config',str(cfg),'--v7-threads','1',
        '--v7-search-jobs-per-method','1','--no-v7-isolate-jobs','--v7-no-plots',
        '--output',str(tmp_path/'run')]


def test_scan_and_final_include_immutable_eighth_model(tiny_engine,tmp_path,monkeypatch):
    import pandas as pd
    args=arguments(tmp_path)
    run(['scan','--scan-only',*args])
    root=tmp_path/'run/synthetic'
    assert not list((root/'search').rglob('*__test.npz'))
    table=pd.read_csv(root/'search/hyperparameter_validation_all.csv')
    assert not any(c.startswith('test_') for c in table)
    run(['final',*args])
    final=pd.read_csv(root/'final_locked_test_all.csv')
    assert set(final.method)=={'Method1','Method4','Method8'} # frozen dataset eligibility matrix
    assert (root/'publication/research_tables.xlsx').is_file()
    assert (root/'publication/tables/objective_coefficients.csv').is_file()
    run(['report','--datasets','synthetic','--output',str(tmp_path/'run'),'--v7-no-plots'])
    def never(*args,**kwargs):raise AssertionError('completed final must not train again')
    monkeypatch.setattr(runtime,'execute_candidate',never)
    run(['final','--resume',*args])


def test_resume_reuses_finished_candidates_after_next_candidate_fails(tiny_engine,tmp_path,monkeypatch):
    import pandas as pd
    args=arguments(tmp_path,'Method1,Method8')
    original=runtime.execute_candidate
    def interrupted(model,*args,**kwargs):
        if type(model).__name__=='Method10':raise RuntimeError('simulate next-candidate failure')
        return original(model,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(runtime,'execute_candidate',interrupted)
        with pytest.raises(RuntimeError,match='next-candidate'):run(['main',*args])
    folder=tmp_path/'run/synthetic/main'
    before=pd.read_csv(folder/'candidate_validation_runs.csv')
    assert set(before.method)=={'Method1'}
    def only_new(model,*args,**kwargs):
        if getattr(model,'v7_method',None)=='Method1':raise AssertionError('finished Method1 retrained')
        return original(model,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(runtime,'execute_candidate',only_new)
        run(['main','--resume',*args])
    after=pd.read_csv(folder/'candidate_validation_runs.csv')
    first=after[after.method=='Method1'].iloc[0]
    assert first.training_reused_from_resume
    assert first.train_wall_sec>0 and first.resume_attempt_wall_sec<first.train_wall_sec
    assert json.loads((folder/'V9_JOB_STATUS.json').read_text())['status']=='COMPLETED'
    assert not list(folder.rglob('candidate_complete.pt'))


def test_paper_recipe_scans_before_final_and_lossstudy_once(tmp_path):
    from paper_v7 import parser,build_commands
    args=parser().parse_args(['--stage','all','--profile','paper','--output',str(tmp_path)])
    commands=build_commands(args);stages=[c[2] for c in commands]
    assert stages.index('scan')<stages.index('final')<stages.index('ablation')
    assert stages.count('lossstudy')==1
    assert '--scan-only' in commands[0]
    assert all('--resume' in c for c in commands if c[2] not in ('report','audit'))


@pytest.mark.parametrize('profile',['common','development','paper'])
@pytest.mark.parametrize('dataset',['synthetic','ml-100k','ml-1m','ml-10m','jester','bookcrossing'])
def test_all_dataset_templates_are_declared_and_valid(profile,dataset,tmp_path):
    from v7.config import StudyConfig
    from v7.worker import hydrate
    from dataclasses import asdict
    root=Path(__file__).resolve().parents[1]
    values=asdict(StudyConfig());payload=json.loads((root/'examples/v7'/profile/(dataset+'.json')).read_text())
    for key,value in payload.items():
        if key=='models':
            for method,overrides in value.items():values['models'][method].update(overrides)
        else:values[key]=value
    study=hydrate(StudyConfig(),values).validate()
    assert study.search_jobs_per_method>0
