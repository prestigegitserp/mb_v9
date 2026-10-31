import json
from pathlib import Path
import pandas as pd
import pytest
import torch

# Frozen V4 CI collects every generation using its own smaller dependency set.
# The V8 matrix installs requirements_v8.txt and runs both integrations fully.
pytest.importorskip('xlsxwriter',reason='V8 publication integrations require requirements_v8.txt')

from v8 import engine,runtime
from v8.entry import run


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


def arguments(tmp_path,methods='Method1,Method2,Method3,Method4,Method5,Method6,Method7,Method8,MBMF-C,MBMF-F,MBMF-N'):
    file=tmp_path/'settings.json'
    file.write_text(json.dumps(dict(split_validation_roles=False,models={
        f'Method{i}':dict(hidden_dim=12,probability_fit=False) for i in range(1,8)},
        method8=dict(hidden_dim=12,decoder_dim=8))))
    return ['--datasets','synthetic','--methods',methods,'--quick','--k','4',
        '--split-seeds','4101','--model-seeds','11','--v8-config',str(file),
        '--v8-threads','1','--v8-search-jobs-per-method','1','--no-v8-isolate-jobs',
        '--v8-no-plots','--v8-storage','resident','--output',str(tmp_path/'run')]


def test_scan_then_final_all_eight_and_all_three_mbmf(tiny_engine,tmp_path,monkeypatch):
    args=arguments(tmp_path)
    run(['scan','--scan-only',*args])
    root=tmp_path/'run/synthetic'
    candidates=pd.read_csv(root/'search/hyperparameter_validation_all.csv')
    assert not any(c.startswith('test_') for c in candidates)
    assert not list((root/'search').rglob('*__test.npz'))
    run(['final',*args])
    final=pd.read_csv(root/'final_locked_test_all.csv')
    expected={f'Method{i}' for i in range(1,9)}|{'MBMF-C','MBMF-F','MBMF-N'}
    assert set(final.method)==expected
    assert len(final)==len(expected)
    assert final.test_RMSE.notna().all()
    assert (root/'publication/research_tables.xlsx').is_file()
    for key in final[final.method.str.startswith('MBMF-')].config_key:
        assert len(json.loads(key)['config']['init_modes'])==1
    def never(*a,**kw):raise AssertionError('Completed final retrained')
    monkeypatch.setattr(runtime,'execute_candidate',never)
    run(['final','--resume',*args])


def test_real_three_initialization_scan_refits_one_frozen_mbmf(tiny_engine,tmp_path):
    args=arguments(tmp_path,'MBMF-C,MBMF-F,MBMF-N')
    args.remove('--quick')
    args+=['--v8-max-epochs','3','--v8-baseline-max-epochs','3','--v8-steps-per-epoch','1']
    run(['scan','--scan-only',*args])
    root=tmp_path/'run/synthetic'
    scan=pd.read_csv(root/'search/hyperparameter_validation_all.csv')
    assert len(scan)==9
    assert scan.groupby('method').init_mode.nunique().eq(3).all()
    run(['test',*args])
    final=pd.read_csv(root/'final_locked_test_all.csv')
    assert len(final)==3 and final.test_RMSE.notna().all()
    from v8.audit import audit_selection
    assert audit_selection(root)['status']=='PASS'
