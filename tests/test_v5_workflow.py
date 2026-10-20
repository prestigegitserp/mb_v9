"""End-to-end scan, frozen final, workbook regeneration and legacy isolation."""
import json
from pathlib import Path
import pandas as pd
import pytest
pytest.importorskip('xlsxwriter',reason='V5 export workflow requires requirements_v5.txt; legacy CI runs its own dependency set')
pytest.importorskip('openpyxl',reason='V5 workbook roundtrip requires requirements_v5.txt')
from v5 import engine
from v5.entry import run


def small(monkeypatch):
    original=engine.make_base
    def build(pub,args,dataset):
        cfg=original(pub,args,dataset)
        cfg.data.synthetic_users=24;cfg.data.synthetic_items=28
        cfg.data.synthetic_rank=3;cfg.data.synthetic_density=.6
        return cfg
    monkeypatch.setattr(engine,'make_base',build)


def test_v5_full_scan_final_and_report_are_complete_and_frozen(tmp_path,monkeypatch):
    small(monkeypatch)
    args=['--datasets','synthetic','--quick','--k','4','--split-seeds','3031','--model-seeds','11',
          '--v5-threads','1','--v5-no-plots','--v5-search-jobs-per-method','1',
          '--v5-retain-candidates','metrics','--output',str(tmp_path/'run')]
    run(['scan','--scan-only',*args]);root=tmp_path/'run/synthetic'
    assert not list(root.rglob('*__test.npz'))
    assert not list((root/'search').rglob('*.pt'))
    workbook=root/'search/publication/research_tables.xlsx';assert workbook.is_file()
    index=pd.read_excel(workbook,sheet_name='INDEX')
    assert 'tables/search_all_candidates.csv' in index.source.values
    assert 'tables/hyperparameters_long.csv' in index.source.values
    winners=pd.read_csv(root/'search/validation_winners.csv')
    # Synthetic's documented support excludes AutoRec/AEMC-NE. Their ML path
    # and full independent penalty grid are covered separately below.
    assert winners.method.nunique()==12
    run(['final',*args]);final=pd.read_csv(root/'final_locked_test_all.csv')
    assert final.method.nunique()==12 and final.test_RMSE.notna().all()
    assert (root/'publication/research_tables.xlsx').is_file()
    assert len(list((root/'publication/tables/per_pair').glob('*__test.csv.gz')))>=12
    statuses={str(p):json.loads(p.read_text()) for p in root.rglob('V9_JOB_STATUS.json')}
    run(['report',*args])
    after={str(p):json.loads(p.read_text()) for p in root.rglob('V9_JOB_STATUS.json')}
    assert statuses==after  # report must never restart or reselect training
    # A changed grid cannot be substituted for the frozen validation winner.
    config=root/'search/jobs/default__Method4/config.json'
    payload=json.loads(config.read_text());payload['v5']['models']['Method4']['hidden_dim']+=1
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='Frozen scan mismatch'):run(['final',*args])


def test_v5_independent_penalty_grid_and_mechanism_ablations(monkeypatch):
    small(monkeypatch)
    import argparse
    from types import SimpleNamespace
    from v5.runtime import installed
    from v5.config import StudyConfig
    from v5.execution import ExecutionConfig
    options=SimpleNamespace(study=StudyConfig(),execution=ExecutionConfig(),max_epochs=None,steps_per_epoch=None,no_plots=True,retain_candidates='all')
    args=argparse.Namespace(data_root='./datasets',backend='auto',split_seeds='3031',model_seeds='11',k='4',methods='AEMC-NE,Method4',eval_every=None,quick=True)
    with installed(engine,options) as (core,pub):
        base=engine.make_base(pub,args,'ml-100k')
        jobs=engine.build_jobs(pub,base,[4],'hybrid')
        pairs={(c.aemcne.lambda_main,c.aemcne.lambda_element) for c in jobs.values() if c.methods==('AEMC-NE',)}
        assert len(pairs)>=64 and (.01,500.) in pairs and (500.,.01) in pairs
        controls=pub.make_ablation_variants_v8(base)
        assert any('symmetric_amplitude_prior' in str(item) for item in controls)


def test_replot_excel_and_heatmap_preserve_source_data(tmp_path):
    from replot_v5 import main
    from v5.export import export_workbook
    tables=tmp_path/'tables';tables.mkdir()
    frame=pd.DataFrame({'method':['Method1','Method1','Method2','Method2'],'K':[10,20,10,20],'metric':['RMSE']*4,'value':[.9,.8,1.,.95]})
    frame.to_csv(tables/'metrics_long.csv',index=False)
    export_workbook(tmp_path)
    main(['--input',str(tmp_path/'research_tables.xlsx'),'--sheet','metrics_long','--where','metric=RMSE',
          '--x','K','--y','method','--value','value','--kind','heatmap','--aggregate','median','--output',str(tmp_path/'heatmap.png')])
    actual=pd.read_csv(tmp_path/'heatmap.csv')
    assert len(actual)==4 and set(actual.method)=={'Method1','Method2'}
    assert json.loads((tmp_path/'heatmap.json').read_text())['training_performed'] is False
    assert all((tmp_path/('heatmap'+suffix)).is_file() for suffix in ('.png','.pdf','.svg'))


def test_baseline_paper_recipe_does_not_require_an_unselected_blend_component(tmp_path,monkeypatch,capsys):
    import sys
    from paper_v5 import main
    monkeypatch.setattr(sys,'argv',['paper_v5.py','--phase','plan','--protocol','aemc-paper','--methods','AutoRec,AEMC-NE','--output',str(tmp_path/'paper'),'--dry-run'])
    main();command=capsys.readouterr().out
    assert '--include-blend' not in command and '--v5-protocol aemc-paper' in command
