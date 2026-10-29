"""A real short scan followed by validation-only objective sensitivity."""
import json
from pathlib import Path
import pytest
pytest.importorskip('xlsxwriter')
from test_v7_workflow import tiny_engine,arguments
from v7.entry import run


def test_loss_study_uses_frozen_winner_and_exports_aggregate_only(tiny_engine,tmp_path,monkeypatch):
    import pandas as pd
    from v7 import lossstudy
    args=arguments(tmp_path,'Method1')
    run(['scan','--scan-only',*args])
    monkeypatch.setattr(lossstudy,'coefficients',lambda cfg:{'bias_prior':None})
    monkeypatch.setattr(lossstudy,'trial_values',lambda cfg,name:(0.,.01))
    run(['lossstudy','--resume',*args])
    folder=tmp_path/'run/synthetic/lossstudy'
    data=pd.read_csv(folder/'loss_coefficient_validation.csv')
    assert set(data.parameter)=={'bias_prior'} and set(data.value)=={0.,.01}
    assert not any(key.startswith('test_') for key in data)
    assert (folder/'loss_sensitivity.xlsx').is_file()
    assert not list((folder/'jobs').rglob('*.xlsx'))
    assert not list(folder.rglob('*__test.npz'))
    protocol=json.loads((folder/'LOSS_STUDY_PROTOCOL.json').read_text())
    assert protocol['test_evaluated'] is False
    assert all(row['frozen']['config_key'] for row in protocol['anchors'])


def test_probability_loss_summary_retains_nll_and_parameter_kind(tmp_path):
    import pandas as pd
    from v7.lossstudy import rebuild
    pd.DataFrame([dict(method='Method7',split_seed=seed,model_seed=1,parameter='probability_rps_weight',
        parameter_kind='probability-fit objective coefficient',value=value,val_RMSE=.9,
        val_Probability_NLL=1.+value+seed*.01) for seed in (1,2) for value in (0.,.1)]).to_csv(
            tmp_path/'loss_coefficient_validation.csv',index=False)
    rebuild(tmp_path,plots=False)
    summary=pd.read_csv(tmp_path/'loss_coefficient_probability_summary.csv')
    assert set(summary['count'])=={2}
    assert 'probability_summary' in pd.ExcelFile(tmp_path/'loss_sensitivity.xlsx').sheet_names
