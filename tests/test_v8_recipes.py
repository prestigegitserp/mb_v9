import json
from pathlib import Path
import pytest
from paper_v8 import parser,resolve_recipe,build_commands,main


def test_saved_recipe_restores_exact_scientific_settings(tmp_path):
    args=['--stage','scan','--output',str(tmp_path),'--budget','light',
          '--data-root',str(tmp_path/'data'),'--split-seeds','1,2','--model-seeds','11,22']
    a,path=resolve_recipe(parser().parse_args(args))
    payload={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    path.write_text(json.dumps({'settings':payload}))
    b,_=resolve_recipe(parser().parse_args(['--stage','test','--output',str(tmp_path)]))
    assert b.max_epochs==120 and b.steps_per_epoch==2
    assert b.model_seeds=='11,22' and b.split_seeds=='1,2'
    command=build_commands(b)[0]
    assert command[2]=='final' and '--scan-only' not in command
    c,_=resolve_recipe(parser().parse_args(args))
    assert c.data_root==a.data_root
    with pytest.raises(ValueError,match='fresh'):
        resolve_recipe(parser().parse_args(['--stage','test','--output',str(tmp_path),'--max-epochs','121']))


def test_all_stages_include_test_before_secondary_studies(tmp_path):
    a,_=resolve_recipe(parser().parse_args(['--stage','all','--output',str(tmp_path)]))
    commands=build_commands(a)
    stages=[c[2] for c in commands]
    assert stages[:2]==['scan','final']
    assert stages.count('lossstudy')==1
    assert '--scan-only' in commands[0]


def test_report_can_enable_plots_after_no_plot_training(tmp_path):
    a,_=resolve_recipe(parser().parse_args(['--stage','report','--output',str(tmp_path),'--plots']))
    assert not a.no_plots


def test_status_reports_pending_test_without_training(tmp_path):
    assert main(['--stage','status','--output',str(tmp_path)])==0
    result=json.loads((tmp_path/'ml-100k/MANUSCRIPT_CHECKLIST.json').read_text())
    assert result['status']=='TEST_PENDING'
    assert not result['journal_acceptance_or_sota_certified']
