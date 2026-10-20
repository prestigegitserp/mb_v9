"""V5.2 recipes and export workflow require V5 spreadsheet dependencies."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
pytest.importorskip("xlsxwriter")
pytest.importorskip("openpyxl")
from v52.entry import run
from v52 import engine


def test_new_recipe_uses_explicit_v52_entry_and_full_paper_protocol(tmp_path,monkeypatch,capsys):
    from paper_v5_2 import main
    monkeypatch.setattr(sys,"argv",["paper_v5_2.py","--phase","plan","--output",str(tmp_path),"--dry-run"])
    main();command=capsys.readouterr().out
    assert "run_v5_2.py" in command and "--v52-protocol aemc-paper" in command
    assert "4101" in command and "4120" in command and "--include-blend" not in command


def test_v5_flag_aliases_are_parsed_by_v52_without_calling_old_implementation(tmp_path):
    run(["plan","--datasets","synthetic","--methods","Method5","--k","4","--split-seeds","4101","--model-seeds","11",
         "--v5-no-plots","--v5-threads","1","--output",str(tmp_path)])


def test_tiny_scan_then_final_exports_and_freezes_v52(tmp_path,monkeypatch):
    original=engine.make_base
    def small(pub,args,dataset):
        cfg=original(pub,args,dataset);cfg.data.synthetic_users=24;cfg.data.synthetic_items=28
        cfg.data.synthetic_rank=3;cfg.data.synthetic_density=.6
        return cfg
    monkeypatch.setattr(engine,"make_base",small)
    config_path=tmp_path/"settings.json"
    config_path.write_text(json.dumps({"models":{method:{"hidden_dim":12,"wide_context_readout":True,"readout_mode":"residual","full_history_weight":.25} for method in ("Method5","Method7")}}))
    args=["--v52-config",str(config_path),"--datasets","synthetic","--methods","MF,Method5,Method7","--quick","--k","4","--split-seeds","4101",
          "--model-seeds","11","--v52-threads","1","--v52-no-plots","--v52-search-jobs-per-method","1",
          "--v52-retain-candidates","metrics","--output",str(tmp_path/"run")]
    run(["scan","--scan-only",*args]);root=tmp_path/"run/synthetic"
    assert not list(root.rglob("*__test.npz"))
    run(["final",*args])
    assert (root/"publication/research_tables.xlsx").is_file()
    assert (root/"final_locked_test_all.csv").is_file()
    config=root/"search/jobs/default__MF/config.json"
    value=json.loads(config.read_text());value["biasmf"]["lr"]*=2;config.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="Frozen scan mismatch"):run(["final",*args])
