"""End-to-end reports, source isolation, retention and recoverable report errors."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
import torch

# Frozen V4 CI installs only requirements_v4.txt.  V6's dedicated workflow
# installs requirements_v6.txt and must exercise all publication integrations.
pytest.importorskip("xlsxwriter", reason="V6 publication tests require requirements_v6.txt")

from v6 import engine
from v6.config import StudyConfig
from v6.entry import run


@pytest.fixture
def tiny_engine(monkeypatch):
    original=engine.make_base
    def small(pub,args,dataset):
        cfg=original(pub,args,dataset)
        cfg.data.synthetic_users=24;cfg.data.synthetic_items=28
        cfg.data.synthetic_rank=3;cfg.data.synthetic_density=.6
        return cfg
    monkeypatch.setattr(engine,"make_base",small)
    torch.set_num_threads(1)


def arguments(tmp_path,methods="MF,Method3,Method4,Method5,Method7",plots=False):
    cfg=tmp_path/"settings.json"
    cfg.write_text(json.dumps({"split_validation_roles":False,"models":{
        "Method"+str(i):{"hidden_dim":12,"probability_fit":False} for i in range(1,8)}}))
    args=["--datasets","synthetic","--methods",methods,"--quick","--k","4",
          "--split-seeds","4101","--model-seeds","11","--v6-config",str(cfg),
          "--v6-threads","1","--v6-search-jobs-per-method","1",
          "--v6-retain-candidates","metrics","--output",str(tmp_path/"run")]
    if not plots:args+=["--v6-no-plots"]
    return args


def test_scan_final_and_report_use_frozen_winners(tiny_engine,tmp_path):
    args=arguments(tmp_path)
    run(["scan","--scan-only",*args])
    root=tmp_path/"run/synthetic"
    assert not list(root.rglob("*__test.npz"))
    assert not list((root/"search").rglob("*__train.npz"))
    assert not list((root/"search").rglob("*.pt"))
    assert (root/"search/best_configurations.json").is_file()
    candidates=pd.read_csv(root/"search/hyperparameter_validation_all.csv")
    assert not any(c.startswith("test_") for c in candidates)
    run(["final",*args])
    final=pd.read_csv(root/"final_locked_test_all.csv")
    assert set(final.method)=={"MF","Method3","Method4","Method5","Method7"}
    assert (root/"publication/research_tables.xlsx").is_file()
    assert (root/"publication/tables/selected_configurations.csv").is_file()
    assert (root/"publication/tables/paper_reference_comparison.csv").is_file()
    assert list((root/"final").rglob("*__test.npz"))
    assert list((root/"final").rglob("*__train.npz"))
    run(["report","--datasets","synthetic","--output",str(tmp_path/"run"),"--v6-no-plots"])
    selected=json.loads((root/"search/best_configurations.json").read_text())
    assert selected["test_used"] is False
    assert "postfit_response_by_model_seed" in selected
    plan=root/"search/V9_SEARCH_PLAN.json";original_plan=plan.read_bytes()
    changed=args.copy();changed[changed.index("--k")+1]="5"
    with pytest.raises(ValueError,match="Different search plan"):
        run(["scan","--scan-only","--resume",*changed])
    assert plan.read_bytes()==original_plan
    protocol=root/"search/jobs/default__Method3/V6_PROTOCOL_4101.json"
    original_protocol=protocol.read_bytes();payload=json.loads(original_protocol)
    payload["data_signature"]["selection_sha256"]="changed-validation-cohort"
    protocol.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="cohort mismatch"):
        run(["scan","--scan-only","--resume",*args])
    protocol.write_bytes(original_protocol)
    config=root/"search/jobs/default__Method5/config.json"
    payload=json.loads(config.read_text());payload["v6"]["models"]["Method5"]["lr"]*=2
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="Frozen scan mismatch"):
        run(["final","--resume",*args])


def test_best_settings_have_real_vector_and_raster_figures(tiny_engine,tmp_path):
    args=arguments(tmp_path,methods="MF,Method4",plots=True)
    run(["scan",*args])
    root=tmp_path/"run/synthetic"
    for suffix in ("png","pdf","svg"):
        assert any(p.stat().st_size>100 for p in (root/"figures").glob("*."+suffix))
        assert any(p.stat().st_size>100 for p in (root/"search/figures").glob("*."+suffix))
    assert (root/"search/publication/research_tables.xlsx").is_file()
    assert (root/"figures/selected_test_rmse.svg").is_file()


def test_report_failure_resumes_without_retraining(tiny_engine,tmp_path,monkeypatch):
    from v6 import reporting
    core,pub=engine.load_v8();original_train=pub.run_publication_study_v8;calls=[]
    def count(*args,**kwargs):
        calls.append(1);return original_train(*args,**kwargs)
    monkeypatch.setattr(pub,"run_publication_study_v8",count)
    original_report=reporting.generate_report
    def fail(*args,**kwargs):raise RuntimeError("simulated spreadsheet failure")
    monkeypatch.setattr(reporting,"generate_report",fail)
    args=arguments(tmp_path,methods="MF")
    with pytest.raises(RuntimeError,match="spreadsheet"):
        run(["main",*args])
    root=tmp_path/"run/synthetic/main"
    status=json.loads((root/"V9_JOB_STATUS.json").read_text())
    assert status["status"]=="REPORT_FAILED" and status["training_complete"]
    trace=(root/"V9_EPOCH_METRICS.csv").read_bytes()
    monkeypatch.setattr(reporting,"generate_report",original_report)
    run(["main","--resume",*args])
    assert len(calls)==1
    assert trace==(root/"V9_EPOCH_METRICS.csv").read_bytes()
    assert (root/"publication/research_tables.xlsx").is_file()


def test_leader_retention_preserves_scores_and_never_touches_final(tmp_path):
    from v6.retention import retain_leaders
    base=SimpleNamespace(v6=StudyConfig(),core_config=SimpleNamespace(split_seeds=(2026,),model_seeds=(1,)))
    rows=[]
    for tag,error in (("a",1.),("b",.9)):
        folder=tmp_path/"search/jobs"/tag
        for child in ("predictions","mechanisms"):
            (folder/child).mkdir(parents=True);(folder/child/"large.bin").write_bytes(b"x"*100)
        (folder/"candidate_validation_runs.csv").write_text("preserve all scores")
        rows.append(dict(dataset="synthetic",method="Method1",split_seed=2026,model_seed=1,
                         scan_job=tag,config_key=tag,val_RMSE=error,K=4,init_mode="NA",artifact_root=str(folder)))
    untouched=tmp_path/"final/keep.bin";untouched.parent.mkdir();untouched.write_bytes(b"keep")
    retain_leaders(tmp_path/"search",pd.DataFrame(rows),base)
    assert not (tmp_path/"search/jobs/a/predictions").exists()
    assert (tmp_path/"search/jobs/a/candidate_validation_runs.csv").is_file()
    assert (tmp_path/"search/jobs/b/predictions/large.bin").is_file()
    assert untouched.read_bytes()==b"keep"
    rows[0]["artifact_root"]=str(tmp_path/"outside")
    with pytest.raises(ValueError,match="outside"):
        retain_leaders(tmp_path/"search",pd.DataFrame(rows),base)


def test_paper_and_development_commands_have_explicit_selection_boundaries(tmp_path):
    from paper_v6 import parser,build_commands
    a=parser().parse_args(["--stage","scan","--profile","paper","--datasets","all","--output",str(tmp_path)])
    commands=build_commands(a)
    assert len(commands)==6 and all("--scan-only" in c for c in commands)
    assert all(c[c.index("--split-seeds")+1].startswith("4101,") for c in commands)
    a.stage="develop";a.profile="development"
    commands=build_commands(a)
    assert all("--scan-only" not in c and c[c.index("--split-seeds")+1]=="2026" for c in commands)


def test_aemc_warm_start_has_accounted_budget_and_ignores_test():
    from test_v3_research import tiny_data,training_pub
    from v6.aemc_baseline import AEMCConfig,AEMCNE,train_aemcne
    data=tiny_data();data.test_ratings_raw[:]=torch.nan
    cfg=AEMCConfig(main_hidden_dim=8,element_hidden_dim=4,initialization="identity-transfer",
                   main_activation="sigmoid",max_epochs=2,warm_start_epochs=2,eval_every=1,item_batch_size=0)
    model=AEMCNE(data,cfg,"cpu")
    info=train_aemcne(training_pub(),model,data,cfg,chunk_size=32)
    assert info["warm_start_optimizer_steps"]==2
    assert info["optimizer_steps"]==info["final_optimizer_steps"]+2
    assert info["prediction_targets_seen"]==4*len(data.train_users)


def test_official_synthetic_residual_forward_matches_sparse_pair_prediction():
    from test_v3_research import tiny_data
    from v6.aemc_baseline import AEMCConfig,AEMCNE
    data=tiny_data();data.train_ratings_raw=data.train_ratings_raw.double()
    cfg=AEMCConfig(main_hidden_dim=8,element_hidden_dim=4,
                   main_activation="tanh",element_activation="tanh",element_skip=.8)
    model=AEMCNE(data,cfg,"cpu").double()
    raw=torch.zeros(data.num_items,data.num_users,dtype=torch.double)
    raw.index_put_((data.train_items,data.train_users),data.train_ratings_raw,accumulate=True)
    layerwise=model.decoder(torch.tanh(model.encoder(raw)))
    z=layerwise[data.val_items,data.val_users]
    # The official supplement's element forward is ENet(z) + 0.8*z.
    expected=model.element2(torch.tanh(model.element1(z[:,None])))[:,0]+.8*z
    actual=model.predict_raw_pairs(data,data.val_users,data.val_items)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    first=torch.autograd.grad(expected.square().sum(),tuple(model.parameters()),retain_graph=True)
    second=torch.autograd.grad(actual.square().sum(),tuple(model.parameters()))
    for a,b in zip(first,second):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)


def test_secondary_workbook_keeps_ablation_variants_separate(tmp_path):
    from v6.secondary_report import rebuild_secondary_reports
    folder=tmp_path/"ablations";folder.mkdir()
    rows=[dict(dataset="synthetic",method="Method1",ablation=cut,split_seed=split,
               model_seed=seed,K=4,init_mode="NA",test_RMSE=score)
          for cut,score in (("Method1__full",1.),("Method1__cut",2.))
          for split in (1,2) for seed in (3,4)]
    pd.DataFrame(rows).to_csv(folder/"ablation_locked_test_runs.csv",index=False)
    rebuild_secondary_reports(tmp_path,plots=False)
    summary=pd.read_csv(folder/"publication/tables/study_summary.csv").set_index("ablation")
    assert summary.loc["Method1__full","test_RMSE__mean"]==1.
    assert summary.loc["Method1__cut","test_RMSE__mean"]==2.
    assert (folder/"publication/research_tables.xlsx").is_file()
    sparse=tmp_path/"sparsity/keep_0.5";sparse.mkdir(parents=True)
    frame=pd.DataFrame(rows)
    frame=frame[frame.ablation=="Method1__full"].drop(columns="ablation")
    frame.to_csv(sparse/"locked_test_runs.csv",index=False)
    density=tmp_path/"synthetic_density/density_0.2";density.mkdir(parents=True)
    frame.to_csv(density/"locked_test_runs.csv",index=False)
    rebuild_secondary_reports(tmp_path,plots=False)
    sparse_summary=pd.read_csv(tmp_path/"sparsity/publication/tables/study_summary.csv")
    assert sparse_summary.train_keep_requested.iloc[0]==.5
    assert "train_keep_requested__mean" not in sparse_summary
    assert (tmp_path/"synthetic_density/publication/research_tables.xlsx").is_file()


def test_common_calibration_can_select_a_historical_or_modern_probability_law(tiny_engine,tmp_path):
    cfg=tmp_path/"probability.json"
    cfg.write_text(json.dumps({"models":{"Method7":{"hidden_dim":12}}}))
    args=["--datasets","synthetic","--methods","Method7","--quick","--k","4",
          "--split-seeds","4101","--model-seeds","11","--v6-config",str(cfg),
          "--v6-threads","1","--v6-search-jobs-per-method","2","--v6-no-plots",
          "--output",str(tmp_path/"probability-run")]
    run(["scan","--scan-only",*args])
    root=tmp_path/"probability-run/synthetic"
    candidates=pd.read_csv(root/"search/hyperparameter_validation_all.csv")
    assert candidates.val_Probability_NLL.notna().all()
    run(["calibration",*args])
    assert (root/"calibration/calibration_locked_test_runs.csv").is_file()
    assert (root/"calibration/publication/research_tables.xlsx").is_file()


def test_temporal_evaluation_cannot_be_labelled_as_aemc_random_split_protocol(tmp_path,monkeypatch):
    from v6.config import StudyConfig
    base=SimpleNamespace(data=SimpleNamespace(dataset="ml-100k",split_mode="random_global"),
                         v6=StudyConfig(protocol="aemc-paper"))
    observed=[]
    monkeypatch.setattr(engine,"scan_dataset",lambda p,c,b,r,a:observed.append((b.data.split_mode,b.v6.protocol)))
    monkeypatch.setattr(engine,"run_final",lambda *a:None)
    engine.run_temporal(None,None,base,tmp_path,SimpleNamespace(dry_run=False))
    assert observed==[("temporal_per_user","common")]
    assert base.v6.protocol=="aemc-paper"


def test_primary_summary_does_not_split_winners_by_K_or_pool_model_seeds_as_outer_repeats():
    from v6.reporting import summarise_locked_runs
    rows=[dict(dataset="ml-100k",method="Method1",split_seed=s,model_seed=r,
               K=k,init_mode="NA",test_RMSE=e)
          for s,k,values in ((1,10,(0.,2.)),(2,50,(6.,8.)))
          for r,e in enumerate(values,1)]
    summary=summarise_locked_runs(pd.DataFrame(rows))
    assert len(summary)==1
    assert summary.test_RMSE_mean.iloc[0]==4.
    assert summary.test_RMSE_std.iloc[0]==pytest.approx(18**.5)
    assert summary.test_RMSE_count.iloc[0]==2
    assert summary.selected_Ks.iloc[0]=="10,50"


def test_comprehensive_search_does_not_overwrite_incumbents_or_warm_start_defaults(tmp_path,monkeypatch):
    from v6 import search
    import paper_v6
    collected={}
    def inspect(pub,base,args):
        collected.update(search.build_jobs(pub,base,engine.numbers(args.k),args.scan_mode))
    monkeypatch.setattr(engine,"plan_for_ds",inspect)
    cfg=Path(paper_v6.ROOT)/"examples/v6/paper/ml-100k.json"
    run(["plan","--datasets","ml-100k","--methods","Method3,AEMC-NE",
         "--split-seeds","4101","--model-seeds","11","--k","4","--v6-config",str(cfg),
         "--v6-no-plots","--output",str(tmp_path)])
    assert collected["default__Method3"].v6.models["Method3"].lineage=="v4"
    assert collected["default__AEMC-NE"].aemcne.warm_start_epochs==100
    assert collected["comprehensive__default__Method3"].v6.models["Method3"].lineage=="v6"
    assert collected["comprehensive__default__AEMC-NE"].aemcne.warm_start_epochs==0
