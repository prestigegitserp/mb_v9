"""Exact split storage and scientific parity with the unchanged V8 runner."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from v7 import engine,runner
from v7.config import StudyConfig
from v7.execution import ExecutionConfig
from v7.split_cache import SplitFrames,OneEntryCache,shared_csv
from test_v3_research import tiny_data


def raw_data():
    data=tiny_data();frames=[]
    for role in ('train','val','test'):
        frames.append(pd.DataFrame(dict(userId=getattr(data,role+'_users').numpy(),
            movieId=getattr(data,role+'_items').numpy(),rating=getattr(data,role+'_ratings_raw').numpy())))
    raw=pd.concat(frames,ignore_index=True);raw['timestamp']=np.arange(len(raw),dtype=np.int64)+1000000000
    return raw


@pytest.mark.parametrize('budget',[1e-6,.01])
def test_split_roles_restore_exact_values_dtypes_and_order(budget,tmp_path):
    _,pub=engine.load_v8();raw=raw_data()
    dc=pub.PublicationDataConfig(dataset='ml-100k',split_seed=91)
    cfg=SimpleNamespace(data=dc,execution=ExecutionConfig(ssd_budget_gb=budget,cache_dir=str(tmp_path/'ssd')),
        core_config=SimpleNamespace(split_seeds=(91,92)),output_dir=str(tmp_path))
    original=cfg.execution;store=SplitFrames(raw,cfg,pub,pub.DATASET_SPECS['ml-100k'])
    try:
        for seed in (91,92):
            selected=copy.deepcopy(dc);selected.split_seed=seed
            frame=pub.make_core_split(raw,pub.DATASET_SPECS['ml-100k'],selected)
            store[seed]=frame
            pd.testing.assert_frame_equal(frame,store[seed],check_exact=True)
        assert store.used<=store.budget
        assert cfg.execution.ssd_budget_gb*1024**3+store.budget<=original.ssd_budget_gb*1024**3+1
        with pytest.raises(KeyError):store[999]
    finally:store.close()
    assert cfg.execution is original and store.raw is None
    assert not hasattr(cfg,'_v7_split_frames')


def test_tensor_data_cache_holds_only_one_split():
    cache=OneEntryCache();cache[1]=object();cache[2]=object()
    assert len(cache)==1 and 1 not in cache and 2 in cache


def test_shared_csv_is_exact_and_copy_fallback_is_recorded(tmp_path,monkeypatch):
    _,pub=engine.load_v8();root=tmp_path/'job';root.mkdir()
    cfg=SimpleNamespace(data=pub.PublicationDataConfig(dataset='ml-100k',data_root=str(tmp_path/'data')),
        v7=StudyConfig(),output_dir=str(root),_v7_shared_data_root=str(tmp_path/'shared'))
    frame=raw_data();shared_csv(frame,root/'first.csv',cfg,'fixed')
    with monkeypatch.context() as patch:
        def unavailable(*args,**kwargs):raise OSError('filesystem has no hardlinks')
        patch.setattr('os.link',unavailable);shared_csv(frame,root/'second.csv',cfg,'fixed')
    expected=frame.to_csv(index=False).encode()
    assert (root/'first.csv').read_bytes()==expected==(root/'second.csv').read_bytes()
    events=[json.loads(line) for line in (root/'V7_SHARED_ARTIFACTS.jsonl').read_text().splitlines()]
    assert events[-1]['mode']=='copy-fallback'


def test_generated_runner_matches_pinned_native_function():
    from experiments.generate_v7_runner import generate
    root=Path(__file__).resolve().parents[1]
    assert generate(root)==(root/'v7/runner.py').read_text()


def test_two_split_metrics_and_predictions_equal_native_orchestration(tmp_path,monkeypatch):
    pytest.importorskip('xlsxwriter')
    from v7.entry import run
    original_base=engine.make_base
    def small(pub,args,dataset):
        cfg=original_base(pub,args,dataset)
        cfg.data.synthetic_users=18;cfg.data.synthetic_items=22
        cfg.data.synthetic_rank=3;cfg.data.synthetic_density=.7
        return cfg
    monkeypatch.setattr(engine,'make_base',small)
    _,pub=engine.load_v8();native=pub.run_publication_study_v8
    path=tmp_path/'settings.json'
    path.write_text(json.dumps(dict(split_validation_roles=False,models={
        'Method'+str(i):dict(hidden_dim=12,probability_fit=False) for i in range(1,8)},
        method8=dict(hidden_dim=12,decoder_dim=8))))
    def arguments(out):
        return ['main','--datasets','synthetic','--methods','MF,Method1,Method4,Method8',
            '--split-seeds','4101,4102','--model-seeds','11','--k','4','--v7-config',str(path),
            '--v7-max-epochs','3','--v7-steps-per-epoch','1','--v7-baseline-max-epochs','3',
            '--v7-no-plots','--no-v7-isolate-jobs','--v7-threads','1','--output',str(out)]
    with monkeypatch.context() as patch:
        patch.setattr(runner,'run_study',lambda pub_,cfg,*args,**kwargs:native(cfg,*args,**kwargs))
        run(arguments(tmp_path/'native'))
    run(arguments(tmp_path/'bounded'))
    old=pd.read_csv(tmp_path/'native/synthetic/main/locked_test_runs.csv')
    new=pd.read_csv(tmp_path/'bounded/synthetic/main/locked_test_runs.csv')
    keys=['method','split_seed','model_seed','K']
    old=old.sort_values(keys).reset_index(drop=True);new=new.sort_values(keys).reset_index(drop=True)
    np.testing.assert_array_equal(old[keys].to_numpy(),new[keys].to_numpy())
    for metric in ('train_RMSE','val_RMSE','test_RMSE','test_MAE'):
        np.testing.assert_array_equal(old[metric].to_numpy(),new[metric].to_numpy())
    for left,right in zip(old.test_prediction_file,new.test_prediction_file):
        a=np.load(left);b=np.load(right)
        for field in ('pair_id','truth','prediction'):np.testing.assert_array_equal(a[field],b[field])
    meta=json.loads((tmp_path/'bounded/synthetic/main/V7_SPLIT_CACHE.json').read_text())
    assert meta['raw_values_and_row_order_preserved'] and meta['cached_role_vectors']==2
