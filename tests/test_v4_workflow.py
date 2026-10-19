"""End-to-end validation locking, resume and paper-protocol integration."""
from pathlib import Path
import json,copy
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from v4 import engine
from v4.entry import run
from v4.config import StudyConfig
from v4.execution import ExecutionConfig
from v4.runtime import installed


def _small(monkeypatch):
    old=engine.make_base
    def make(pub,args,dataset):
        cfg=old(pub,args,dataset)
        cfg.data.synthetic_users=24;cfg.data.synthetic_items=28
        cfg.data.synthetic_rank=3;cfg.data.synthetic_density=.6
        return cfg
    monkeypatch.setattr(engine,'make_base',make)


def test_scan_freezes_winners_runs_final_and_resumes_without_training(tmp_path,monkeypatch):
    _small(monkeypatch)
    args=['scan','--datasets','synthetic','--quick','--methods','Method5,Method7','--k','4','--split-seeds','2026','--model-seeds','1',
          '--v4-no-plots','--v4-threads','2','--v4-search-jobs-per-method','2','--output',str(tmp_path/'run')]
    run(args);root=tmp_path/'run'/'synthetic'
    candidates=pd.read_csv(root/'search'/'hyperparameter_validation_all.csv')
    assert len(candidates)==4 and not any(c.startswith('test_') for c in candidates)
    assert len(pd.read_csv(root/'final_locked_test_all.csv'))==2
    winners=pd.read_csv(root/'search'/'validation_winners.csv')
    for _,row in winners.iterrows():
        final=pd.read_csv(root/'final'/row.method/'split_2026'/'locked_test_runs.csv')
        assert json.loads(row.config_key)==json.loads(final.config_key.iloc[0])
    for job in (root/'search'/'jobs').iterdir():
        assert not list((job/'predictions').glob('*__test.npz'))
        assert list(job.rglob('*.pt'))
        manifest=json.loads((job/'publication'/'REPORT_MANIFEST.json').read_text())
        assert manifest['stage']=='validation_only' and manifest['smoke_only']
        for item in manifest['sources']:assert item['partition'] in ('train','val')
    m7=root/'search'/'jobs'/'default__Method7'
    for path in (m7/'predictions').glob('*__val.npz'):
        with np.load(path) as archive:
            np.testing.assert_allclose(archive['probabilities']@archive['levels'],archive['prediction'],atol=2e-5)
            assert archive['member_means'].shape[1]==3
    assert list((m7/'predictions'/'calibration').glob('*.npz'))
    before={p:p.stat().st_mtime_ns for p in root.rglob('*.pt')}
    for job in (root/'search'/'jobs').iterdir():
        assert json.loads((job/'V9_JOB_STATUS.json').read_text())['status']=='COMPLETED'
        assert (job/'candidate_validation_runs.csv').is_file()
    import v4.runtime
    def forbidden_training(*args,**kwargs):raise AssertionError('Completed resume must not train')
    monkeypatch.setattr(v4.runtime,'train_v4',forbidden_training)
    run(args+['--resume'])
    assert before=={p:p.stat().st_mtime_ns for p in root.rglob('*.pt')}
    with pytest.raises(ValueError,match='DIFFERENT config'):
        run(args+['--resume','--v4-storage','host'])
    run(['report','--datasets','synthetic','--v4-no-plots','--output',str(tmp_path/'run'),'--v4-threads','2'])
    assert before=={p:p.stat().st_mtime_ns for p in root.rglob('*.pt')}


@pytest.mark.parametrize('protocol,train_fraction,val_fraction',[('autorec-paper',.81,.09),('aemc-paper',.855,.045)])
def test_explicit_global_paper_split_does_not_reuse_m7_roles(protocol,train_fraction,val_fraction,tmp_path,monkeypatch):
    _small(monkeypatch)
    run(['main','--datasets','synthetic','--quick','--methods','MF','--k','4','--split-seeds','2026','--model-seeds','1',
         '--v4-protocol',protocol,'--v4-no-plots','--v4-threads','2','--output',str(tmp_path/protocol)])
    root=tmp_path/protocol/'synthetic'/'main'
    frame=pd.read_csv(root/'split_assignments_2026.csv');n=len(frame)
    assert (frame.split=='train').sum()==round(n*train_fraction)
    assert (frame.split=='val').sum()==round(n*val_fraction)
    meta=json.loads((root/'V4_PROTOCOL_2026.json').read_text())
    assert meta['selection']==meta['validation_original'] and meta['calibration_fit']==meta['calibration_hold']==0
    with pytest.raises(ValueError,match='exclude Method7'):
        run(['plan','--datasets','synthetic','--methods','Method7','--v4-protocol',protocol,'--v4-threads','2','--output',str(tmp_path/'invalid')])


def test_complete_baseline_grid_and_mechanism_ablations():
    options=SimpleNamespace(study=StudyConfig(),execution=ExecutionConfig(),retain_candidates='all',max_epochs=None,steps_per_epoch=None,no_plots=True)
    args=SimpleNamespace(data_root='datasets',backend='auto',split_seeds='2026',model_seeds='1',k='20',methods='Method1,Method2,Method3,Method4,Method5,Method6,Method7,AutoRec,AEMC-NE',eval_every=None,quick=False)
    with installed(engine,options) as (_,pub):
        base=engine.make_base(pub,args,'ml-100k');jobs=engine.build_jobs(pub,base,(20,),'hybrid')
        ar={c.autorec.lambda_l2 for c in jobs.values() if c.methods==('AutoRec',)}
        aemc={(c.aemcne.lambda_main,c.aemcne.lambda_element) for c in jobs.values() if c.methods==('AEMC-NE',)}
        assert {.001,.01,.1,1.,100.,1000.}<=ar
        grid=(.01,.1,1.,10.,50.,100.,200.,500.)
        assert {(a,b) for a in grid for b in grid}<=aemc
        variants=pub.make_ablation_variants_v8(base)
        tags={tag for tag,_ in variants}
        assert 'Method6__no_reference_balance' not in tags  # same soft-default intervention as no_moment_prior
        for tag in ('Method1__no_cavity','Method2__coarse_conditional_risk','Method3__global_radius','Method4__no_moment_refinement','Method5__plugin_mean','Method6__hard_balance','Method7__statistics_only'):assert tag in tags


def test_render_scientific_figures_and_no_one_split_confidence_band(tmp_path):
    from v4.figures import plot_k_rmse
    from PIL import Image
    for k in (4,8):
        folder=tmp_path/f'K_{k}';folder.mkdir()
        pd.DataFrame(dict(dataset=['tiny']*2,method=['Method1']*2,split_seed=[2026]*2,model_seed=[1,2],K=[k,k],train_RMSE=[.8,.9],val_RMSE=[1.,1.1],test_RMSE=[1.,1.2])).to_csv(folder/'locked_test_runs.csv',index=False)
        (folder/'V9_JOB_STATUS.json').write_text(json.dumps({'command_arguments':{'quick':True}}))
    out=tmp_path/'figures';names=plot_k_rmse(tmp_path,out)
    assert names and len(list(out.glob('*.png')))==len(list(out.glob('*.svg')))==len(list(out.glob('*.pdf')))==1
    image=Image.open(next(out.glob('*.png')));image.load();assert image.width>1000
    summary=pd.read_csv(out/'k_rmse_summary.csv');assert summary.test_RMSE__std.isna().all()
    assert summary.test_RMSE__count.eq(1).all()
    assert json.loads((out/'FIGURE_PROVENANCE.json').read_text())['smoke']


def test_v4_ablation_names_pair_against_actual_full_reference(tmp_path):
    from v4.figures import plot_ablation_effects
    data=pd.DataFrame(dict(dataset=['tiny']*4,method=['Method1']*4,split_seed=[1]*4,model_seed=[1,2,1,2],K=[4]*4,ablation=['Method1__full']*2+['Method1__no_cavity']*2,test_RMSE=[1.,1.1,1.2,1.3]))
    assert plot_ablation_effects(data,tmp_path)==['tiny_ablation_Method1']
    pairs=pd.read_csv(tmp_path/'ablation_paired_effects.csv')
    np.testing.assert_allclose(pairs.delta_RMSE,.2);assert pairs.matched_K.all()
    with pytest.raises(ValueError,match='Missing full'):
        plot_ablation_effects(data.iloc[2:],tmp_path/'missing')
