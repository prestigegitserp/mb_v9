"""V4.2 regressions: stronger nulls, frozen studies, storage and competitor fairness."""
import copy,json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from test_v3_research import tiny_data,training_pub
from test_v4_research import config
from test_v4_review import public_data
from test_v4_workflow import _small
from v4.config import StudyConfig
from v4.models import instantiate
from v4.protocol import validation_roles
from v4.training import train_v4
from v4.checkpoints import save,load
from v4.bounds import structural_bounds
from v4.execution import ExecutionConfig,SavedTensorStore
from v4.entry import run
from v9.engine import load_v8


@pytest.mark.parametrize('method',[f'Method{i}' for i in range(1,8)])
@pytest.mark.parametrize('readout',['coupled','decoder'])
def test_new_readout_training_checkpoint_and_mean_law(method,readout,tmp_path):
    torch.set_num_threads(1)
    data=validation_roles(tiny_data(),StudyConfig());data.test_ratings_raw[:]=torch.nan
    cfg=replace(config(method),readout_mode=readout)
    model=instantiate(data,cfg,31);info=train_v4(training_pub(),model,data,cfg,None)
    model.eval()
    with torch.no_grad():pred=model.predict_raw_pairs(data,data.val_users,data.val_items)
    low,high=structural_bounds(model,data,data.val_users,data.val_items)
    assert np.isfinite(info['best_val_RMSE']) and np.all(pred.numpy()>=low-1e-5) and np.all(pred.numpy()<=high+1e-5)
    path=tmp_path/'model.pt';save(path,model,config=cfg,data=data,train_info=info)
    restored,_=load(path,data,torch.device('cpu'))
    with torch.no_grad():torch.testing.assert_close(pred,restored.predict_raw_pairs(data,data.val_users,data.val_items),atol=0,rtol=0)
    if method=='Method7':
        distribution=restored.distribution(data,data.val_users,data.val_items)
        torch.testing.assert_close(distribution['probabilities']@restored.levels,pred,atol=2e-5,rtol=1e-6)
        assert info['v4_probability']['mean_parameters_unchanged']


def test_m4_linear_refinement_keeps_feasible_moments_and_target_exclusion():
    data=tiny_data();cfg=replace(config('Method4'),moment_refinement_link='linear',moment_refinement_scale=1.,moment_projection_mode='mean-priority')
    m=instantiate(data,cfg,13)
    from v4.history import masked_targets
    keep,query=masked_targets(data,.2,torch.Generator().manual_seed(2))
    h=m.history(keep);state=m.prepare(h);u,i=data.train_users[query],data.train_items[query]
    p,extra=m.predict(h,state,u,i)
    assert torch.isfinite(p).all() and (extra['positive_moment']>=-1e-6).all() and (extra['negative_moment']>=-1e-6).all()
    before=p.detach().clone();data.train_ratings_raw[query]=99
    h=m.history(keep);state=m.prepare(h)
    torch.testing.assert_close(m.predict(h,state,u,i)[0],before,atol=0,rtol=0)


@pytest.mark.parametrize('method',['MF','AEMC-NE'])
def test_competitors_validate_nonmultiple_last_epoch(method):
    from v4.baselines import train_biasmf,train_aemcne
    _,pub=load_v8();data=public_data();torch.set_num_threads(1)
    study=pub.PublicationExperimentConfigV8()
    if method=='MF':study.biasmf.max_epochs=3;study.biasmf.eval_every=2
    else:study.aemcne.max_epochs=3;study.aemcne.eval_every=2;study.aemcne.main_hidden_dim=8;study.data.dataset='synthetic'
    model,cfg=pub.instantiate_model_v8(method,data,study,4,17,torch.device('cpu'))
    scores=iter((3.,2.,1.));snapshots=[]
    def evaluate(model,data,partition,**kwargs):
        assert partition=='val';snapshots.append(copy.deepcopy(model.state_dict()));return dict(RMSE=next(scores))
    evaluator=SimpleNamespace(MetricConfig=lambda **kw:None,evaluate_publication_model=evaluate)
    info=(train_biasmf(evaluator,model,data,cfg) if method=='MF' else train_aemcne(evaluator,model,data,cfg,17))
    assert info['best_epoch']==3 and [r['epoch'] for r in info['history']]==[1,2,3]
    for name,value in model.state_dict().items():torch.testing.assert_close(value,snapshots[-1][name],atol=0,rtol=0)


def test_mbmf_adam_is_explicit_same_objective_optimizer_control():
    from v4.baselines import paper_config,train_mbmf
    core,pub=load_v8();data=public_data();torch.manual_seed(17)
    original=core.PaperBaselineConfig(latent_dim=4,max_epochs=1,eval_every=1)
    a=core.PaperMBMF(data,original,'MBMF-F','isotropic_gaussian',torch.device('cpu'))
    b=copy.deepcopy(a);cfg=paper_config(original,optimizer='adam',adam_lr=.003)
    opt=torch.optim.Adam(b.parameters(),lr=.003)
    u,i=data.train_users,data.train_items
    target=b.native_targets(data.train_ratings_raw,u,i)
    loss=.5*(b.native_predict_pairs(u,i)-target).square().sum();loss.backward();opt.step()
    evaluator=SimpleNamespace(MetricConfig=lambda **kw:None,evaluate_publication_model=lambda *args,**kw:dict(RMSE=1.))
    info=train_mbmf(evaluator,a,data,cfg,chunk_size=len(u))
    assert info['optimizer']=='adam_common_protocol'
    for name,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[name],atol=0,rtol=0)


def test_storage_reuse_respects_mutations_aliases_and_persistent_parameters(tmp_path):
    p=torch.nn.Parameter(torch.tensor([2.,3.]))
    with SavedTensorStore(ExecutionConfig(storage='ssd',cache_dir=str(tmp_path)),persistent=[p]) as store:
        resident=store.pack(p);assert torch.is_tensor(resident) and store.disk_bytes==0
        x=torch.tensor([1.,4.]);a=store.pack(x);b=store.pack(x)
        assert a is b and store.cache_hits==1
        x.add_(2);c=store.pack(x);assert c is not a
        torch.testing.assert_close(store.unpack(a),torch.tensor([1.,4.]),atol=0,rtol=0)
        torch.testing.assert_close(store.unpack(c),torch.tensor([3.,6.]),atol=0,rtol=0)
        alias=x.view_as(x);d=store.pack(alias);assert d is not c
    assert not list(tmp_path.glob('v4_graph_*'))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires real single-CUDA hardware')
def test_cuda_storage_does_not_change_optimizer_updates(tmp_path):
    states=[]
    for policy in ('resident','host','ssd'):
        torch.manual_seed(2);torch.cuda.manual_seed(2)
        model=torch.nn.Sequential(torch.nn.Linear(7,8),torch.nn.GELU(),torch.nn.Linear(8,1)).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.003)
        x=torch.arange(56,device='cuda',dtype=torch.float32).reshape(8,7)/50
        with SavedTensorStore(ExecutionConfig(storage=policy,cache_dir=str(tmp_path))):
            for _ in range(3):
                opt.zero_grad(set_to_none=True);model(x).square().mean().backward();opt.step()
        states.append({k:v.detach().cpu() for k,v in model.state_dict().items()})
    for state in states[1:]:
        for key,value in state.items():torch.testing.assert_close(value,states[0][key],atol=0,rtol=0)


def test_scan_dossier_and_ablations_inherit_winners_without_retaining_weights(tmp_path,monkeypatch):
    _small(monkeypatch)
    cfgfile=tmp_path/'settings.json';cfgfile.write_text(json.dumps(dict(models=dict(Method4=dict(moment_refinement_link='linear',moment_refinement_scale=1.,moment_dispersion_weight=.23)))))
    common=['--datasets','synthetic','--quick','--methods','Method4','--k','4','--split-seeds','2026','--model-seeds','1',
        '--v4-threads','1','--v4-no-plots','--v4-search-jobs-per-method','1','--v4-config',str(cfgfile),'--v4-retain-candidates','metrics','--output',str(tmp_path/'run')]
    run(['scan','--scan-only',*common]);root=tmp_path/'run'/'synthetic'
    assert not list(root.rglob('*__test.npz')) and not list((root/'search').rglob('*.pt'))
    tables=root/'search'/'publication'/'tables'
    for name in ('search_all_candidates.csv','hyperparameters_long.csv','search_budget.csv','selection_trace.csv','training_budget.csv','point_prediction_calibration.csv'):
        assert (tables/name).is_file()
    assert list((tables/'per_pair').glob('*__train.csv.gz'))
    assert len(pd.read_csv(tables/'selection_trace.csv'))==1
    run(['ablation',*common]);protocol=json.loads((root/'ablations'/'ABLATION_PROTOCOL.json').read_text())
    assert protocol['reselection'] is False and protocol['anchors'][0]['frozen']['method']=='Method4'
    full=json.loads((root/'ablations'/'Method4__full'/'Method4'/'split_2026'/'config.json').read_text())
    actual=full['v4']['models']['Method4'];assert actual['moment_refinement_link']=='linear' and actual['moment_refinement_scale']==1. and actual['moment_dispersion_weight']==.23
    for path in (root/'ablations').rglob('locked_test_runs.csv'):
        assert pd.read_csv(path).K.eq(4).all()
    # Fail before training if a frozen scientific setting was changed on disk.
    path=root/'search'/'jobs'/'default__Method4'/'config.json';payload=json.loads(path.read_text());payload['v4']['models']['Method4']['hidden_dim']=999;path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='Frozen scan mismatch'):
        run(['kstudy',*common])


def test_blend_uses_only_validation_and_aligns_pairs(tmp_path):
    from v4.ensemble import aligned,paired,convex_weight,fit_validation_blend
    truth=np.array([1.,2.,3.,4.]);first=truth+np.array([1.,1.,-1.,-1.]);second=truth-np.array([1.,1.,-1.,-1.])
    assert convex_weight(first,second,truth)==.5
    rows=[]
    for method,prediction in [('AutoRec',first),('Method4',second)]:
        path=tmp_path/(method+'__val.npz');order=np.array([2,0,3,1])
        np.savez(path,pair_id=order,truth=truth[order],prediction=prediction[order])
        rows.append(dict(dataset='tiny',method=method,split_seed=1,model_seed=2,scan_job=method,config_key=json.dumps(dict(config={},method=method)),val_prediction_file=str(path)))
    frame=pd.DataFrame(rows);winners=frame[['dataset','method','split_seed','scan_job','config_key']]
    frozen=fit_validation_blend(tmp_path,frame,winners,('AutoRec','Method4'))
    assert not frozen['test_read_during_fit'] and frozen['splits'][0]['validation_RMSE']==0.
    pair=aligned(rows[0]['val_prediction_file']);other=aligned(rows[1]['val_prediction_file']);paired(pair,other)
    other['truth'][0]+=1
    with pytest.raises(ValueError,match='cohorts'):paired(pair,other)
    with pytest.raises(ValueError,match='TEST'):fit_validation_blend(tmp_path,frame.assign(test_RMSE=1),winners,('AutoRec','Method4'))


def test_collaborative_regularization_matches_declared_raw_scale():
    data=tiny_data();cfg=config('Method4');a=instantiate(data,cfg,31);b=instantiate(data,replace(cfg,collaborative_l2=100),31)
    h=a.history();sa=a.prepare(h);sb=b.prepare(h)
    increase=b.common_prior(sb)-a.common_prior(sa)
    weights=sum(p.square().sum() for name,p in b.named_parameters() if p.ndim>1 and ('encoder' in name or 'decoder' in name))
    expected=.5*100*weights/(len(data.train_users)*(data.rating_max-data.rating_min)**2)
    torch.testing.assert_close(increase,expected,atol=1e-7,rtol=2e-6)


def test_frozen_blend_complete_scan_to_locked_archives(tmp_path,monkeypatch):
    _small(monkeypatch)
    args=['scan','--datasets','synthetic','--quick','--methods','MF,Method4','--k','4',
        '--split-seeds','2026','--model-seeds','1','--v4-threads','1','--v4-no-plots',
        '--v4-search-jobs-per-method','1','--include-blend','--blend-methods','MF,Method4','--output',str(tmp_path/'run')]
    run(args);root=tmp_path/'run'/'synthetic'
    frozen=json.loads((root/'search'/'ensemble'/'VALIDATION_WEIGHTS.json').read_text())
    assert not frozen['test_read_during_fit'] and not frozen['novelty_claim']
    row=pd.read_csv(root/'ensemble'/'ensemble_locked_test_runs.csv').iloc[0]
    assert np.isfinite(row.test_RMSE) and row.method=='Blend[MF+Method4]'
    for split in ('train','val','test'):
        with np.load(row[split+'_prediction_file']) as archive:
            error=archive['prediction'].astype(float)-archive['truth']
            assert abs(np.sqrt(np.mean(error**2))-row[split+'_RMSE'])<1e-7
    # Validation archives are content-addressed: editing one invalidates the
    # blend before it can read or produce any new test result.
    part=frozen['splits'][0]['sources'][0]['components'][0]
    path=root/'search'/part['validation_file']
    original=path.read_bytes();path.write_bytes(original+b'changed')
    from v4.ensemble import apply_locked_blend
    with pytest.raises(ValueError,match='archive changed'):
        apply_locked_blend(root,pd.read_csv(root/'final_locked_test_all.csv'),('MF','Method4'),plots=False)


def test_small_search_budget_includes_strong_competitors():
    from v4 import engine
    from v4.runtime import installed
    from v4.execution import ExecutionConfig
    settings=StudyConfig(search_jobs_per_method=2)
    options=SimpleNamespace(study=settings,execution=ExecutionConfig(),retain_candidates='metrics',max_epochs=None,steps_per_epoch=None,no_plots=True)
    args=SimpleNamespace(data_root='datasets',backend='auto',split_seeds='2026',model_seeds='1',k='20',methods='AutoRec,MBMF-F,AEMC-NE',eval_every=None,quick=False)
    with installed(engine,options) as (_,pub):
        base=engine.make_base(pub,args,'ml-100k');jobs=engine.build_jobs(pub,base,(20,),'hybrid')
        assert any('AutoRec' in c.methods and c.autorec.lambda_l2==100. for c in jobs.values())
        assert any('AEMC-NE' in c.methods and c.aemcne.lambda_main==c.aemcne.lambda_element==100. for c in jobs.values())
        assert any('MBMF-F' in c.methods and c.v4.mbmf_optimizer=='adam' for c in jobs.values())
        assert all(sum(method in c.methods for c in jobs.values())==2 for method in base.methods)
        base.v4.search_jobs_per_method=0
        complete=engine.build_jobs(pub,base,(20,),'hybrid')
        controls={(c.autorec.hidden_dim,c.autorec.lambda_l2) for c in complete.values() if c.methods==('AutoRec',)}
        assert {(146,10.),(146,100.),(265,10.),(265,100.)}<=controls
