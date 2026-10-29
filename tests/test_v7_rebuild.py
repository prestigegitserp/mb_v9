"""V7 scientific and recovery contracts; frozen version tests remain untouched."""
import copy
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from test_v3_research import tiny_data,training_pub,config as tiny_config
from v7.config import StudyConfig,default_models
from v7.models import instantiate
from v7.protocol import validation_roles
from v7.training import train_v7,train_mean
from v7.checkpoints import save,load
from v7.recovery import EpochRecovery,candidate_identity,save_candidate,load_candidate


def small(method,modern=False,**overrides):
    values=vars(tiny_config(method)).copy()
    values.update(hidden_dim=12,probability_fit=False)
    if modern:values['lineage']='v7'
    values.update(overrides)
    return replace(default_models()[method],**values)


@pytest.mark.parametrize('method',['Method'+str(i) for i in range(1,8)])
@pytest.mark.parametrize('modern',[False,True])
def test_all_models_train_without_test_and_restore_exactly(method,modern,tmp_path):
    torch.set_num_threads(1)
    data=validation_roles(tiny_data(),StudyConfig(protocol='aemc-paper',split_validation_roles=False))
    data.test_ratings_raw[:]=torch.nan
    model=instantiate(data,small(method,modern),23)
    train_v7(training_pub(),model,data,model.cfg,None)
    point=model.predict_raw_pairs(data,data.val_users,data.val_items).detach()
    assert torch.isfinite(point).all()
    path=tmp_path/'model.pt';save(path,model,config=model.cfg,data=data)
    restored,_=load(path,data,'cpu')
    torch.testing.assert_close(point,restored.predict_raw_pairs(data,data.val_users,data.val_items),atol=0,rtol=0)


@pytest.mark.parametrize('method',['Method'+str(i) for i in range(1,8)])
def test_gates_are_trainable_and_exclude_queries(method):
    data=tiny_data();model=instantiate(data,small(method,True,history_signal='gated'),9)
    keep=torch.arange(len(data.train_users))%3!=0
    h=model.history(keep);zu,zi=model.encode(h)
    (zu.square().sum()+zi.square().sum()).backward()
    assert model.history_mix_user.grad.abs().sum()>0
    assert model.history_mix_item.grad.abs().sum()>0
    def encoded():
        with torch.no_grad():return model.encode(model.history(keep))
    before=encoded();data.train_ratings_raw[~keep]=999.;data.train_timestamps[~keep]=-1e12
    after=encoded()
    for x,y in zip(before,after):torch.testing.assert_close(x,y,atol=0,rtol=0)


@pytest.mark.parametrize('method,modern',[('Method1',True),('Method2',True),('Method3',False),('Method4',False),('Method5',False),('Method1','v52'),('Method2','v52'),('Method6','v52'),('Method7','v52')])
def test_interrupted_optimizer_resume_matches_uninterrupted(method,modern,tmp_path,monkeypatch):
    torch.set_num_threads(1)
    data=validation_roles(tiny_data(),StudyConfig(protocol='aemc-paper',split_validation_roles=False))
    cfg=small(method,modern,max_epochs=5,val_patience=10,router_enabled=False,response_shrinkage=False)
    if isinstance(modern,str):cfg=replace(cfg,lineage=modern)
    def build(folder):
        model=instantiate(data,cfg,41);model._v7_artifact_dir=folder
        model._v7_resume=True;model._v7_recovery_interval=1
        return model
    reference=build(tmp_path/'reference')
    info=train_v7(training_pub(),reference,data,cfg,None)
    interrupted=build(tmp_path/'interrupted')
    original=EpochRecovery.save
    def interrupt(self,model,opt,rng,state,force=False):
        original(self,model,opt,rng,state,force)
        if state['epoch']==2:raise KeyboardInterrupt('simulated interruption')
    with monkeypatch.context() as patch:
        patch.setattr(EpochRecovery,'save',interrupt)
        with pytest.raises(KeyboardInterrupt):train_v7(training_pub(),interrupted,data,cfg,None)
    resumed=build(tmp_path/'interrupted')
    replay=train_v7(training_pub(),resumed,data,cfg,None)
    assert [r['val_RMSE'] for r in info['history']]==[r['val_RMSE'] for r in replay['history']]
    assert info['mean_optimizer_steps']==replay['mean_optimizer_steps']
    for key,value in reference.state_dict().items():torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)


def test_completed_candidate_identity_and_cache(tmp_path):
    data=tiny_data();cfg=small('Method1',True);model=instantiate(data,cfg,1)
    model._v7_artifact_dir=tmp_path;model._v7_model_seed=1;model._v7_resume=True
    identity=candidate_identity(model,data,cfg)
    save_candidate(model,identity,dict(best_epoch=2,history=[]))
    assert load_candidate(model,identity)['v7_resume']['reused']
    with pytest.raises(ValueError,match='identity'):load_candidate(model,'changed')
    data.val_ratings_raw[0]+=.1
    assert candidate_identity(model,data,cfg)!=identity


def test_loss_coefficients_present_even_in_bounded_search():
    from v7.entry import run
    from v7.runtime import installed
    from v7.execution import ExecutionConfig
    from v7 import engine,search
    options=SimpleNamespace(study=StudyConfig(search_jobs_per_method=8),execution=ExecutionConfig(),
        max_epochs=None,baseline_max_epochs=None,steps_per_epoch=None,no_plots=True,retain_candidates='metrics',
        v7_recovery_interval=5,v7_isolate_jobs=False)
    args=SimpleNamespace(data_root='.',backend='auto',split_seeds='1',model_seeds='1',k='20',methods='Method1',
                         eval_every=None,quick=False,resume=True)
    with installed(engine,options) as (_,pub):
        base=engine.make_base(pub,args,'ml-100k');jobs=search.build_jobs(pub,base,(20,))
        assert any(tag.startswith('loss_joint_') for tag in jobs)
        assert any(tag.startswith('gated_context') for tag in jobs)
        assert list(jobs)==list(search.build_jobs(pub,base,(20,)))


def test_caps_override_json_and_method8_remains_original():
    from v7.runtime import installed
    from v7.execution import ExecutionConfig
    from v7 import engine,method8
    options=SimpleNamespace(study=StudyConfig(baseline_overrides={'aemcne':{'max_epochs':999,'warm_start_epochs':100}}),
        execution=ExecutionConfig(),max_epochs=7,baseline_max_epochs=5,steps_per_epoch=None,no_plots=True,
        retain_candidates='metrics',v7_recovery_interval=5,v7_isolate_jobs=False)
    args=SimpleNamespace(data_root='.',backend='auto',split_seeds='1',model_seeds='1',k='20',methods='Method8,AEMC-NE',
                         eval_every=None,quick=True,resume=True)
    with installed(engine,options) as (_,pub):
        cfg=engine.make_base(pub,args,'ml-100k')
        assert cfg.aemcne.max_epochs==3 and cfg.aemcne.warm_start_epochs==3
        assert cfg.v7.method8['max_epochs']==3
    method8.verify_source()
    assert method8.Method10.__module__=='vendor_method10.method10'


def test_manual_seed_conflict_is_explicit():
    from v7.entry import run
    with pytest.raises(SystemExit):run(['plan','--manual-seed','2','--model-seeds','3'])


def test_worker_hydrates_nested_models_and_tuples():
    from v7.worker import hydrate
    original=StudyConfig();payload=asdict(original);payload['models']['Method1']['lr']=.003
    recovered=hydrate(original,payload)
    assert recovered.models['Method1'].lr==.003
    assert recovered.models['Method1'].method=='Method1'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA hardware required')
def test_cuda_releases_live_allocations():
    from v7.lifecycle import release_model
    before=torch.cuda.memory_allocated()
    model=torch.nn.Linear(4096,4096,device='cuda')
    record=release_model(model)
    assert record['after']['allocated_bytes']<=before+1024*1024
