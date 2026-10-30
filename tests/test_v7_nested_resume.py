"""Recover interrupted OOF expert training without changing final predictions."""
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from test_v3_research import tiny_data,training_pub
from test_v7_rebuild import small
from v7.config import StudyConfig
from v7.models import instantiate
from v7.protocol import validation_roles
from v7.training import train_v7
from v7.recovery import EpochRecovery,recovered_training_time


@pytest.mark.parametrize('lineage',['v7','v52'])
def test_resume_inside_second_oof_expert_is_exact(lineage,tmp_path,monkeypatch):
    torch.set_num_threads(1)
    data=validation_roles(tiny_data(),StudyConfig(protocol='aemc-paper',split_validation_roles=False))
    cfg=small('Method2',True,lineage=lineage,max_epochs=3,val_patience=10,router_enabled=True,
              router_folds=2,router_expert_epochs=3,router_epochs=4,response_shrinkage=False)
    def create(folder):
        model=instantiate(data,cfg,19);model._v7_artifact_dir=folder
        model._v7_resume=True;model._v7_recovery_interval=1
        return model
    full=create(tmp_path/'full');train_v7(training_pub(),full,data,cfg,None)
    original=EpochRecovery.save
    def interrupt(self,model,opt,generator,state,force=False):
        original(self,model,opt,generator,state,force)
        if self.path and 'oof_fold_1' in str(self.path) and state['epoch']==1:raise KeyboardInterrupt('OOF interruption')
    partial=create(tmp_path/'partial')
    with monkeypatch.context() as patch:
        patch.setattr(EpochRecovery,'save',interrupt)
        with pytest.raises(KeyboardInterrupt):train_v7(training_pub(),partial,data,cfg,None)
    resumed=create(tmp_path/'partial')
    with recovered_training_time() as recovered:
        train_v7(training_pub(),resumed,data,cfg,None)
    assert len(recovered)==3 # main mean + first complete expert + interrupted second expert
    assert all(seconds>0 for seconds in recovered.values())
    for key,value in full.state_dict().items():torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)


def test_resumed_warm_start_time_survives_a_second_main_interruption(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import v7.recovery as recovery
    monkeypatch.setattr(recovery.time,'perf_counter',lambda:52.)
    def create(folder):
        model=torch.nn.Linear(2,1);model.cfg=SimpleNamespace(lr=.001)
        model._v7_artifact_dir=folder;model._v7_resume=True
        return model
    main=create(tmp_path);warm=create(tmp_path/'warm_start')
    with recovered_training_time() as phases:
        parent=EpochRecovery(main,phase='baseline');parent.started=50.
        old=EpochRecovery(warm,phase='baseline');old.elapsed_before=4.
        old.save(warm,None,None,dict(epoch=1),force=True)
        EpochRecovery(warm,phase='baseline')
        assert sum(phases.values())==4.
        parent.save(main,None,None,dict(epoch=1),force=True)
    with recovered_training_time() as phases:
        EpochRecovery(main,phase='baseline')
        assert len(phases)==1 and sum(phases.values())==6.
