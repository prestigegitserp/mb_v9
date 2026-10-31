from dataclasses import replace
import pytest
import torch
from test_v3_research import tiny_data,training_pub
from test_v8_scientific_contracts import small
from v8.config import StudyConfig
from v8.models import instantiate
from v8.protocol import validation_roles
from v8.training import train_v8
from v8.recovery import EpochRecovery


@pytest.mark.parametrize('method',['Method1','Method6','Method7'])
def test_interrupted_mean_resume_matches_with_reliability_and_calibration(method,tmp_path,monkeypatch):
    torch.set_num_threads(1)
    data=validation_roles(tiny_data(),StudyConfig())
    cfg=small(method,max_epochs=5,val_patience=10,router_enabled=False)
    def build(folder):
        model=instantiate(data,cfg,41)
        model._v8_artifact_dir=folder;model._v8_resume=True;model._v8_recovery_interval=1
        return model
    reference=build(tmp_path/'reference')
    expected=train_v8(training_pub(),reference,data,cfg,None)
    interrupted=build(tmp_path/'interrupted')
    original=EpochRecovery.save
    def stop(self,model,opt,rng,state,force=False):
        original(self,model,opt,rng,state,force)
        if state['epoch']==2:raise KeyboardInterrupt('simulated interruption')
    with monkeypatch.context() as patch:
        patch.setattr(EpochRecovery,'save',stop)
        with pytest.raises(KeyboardInterrupt):train_v8(training_pub(),interrupted,data,cfg,None)
    resumed=build(tmp_path/'interrupted')
    result=train_v8(training_pub(),resumed,data,cfg,None)
    assert [r['val_RMSE'] for r in expected['history']]==[r['val_RMSE'] for r in result['history']]
    for name,value in reference.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[name],atol=0,rtol=0)
