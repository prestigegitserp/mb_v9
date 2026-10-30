"""Interrupt baselines after epoch two and compare their recovered trajectories."""
from pathlib import Path
import pytest
import torch
from test_v4_review import public_data
from test_v3_research import training_pub
from v7.engine import load_v8
from v7.recovery import EpochRecovery
from v7 import baselines
from v7.aemc_baseline import AEMCConfig,AEMCNE,train_aemcne


@pytest.mark.parametrize('family',['mf','autorec-rprop','autorec-adam','aemc-direct','aemc-warm','mbmf-paper','mbmf-adam'])
def test_baseline_epoch_resume_is_exact(family,tmp_path,monkeypatch):
    torch.set_num_threads(1)
    core,pub=load_v8();data=public_data()
    if family=='mf':
        cfg=pub.BiasMFConfig(latent_dim=4,max_epochs=5,eval_every=1,val_patience=10,batch_size=30)
        make=lambda:pub.BiasMF(data.num_users,data.num_items,cfg,torch.device('cpu'))
        fit=lambda m:baselines.train_biasmf(training_pub(),m,data,cfg)
    elif family.startswith('autorec'):
        cfg=pub.AutoRecConfig(hidden_dim=8,max_epochs=5,eval_every=1,val_patience=10,optimizer=family.split('-')[1])
        make=lambda:pub.ItemAutoRec(data,cfg,torch.device('cpu'))
        fit=lambda m:baselines.train_autorec(training_pub(),m,data,cfg,17)
    elif family.startswith('aemc'):
        cfg=AEMCConfig(main_hidden_dim=8,element_hidden_dim=4,max_epochs=5,eval_every=1,val_patience=10,
                       main_activation='sigmoid',initialization='identity-transfer',
                       warm_start_epochs=3 if family=='aemc-warm' else 0,item_batch_size=4)
        make=lambda:AEMCNE(data,cfg,'cpu')
        fit=lambda m:train_aemcne(training_pub(),m,data,cfg,17)
    else:
        cfg=baselines.paper_config(core.PaperBaselineConfig(latent_dim=4,max_epochs=5,eval_every=1),patience=10,
                                   optimizer='adam' if family=='mbmf-adam' else 'paper')
        make=lambda:pub._make_paper_baseline_v8(data,cfg,'MBMF-F','isotropic_gaussian',torch.device('cpu'))
        fit=lambda m:baselines.train_mbmf(pub,m,data,cfg,17)
    def build(folder):
        torch.manual_seed(57);model=make();model._v7_artifact_dir=folder
        model._v7_resume=True;model._v7_recovery_interval=1
        return model
    original=build(tmp_path/'full');info=fit(original)
    interrupted=build(tmp_path/'partial')
    save=EpochRecovery.save
    def stop(self,model,opt,generator,state,force=False):
        save(self,model,opt,generator,state,force)
        if state['epoch']==2:raise KeyboardInterrupt('interrupt epoch')
    with monkeypatch.context() as patch:
        patch.setattr(EpochRecovery,'save',stop)
        with pytest.raises(KeyboardInterrupt):fit(interrupted)
    resumed=build(tmp_path/'partial');result=fit(resumed)
    assert [r['val_RMSE'] for r in info['history']]==[r['val_RMSE'] for r in result['history']]
    for name,value in original.state_dict().items():torch.testing.assert_close(value,resumed.state_dict()[name],atol=0,rtol=0)
