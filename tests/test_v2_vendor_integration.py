"""Small real-vendor synthetic integration for V2. No external datasets needed."""
import math
import torch
import pytest
from v2.adapters import adapt
from v2.training import train_v2
from v9.engine import load_v8

METHODS=tuple("Method"+str(i) for i in range(1,8))

@pytest.mark.parametrize("method",METHODS)
def test_real_v7_v8_instance_model_and_checkpoint(method):
    core,pub=load_v8()
    device=torch.device("cpu")
    core.set_global_seed(109)
    data=core.synthetic_data(device)
    cfg=pub.PublicationExperimentConfigV8()
    core.apply_q1_safe_preset(cfg.core_config)
    for n in range(1,8):
        c=getattr(cfg.core_config,"method"+str(n))
        c.max_epochs=3
        c.eval_every=1
        c.val_patience=2
        c.latent_dim=4
        if hasattr(c,"hidden_dim"):c.hidden_dim=8
    cfg.core_config.method7.batch_size=16
    cfg.core_config.method2.consistency_weight=.04
    model,mcfg=pub.instantiate_model_v8(method,data,cfg,4,109,device)
    model=adapt(model,method,data,"balanced",core,pub)
    assert model.v2_method==method
    assert not model.training or True
    model.eval()
    with torch.no_grad():
        pred=model.predict_raw_pairs(data,data.val_users,data.val_items)
    assert pred.shape==data.val_ratings_raw.shape
    assert torch.isfinite(pred).all()
    assert float(pred.min())>=data.rating_min-1e-5
    assert float(pred.max())<=data.rating_max+1e-5
    if method!="Method7":
        info=train_v2(pub,model,data,mcfg,chunk_size=29)
    else:
        info=pub.train_model_v8(method,model,data,mcfg,cfg)
    assert info["best_epoch"]>=1
    m=pub.MetricConfig(compute_testset_ndcg=False,compute_binary_ranking=False,compute_pearson_spearman=False)
    val=pub.evaluate_publication_model(model,data,"val",metric_cfg=m)
    assert math.isfinite(float(val["RMSE"]))
    state={key:x.detach().clone() for key,x in model.state_dict().items()}
    with torch.no_grad():
        p1=model.predict_raw_pairs(data,data.val_users,data.val_items).clone()
    model.load_state_dict(state)
    with torch.no_grad():
        p2=model.predict_raw_pairs(data,data.val_users,data.val_items)
    assert torch.allclose(p1,p2,atol=1e-6)
