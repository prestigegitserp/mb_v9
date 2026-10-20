"""Primary-source baseline ledger: paper scores are not target values to force."""
import json
from pathlib import Path
from dataclasses import asdict
import pandas as pd

PAPERS={
 'AEMC-NE':dict(source='https://proceedings.iclr.cc/paper_files/paper/2024/file/6cd3ac24cdb789beeaa9f7145670fcae-Paper-Conference.pdf',
   location='Section 5.2 and Table 2',ml100k_test_RMSE=.8767,reported_sd=.0089,reported_splits=20,
   paper_train_fraction=.855,paper_validation_fraction=.045,paper_test_fraction=.1,
   implementation='independent equation-level reimplementation; official-code parity NOT established'),
 'AutoRec':dict(source='https://users.cecs.anu.edu.au/~ssanner/Papers/www15.pdf',
   location='Section 3: original paper reports ML-1M, ML-10M and Netflix; no ML-100K score',
   ml100k_test_RMSE=None,reported_splits=5,paper_train_fraction=.81,paper_validation_fraction=.09,paper_test_fraction=.1,
   implementation='independent I-AutoRec; sigmoid hidden, linear output, Rprop; cold-start policy may differ'),
 'MBMF':dict(source='https://arxiv.org/pdf/1807.05515',location='arXiv v1, experiments; archived accepted-version audit is in docs/V4_2_REVIEW_FA.md',
   ml100k_test_RMSE=None,implementation='independent spherical-coordinate factors; common-protocol validation checkpoint; paper/Adam optimizer controls explicitly distinguished')}


def write_ledger(folder,cfg):
    rows=[]
    for method in cfg.methods:
        if method.startswith('Method'):continue
        reference=PAPERS.get('MBMF' if method.startswith('MBMF-') else method,{})
        if method=='AEMC-NE':operative=asdict(cfg.aemcne)
        elif method=='AutoRec':operative=asdict(cfg.autorec)
        elif method in ('MF','BiasMF'):operative=asdict(cfg.biasmf)
        else:operative=dict(checkpoint_policy=cfg.v52.mbmf_checkpoint_policy,optimizer=cfg.v52.mbmf_optimizer,max_epochs=cfg.core_config.paper.max_epochs)
        rows.append(dict(method=method,dataset=cfg.data.dataset,protocol=cfg.v52.protocol,
            operative_config_json=json.dumps(operative,sort_keys=True),
            train_fraction=cfg.data.train_fraction,validation_fraction=cfg.data.val_fraction,test_fraction=cfg.data.test_fraction,
            selection_role_fraction=cfg.v52.validation_selection_fraction if cfg.v52.split_validation_roles else 1.,
            status='NOT_VERIFIED_AS_PAPER_REPRODUCTION',**reference))
    path=Path(folder);path.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path/'BASELINE_FIDELITY.csv',index=False)
    (path/'BASELINE_FIDELITY.json').write_text(json.dumps(dict(
        score_policy='Observed scores are never rescaled or changed to match papers',
        status='Protocol settings alone do not establish reproduction; compare multi-split test distributions and official code',
        aemc_unverified_details=['Main activation','Scalar-network activation','Initialization','Item minibatch size and optimizer update count'],
        method7_probability_fitting=cfg.v52.models['Method7'].probability_fit,
        model7_point_only_has_no_extra_calibration_data=not cfg.v52.models['Method7'].probability_fit,
        autorec_ml100k_reproduced_elsewhere=dict(source=PAPERS['AEMC-NE']['source'],location='Table 2',mean=.8818,sd=.0082,splits=20),rows=rows),indent=2))
