"""Primary-source baseline ledger: paper scores are not target values to force."""
import json
from pathlib import Path
from dataclasses import asdict
import pandas as pd

PAPERS={
 'AEMC-NE':dict(source='https://proceedings.iclr.cc/paper_files/paper/2024/file/6cd3ac24cdb789beeaa9f7145670fcae-Paper-Conference.pdf',
   location='Section 5.2 and Table 2',ml100k_test_RMSE=.8767,reported_plusminus=.0089,reported_splits=20,
   paper_train_fraction=.855,paper_validation_fraction=.045,paper_test_fraction=.1,
   implementation='independent; official synthetic tanh+0.8z forward audited; MovieLens training parity NOT established'),
 'AutoRec':dict(source='https://users.cecs.anu.edu.au/~ssanner/Papers/www15.pdf',
   location='Section 3: original paper reports ML-1M, ML-10M and Netflix; no ML-100K score',
   ml100k_test_RMSE=None,reported_splits=5,paper_train_fraction=.81,paper_validation_fraction=.09,paper_test_fraction=.1,
   implementation='independent I-AutoRec; sigmoid hidden, linear output, Rprop; cold-start policy may differ'),
 'MBMF':dict(source='https://arxiv.org/pdf/1807.05515',location='arXiv v1, experiments; archived accepted-version audit is in docs/V4_2_REVIEW_FA.md',
   ml100k_test_RMSE=None,implementation='independent spherical-coordinate factors; common-protocol validation checkpoint; paper/Adam optimizer controls explicitly distinguished')}


def assess_final(root, runs, protocol):
    """Descriptive paper deltas, without declaring independent-code reproduction."""
    root=Path(root);rows=[];configs=[]
    references={
        "aemc-paper":{
            "AEMC-NE":{"ml-100k":(.8767,.0089),"ml-1m":(.8248,.0024),"ml-10m":(.7723,.0025)},
            "AutoRec":{"ml-100k":(.8818,.0082),"ml-1m":(.8291,.0021),"ml-10m":(.7780,.0024)}},
        "autorec-paper":{"AutoRec":{"ml-1m":(.831,.003),"ml-10m":(.782,.003)}}}
    outer=runs.groupby(["dataset","method","split_seed"],as_index=False).test_RMSE.mean()
    for (dataset,method),frame in outer.groupby(["dataset","method"]):
        ref=references.get(protocol,{}).get(method,{}).get(dataset)
        required=20 if protocol=="aemc-paper" else 5
        rows.append(dict(dataset=dataset,method=method,protocol=protocol,
            observed_mean=frame.test_RMSE.mean(),observed_sd=frame.test_RMSE.std(ddof=1),
            observed_outer_splits=len(frame),reference_mean=ref[0] if ref else None,
            reference_plusminus_as_printed=ref[1] if ref else None,
            reference_uncertainty_kind=("paper_table_plusminus_no_CI_claim" if protocol=="aemc-paper" else "95pct_CI_halfwidth_bound") if ref else None,
            delta_mean=frame.test_RMSE.mean()-ref[0] if ref else None,
            required_reference_splits=required if ref else None,
            enough_repetitions=(len(frame)>=required) if ref else False,
            comparison="DESCRIPTIVE_NUMERIC_ONLY" if ref else "NO_MATCHED_PAPER_REFERENCE",
            official_implementation_verified=False,
            confirmatory_significance=False,
            source=PAPERS["AEMC-NE"]["source"] if protocol=="aemc-paper" and ref else PAPERS.get(method,{}).get("source")))
    for _,row in runs.iterrows():
        value=json.loads(row.config_key)["config"]
        response={}
        history_file=root/"final"/str(row.method)/f"split_{int(row.split_seed)}"/"training_histories.json"
        if history_file.is_file():
            for identity,history in json.loads(history_file.read_text()).items():
                if identity.startswith(str(row.method)+"|") and f"|seed={int(row.model_seed)}|" in identity and history:
                    response=history[-1].get("v6_response",{})
                    break
        configs.append(dict(dataset=row.dataset,method=row.method,split_seed=row.split_seed,
            model_seed=row.model_seed,K=row.K,source_lineage=value.get("lineage","baseline"),
            hidden_dim=value.get("hidden_dim",value.get("main_hidden_dim")),
            response_shrinkage=value.get("response_shrinkage",False),
            fitted_response_scale=response.get("selected_scale"),
            raw_val_RMSE_before_response=response.get("raw_val_RMSE"),
            corrected_val_RMSE=response.get("corrected_val_RMSE"),
            response_degenerates_to_bias=response.get("degenerates_to_bias_only"),
            selected_search_job=row.get("selected_search_job"),config_json=json.dumps(value,sort_keys=True)))
    tables=root/"publication"/"tables";tables.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(tables/"paper_reference_comparison.csv",index=False)
    pd.DataFrame(configs).to_csv(tables/"selected_configurations.csv",index=False)


def write_ledger(folder,cfg):
    rows=[]
    for method in cfg.methods:
        if method.startswith('Method'):continue
        reference=PAPERS.get('MBMF' if method.startswith('MBMF-') else method,{})
        if method=='AEMC-NE':operative=asdict(cfg.aemcne)
        elif method=='AutoRec':operative=asdict(cfg.autorec)
        elif method in ('MF','BiasMF'):operative=asdict(cfg.biasmf)
        else:operative=dict(checkpoint_policy=cfg.v6.mbmf_checkpoint_policy,optimizer=cfg.v6.mbmf_optimizer,max_epochs=cfg.core_config.paper.max_epochs)
        rows.append(dict(method=method,dataset=cfg.data.dataset,protocol=cfg.v6.protocol,
            cold_start_policy="fixed rating 3 on unseen user/item" if method=="AutoRec" else "see implementation",
            operative_config_json=json.dumps(operative,sort_keys=True),
            train_fraction=cfg.data.train_fraction,validation_fraction=cfg.data.val_fraction,test_fraction=cfg.data.test_fraction,
            selection_role_fraction=cfg.v6.validation_selection_fraction if cfg.v6.split_validation_roles else 1.,
            status='NOT_VERIFIED_AS_PAPER_REPRODUCTION',**reference))
    path=Path(folder);path.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path/'BASELINE_FIDELITY.csv',index=False)
    (path/'BASELINE_FIDELITY.json').write_text(json.dumps(dict(
        score_policy='Observed scores are never rescaled or changed to match papers',
        status='Protocol settings alone do not establish reproduction; compare multi-split test distributions and official code',
        official_synthetic_forward_audited=True,
        official_supplement='docs/v6_evidence/AEMC_SUPPLEMENT_AUDIT.json',
        aemc_unverified_details=['MovieLens-specific activations','Initialization','Item minibatch size and optimizer update count'],
        method7_probability_fitting=cfg.v6.models['Method7'].probability_fit,
        model7_point_only_has_no_extra_calibration_data=not cfg.v6.models['Method7'].probability_fit,
        autorec_ml100k_reproduced_elsewhere=dict(source=PAPERS['AEMC-NE']['source'],location='Table 2',mean=.8818,sd=.0082,splits=20),rows=rows),indent=2))
