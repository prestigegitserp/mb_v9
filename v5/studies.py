"""Secondary studies inherit each outer split's frozen validation winner."""
import copy
import json
from pathlib import Path


def frozen_configurations(pub,base,root,args):
    from . import engine,search
    from .selection import prepare_locked_plan
    import pandas as pd
    if getattr(args,'analysis_source','winners')=='defaults':
        return [('defaults',copy.deepcopy(base))]
    path=Path(root)/'search'/'validation_winners.csv'
    if not path.is_file():
        raise ValueError('Run scan --scan-only before secondary studies, or explicitly use --analysis-source defaults')
    winners=pd.read_csv(path)
    jobs=search.build_jobs(pub,base,engine.numbers(args.k),args.scan_mode)
    plan,specs=prepare_locked_plan(base,root,winners,jobs)
    result=[]
    for _,row in winners.iterrows():
        method,seed=str(row.method),int(row.split_seed)
        cfg=copy.deepcopy(plan[str(row.scan_job)])
        cfg.methods=(method,);cfg.core_config.split_seeds=(seed,)
        cfg.core_config.latent_dims=(int(row.K) if pd.notna(row.K) else int(base.core_config.latent_dims[0]),)
        if method.startswith('MBMF-'):cfg.core_config.paper.init_modes=(str(row.init_mode),)
        cfg.v5.analysis_anchor=dict(method=method,split_seed=seed,scan_job=str(row.scan_job),
            config_key=str(row.config_key),data_signature=specs[(method,seed)]['data_signature'])
        result.append((method+'/split_'+str(seed),cfg))
    return result


def anchor_protocol(configurations,*,kind,**changes):
    return dict(study=kind,selection='per-outer-split validation winner; no test tuning',
        anchors=[dict(path=tag,methods=list(cfg.methods),K=list(cfg.core_config.latent_dims),
                      frozen=cfg.v5.analysis_anchor) for tag,cfg in configurations],**changes)
