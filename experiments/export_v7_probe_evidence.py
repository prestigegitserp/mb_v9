"""Export compact, auditable development evidence without model checkpoints."""
import argparse
import json
from pathlib import Path
import pandas as pd


def export(roots,output,warm_scan=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frames=[];histories={};protocols=[]
    for root in map(Path,roots):
        protocol=json.loads((root/'PROBE_PROTOCOL.json').read_text())
        completion=json.loads((root/'PROBE_COMPLETION.json').read_text())
        if not completion['completed'] or completion['test_evaluated']:raise ValueError('Incomplete or TEST probe')
        protocols.append(dict(study=root.name,protocol=protocol,completion=completion))
        table=pd.read_csv(root/'VALIDATION_PROBES.csv');table['source_study']=root.name
        for tag in table.probe_tag.unique():
            folder=root/'ml-100k/validation_probes/jobs'/tag
            histories[root.name+'/'+tag]=json.loads((folder/'training_histories.json').read_text())
        frames.append(table)
    if warm_scan is not None:
        warm_scan=Path(warm_scan)
        table=pd.read_csv(warm_scan/'ml-100k/search/hyperparameter_validation_all.csv')
        table['probe_family']='warm_start_control';table['probe_tag']=table.scan_job
        table['source_study']=warm_scan.name
        for _,row in table.iterrows():
            histories[warm_scan.name+'/'+str(row.scan_job)]=json.loads((Path(row.artifact_root)/'training_histories.json').read_text())
        frames.append(table)
    rows=[]
    for _,row in pd.concat(frames,ignore_index=True).iterrows():
        cfg=json.loads(row.config_key)['config']
        lineage=cfg.get('lineage','original Method10' if row.method=='Method8' else 'independent baseline')
        inactive=row.probe_family=='gated' and lineage!='v7'
        record={key:row.get(key) for key in ('dataset','method','probe_tag','source_study','split_seed','model_seed','K',
                    'train_RMSE','val_RMSE','train_wall_sec','parameter_count','best_epoch','config_key')}
        record.update(probe_family='inactive_gate_control' if inactive else row.probe_family,
            source_lineage=lineage,history_signal=cfg.get('history_signal'),
            gate_active=lineage=='v7' and cfg.get('history_signal')=='gated',
            exclude_from_architecture_claims=inactive,hidden_dim=cfg.get('hidden_dim',cfg.get('main_hidden_dim')),
            max_epochs_cap=cfg.get('max_epochs'),steps_per_epoch=cfg.get('steps_per_epoch'),
            note='Historical Method3 ignores the new gate; replaced by separately executed modern/gated controls' if inactive else '')
        traces=histories[str(row.source_study)+'/'+str(row.probe_tag)]
        budget=next(iter(traces.values()))[-1].get('v7_training_budget',{})
        record.update({'actual_'+key:value for key,value in budget.items()})
        rows.append(record)
    full=pd.DataFrame(rows)
    if full.val_RMSE.isna().any():raise ValueError('Missing validation result')
    full.to_csv(output/'ML100K_VALIDATION_PROBES.csv',index=False,float_format='%.17g')
    eligible=full[~full.exclude_from_architecture_claims]
    best=eligible.loc[eligible.groupby('method').val_RMSE.idxmin()].sort_values('method')
    best.to_csv(output/'ML100K_OBSERVED_VALIDATION_BEST.csv',index=False,float_format='%.17g')
    (output/'ML100K_VALIDATION_HISTORIES.json').write_text(json.dumps(histories,indent=2)+'\n')
    statement=dict(candidate_trainings=len(full),comparison='within-method bounded validation development; not equal total compute',
        study_protocols=protocols,first_driver_commit='c38ef9494ec95e855a29dbfecd5abf151a753779',
        first_driver_inactive_gate_excluded=True,modern_method3_controls_executed_separately=True,
        warm_start_control='extra AutoRec initialization budget is recorded, not an author MovieLens implementation claim',
        test_evaluated=False,paper_reproduction_established=False,rtx4090_benchmark=False,
        all_six_datasets_fully_trained=False,all_candidate_results_retained=True,
        limitation='One split/seed, short caps, validation-selected observations; use locked TEST and independent splits for claims')
    (output/'ML100K_PROBE_PROTOCOL.json').write_text(json.dumps(statement,indent=2)+'\n')
    print(best[['method','probe_family','source_lineage','val_RMSE']].to_string(index=False))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-root',type=Path,action='append',required=True)
    parser.add_argument('--warm-scan',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    export(args.probe_root,args.output,args.warm_scan)


if __name__=='__main__':main()
