"""Aggregate a completed frozen validation study and preserve negative results."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--metric',type=Path,action='append',default=[])
    parser.add_argument('--baseline-budget',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--recipe',type=Path,required=True)
    args=parser.parse_args()
    plan_path=args.study/'PLAN.json'
    plan=json.loads(plan_path.read_text())
    plan_hash=hashlib.sha256(plan_path.read_bytes()).hexdigest()
    records=[]
    winner_sources={}
    canonical=None
    def archive_metrics(path):
        nonlocal canonical
        with np.load(path,allow_pickle=False) as archive:
            pair,y,p=[archive[k] for k in ('pair_id','truth','prediction')]
        if canonical is None:canonical=(pair.copy(),y.copy())
        np.testing.assert_array_equal(pair,canonical[0]);np.testing.assert_array_equal(y,canonical[1])
        if not np.isfinite(p).all():raise ValueError('Nonfinite prediction archive')
        return dict(val_MAE=float(np.abs(p.astype(float)-y).mean()),
                    val_clipped_RMSE=float(np.mean((p.clip(1,5).astype(float)-y)**2)**.5),
                    val_global_OOB_rate=float(((p<1)|(p>5)).mean()))
    for job in plan['jobs']:
        path=args.study/job['id']/'result.json'
        if not path.is_file():raise ValueError('Incomplete study: '+job['id'])
        result=json.loads(path.read_text())
        if result.get('status')!='COMPLETED' or result['plan_sha256']!=plan_hash or result['job']!=job:
            raise ValueError('Result does not match frozen job: '+str(path))
        if result['test_evaluated'] or not result['test_labels_poisoned']:
            raise ValueError('Unexpected test access')
        info=result['info']
        row=dict(job=job['id'],method=job['method'],arm=job['arm'],init_mode=job['init_mode'],
                 val_RMSE=result['validation_RMSE'],parameter_count=result['parameter_count'],
                 best_epoch=info.get('best_epoch',info.get('final_epoch')),
                 last_epoch=info.get('last_epoch',info.get('final_epoch')),
                 optimizer_steps=info.get('mean_optimizer_steps'),
                 targets_seen=info.get('prediction_targets_seen'),
                 oof_optimizer_steps=info.get('v4_routing',{}).get('oof_expert_optimizer_steps',0),
                 router_optimizer_steps=info.get('v4_routing',{}).get('router_optimizer_steps',0),
                 wall_seconds=result['wall_seconds'],result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        row.update(archive_metrics(path.parent/'validation.npz'))
        records.append(row)
        config=dict(result['actual_config'])
        if job['method'].startswith('Method'):
            config['moment_projection_weight']=1.
            config['moment_projection_mode']='metric'
        winner_sources[job['id']]=dict(result=str(path),checkpoint=str(path.parent/'checkpoint.pt'),
                                     predictions=str(path.parent/'validation.npz'),config=config,kind='frozen-runtime')
    for directory in args.metric:
        path=directory/'result.json'
        result=json.loads(path.read_text())
        if result['test_evaluated'] or not result['test_labels_poisoned']:
            raise ValueError('Unexpected prototype test access')
        info=result['info']
        mode=result['plan'].get('projection','metric')
        tag='Method4__'+('priority' if mode=='mean-priority' else 'metric')+('__density' if result['config']['mask_fraction_jitter'] else '')
        control=next(r for r in records if r['job']=='Method4__control')
        records.append(dict(job=tag,method='Method4',arm=mode,init_mode='NA',
                            val_RMSE=result['validation_RMSE'],parameter_count=control['parameter_count'],
                            best_epoch=info['best_epoch'],last_epoch=info['last_epoch'],
                            optimizer_steps=info['mean_optimizer_steps'],targets_seen=info['prediction_targets_seen'],
                            oof_optimizer_steps=0,router_optimizer_steps=0,wall_seconds=result['wall_seconds'],
                            result_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        records[-1].update(archive_metrics(directory/'validation.npz'))
        config=dict(result['config']);config['moment_projection_weight']=result['plan']['gamma'];config['moment_projection_mode']=mode
        winner_sources[tag]=dict(result=str(path),checkpoint=str(directory/'prototype.pt'),
                                 predictions=str(directory/'validation.npz'),config=config,kind='moment-projection-prototype')
    for directory in args.baseline_budget:
        path=directory/'result.json';result=json.loads(path.read_text());info=result['info']
        if result['test_evaluated'] or not result['test_labels_poisoned']:raise ValueError('Unexpected baseline test access')
        method=result['plan']['method']
        tag=method+'__long_budget'+('_wd='+str(result['actual_config']['weight_decay']) if method=='MF' else '')
        last=info.get('final_epoch',info['history'][-1]['epoch'])
        records.append(dict(job=tag,method=method,arm='long_budget',init_mode=result['plan'].get('init_mode','NA'),val_RMSE=result['validation_RMSE'],
                            parameter_count=result['parameter_count'],best_epoch=info['best_epoch'],
                            last_epoch=last,optimizer_steps=last,
                            targets_seen=last*80053,oof_optimizer_steps=0,router_optimizer_steps=0,
                            wall_seconds=result['wall_seconds'],result_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        records[-1].update(archive_metrics(directory/'validation.npz'))
        winner_sources[tag]=dict(result=str(path),checkpoint=str(directory/'checkpoint.pt'),
                                 predictions=str(directory/'validation.npz'),config=result['actual_config'],kind='baseline-budget-pilot')
    table=pd.DataFrame(records)
    if not np.isfinite(table.val_RMSE).all():raise ValueError('Nonfinite validation results')
    args.output.mkdir(parents=True,exist_ok=True)
    table.to_csv(args.output/'validation_candidates.csv',index=False)
    winners=table.sort_values(['val_RMSE','job'],kind='mergesort').drop_duplicates('method')
    winners.sort_values('method').to_csv(args.output/'validation_winners.csv',index=False)
    changes=[]
    for method in ['Method'+str(i) for i in range(1,8)]:
        control=table[table.job.eq(method+'__control')].iloc[0]
        win=winners[winners.method.eq(method)].iloc[0]
        changes.append(dict(method=method,control_val_RMSE=control.val_RMSE,
                            selected_job=win.job,selected_val_RMSE=win.val_RMSE,
                            relative_gain_percent=100*(control.val_RMSE-win.val_RMSE)/control.val_RMSE,
                            control_parameter_count=control.parameter_count,selected_parameter_count=win.parameter_count,
                            control_optimizer_steps=control.optimizer_steps,selected_optimizer_steps=win.optimizer_steps,
                            control_targets=control.targets_seen,selected_targets=win.targets_seen))
    pd.DataFrame(changes).to_csv(args.output/'proposed_control_comparison.csv',index=False)
    registry={r.method:dict(job=r.job,**winner_sources[r.job]) for r in winners.itertuples()}
    (args.output/'WINNER_SOURCES.json').write_text(json.dumps(registry,indent=2))
    strata=[]
    for method,record in registry.items():
        with np.load(record['predictions'],allow_pickle=False) as archive:y,p=archive['truth'],archive['prediction']
        for rating in np.unique(y):
            mask=y==rating;error=p[mask].astype(float)-y[mask]
            strata.append(dict(method=method,rating=float(rating),n=int(mask.sum()),val_RMSE=float(np.mean(error**2)**.5),
                               val_MAE=float(np.abs(error).mean()),val_mean_error=float(error.mean())))
    pd.DataFrame(strata).to_csv(args.output/'selected_error_by_rating.csv',index=False)
    recipe=dict(mbmf_checkpoint_policy='validation',mbmf_val_patience=16,mf_max_epochs=1000,mbmf_max_epochs=2000,
                models={m:registry[m]['config'] for m in registry if m.startswith('Method')})
    args.recipe.parent.mkdir(parents=True,exist_ok=True)
    args.recipe.write_text(json.dumps(recipe,indent=2))
    status=dict(completed_candidates=len(table),fixed_split_seed=2026,fixed_model_seed=1,
                selection_pairs=5980,source_plan_sha256=plan_hash,all_new_training_test_labels_poisoned=True,
                new_real_test_evaluated=False,statistical_claim='descriptive one-split, one-seed validation only',
                capacity_matched_within_each_proposed_method=True,
                equal_compute_claim=False,reason='Early stopping, target counts and OOF costs differ; raw concurrent wall times are descriptive.')
    (args.output/'STUDY_STATUS.json').write_text(json.dumps(status,indent=2))
    print(pd.DataFrame(changes).to_string(index=False))


if __name__=='__main__':main()
