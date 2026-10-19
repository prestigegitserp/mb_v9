"""Recompute blinded validation evidence and retain all declared negative arms."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--require-complete',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 groups=['v4_2_runs','v4_2_baselines','v4_2_regularized','v4_2_m7_corrected','v4_2_factorial','v4_2_m6_factorial','v4_2_capacity']
 ledger_path=a.output/'DECLARED_FAILURES.json'
 ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
 rows=[];pending=[];failures=[];cohort=None;sources={}
 def record(directory,study,role=None):
  nonlocal cohort
  file=directory/'result.json'
  if not file.is_file():
   key=directory.parent.name+'/'+directory.name
   if key in ledger:failures.append(dict(trial=key,**ledger[key]))
   else:pending.append(str(file))
   return
  d=json.loads(file.read_text());job=d.get('job',d.get('plan',{}));cfg=d['actual_config'];method=job.get('method',cfg.get('method'))
  if d.get('test_evaluated') is not False or d.get('test_labels_poisoned') is not True:raise ValueError('Unblinded evidence: '+str(file))
  checkpoint=directory/'checkpoint.pt'
  if sha(checkpoint)!=d['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch '+str(checkpoint))
  archive=directory/'validation.npz'
  with np.load(archive,allow_pickle=False) as z:
   order=np.argsort(z['pair_id']);pair,truth,pred=[z[k][order] for k in ('pair_id','truth','prediction')]
  if not np.isfinite(pred).all() or not np.isfinite(truth).all():raise ValueError('Nonfinite archive')
  if cohort is None:cohort=(pair,truth)
  np.testing.assert_array_equal(cohort[0],pair);np.testing.assert_array_equal(cohort[1],truth)
  error=pred.astype(float)-truth;clipped_error=np.clip(pred.astype(float),1.,5.)-truth
  rmse=float(np.sqrt(np.mean(error**2)));arm=job.get('arm',directory.name)
  info=d['info'];last=info.get('last_epoch',info.get('final_epoch'))
  role=role or ('null' if 'null' in directory.name else 'full')
  name=study+'/'+directory.name
  row=dict(trial=name,study=study,method=method,role=role,arm=arm,validation_RMSE=rmse,validation_MAE=float(np.mean(abs(error))),
    validation_RMSE_clipped_1_5=float(np.sqrt(np.mean(clipped_error**2))),validation_MAE_clipped_1_5=float(np.mean(abs(clipped_error))),
    predictions_outside_documented_range=float(np.mean((pred<1.)|(pred>5.))),selection_metric='raw validation RMSE; uniform clipping also reported',
    validation_RMSE_recorded=d['validation_RMSE'],validation_pairs=len(pair),parameter_count=d['parameter_count'],best_epoch=info.get('best_epoch'),last_epoch=last,
    max_epochs=cfg['max_epochs'],mean_optimizer_steps=info.get('mean_optimizer_steps',info.get('optimizer_steps')),
    prediction_targets_seen=info.get('prediction_targets_seen'),wall_seconds=d['wall_seconds'],K=cfg.get('latent_dim'),hidden_dim=cfg.get('hidden_dim',cfg.get('main_hidden_dim',cfg.get('hidden_dim'))),
    readout_mode=cfg.get('readout_mode','geometric' if method.startswith('Method') else ''),collaborative_l2=cfg.get('collaborative_l2',0.),
    moment_refinement_link=cfg.get('moment_refinement_link','tanh'),moment_refinement_scale=cfg.get('moment_refinement_scale',.25),
    optimizer=info.get('optimizer',cfg.get('optimizer','')),best_checkpoint_at_budget=info.get('best_checkpoint_at_budget',info.get('best_epoch')==cfg['max_epochs']),
    prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),config_json=json.dumps(cfg,sort_keys=True),test_evaluated=False)
  if abs(rmse-d['validation_RMSE'])>2e-6:raise ValueError('Metric/NPZ mismatch '+name)
  rows.append(row);sources[name]=dict(method=method,result=str(file.resolve()),checkpoint=str(checkpoint.resolve()),validation=str(archive.resolve()))
 for group in groups:
  path=a.workspace/group;plan=json.loads((path/'PLAN.json').read_text())
  for job in plan['jobs']:
   if group=='v4_2_runs' and job['method']=='Method7':continue # original probability-state bug, replaced by corrected frozen study
   record(path/job['id'],group)
 fresh_count=len(rows)
 controls={f'Method{i}':a.workspace/'review_runs'/f'Method{i}__density' for i in (1,2,3,5,7)}
 controls['Method4']=a.workspace/'v4_2_runs'/'Method4__control';controls['Method6']=a.workspace/'review_runs'/'Method6__decoder_context'
 for method,path in controls.items():record(path,'v4_1_selected_reference')
 for method,path in [('BiasMF',a.workspace/'review_runs'/'BiasMF__control'),('MF',a.workspace/'mf_budget_0.0001'),('MBMF-F',a.workspace/'mbmf_budget_F'),('MBMF-C',a.workspace/'mbmf_budget_C'),('MBMF-N',a.workspace/'mbmf_budget_N')]:record(path,'previous_competitor_reference')
 frame=pd.DataFrame(rows);frame.to_csv(a.output/'validation_trials.csv',index=False,float_format='%.10g')
 selected=[]
 for method,group in frame[frame.role=='full'].groupby('method'):
  best=group.sort_values(['validation_RMSE','trial'],kind='stable').iloc[0]
  if method.startswith('Method'):
   prior=frame[(frame.method==method)&(frame.study=='v4_1_selected_reference')].iloc[0]
   selected.append(dict(method=method,prior_validation_RMSE=prior.validation_RMSE,validation_RMSE=best.validation_RMSE,
      relative_improvement_percent=100*(prior.validation_RMSE-best.validation_RMSE)/prior.validation_RMSE,selected_trial=best.trial,
      prior_parameters=prior.parameter_count,selected_parameters=best.parameter_count))
 winners=frame[frame.role=='full'].sort_values(['validation_RMSE','trial']).drop_duplicates('method')
 winners.to_csv(a.output/'selected_validation_models.csv',index=False,float_format='%.10g')
 pd.DataFrame(selected).to_csv(a.output/'seven_model_comparison.csv',index=False,float_format='%.10g')
 manifest=dict(entries=[sources[row.trial] for _,row in winners.iterrows()])
 (a.output/'WINNER_SOURCES_WORKSPACE.json').write_text(json.dumps(manifest,indent=2))
 # Strong reductive null: two predeclared families, both selected on validation.
 auto=winners[winners.method=='AutoRec'].iloc[0];m4=winners[winners.method=='Method4'].iloc[0]
 first,second=[dict(np.load(sources[r.trial]['validation'])) for r in (auto,m4)]
 for z in (first,second):
  order=np.argsort(z['pair_id']);z.update({key:value[order] for key,value in z.items()})
 np.testing.assert_array_equal(first['pair_id'],second['pair_id']);np.testing.assert_array_equal(first['truth'],second['truth'])
 truth=first['truth'].astype(float);x=first['prediction'].astype(float);delta=second['prediction'].astype(float)-x
 weight=float(np.clip((truth-x)@delta/(delta@delta),0,1));prediction=x+weight*delta
 blend=dict(method='Blend[AutoRec+Method4]',second_weight=weight,validation_RMSE=float(np.sqrt(np.mean((prediction-truth)**2))),
      validation_RMSE_clipped_1_5=float(np.sqrt(np.mean((np.clip(prediction,1.,5.)-truth)**2))),
      components=[auto.trial,m4.trial],test_evaluated=False,novelty_claim=False,validation_selection='exploratory one-scalar convex MSE fit; same validation also used for component settings')
 (a.output/'ENSEMBLE_NULL.json').write_text(json.dumps(blend,indent=2))
 np.savez_compressed(a.output/'ensemble_validation.npz',pair_id=first['pair_id'],truth=truth,prediction=prediction.astype(np.float32))
 # Preserve the earlier H=500 ensemble control even when H=146 becomes the
 # separately selected AutoRec. These one-scalar fits are exploratory, not TEST.
 comparisons=[dict(AutoRec_trial=auto.trial,Method4_trial=m4.trial,Method4_weight=weight,
   validation_RMSE=blend['validation_RMSE'],primary_family_winner_rule=True,test_evaluated=False)]
 alternative=frame[(frame.study=='v4_2_baselines')&(frame.method=='AutoRec')].sort_values('validation_RMSE').iloc[0]
 with np.load(sources[alternative.trial]['validation'],allow_pickle=False) as z:
  idx=np.argsort(z['pair_id']);np.testing.assert_array_equal(first['pair_id'],z['pair_id'][idx]);np.testing.assert_array_equal(truth,z['truth'][idx])
  other=z['prediction'][idx].astype(float)
 change=second['prediction'].astype(float)-other;alternative_weight=float(np.clip((truth-other)@change/(change@change),0,1))
 comparisons.append(dict(AutoRec_trial=alternative.trial,Method4_trial=m4.trial,Method4_weight=alternative_weight,
   validation_RMSE=float(np.sqrt(np.mean((other+alternative_weight*change-truth)**2))),primary_family_winner_rule=False,test_evaluated=False))
 pd.DataFrame(comparisons).to_csv(a.output/'ensemble_null_comparisons.csv',index=False,float_format='%.10g')
 pd.DataFrame(failures,columns=['trial','method','role','status','last_flushed_epoch','last_flushed_optimizer_step','cause','reason','test_evaluated']).to_csv(a.output/'failed_training_attempts.csv',index=False)
 status=dict(fresh_completed_trainings=fresh_count,planned_canonical_trainings=55,declared_failed_or_censored=len(failures),pending=pending,
       matched_validation_pairs=len(cohort[0]),split_seed=2026,model_seed=1,test_evaluated=False,
       historical_controls_are_not_new_repetitions=True,original_m7_failures='replaced by corrected same-config runs; failure logs retained')
 (a.output/'STUDY_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));print(pd.DataFrame(selected).to_string(index=False))
 if pending and a.require_complete:raise SystemExit('Incomplete declared studies')
if __name__=='__main__':main()
