"""Bounded, validation-only V4/V5 pilot on a supplied split assignment.

This is diagnostic development on an already observed split, not a final paper
benchmark. TEST tensors are poisoned; all trials and failures are retained.
"""
import argparse,hashlib,json,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict,replace
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def worker(plan_path,job_id):
    import numpy as np,pandas as pd,torch
    from v9.engine import load_v8
    plan=json.loads(plan_path.read_text());job=next(j for j in plan['jobs'] if j['id']==job_id)
    if plan.get('source_directory'):sys.path.insert(0,plan['source_directory'])
    folder=plan_path.parent/job_id;folder.mkdir(exist_ok=True)
    if (folder/'result.json').is_file():return
    torch.set_num_threads(1);core,pub=load_v8();core.set_global_seed(1)
    frame=pd.read_csv(plan['assignment'])
    data,_=pub.build_publication_matrix_data_v8(frame,pub.DATASET_SPECS['ml-100k'],pub.PublicationDataConfig(dataset='ml-100k',force_backend='dense'),torch.device('cpu'))
    data.train_timestamps=torch.tensor(frame.loc[frame.split=='train','timestamp'].to_numpy(),dtype=torch.float64)
    data.test_ratings_raw[:]=torch.nan
    from v5.protocol import validation_roles
    from v5.config import StudyConfig,implementation_signature
    if implementation_signature()!=plan['source_sha256']:raise ValueError('Pilot source changed')
    data=validation_roles(data,StudyConfig(),2026)
    def evaluate(model,data,partition,**kw):
        if partition!='val':raise AssertionError('Pilot only permits validation')
        model.eval()
        with torch.no_grad():p=model.predict_raw_pairs(data,data.val_users,data.val_items)
        return dict(RMSE=float((p-data.val_ratings_raw).square().mean().sqrt()))
    evaluator=SimpleNamespace(MetricConfig=lambda **kw:None,evaluate_publication_model=evaluate)
    method=job['method'];started=time.perf_counter()
    if method.startswith('Method'):
        if job['profile']=='V4':
            from v4.config import StudyConfig as Config
            from v4.models import instantiate
            from v4.training import train_v4 as train
        else:
            from v5.config import StudyConfig as Config
            from v5.models import instantiate
            from v5.training import train_v5 as train
        cfg=replace(Config().models[method],max_epochs=plan['epochs'] if method!='Method5' else min(8,plan['epochs']),steps_per_epoch=2,
            warmup_steps=20,eval_every=1 if method=='Method5' else 5,val_patience=8,
            router_expert_epochs=10,router_epochs=30,shape_epochs=30,shape_patience=8,graph_probes=3,posterior_probes=2)
        if job.get('overrides'):cfg=replace(cfg,**job['overrides'])
        # Reuse exactly the same selection/calibration assignment in both
        # namespaces; V4's names are a serialization compatibility detail.
        if job['profile']=='V4':
            for name,value in list(vars(data).items()):
                if name.startswith('v5_'):setattr(data,name.replace('v5_','v4_',1),value)
        model=instantiate(data,cfg,1);setattr(model,'_'+job['profile'].lower()+'_artifact_dir',folder/'trace')
        info=train(evaluator,model,data,cfg,None)
    else:
        from v5.baselines import train_autorec,train_aemcne
        study=pub.PublicationExperimentConfigV8();study.data.dataset='ml-100k'
        cfg=study.autorec if method=='AutoRec' else study.aemcne
        cfg.max_epochs=300;cfg.eval_every=5;cfg.val_patience=20
        if method=='AutoRec':cfg.lambda_l2=job['penalty'];cfg.lr=.01
        else:cfg.lambda_main=cfg.lambda_element=job['penalty']
        model,cfg=pub.instantiate_model_v8(method,data,study,20,1,torch.device('cpu'))
        info=(train_autorec if method=='AutoRec' else train_aemcne)(evaluator,model,data,cfg,8192)
    model.eval()
    with torch.no_grad():prediction=model.predict_raw_pairs(data,data.val_users,data.val_items)
    result=dict(job=job,config=asdict(cfg),info=info,validation_RMSE=float((prediction-data.val_ratings_raw).square().mean().sqrt()),
        test_evaluated=False,test_labels_poisoned=bool(torch.isnan(data.test_ratings_raw).all()),
        wall_seconds=time.perf_counter()-started,source_sha256=plan['source_sha256'],
        scope='bounded diagnostic, one reused outer split and one model seed; not a paper reproduction')
    (folder/'result.json').write_text(json.dumps(result,indent=2,default=str))
    np.savez_compressed(folder/'validation.npz',pair_id=(data.val_users*data.num_items+data.val_items).numpy(),truth=data.val_ratings_raw.numpy(),prediction=prediction.numpy())
    print(job_id,result['validation_RMSE'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assignment',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--plan',type=Path);p.add_argument('--job');p.add_argument('--workers',type=int,default=2);p.add_argument('--epochs',type=int,default=40)
    a=p.parse_args()
    if a.job:worker(a.plan,a.job);return
    if a.plan is None:
        if a.assignment is None or a.output is None:p.error('Provide assignment and output, or an existing plan')
        from v5.config import implementation_signature
        a.output.mkdir(parents=True,exist_ok=True);a.plan=a.output/'PLAN.json'
        if a.plan.exists():raise ValueError('Plan already exists; resume with --plan')
        jobs=[dict(id=f'{profile}__Method{i}',method=f'Method{i}',profile=profile) for i in range(1,8) for profile in ('V4','V5')]
        jobs += [dict(id=f'{method}__l2_{penalty}',method=method,profile='baseline',penalty=penalty) for method,values in [('AutoRec',[.001,100.]),('AEMC-NE',[.1,100.])] for penalty in values]
        a.plan.write_text(json.dumps(dict(assignment=str(a.assignment.resolve()),assignment_sha256=hashlib.sha256(a.assignment.read_bytes()).hexdigest(),epochs=a.epochs,jobs=jobs,source_sha256=implementation_signature()),indent=2))
    plan=json.loads(a.plan.read_text())
    if hashlib.sha256(Path(plan['assignment']).read_bytes()).hexdigest()!=plan['assignment_sha256']:raise ValueError('Assignment changed')
    def execute(job):
        log=a.plan.parent/(job['id']+'.log')
        with log.open('a') as stream:result=subprocess.run([sys.executable,__file__,'--plan',str(a.plan.resolve()),'--job',job['id']],stdout=stream,stderr=subprocess.STDOUT)
        print(job['id'],'PASS' if result.returncode==0 else 'FAILED',flush=True)
        if result.returncode:
            (a.plan.parent/(job['id']+'.failure.json')).write_text(json.dumps(dict(job=job,exit_code=result.returncode,log=str(log),result_valid=False),indent=2))
        return result.returncode
    with ThreadPoolExecutor(max_workers=a.workers) as pool:failures=[f.result() for f in as_completed([pool.submit(execute,j) for j in plan['jobs']])]
    if any(failures):raise SystemExit('Pilot has declared failures; inspect logs')


if __name__=='__main__':main()
