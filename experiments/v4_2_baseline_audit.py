"""Blinded competitor optimization audit; keep every declared trial."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(assignment,output):
    from v4.config import implementation_signature
    output.mkdir(parents=True,exist_ok=True);path=output/'PLAN.json'
    if path.exists():raise ValueError('Existing frozen plan')
    jobs=[]
    for method in ('MBMF-F','MBMF-C','MBMF-N'):
        for optimizer,lr,epochs in [('paper',.003,5000),('adam',.003,2000),('adam',.01,2000)]:
            jobs.append(dict(id=method+'__'+optimizer+'__'+str(lr),method=method,
                config=dict(optimizer=optimizer,adam_lr=lr,max_epochs=epochs),seed=1,init='isotropic_gaussian'))
    jobs.append(dict(id='MF__audited',method='MF',config=dict(max_epochs=1000),seed=1,init='NA'))
    for value in (10.,100.,1000.):
        jobs.append(dict(id='AutoRec__'+str(value),method='AutoRec',config=dict(lambda_l2=value,lr=.01),seed=1,init='NA'))
    for main,element in ((.1,.1),(1.,1.),(10.,10.),(100.,100.),(100.,1.),(1.,100.)):
        jobs.append(dict(id='AEMC-NE__'+str(main)+'__'+str(element),method='AEMC-NE',
                         config=dict(lambda_main=main,lambda_element=element),seed=1,init='NA'))
    path.write_text(json.dumps(dict(assignment=str(assignment.resolve()),assignment_sha256=sha(assignment),
        implementation_sha256=implementation_signature(),runner_sha256=sha(__file__),jobs=jobs,
        protocol='V4 common selection; TEST labels NaN before construction; no test evaluation',
        limitation='one split, one seed; unequal family-specific tuning budgets'),indent=2));return path


def worker(plan_path,job_id):
    import numpy as np,pandas as pd,torch
    from v4.config import StudyConfig,implementation_signature
    from v4.protocol import validation_roles
    from v4.baselines import paper_config,train_mbmf,train_autorec,train_biasmf,train_aemcne
    from v9.engine import load_v8
    plan=json.loads(plan_path.read_text());job=next(j for j in plan['jobs'] if j['id']==job_id)
    if plan['implementation_sha256']!=implementation_signature() or plan['runner_sha256']!=sha(__file__):raise ValueError('Frozen source mismatch')
    if plan['assignment_sha256']!=sha(plan['assignment']):raise ValueError('Frozen data mismatch')
    out=plan_path.parent/job_id;out.mkdir(exist_ok=True)
    if (out/'result.json').exists():
        old=json.loads((out/'result.json').read_text())
        if old['plan_sha256']!=sha(plan_path):raise ValueError('Mismatched resume')
        return
    core,pub=load_v8();torch.set_num_threads(1)
    frame=pd.read_csv(plan['assignment']);data,_=pub.build_publication_matrix_data_v8(frame,pub.DATASET_SPECS['ml-100k'],pub.PublicationDataConfig(dataset='ml-100k',force_backend='dense'),torch.device('cpu'))
    data.train_timestamps=torch.tensor(frame.loc[frame.split=='train','timestamp'].to_numpy(),dtype=torch.float64)
    data.test_ratings_raw[:]=torch.nan;data=validation_roles(data,StudyConfig(),2026)
    def evaluate(model,data,split,**kw):
        if split!='val':raise AssertionError('Only selection validation is accessible')
        with torch.no_grad():
            model.eval();p=model.predict_raw_pairs(data.val_users,data.val_items) if isinstance(model,core.PaperMBMF) else model.predict_raw_pairs(data,data.val_users,data.val_items)
            return dict(RMSE=float((p-data.val_ratings_raw).square().mean().sqrt()))
    evaluator=SimpleNamespace(MetricConfig=lambda **kw:None,evaluate_publication_model=evaluate)
    study=pub.PublicationExperimentConfigV8();method=job['method'];values=job['config']
    if method.startswith('MBMF-'):study.core_config.paper.max_epochs=values['max_epochs']
    else:
        obj=study.autorec if method=='AutoRec' else study.aemcne if method=='AEMC-NE' else study.biasmf
        for key,value in values.items():setattr(obj,key,value)
    model,cfg=pub.instantiate_model_v8(method,data,study,20,job['seed'],torch.device('cpu'),None if job['init']=='NA' else job['init'])
    start=time.perf_counter()
    if method.startswith('MBMF-'):
        cfg=paper_config(cfg,optimizer=values['optimizer'],adam_lr=values['adam_lr']);info=train_mbmf(evaluator,model,data,cfg)
    elif method=='AutoRec':info=train_autorec(evaluator,model,data,cfg)
    elif method=='AEMC-NE':info=train_aemcne(evaluator,model,data,cfg)
    else:info=train_biasmf(evaluator,model,data,cfg)
    with torch.no_grad():
        pred=model.predict_raw_pairs(data.val_users,data.val_items) if method.startswith('MBMF-') else model.predict_raw_pairs(data,data.val_users,data.val_items)
    np.savez_compressed(out/'validation.npz',pair_id=(data.val_users*data.num_items+data.val_items).numpy(),truth=data.val_ratings_raw.numpy(),prediction=pred.numpy())
    pub.save_model_checkpoint_v8(out/'checkpoint.pt',model,config=cfg,data=data,train_info=info,extra=dict(method=method,init_mode=job['init']))
    if plan['implementation_sha256']!=implementation_signature():raise ValueError('Source changed during training')
    result=dict(job=job,plan_sha256=sha(plan_path),actual_config=asdict(cfg),info=info,validation_RMSE=float((pred-data.val_ratings_raw).square().mean().sqrt()),
        parameter_count=sum(p.numel() for p in model.parameters()),wall_seconds=time.perf_counter()-start,protocol=data.v4_protocol,
        checkpoint_sha256=sha(out/'checkpoint.pt'),test_evaluated=False,test_labels_poisoned=True)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(job_id,result['validation_RMSE'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assignment',type=Path);p.add_argument('--output',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--job');p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    if a.assignment:
        if a.output is None:p.error('Provide --output')
        print(prepare(a.assignment,a.output));return
    if a.plan is None:p.error('Provide --plan')
    if a.job:worker(a.plan,a.job);return
    if a.workers<1:p.error('Positive --workers required')
    jobs=json.loads(a.plan.read_text())['jobs']
    def execute(job):
        with (a.plan.parent/(job['id']+'.log')).open('a') as log:
            result=subprocess.run([sys.executable,__file__,'--plan',str(a.plan.resolve()),'--job',job['id']],stdout=log,stderr=subprocess.STDOUT)
        print(job['id'],'PASS' if result.returncode==0 else 'FAILED',flush=True);return result.returncode
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        failures=[f.result() for f in as_completed([pool.submit(execute,j) for j in jobs])]
    if any(failures):raise SystemExit('Incomplete baseline audit; inspect failed logs')
if __name__=='__main__':main()
