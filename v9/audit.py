"""Additional scientifically meaningful publication checks."""
from __future__ import annotations
import json,pathlib
import pandas as pd

def audit_selection(dataset_root):
    dataset_root=pathlib.Path(dataset_root)
    root=dataset_root/'search'
    valf=root/'hyperparameter_validation_all.csv'; winf=root/'validation_winners.csv'
    errs=[];checks={};warnings=[]
    if not valf.is_file():errs.append('missing validation candidate aggregate: run scan first')
    if not winf.is_file():errs.append('missing per-outer-split validation winners: run scan first')
    if valf.exists():
        a=pd.read_csv(valf)
        if any(x.startswith('test_') for x in a.columns):errs.append('TEST LEAK: hyperparameter table has test metrics')
        checks['candidate_rows']=len(a)
        if winf.exists():
            w=pd.read_csv(winf)
            for _,r in w.iterrows():
                g=a[(a.method==r.method)&(a.split_seed==r.split_seed)&(a.scan_job==r.scan_job)&(a.config_key==r.config_key)]
                if g.empty or abs(g.val_RMSE.mean()-r.val_mean)>1e-7:errs.append(f'{r.method} winner not recoverable from validation')
    finals=list((dataset_root/'final').glob('*/split_*/locked_test_runs.csv'))
    checks['final_method_split_jobs']=len(finals)
    if winf.exists():
        expected_jobs=len(pd.read_csv(winf))
        checks['expected_final_jobs']=expected_jobs
        if expected_jobs!=len(finals):errs.append(f'final job coverage incomplete {len(finals)} / {expected_jobs}')
    for f in finals:
        d=pd.read_csv(f)
        status=f.parent/'V9_JOB_STATUS.json'
        if not status.is_file() or json.loads(status.read_text()).get('status')!='COMPLETED':
            errs.append(f'incomplete final job (status missing or not COMPLETED): {f}')
        if d.split_seed.nunique()!=1:errs.append(f'final must contain exactly one split: {f}')
        if 'test_RMSE' not in d.columns or d.test_RMSE.isna().any():errs.append(f'final missing test RMSE: {f}')
        if d.K.notna().any() and d.K.nunique()>1:errs.append(f'final mixes multiple K values within the SAME outer split: {f}')
        if d.init_mode.nunique()>1:errs.append(f'final mixes initializations: {f}')
    for prefix in ('ablations','ablations_matched_K_20'):
        study=dataset_root/prefix
        if study.exists():
            jobs=[x for x in study.iterdir() if x.is_dir() and x.name.startswith('m')]
            checks[prefix+'_variants']=len(jobs)
            if len(jobs)!=38:errs.append(f'{prefix}: incomplete 38 ablation variants: {len(jobs)}')
    out={'status':'PASS' if not errs else 'FAIL','errors':errs,'warnings':warnings,'checks':checks}
    (dataset_root/'v9_validation_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    return out
