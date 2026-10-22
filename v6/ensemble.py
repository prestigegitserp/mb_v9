"""A deliberately simple strong null; components stay separately reported.

Fit one convex weight on validation per OUTER split, before reading any TEST
archive. The result is an ensemble control, never a renamed proposed mechanism.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd


def aligned(path):
    with np.load(path,allow_pickle=False) as z:
        order=np.argsort(z['pair_id'],kind='stable')
        result={key:z[key][order] for key in ('pair_id','truth','prediction')}
    if not all(np.isfinite(result[key]).all() for key in ('truth','prediction')):raise ValueError('Nonfinite ensemble inputs')
    if len(np.unique(result['pair_id']))!=len(result['pair_id']):raise ValueError('Duplicated ensemble pairs')
    return result


def paired(left,right):
    if not np.array_equal(left['pair_id'],right['pair_id']) or not np.array_equal(left['truth'],right['truth']):
        raise ValueError('Ensemble cohorts do not match')


def convex_weight(first,second,truth):
    first,second,truth=[np.asarray(v,dtype=np.float64) for v in (first,second,truth)]
    delta=second-first;denominator=float(delta@delta)
    return float(np.clip(((truth-first)@delta)/denominator,0.,1.)) if denominator else 0.


def fit_validation_blend(root,table,winners,methods):
    if len(methods)!=2 or len(set(methods))!=2:raise ValueError('Two distinct predeclared blend components required')
    if any(str(c).startswith('test_') for c in table):raise ValueError('No TEST columns allowed in blend fitting')
    root=Path(root);folder=root/'ensemble';folder.mkdir(parents=True,exist_ok=True)
    keys=['dataset','method','split_seed','scan_job','config_key']
    selected=table.merge(winners[keys],on=keys,validate='many_to_one')
    records=[]
    for seed,group in selected.groupby('split_seed'):
        a=group[group.method==methods[0]];b=group[group.method==methods[1]]
        if a.empty or b.empty:continue
        pairs=a.merge(b,on=['dataset','split_seed','model_seed'],suffixes=('_first','_second'),validate='one_to_one')
        if len(pairs)!=len(a) or len(pairs)!=len(b):raise ValueError('Incomplete ensemble replicate pairing')
        first,second,truth,sources=[],[],[],[]
        for _,row in pairs.iterrows():
            x,y=[aligned(row['val_prediction_file_'+part]) for part in ('first','second')];paired(x,y)
            first.append(x['prediction']);second.append(y['prediction']);truth.append(x['truth'])
            components=[]
            for method,part in zip(methods,('first','second')):
                file=Path(row['val_prediction_file_'+part]).resolve()
                components.append(dict(method=method,config_key=row['config_key_'+part],
                    validation_file=str(file.relative_to(root.resolve())) if file.is_relative_to(root.resolve()) else str(file),
                    sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
            sources.append(dict(model_seed=int(row.model_seed),components=components))
        first,second,truth=map(np.concatenate,(first,second,truth));weight=convex_weight(first,second,truth)
        pred=(1-weight)*first.astype(float)+weight*second.astype(float)
        records.append(dict(split_seed=int(seed),second_weight=weight,validation_RMSE=float(np.sqrt(np.mean((pred-truth)**2))),
                            sources=sources))
    if not records:raise ValueError('Blend components absent from frozen validation winners')
    payload=dict(format='V6-validation-ensemble-null',methods=list(methods),weight_selection='one convex coefficient per outer split; pooled validation MSE across identical model seeds',
        test_read_during_fit=False,novelty_claim=False,cost='two trained component models plus one fitted scalar per split',splits=records)
    (folder/'VALIDATION_WEIGHTS.json').write_text(json.dumps(payload,indent=2));return payload


def apply_locked_blend(root,runs,methods,*,plots=True):
    from .reporting import generate_report
    root=Path(root);path=root/'search'/'ensemble'/'VALIDATION_WEIGHTS.json'
    if not path.is_file():raise ValueError('Freeze blend weights with scan --scan-only --include-blend first')
    payload=json.loads(path.read_text())
    if list(methods)!=payload['methods'] or payload['test_read_during_fit'] is not False:raise ValueError('Blend provenance mismatch')
    out=root/'ensemble';out.mkdir(parents=True,exist_ok=True);rows=[]
    for entry in payload['splits']:
        seed=entry['split_seed'];weight=entry['second_weight']
        group=runs[runs.split_seed==seed];first=group[group.method==methods[0]];second=group[group.method==methods[1]]
        pairs=first.merge(second,on=['dataset','split_seed','model_seed'],suffixes=('_first','_second'),validate='one_to_one')
        if len(pairs)!=len(entry['sources']):raise ValueError('Incomplete locked ensemble replicas')
        for _,row in pairs.iterrows():
            origin=next(s for s in entry['sources'] if s['model_seed']==int(row.model_seed))
            # Check final refits against the frozen validation configuration and
            # predictions BEFORE opening test files.
            for part,source in zip(('first','second'),origin['components']):
                if json.loads(row['config_key_'+part])!=json.loads(source['config_key']):raise ValueError('Ensemble component config changed')
                file=Path(source['validation_file'])
                if not file.is_absolute():file=root/'search'/file
                if hashlib.sha256(file.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('Frozen ensemble validation archive changed')
                frozen=aligned(file);current=aligned(row['val_prediction_file_'+part]);paired(frozen,current)
                if not np.allclose(frozen['prediction'],current['prediction'],atol=2e-5,rtol=0):raise ValueError('Ensemble refit differs from frozen validation predictions')
            name='Blend['+'+'.join(methods)+']';identity=f"{row.dataset}__{name}__split{seed}__seed{int(row.model_seed)}"
            record=dict(dataset=row.dataset,method=name,split_seed=seed,model_seed=int(row.model_seed),K=np.nan,init_mode='NA',
                artifact_root=str(out.resolve()),second_weight=weight,config_key=json.dumps(dict(method=name,config=dict(second_weight=weight,components=[s['config_key'] for s in origin['components']]))),
                parameter_count=int(row.parameter_count_first+row.parameter_count_second),
                train_wall_sec=float(row.train_wall_sec_first+row.train_wall_sec_second),novelty_claim=False)
            for partition in ('train','val','test'):
                x,y=[aligned(row[partition+'_prediction_file_'+part]) for part in ('first','second')];paired(x,y)
                pred=((1-weight)*x['prediction'].astype(float)+weight*y['prediction'].astype(float)).astype(np.float32)
                target=out/'predictions'/(identity+'__'+partition+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target,pair_id=x['pair_id'],truth=x['truth'],prediction=pred)
                record[partition+'_prediction_file']=str(target.resolve());error=pred.astype(float)-x['truth']
                record[partition+'_RMSE']=float(np.sqrt(np.mean(error**2)));record[partition+'_MAE']=float(np.mean(abs(error)))
            rows.append(record)
    result=pd.DataFrame(rows);result.to_csv(out/'ensemble_locked_test_runs.csv',index=False)
    (out/'ENSEMBLE_PROTOCOL.json').write_text(json.dumps(payload,indent=2))
    generate_report(out,result,plots=plots);return result
