"""Audit ZIP prediction arrays against reported metrics, then rerender honestly.

Does not train, tune, change ratings, or change any delivered predictions.
"""
import argparse,hashlib,io,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import numpy as np,pandas as pd
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figures-zip',type=Path,required=True);p.add_argument('--predictions-zip',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.figures_zip) as figures,zipfile.ZipFile(a.predictions_zip) as predictions:
        for archive in (figures,predictions):
            names=archive.namelist()
            if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP members')
            for name in names:
                if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe archive member')
        runs=pd.read_csv(io.BytesIO(figures.read('locked_test_runs.csv'))).fillna({'init_mode':'NA'})
        assignment=pd.read_csv(io.BytesIO(figures.read('split_assignments_2026.csv')))
        checks=[];cohorts={}
        for _,row in runs.iterrows():
            k='NA' if pd.isna(row.K) else str(int(row.K))
            for part in ('train','val','test'):
                name=f'predictions/{row.dataset}__{row.method}__split{int(row.split_seed)}__seed{int(row.model_seed)}__K{k}__{row.init_mode}__{part}.npz'
                raw=predictions.read(name)
                with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
                    pair,truth,prediction=[archive[key] for key in ('pair_id','truth','prediction')]
                    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():raise ValueError('Nonfinite array '+name)
                    if len(np.unique(pair))!=len(pair):raise ValueError('Duplicated evaluation pair '+name)
                    order=np.argsort(pair);signature=(pair[order].tobytes(),truth[order].tobytes())
                    if part in cohorts and signature!=cohorts[part]:raise ValueError('Methods evaluated on different cohorts')
                    cohorts[part]=signature
                    error=prediction.astype(np.float64)-truth
                    rmse=float(np.sqrt(np.mean(error**2)));reported=float(row[part+'_RMSE'])
                    if abs(rmse-reported)>2e-5:raise ValueError('Reported RMSE does not match archive: '+name)
                    checks.append(dict(method=row.method,partition=part,n=len(pair),recomputed_RMSE=rmse,reported_RMSE=reported,
                        difference=rmse-reported,archive=name,sha256=hashlib.sha256(raw).hexdigest()))
                target=a.output/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        for name in ('split_assignments_2026.csv','V4_ROLE_ASSIGNMENTS_2026.csv','V9_EPOCH_METRICS.csv'):
            if name in figures.namelist():(a.output/name).write_bytes(figures.read(name))
        (a.output/'dataset_manifest.json').write_text(json.dumps(dict(documented_range=[1,5],dataset='ml-100k',n_interactions=len(assignment))))
        pd.DataFrame(checks).to_csv(a.output/'prediction_integrity.csv',index=False)
        runs.to_csv(a.output/'locked_test_runs.csv',index=False)
        history=pd.read_csv(io.BytesIO(figures.read('V9_EPOCH_METRICS.csv')))
        rows=[]
        for _,row in runs.iterrows():
            group=history[(history.method==row.method)&(history.split_seed==row.split_seed)&(history.model_seed==row.model_seed)]
            if row.method.startswith('MBMF-'):group=group[group.init.fillna('NA')==row.init_mode]
            if pd.notna(row.K):group=group[group.K==row.K]
            kval='NA' if pd.isna(row.K) else str(int(row.K))
            identity=f'{row.method}|split={int(row.split_seed)}|seed={int(row.model_seed)}|K={kval}|init={row.init_mode}'
            rows.append((identity,group[['epoch','train_RMSE','val_RMSE']].to_dict(orient='records')))
        (a.output/'training_histories.json').write_text(json.dumps(dict(rows),indent=2))
        from v5.reporting import generate_report
        generate_report(a.output,runs,plots=True)
        summary=runs[['method','train_RMSE','val_RMSE','test_RMSE','best_epoch','train_wall_sec','PeakGPUAllocated_MB']].copy()
        summary.to_csv(a.output/'observed_v4_results.csv',index=False)
        report=dict(status='PASS',scope='Uploaded V4 output audit and V5 rendering only; no new model training',
            audited_prediction_archives=len(checks),max_reported_RMSE_difference=max(abs(r['difference']) for r in checks),
            source_zips={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.figures_zip,a.predictions_zip)},
            assignment_counts=assignment.split.value_counts().to_dict(),
            all_reported_GPU_peaks_zero=bool(runs.PeakGPUAllocated_MB.eq(0).all()),
            device_conclusion='Zero telemetry alone does not prove CPU execution; original environment/device records not attached',
            test_scope='Already observed exploratory TEST; never use these test values for V5 tuning')
        (a.output/'UPLOAD_AUDIT.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
