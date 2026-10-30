"""Read-only audit of the two supplied archives and the frozen V6 parent."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def audit(pilot,method10):
    rows=[]
    with zipfile.ZipFile(pilot) as archive:
        if archive.testzip() is not None:raise ValueError('Pilot ZIP CRC failed')
        names=archive.namelist()
        csv=next(name for name in names if name.endswith('/locked_test_runs.csv'))
        table=pd.read_csv(io.BytesIO(archive.read(csv)))
        for _,row in table.iterrows():
            filename=str(row.test_prediction_file).replace('\\','/').rsplit('/',1)[-1]
            member=next(name for name in names if name.endswith('/predictions/'+filename))
            with np.load(io.BytesIO(archive.read(member)),allow_pickle=False) as arrays:
                actual=float(np.sqrt(np.mean((arrays['truth'].astype(float)-arrays['prediction'].astype(float))**2)))
            difference=abs(actual-float(row.test_RMSE))
            if difference>1e-12:raise ValueError('Pilot RMSE mismatch: '+str(row.method))
            rows.append(dict(method=row.method,reported_test_RMSE=row.test_RMSE,
                recomputed_test_RMSE=actual,absolute_difference=difference))
        devices=[json.loads(archive.read(name)) for name in names if name.endswith('/DEVICE_AND_STORAGE.json')]
        pilot_info=dict(sha256=sha256(pilot),bytes=pilot.stat().st_size,zip_entries=len(names),crc='PASS',
            selected_rows=len(rows),max_rmse_difference=max(row['absolute_difference'] for row in rows),
            gpu_names=sorted({record['device'] for record in devices}),resource_records=len(devices),
            all_actual_cuda0=all(record['placement']['actual_device']=='cuda:0' for record in devices),
            cpu_fallback_count=sum(record['placement'].get('fallback') is not None for record in devices),
            peak_allocated_bytes=max(record['peak_allocated_bytes'] for record in devices),
            peak_reserved_bytes=max(record['peak_reserved_bytes'] for record in devices),metrics=rows)
    with zipfile.ZipFile(method10) as archive:
        if archive.testzip() is not None:raise ValueError('Method10 ZIP CRC failed')
        member='MBMF-SI2/v9/method10.py';source=archive.read(member)
        if source!=(ROOT/'vendor_method10/method10.py').read_bytes():raise ValueError('Method10 was altered')
        model_info=dict(archive_sha256=sha256(method10),bytes=method10.stat().st_size,
            zip_entries=len(archive.namelist()),crc='PASS',member=member,member_bytes=len(source),
            member_sha256=hashlib.sha256(source).hexdigest(),vendor_bytes_identical=True)
    protected=json.loads((ROOT/'V7_BASELINE_MANIFEST.json').read_text())
    for name,expected in protected['protected_parent_blobs'].items():
        data=(ROOT/name).read_bytes()
        variants=[data,data.replace(b'\r\n',b'\n')]
        if not any(hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest()==expected for value in variants):
            raise ValueError('Frozen parent changed: '+name)
    return dict(pilot=pilot_info,method10=model_info,parent_commit=protected['parent_commit'],
        frozen_parent_files_verified=len(protected['protected_parent_blobs']),
        archived_test_audited_only=True,archived_test_used_for_v7_selection=False,
        new_gpu_benchmark=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--method10',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    result=audit(args.pilot,args.method10)
    value=json.dumps(result,indent=2)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(value+'\n')
    print(value)


if __name__=='__main__':main()
