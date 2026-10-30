"""Finished jobs may drop recovery; interrupted/report-failed jobs must keep it."""
import json
import pytest
from v7.retention import prune_completed_recovery


@pytest.mark.parametrize('state',['RUNNING','FAILED','REPORT_FAILED','COMPLETED'])
def test_only_completed_jobs_drop_all_nested_recovery(tmp_path,state):
    paths=[tmp_path/'mechanisms'/name/'recovery' for name in ('Method1','Method2/oof_fold_1')]
    for folder in paths:
        folder.mkdir(parents=True);(folder/'mean.pt').write_bytes(b'checkpoint')
    durable=tmp_path/'candidate_validation_runs.csv';durable.write_text('method,val_RMSE\nMethod1,.9\n')
    (tmp_path/'V9_JOB_STATUS.json').write_text(json.dumps(dict(status=state)))
    prune_completed_recovery(tmp_path)
    assert [folder.exists() for folder in paths]==[state!='COMPLETED']*2
    assert durable.is_file()
    prune_completed_recovery(tmp_path)
