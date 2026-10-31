"""The documented estimator must work without an untracked timing file."""
import math
from pathlib import Path
import pytest
from estimate_v8 import main


def test_default_estimator_loads_the_frozen_reference_and_counts_a_real_plan(capsys):
    report=main(['--profile','common','--methods','Method1,Method8',
                 '--search-jobs-per-method','1','--max-epochs','120',
                 '--steps-per-epoch','2','--baseline-max-epochs','200'])
    assert Path(report['calibration_file']).is_file()
    assert not report['user_supplied_calibration']
    stages={row['stage']:row for row in report['stages']}
    assert stages['hyperparameter_scan']['candidate_trainings']==2
    assert stages['final']['candidate_trainings']==2
    assert all(math.isfinite(row['reference_hours']) and row['reference_hours']>=0
               for row in report['stages'])


def test_estimator_rejects_a_negative_search_budget():
    with pytest.raises(SystemExit) as error:
        main(['--search-jobs-per-method','-1'])
    assert error.value.code==2
