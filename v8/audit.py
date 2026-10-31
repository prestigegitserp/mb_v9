"""V8 audit without fixed assumptions about an ablation count."""
from pathlib import Path
import json
import pandas as pd
from .readiness import build_checklist


def audit_selection(root):
    root=Path(root)
    report=build_checklist(root)
    errors=list(report['errors'])
    candidates=root/'search/hyperparameter_validation_all.csv'
    winners=root/'search/validation_winners.csv'
    if not candidates.is_file() or not winners.is_file():
        errors.append('Missing validation search; run scan before auditing a tuned study')
    else:
        a,w=pd.read_csv(candidates),pd.read_csv(winners)
        for row in w.itertuples():
            group=a[(a.method==row.method)&(a.split_seed==row.split_seed)&
                    (a.scan_job==row.scan_job)&(a.config_key==row.config_key)]
            if group.empty or abs(group.val_RMSE.mean()-row.val_mean)>1e-7:
                errors.append('Winner cannot be recovered from validation: '+row.method)
    if not report['complete_primary']:errors.append('Primary TEST is missing or incomplete')
    result=dict(status='FAIL' if errors else 'PASS',errors=errors,
        warnings=['A passing artifact audit is not a confirmatory statistical or novelty claim'],
        checks=report)
    (root/'V8_VALIDATION_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
