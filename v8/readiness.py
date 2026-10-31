"""Artifact completeness is checked separately from scientific adequacy."""
import csv
import json
import math
import hashlib
import re
from pathlib import Path


def records(path):
    if not path.is_file():return []
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))


def build_checklist(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    winners=records(root/'search/validation_winners.csv')
    candidate_rows=records(root/'search/hyperparameter_validation_all.csv')
    final_file=root/'final_locked_test_all.csv'
    fixed_main=False
    if not final_file.is_file() and not winners and (root/'main/locked_test_runs.csv').is_file():
        final_file=root/'main/locked_test_runs.csv';fixed_main=True
    errors=[]
    final=records(final_file)
    if winners and not final_file.is_file():
        children=list((root/'final').glob('*/split_*/locked_test_runs.csv'))
        if children:
            final=[r for path in children for r in records(path)]
            errors.append('Combined final table missing; available child TEST jobs are partial')
    expected={(r['method'],r['split_seed']) for r in winners}
    actual={(r['method'],r['split_seed']) for r in final}
    if any(k.startswith('test_') for r in candidate_rows for k in r):
        errors.append('TEST columns in validation-search table')
    if expected and actual!=expected:
        errors.append(f'Final method/split coverage incomplete: {len(actual)}/{len(expected)}')
    if final and any(not math.isfinite(float(r.get('test_RMSE','nan'))) for r in final):
        errors.append('Nonfinite or missing final TEST RMSE')
    for winner in winners:
        matched=[r for r in final if (r['method'],r['split_seed'])==(winner['method'],winner['split_seed'])]
        if not matched:continue
        expected_n=int(float(winner.get('n') or 1))
        if len(matched)!=expected_n:errors.append('Final model-seed coverage mismatch: '+winner['method'])
        if any(json.loads(r['config_key'])!=json.loads(winner['config_key']) for r in matched):
            errors.append('Final configuration differs from winner: '+winner['method'])
        folder=root/'final'/winner['method']/('split_'+winner['split_seed'])
        state=folder/'V8_JOB_STATUS.json'
        if not state.is_file() or json.loads(state.read_text()).get('status')!='COMPLETED':
            errors.append('Incomplete final job state: '+winner['method'])
        elif json.loads(state.read_text()).get('results_sha256'):
            expected_hash=json.loads(state.read_text())['results_sha256']
            if hashlib.sha256((folder/'locked_test_runs.csv').read_bytes()).hexdigest()!=expected_hash:
                errors.append('Final result checksum mismatch: '+winner['method'])
        tag=re.sub('[^a-zA-Z0-9_.=-]','_',winner['scan_job'])
        reference=root/'search/jobs'/tag/('V8_PROTOCOL_'+winner['split_seed']+'.json')
        protocol=folder/('V8_PROTOCOL_'+winner['split_seed']+'.json')
        if not reference.is_file() or not protocol.is_file():
            errors.append('Missing final/search data provenance: '+winner['method'])
        elif json.loads(reference.read_text())['data_signature']!=json.loads(protocol.read_text())['data_signature']:
            errors.append('Final/search cohort mismatch: '+winner['method'])
    primary_root=root/'main' if fixed_main else root
    report=primary_root/'publication'
    figure=report/'figures/selected_test_rmse.png'
    workbook=report/'research_tables.xlsx'
    modes=[json.loads(p.read_text(encoding='utf-8')).get('quick',False) for p in root.rglob('V8_STUDY_MODE.json')]
    smoke=any(modes)
    splits=len({r['split_seed'] for r in final})
    stages={
        'validation_search':bool(winners),
        'locked_test':bool(final) and not errors,
        'test_figures':figure.is_file(),
        'research_workbook':workbook.is_file(),
        'trained_ablations':(root/'ablations/ablation_locked_test_runs.csv').is_file(),
        'matched_k_ablations':any(root.glob('ablations_matched_K_*/ablation_locked_test_runs.csv')),
        'loss_sensitivity_validation':(root/'lossstudy/loss_coefficient_validation.csv').is_file(),
        'fixed_k_sensitivity':(root/'kstudy/all_fixed_k_locked_runs.csv').is_file(),
        'sparsity':(root/'sparsity/SPARSITY_PROTOCOL.json').is_file(),
        'convergence':(root/'convergence/CONVERGENCE_PROTOCOL.json').is_file(),
        'calibration':(root/'calibration/CALIBRATION_PROTOCOL.json').is_file(),
        'temporal':(root/'temporal/final_locked_test_all.csv').is_file(),
    }
    if not final:status='TEST_PENDING'
    elif errors:status='INCOMPLETE_OR_INVALID'
    elif smoke:status='SMOKE_COMPLETE_NOT_MANUSCRIPT_EVIDENCE'
    elif splits<2:status='PRIMARY_COMPLETE_SINGLE_SPLIT_DESCRIPTIVE'
    else:status='PRIMARY_COMPLETE_SCIENTIFIC_REVIEW_REQUIRED'
    result=dict(status=status,primary_partition='test' if final else 'val' if winners else None,
        expected_final_method_splits=len(expected),completed_final_method_splits=len(actual),
        final_rows=len(final),outer_splits=splits,smoke=smoke,fixed_default_pilot=fixed_main,
        complete_primary=bool(final) and not errors,errors=errors,stages=stages,
        primary_test_table=str(final_file) if final else None,
        test_figure_directory=str(report/'figures'),workbook=str(workbook),
        journal_acceptance_or_sota_certified=False,
        note='Completeness only. Baseline strength, novelty, statistical dependence and scientific claims need review.')
    (root/'MANUSCRIPT_CHECKLIST.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# وضعیت خروجی‌های پژوهش نسخه۸','',f'وضعیت: `{status}`','',
        'این فهرست کامل‌بودن فایل‌ها را بررسی می‌کند؛ تأیید نوآوری، پیروزی بر رقبا یا کفایت داوری ژورنال نیست.','',
        f'- تعداد ردیف TEST: {len(final)}',f'- تعداد split داده: {splits}',
        f'- جدول اصلی: `{final_file}`',f'- نمودارهای TEST: `{report / "figures"}`',
        f'- فایل Excel: `{workbook}`','',
        '| بخش | موجود است؟ |','|---|---|']
    lines += [f'| {name} | {"بله" if exists else "خیر / هنوز اجرا نشده"} |' for name,exists in stages.items()]
    lines += ['', 'شکل‌های search متعلق به VAL هستند. شکل selected_test_rmse و جدول اصلی فقط پس از اجرای موفق TEST ساخته می‌شوند.',
        'یک split شاهد توصیفی است. برای ادعای مقاله، پروتکل ازپیش‌مشخص، baselineهای تنظیم‌شده، تکرار و تحلیل سهم سازوکار لازم‌اند.']
    if errors:lines += ['', 'خطاهای کنترل:']+['- '+e for e in errors]
    (root/'MANUSCRIPT_CHECKLIST.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
