"""Leakage-safe per-outer-split validation selection.

CRITICAL: selecting a single hyperparameter globally from ALL random splits can
leak the test labels of split A via validation of split B. We avoid this by
choosing the hyperparameters of split A ONLY from its own validation block.
"""
from __future__ import annotations
import dataclasses, hashlib, json, math
import pandas as pd

REPLICATE_KEY=('dataset','method','split_seed','scan_job','config_key')


def select_validation_winners(table:pd.DataFrame,expected_splits:int,expected_seeds:int,metric:str='val_RMSE'):
    if table.empty:raise ValueError('No completed validation candidates')
    if any(str(c).startswith('test_') for c in table.columns):
        raise ValueError('TEST LEAKAGE: test columns forbidden in hyperparameter selection')
    if not metric.startswith('val_'):raise ValueError('Selection criterion must be VALIDATION only')
    cols=[*REPLICATE_KEY,metric,'model_seed']
    missing=set(cols)-set(table.columns)
    if missing:raise ValueError(f'Missing validation columns: {missing}')
    if table[metric].isna().any() or not table[metric].map(math.isfinite).all():
        raise ValueError(f'Non-finite {metric}')
    if table.duplicated([*REPLICATE_KEY,'model_seed']).any():
        raise ValueError('Duplicated candidate replicate rows; refusing ambiguous selection')
    base_key=['dataset','method','scan_job','config_key']
    covered=table.groupby(base_key,dropna=False)['split_seed'].nunique()
    if not (covered==expected_splits).all():
        raise ValueError('Unequal outer-split coverage across candidate configs')
    covered_seeds=table.groupby(list(REPLICATE_KEY),dropna=False)['model_seed'].nunique()
    if not (covered_seeds==expected_seeds).all():
        raise ValueError('Unequal model-seed coverage across candidate configs')
    g=table.groupby(list(REPLICATE_KEY),dropna=False).agg(
      val_mean=(metric,'mean'),val_std=(metric,'std'),
      n=('val_RMSE','size'),K=('K','first'),init_mode=('init_mode','first')).reset_index()
    if not (g.n==expected_seeds).all():raise ValueError('Incomplete candidate replicate set')
    g['sort_k']=pd.to_numeric(g.K,errors='coerce').fillna(0)
    g['val_std']=g.val_std.fillna(0)
    ranked=g.sort_values(['dataset','method','split_seed','val_mean','val_std','sort_k','scan_job','config_key'],kind='mergesort')
    winners=ranked.drop_duplicates(['dataset','method','split_seed'],keep='first').reset_index(drop=True)
    if winners.groupby(['dataset','method']).split_seed.nunique().min()!=expected_splits:
        raise ValueError('Winner coverage incomplete')
    return winners,ranked


def fingerprint_config(obj):
    s=json.dumps(dataclasses.asdict(obj),sort_keys=True,default=str,ensure_ascii=True)
    return hashlib.sha256(s.encode('utf8')).hexdigest()
