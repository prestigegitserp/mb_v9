"""Synthetic fixtures test audit logic; they are not benchmark accuracy."""
import pandas as pd
import pytest
from v52.goal import METHODS,BASELINES,summarize,pair_hash


def rows(splits=20):
    return pd.DataFrame([dict(method=method,split_seed=4101+split,model_seed=11,
        verified_test_RMSE=.86+.0001*index if method in METHODS else .88+.0001*index)
        for split in range(splits) for index,method in enumerate(METHODS+BASELINES)])


def test_goal_requires_complete_twenty_split_comparison():
    table,paired=summarize(rows())
    assert len(table)==7 and len(paired)==7*7*20
    assert table.numeric_target_met.all() and table.beats_all_predeclared_baselines.all()
    assert not table.publication_novelty_or_significance_proven.any()


def test_single_split_cannot_become_a_paper_claim_even_with_low_threshold_option():
    table,_=summarize(rows(1),minimum_splits=1)
    assert not table.at_least_20_outer_splits.any()
    assert set(table.status)=={"EXPLORATORY_TARGET_ONLY"}


def test_weaker_reference_is_flagged_instead_of_hiding_it():
    frame=rows();frame.loc[frame.method=="AEMC-NE","verified_test_RMSE"]=.99
    table,_=summarize(frame)
    assert set(table.status)=={"TARGET_MET_REFERENCE_TOO_WEAK"}


def test_every_model_is_reported_instead_of_selecting_the_best_test_row():
    frame=rows();frame.loc[frame.method=="Method7","verified_test_RMSE"]=.95
    table,_=summarize(frame)
    assert len(table)==7 and table.loc[table.method=="Method7","status"].iloc[0]=="GOAL_NOT_MET"


def test_missing_baseline_duplicate_rows_and_seed_mismatch_are_rejected():
    frame=rows()
    with pytest.raises(ValueError,match="Missing"):summarize(frame[frame.method!="AutoRec"])
    with pytest.raises(ValueError,match="Duplicate"):summarize(pd.concat([frame,frame.iloc[:1]]))
    frame.loc[0,"model_seed"]=33
    with pytest.raises(ValueError,match="seed populations"):summarize(frame)


def test_test_pair_digest_matches_the_freezing_function():
    import torch
    from v52.protocol import digest_tensors
    pair=torch.tensor([0,17,33,65],dtype=torch.long)
    assert pair_hash(pair.numpy(),16)==digest_tensors(pair//16,pair%16)


def test_nonfinite_or_negative_reported_scores_are_rejected():
    frame=rows();frame.loc[0,"verified_test_RMSE"]=float("nan")
    with pytest.raises(ValueError,match="Invalid numerical"):summarize(frame)
    frame=rows();frame.loc[0,"verified_test_RMSE"]=-1.
    with pytest.raises(ValueError,match="Invalid numerical"):summarize(frame)
