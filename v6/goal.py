"""Read-only assessment of the predeclared ML-100K TEST target.

No tuning/training is performed. All seven proposed models are reported; the
best TEST row is never selected and handed back to the optimizer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

METHODS=tuple("Method"+str(i) for i in range(1,8))
BASELINES=("AEMC-NE","AutoRec","MF","BiasMF","MBMF-F","MBMF-C","MBMF-N")
TARGET=.87
REFERENCE_CEILING=.89  # Predeclared diagnostic tolerance, not a confidence bound.


def pair_hash(pair,items):
    digest=hashlib.sha256()
    for value in (np.asarray(pair,dtype=np.int64)//items,np.asarray(pair,dtype=np.int64)%items):
        value=np.ascontiguousarray(value)
        digest.update(str(value.dtype).encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def summarize(frame,minimum_splits=20):
    if frame.duplicated(["method","split_seed","model_seed"]).any():raise ValueError("Duplicate model/split/seed row")
    if not np.isfinite(frame.verified_test_RMSE).all() or (frame.verified_test_RMSE<0).any():raise ValueError("Invalid numerical RMSE")
    required=set(METHODS+BASELINES)
    if not required<=set(frame.method):raise ValueError("Missing predeclared model families: "+str(sorted(required-set(frame.method))))
    seed_sets=frame.groupby(["method","split_seed"]).model_seed.apply(lambda values:tuple(sorted(values)))
    if len(set(seed_sets))!=1:raise ValueError("Model-seed populations differ")
    outer=frame.groupby(["method","split_seed"],as_index=False).verified_test_RMSE.mean()
    reference=outer[outer.method=="AEMC-NE"].set_index("split_seed").verified_test_RMSE
    expected=set(reference.index)
    rows=[];comparisons=[]
    for method in METHODS:
        values=outer[outer.method==method].set_index("split_seed").verified_test_RMSE
        if set(values.index)!=expected:raise ValueError("Outer split populations differ")
        mean=float(values.mean());deltas=[]
        for baseline in BASELINES:
            rival=outer[outer.method==baseline].set_index("split_seed").verified_test_RMSE
            if set(rival.index)!=expected:raise ValueError("Baseline split populations differ")
            delta=values.sort_index()-rival.sort_index();deltas.append(float(delta.mean()))
            for split,value in delta.items():comparisons.append(dict(method=method,reference=baseline,split_seed=int(split),delta_RMSE=float(value)))
        enough=len(expected)>=max(20,minimum_splits)
        numeric=mean<TARGET;beats_all=max(deltas)<0;strong_reference=float(reference.mean())<=REFERENCE_CEILING
        status=("GOAL_NOT_MET" if not numeric else "EXPLORATORY_TARGET_ONLY" if not enough
                else "TARGET_MET_REFERENCE_TOO_WEAK" if not strong_reference
                else "TARGET_MET_BUT_BASELINE_NOT_BEATEN" if not beats_all
                else "NUMERIC_AND_COMPARATIVE_GOAL_MET_AUDIT_REQUIRED")
        rows.append(dict(method=method,test_RMSE_mean=mean,test_RMSE_SD_outer=float(values.std(ddof=1)) if len(values)>1 else None,
            n_outer_splits=len(values),target=TARGET,numeric_target_met=numeric,beats_all_predeclared_baselines=beats_all,
            aemc_reference_mean=float(reference.mean()),reference_within_diagnostic_tolerance=strong_reference,
            at_least_20_outer_splits=enough,status=status,publication_novelty_or_significance_proven=False))
    return pd.DataFrame(rows),pd.DataFrame(comparisons)


def audit(root,minimum_splits=20):
    from .config import implementation_signature
    from .reporting import _archive,_write
    root=Path(root);frame=pd.read_csv(root/"final_locked_test_all.csv").fillna({"init_mode":"NA"})
    if set(frame.dataset)!={"ml-100k"}:raise ValueError("The 0.87 target is specifically ML-100K")
    selection=json.loads((root/"final_selection_provenance.json").read_text())
    manifest=json.loads((root/"search/validation_selection_manifest.json").read_text())
    if selection.get("selected_on")!="validation ONLY" or selection.get("test_used_for_selection") is not False or manifest.get("test_seen") is not False:
        raise ValueError("Missing validation-only frozen-selection provenance")
    winners=pd.read_csv(root/"search/validation_winners.csv")
    if any(str(column).startswith("test_") for column in winners):raise ValueError("TEST found in winner selection table")
    if winners.duplicated(["method","split_seed"]).any():raise ValueError("Duplicated frozen winner")
    cohorts={};verified=[];archive_records=[];checks=[];null_selected=set()
    wanted=frame[frame.method.isin(METHODS+BASELINES)]
    for _,row in wanted.iterrows():
        method=str(row.method);split=int(row.split_seed);seed=int(row.model_seed)
        winner=winners[(winners.method==method)&(winners.split_seed==split)]
        if len(winner)!=1:raise ValueError("Missing exact validation winner")
        winner=winner.iloc[0]
        if str(row.selected_search_job)!=str(winner.scan_job) or json.loads(str(row.config_key))!=json.loads(str(winner.config_key)):
            raise ValueError("Final configuration differs from frozen validation winner")
        folder=root/"final"/method/("split_"+str(split))
        config=json.loads((folder/"config.json").read_text())
        study=config.get("v6",{})
        if study.get("protocol")!="aemc-paper" or study.get("split_validation_roles") is not False:raise ValueError("Target requires the matched 85.5/4.5/10 point-prediction protocol")
        if study.get("implementation_sha256")!=implementation_signature():raise ValueError("Final run source differs from assessor source")
        if method in METHODS:
            model_cfg=study["models"][method]
            pure_decoder=(model_cfg.get("readout_mode")=="decoder" and model_cfg.get("mechanism_auxiliary_weight",1.)==0
                and model_cfg.get("code_prior",0.)==0 and model_cfg.get("direction_prior",0.)==0)
            if pure_decoder:null_selected.add(method)
        fractions=config["data"]
        if any(abs(float(fractions[name])-value)>1e-10 for name,value in (("train_fraction",.855),("val_fraction",.045),("test_fraction",.1))):raise ValueError("Wrong declared data fractions")
        protocol=json.loads((folder/("V6_PROTOCOL_"+str(split)+".json")).read_text())
        counts=(protocol.get("train_pairs"),protocol.get("validation_original"),protocol.get("test_pairs"),protocol.get("num_users"),protocol.get("num_items"))
        if counts!=(85500,4500,10000,943,1682):raise ValueError("Actual ML-100K population/counts differ from the target protocol")
        if protocol.get("selection")!=4500 or protocol.get("calibration_fit")!=0 or protocol.get("calibration_hold")!=0:raise ValueError("Unmatched validation roles")
        if method=="Method7" and study["models"][method].get("probability_fit") is not False:raise ValueError("Method7 consumed extra calibration data")
        path=_archive(folder,row,"test")
        if path is None:raise ValueError("Missing final TEST prediction archive")
        with np.load(path,allow_pickle=False) as archive:
            pair=np.asarray(archive["pair_id"],dtype=np.int64);truth=np.asarray(archive["truth"],dtype=np.float64);prediction=np.asarray(archive["prediction"],dtype=np.float64)
            if len(pair)!=10000 or len(np.unique(pair))!=10000 or truth.shape!=pair.shape or prediction.shape!=pair.shape:raise ValueError("Invalid TEST cohort")
            if not np.isfinite(truth).all() or not np.isfinite(prediction).all():raise ValueError("Nonfinite TEST arrays")
            if not ((truth>=1)&(truth<=5)&(truth==np.round(truth))).all():raise ValueError("Invalid ML-100K target scale")
            if pair_hash(pair,1682)!=protocol["data_signature"].get("test_pair_sha256"):raise ValueError("TEST membership changed after freezing")
            order=np.argsort(pair)
            cohort=hashlib.sha256(pair[order].astype("<i8").tobytes()+truth[order].astype("<f8").tobytes()).hexdigest()
            key=(split,seed)
            if key in cohorts and cohorts[key]!=cohort:raise ValueError("Methods used different TEST pairs or truth")
            cohorts[key]=cohort
            score=float(np.sqrt(np.mean((prediction-truth)**2)))
        if not np.isfinite(score) or not np.isfinite(float(row.test_RMSE)) or abs(score-float(row.test_RMSE))>2e-5:raise ValueError("Reported TEST RMSE does not match predictions")
        verified.append(dict(method=method,split_seed=split,model_seed=seed,verified_test_RMSE=score))
        archive_records.append(dict(method=method,split_seed=split,model_seed=seed,archive=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cohort_sha256=cohort))
        checks.append(dict(method=method,split_seed=split,model_seed=seed,source_sha256=study["implementation_sha256"],configuration_frozen=True,protocol_verified=True))
    by_outer={}
    for (split,seed),cohort in cohorts.items():by_outer.setdefault(split,set()).add(cohort)
    if any(len(values)!=1 for values in by_outer.values()):raise ValueError("Model seeds used different data within an outer split")
    if len({next(iter(values)) for values in by_outer.values()})!=len(by_outer):raise ValueError("Repeated TEST cohort counted as multiple outer splits")
    results,paired=summarize(pd.DataFrame(verified),minimum_splits)
    results["pure_decoder_control_selected_on_any_split"]=results.method.isin(null_selected)
    results.loc[results.method.isin(null_selected)&results.numeric_target_met,"status"]="TARGET_MET_BY_DECODER_CONTROL_NOT_GEOMETRIC_EVIDENCE"
    output=root/"target_087";output.mkdir(exist_ok=True)
    _write(results,output/"goal_by_method.csv");_write(paired,output/"paired_deltas.csv")
    _write(pd.DataFrame(archive_records),output/"archive_integrity.csv");_write(pd.DataFrame(checks),output/"protocol_checks.csv")
    payload=dict(status="ASSESSMENT_COMPLETED",target=TARGET,results=results.astype(object).where(pd.notna(results),None).to_dict(orient="records"),
        training_performed=False,hyperparameters_changed=False,test_used_for_selection=False,
        evidence_scope="Mean over fixed model seeds within outer split, then mean over outer splits. Overlapping random splits are not asserted independent.",
        caution="Passing the numeric target does not prove novelty, statistical significance or official-author implementation parity.",
        aemc_reference_tolerance=REFERENCE_CEILING,reference_tolerance_is_not_a_confidence_bound=True)
    temporary=output/".GOAL_ASSESSMENT.json.partial"
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False));temporary.replace(output/"GOAL_ASSESSMENT.json")
    return payload


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True);args=parser.parse_args(argv)
    result=audit(args.root)
    for row in result["results"]:print(row["method"],format(row["test_RMSE_mean"],".6f"),row["status"])
