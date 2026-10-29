"""Scoped V7 integration: all monkeypatches are restored, including on failure."""
from __future__ import annotations

import copy
import csv
import json
from contextlib import contextmanager
from dataclasses import field, fields, make_dataclass, replace
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .config import StudyConfig, implementation_signature
from .models import BaseModel, Method7, instantiate
from .training import train_v7
from .protocol import validation_roles, data_signature
from .selection import prepare_locked_plan
from . import checkpoints, evaluation, search
from .execution import ExecutionConfig,run_training
from .resources import move_data_, resolve_device, execute_candidate
from .aemc_baseline import AEMCConfig, AEMCNE, train_aemcne as train_aemc_control
from importlib.metadata import version as package_version, PackageNotFoundError
import platform


@contextmanager
def installed(engine, options):
    old_load, old_base = engine.load_v8, engine.make_base
    from . import method8
    from .lifecycle import release_model
    from .recovery import candidate_identity,load_candidate,save_candidate
    old_job=engine.run_job
    core, pub = old_load()
    saved = []
    active = {"settings": None, "output": None, "locked_specs": None, "execution": None, "validation_only": False}
    archive_cache = {}
    def patch(obj, name, value):
        saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)
    Config = make_dataclass("PublicationExperimentConfigV7", [("v7", StudyConfig, field(default_factory=StudyConfig)),("execution",ExecutionConfig,field(default_factory=ExecutionConfig))],
                            bases=(pub.PublicationExperimentConfigV8,))
    def make_base(_pub, args, dataset):
        old = old_base(pub, args, dataset)
        cfg = Config(**{f.name: getattr(old, f.name) for f in fields(old)})
        cfg.v7 = copy.deepcopy(options.study)
        cfg.aemcne=AEMCConfig.from_legacy(cfg.aemcne,
            main_activation=cfg.v7.aemc_main_activation,element_activation=cfg.v7.aemc_element_activation,
            initialization=cfg.v7.aemc_initialization,item_batch_size=cfg.v7.aemc_item_batch_size,element_skip=cfg.v7.aemc_element_skip)
        cfg.aemcne.main_hidden_dim={"ml-100k":500,"ml-1m":900,"ml-10m":1000}.get(dataset,cfg.aemcne.main_hidden_dim)
        # Strong, explicit starting points; the full validation grid remains
        # necessary. These are not claimed to reproduce the published scores.
        cfg.autorec.lambda_l2=100.0
        cfg.aemcne.lambda_main=cfg.aemcne.lambda_element=100.0
        if not args.quick:
            cfg.autorec.max_epochs=cfg.aemcne.max_epochs=1000
            cfg.autorec.val_patience=cfg.aemcne.val_patience=20
        if not args.quick:cfg.biasmf.max_epochs=cfg.v7.mf_max_epochs
        if not args.quick and cfg.v7.mbmf_checkpoint_policy=='validation':
            cfg.core_config.paper.max_epochs=cfg.v7.mbmf_max_epochs
        # Preserve the legacy effective Rprop default, now explicitly recorded.
        if cfg.autorec.optimizer == "rprop":cfg.autorec.lr = .01
        cfg.execution=copy.deepcopy(options.execution)
        cfg.save_candidate_checkpoints=options.retain_candidates=="all"
        cfg.save_predictions=True;cfg.save_selected_checkpoints=True
        if cfg.v7.protocol!="common":
            # Point-only Method7 uses the SAME complete selection population.
            # Its probability heads are neither fitted nor reported here.
            cfg.v7.models["Method7"].probability_fit=False
            train,val=(.855,.045) if cfg.v7.protocol=="aemc-paper" else (.81,.09)
            cfg.data.train_fraction,cfg.data.val_fraction,cfg.data.test_fraction=train,val,.1
        for method, m in cfg.v7.models.items():
            if options.max_epochs is not None:
                m.max_epochs = options.max_epochs
            if options.steps_per_epoch is not None:
                m.steps_per_epoch = options.steps_per_epoch
            if args.eval_every is not None:
                m.eval_every = args.eval_every
            if args.quick:
                m.max_epochs, m.eval_every, m.val_patience = 3, 1, 2
                m.steps_per_epoch, m.warmup_steps = 1, 2
                m.router_expert_epochs, m.router_epochs = 3, 5
                m.shape_epochs, m.shape_patience = 5, 3
                m.graph_probes, m.cg_steps = 2, 6
                m.posterior_probes, m.posterior_cg_steps = 2, 32
                m.posterior_solver_tolerance=.001
                m.angular_reference_pairs = 512
            m.validate()
        baseline_epochs=getattr(options,"baseline_max_epochs",None)
        if baseline_epochs is not None and not args.quick:
            cfg.autorec.max_epochs=cfg.aemcne.max_epochs=cfg.biasmf.max_epochs=baseline_epochs
            cfg.core_config.paper.max_epochs=baseline_epochs
        cfg.aemcne.validate()
        cfg.auto_generate_plots = not options.no_plots
        from dataclasses import fields as config_fields
        for name, overrides in cfg.v7.baseline_overrides.items():
            obj = getattr(cfg, name)
            allowed = {f.name for f in config_fields(obj)}
            if not isinstance(overrides, dict) or set(overrides) - allowed:
                raise ValueError("Unknown baseline setting: " + name)
            for setting, value in overrides.items():
                setattr(obj, setting, value)
            if hasattr(obj, "validate"):
                obj.validate()
        if args.quick:
            cfg.aemcne.warm_start_epochs=min(cfg.aemcne.warm_start_epochs,3)
        elif baseline_epochs is not None:
            cfg.aemcne.warm_start_epochs=min(cfg.aemcne.warm_start_epochs,baseline_epochs)
        if cfg.v7.protocol != "common":
            cfg.data.split_mode = "random_global"
            cfg.data.enforce_train_support_for_all_items = False
        # Explicit CLI limits take precedence over JSON overrides.
        for obj in (cfg.autorec,cfg.aemcne,cfg.biasmf,cfg.core_config.paper):
            if args.quick: obj.max_epochs=min(obj.max_epochs,3);obj.eval_every=1
            elif baseline_epochs is not None: obj.max_epochs=baseline_epochs
        cfg.v7.method8=dict(cfg.v7.method8)
        if options.max_epochs is not None:cfg.v7.method8["max_epochs"]=options.max_epochs
        if args.eval_every is not None:cfg.v7.method8["eval_every"]=args.eval_every
        if args.quick:cfg.v7.method8.update(max_epochs=3,eval_every=1,val_patience=2)
        cfg.aemcne.warm_start_epochs=min(cfg.aemcne.warm_start_epochs,cfg.aemcne.max_epochs)
        cfg._v7_resume=bool(args.resume)
        cfg._v7_locked_specs={}
        return cfg
    old_inst, old_train = pub.instantiate_model_v8, pub.train_model_v8
    old_build, old_run = pub.build_publication_matrix_data_v8, pub.run_publication_study_v8
    old_save, old_restore = pub.save_model_checkpoint_v8, pub.load_model_checkpoint_v8
    old_bounds, old_count = pub.individual_bound_arrays, core.parameter_count
    old_archive = pub.save_prediction_archive_v8
    old_final = engine.run_final
    from v9.progress import CandidateProgress
    old_progress_eval = CandidateProgress.on_eval
    old_instrument=CandidateProgress.instrument
    old_split=pub.make_core_split
    @contextmanager
    def instrument(progress,*args,**kwargs):
        previous=active.get("progress");active["progress"]=progress
        try:
            with old_instrument(progress,*args,**kwargs):yield progress
        finally:active["progress"]=previous
    def split(raw,spec,dc):
        if active["settings"] is None or active["settings"].protocol=="common" or dc.split_mode=="temporal_per_user":return old_split(raw,spec,dc)
        out=raw.copy().reset_index(drop=True);order=np.random.default_rng(dc.split_seed).permutation(len(out))
        nt=round(len(out)*dc.train_fraction);nv=round(len(out)*dc.val_fraction)
        role=np.full(len(out),"test",dtype=object);role[order[:nt]]="train";role[order[nt:nt+nv]]="val"
        out["split"]=role;return out
    def inst(method, data, cfg, k, seed, device, init_mode=None):
        progress=active.get("progress")
        if progress is not None and progress.active and progress.active.get("model") is not None:
            previous_model=progress.active.pop("model")
            release_model(previous_model,active.get("output"))
        def construct(target):
            move_data_(data,target)
            if method=="Method8":return method8.create(data,cfg.v7.method8,k,seed,target)
            if method not in cfg.v7.models:
                m,c=old_inst(method,data,cfg,k,seed,target,init_mode)
                if method.startswith('MBMF-'):
                    from .baselines import paper_config
                    policy=cfg.v7.mbmf_checkpoint_policy
                    c=paper_config(c,policy,cfg.v7.mbmf_val_patience,cfg.v7.mbmf_optimizer,cfg.v7.mbmf_adam_lr)
                return m,c
            c=replace(cfg.v7.models[method],latent_dim=int(k))
            return instantiate(data,c,seed,target),c
        initial_fallback=None
        try:model,model_cfg=construct(device)
        except torch.cuda.OutOfMemoryError as exc:
            if not cfg.execution.cpu_fallback or torch.device(device).type!='cuda':raise
            initial_fallback=str(exc);exc.__traceback__=None
            import gc
            gc.collect();torch.cuda.empty_cache();device=torch.device('cpu')
            model,model_cfg=construct(device)
        model._v7_resume=bool(getattr(cfg,"_v7_resume",False))
        model._v7_recovery_interval=options.v7_recovery_interval
        model._v7_model_seed=int(seed)
        model._v7_initialization_fallback=initial_fallback
        model._v7_rebuild_spec=(method,cfg,k,seed,device,init_mode)
        if active["output"] is not None:
            split_seed=data.v7_protocol["split_seed"]
            model._v7_artifact_dir=Path(active["output"])/"mechanisms"/f"{method}__split{split_seed}__seed{seed}__K{k}__{init_mode or 'NA'}"
        if active["locked_specs"] is not None:
            split_seed = data.v7_protocol["split_seed"]
            specification = active["locked_specs"].get((method, split_seed))
            if specification is None:
                raise ValueError("No frozen winner for this locked refit")
            actual_key = pub._candidate_config_key_v8(method, model_cfg, init_mode or "NA")
            if json.loads(actual_key) != json.loads(specification["config_key"]):
                raise ValueError("Locked refit configuration differs from the validation winner")
            if data_signature(data) != specification["data_signature"]:
                raise ValueError("Locked refit data differ from the frozen scan")
        return model, model_cfg
    @contextmanager
    def locked_scope(base, root, winners, jobs):
        plan, specs = prepare_locked_plan(base, root, winners, jobs)
        previous = active["locked_specs"]
        active["locked_specs"] = specs
        try:
            yield plan
        finally:
            active["locked_specs"] = previous
    def final(pub, core, base, root, args, scan_results=None):
        if scan_results is None:
            import pandas as pd
            winners = pd.read_csv(Path(root)/"search"/"validation_winners.csv")
            jobs = search.build_jobs(pub, base, engine.numbers(args.k), args.scan_mode)
        else:
            jobs, winners = scan_results
        with locked_scope(base, root, winners, jobs) as plan:
            return old_final(pub, core, base, root, args, (plan, winners))
    def calibration(pub, core, base, root, args):
        return run_calibration(pub, core, base, root, args, locked_scope=locked_scope)
    def remember_training(model,data,result):
        method,cfg,k,seed,device,mode=model._v7_rebuild_spec
        active.setdefault("training_records",[]).append(dict(method=method,split_seed=data.v7_protocol["split_seed"],
            K=k,model_seed=seed,init_mode=mode or "NA",seconds=result["v7_original_training_seconds"],
            reused=bool(result.get("v7_resume",{}).get("reused",False))))
    def trainer(method,model,data,config,study):
        import time
        training_started=time.perf_counter()
        identity=candidate_identity(model,data,config)
        cached=load_candidate(model,identity)
        if cached is not None:
            remember_training(model,data,cached)
            return cached
        spec=model._v7_rebuild_spec
        def callback():
            if method=="Method8":return method8.train(model,data)
            if isinstance(model,BaseModel):return train_v7(pub,model,data,config,study)
            from .baselines import train_autorec,train_mbmf,train_biasmf,train_aemcne
            if method == "AutoRec":return train_autorec(pub,model,data,config,study.scalable_chunk_size)
            if method in ("MF","BiasMF"):return train_biasmf(pub,model,data,config)
            if method == "AEMC-NE":return train_aemc_control(pub,model,data,config,study.scalable_chunk_size)
            if method.startswith("MBMF-") and config.checkpoint_policy == "validation":
                return train_mbmf(pub,model,data,config,study.scalable_chunk_size)
            return old_train(method,model,data,config,study)
        def rebuild(target_device=None):
            import shutil
            folder=getattr(model,"_v7_artifact_dir",None)
            if folder:
                for path in Path(folder).rglob("recovery"):
                    if path.is_dir() and not path.is_symlink():shutil.rmtree(path)
            name,cfg,k,seed,device,mode=spec
            return inst(name,data,cfg,k,seed,target_device or data.train_users.device,mode)[0]
        progress=active.get("progress");rollback=None
        if progress is not None:
            offset=progress.log_path.stat().st_size if progress.log_path.exists() else 0
            before=dict(progress.active) if progress.active else None
            count=progress.n_epoch_records
            def rollback():
                if progress.log_path.exists():
                    with progress.log_path.open("r+b") as handle:handle.truncate(offset)
                progress.active=dict(before) if before else None;progress.n_epoch_records=count
        result=execute_candidate(model,data,callback,rebuild,study.execution,on_replay=rollback)
        if result.get("history"):
            result["history"][-1]['v7_training_budget']={key:result[key] for key in (
                'best_epoch','best_val_RMSE','last_epoch','final_epoch','stopping_reason','best_checkpoint_at_budget',
                'optimizer_steps','mean_optimizer_steps','prediction_targets_seen','full_observed_targets_seen','objective','radius_block_updates','bias_warmup_steps') if key in result}
            for key in ("v7_routing","v7_probability","v7_execution","v7_response","source_lineage"):
                if key in result:result["history"][-1][key]=result[key]
        prior=getattr(model,"_v7_prior_training_seconds",getattr(getattr(model,"inner",None),"_v7_prior_training_seconds",0.))
        result["v7_original_training_seconds"]=time.perf_counter()-training_started+prior
        remember_training(model,data,result)
        save_candidate(model,identity,result)
        return result
    def build(split, spec, data_cfg, device):
        previous_data=active.pop("resident_data",None)
        if previous_data is not None:move_data_(previous_data,"cpu")
        try:data,backend=old_build(split,spec,data_cfg,device)
        except torch.cuda.OutOfMemoryError as exc:
            if active['execution'] is None or not active['execution'].cpu_fallback:raise
            exc.__traceback__=None
            import gc
            gc.collect();torch.cuda.empty_cache();device=torch.device('cpu')
            data,backend=old_build(split,spec,data_cfg,device)
            print('[V7 DEVICE] data construction exceeded CUDA memory; building the same data on CPU',flush=True)
        if getattr(data, "train_timestamp_status", None) == "complete":
            # The legacy attachment casts Unix timestamps to float32 before
            # centering. Keep their precision in V7, without editing V8 code.
            import pandas as pd
            tr = split[split["split"] == "train"]
            data.train_timestamps = torch.as_tensor(
                pd.to_numeric(tr["timestamp"], errors="raise").to_numpy(dtype=np.float64), device=device)
        if active["settings"] is None:
            raise RuntimeError("V7 data construction requires an active study")
        data=validation_roles(data, active["settings"], data_cfg.split_seed, active["output"])
        anchor=active["settings"].analysis_anchor
        if anchor:
            current=data_signature(data)
            for name,expected in anchor["data_signature"].items():
                if current.get(name)!=expected:raise ValueError("Frozen analysis data mismatch for "+name)
        active["resident_data"]=data
        return data, backend
    def run(cfg, *args, **kwargs):
        if cfg.v7.implementation_sha256 != implementation_signature():
            raise ValueError("V7 source changed since configuration creation")
        previous = dict(active)
        active.update(settings=cfg.v7, output=cfg.output_dir, execution=cfg.execution, validation_only=cfg.validation_only,training_records=[])
        supplied=getattr(cfg,"_v7_locked_specs",None)
        if supplied:active["locked_specs"]=supplied
        try:
            selected,reason=resolve_device(cfg.execution.device)
            kwargs["device"]=selected
            print(f"[V7 DEVICE] requested={cfg.execution.device}; selected={selected}; {reason}",flush=True)
            environment=dict(architecture=__version__,implementation_sha256=implementation_signature(),python=platform.python_version(),
                torch_threads=torch.get_num_threads(),cuda_available=torch.cuda.is_available(),cuda_runtime=torch.version.cuda,
                float32_matmul_precision=torch.get_float32_matmul_precision(),cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                execution_policy=vars(cfg.execution))
            import sys
            environment.update(python_executable=sys.executable,selected_device=str(selected),device_selection_reason=reason)
            environment["packages"]={}
            for name in ("torch","numpy","pandas","scipy","matplotlib"):
                try:environment["packages"][name]=package_version(name)
                except PackageNotFoundError:environment["packages"][name]="unknown"
            Path(cfg.output_dir).mkdir(parents=True,exist_ok=True)
            from .fidelity import write_ledger
            write_ledger(cfg.output_dir,cfg)
            (Path(cfg.output_dir)/"V7_ENVIRONMENT.json").write_text(json.dumps(environment,indent=2))
            result = old_run(cfg, *args, **kwargs)
            # A reused model must not be reported as a millisecond training result.
            def annotate(table):
                if table.empty:return table
                table=table.copy()
                table["resume_attempt_wall_sec"]=table.train_wall_sec
                table["training_reused_from_resume"]=False
                for event in active["training_records"]:
                    chosen=(table.method==event["method"]) & (table.split_seed==event["split_seed"]) & (table.model_seed==event["model_seed"]) & (table.init_mode.fillna("NA")==event["init_mode"]) & (table.K.isna() | (table.K==event["K"]))
                    table.loc[chosen,"train_wall_sec"]=event["seconds"]
                    table.loc[chosen,"training_reused_from_resume"]=event["reused"]
                return table
            candidates,locked=map(annotate,result)
            candidates.to_csv(Path(cfg.output_dir)/"candidate_validation_runs.csv",index=False)
            if not locked.empty:locked.to_csv(Path(cfg.output_dir)/"locked_test_runs.csv",index=False)
            result=(candidates,locked)
            if cfg.v7.implementation_sha256 != implementation_signature():
                raise RuntimeError("V7 source changed during this experiment; results are not a reproducible completed job")
            return result
        finally:
            progress=active.get("progress")
            if progress is not None and progress.active and progress.active.get("model") is not None:
                release_model(progress.active.pop("model"),cfg.output_dir)
            resident=active.pop("resident_data",None)
            if resident is not None:move_data_(resident,"cpu")
            active.update(previous)
    def save(path,model,**kwargs):
        if isinstance(model,method8.Method10):return method8.save(path,model,**kwargs)
        result=checkpoints.save(path,model,**kwargs) if isinstance(model,BaseModel) else old_save(path,model,**kwargs)
        data=kwargs.get("data")
        if active["settings"] is not None and active["settings"].artifact_policy == "selected" and active["validation_only"]:
            return result
        if data is not None and active["output"] is not None:
            saved_cache=dict(archive_cache)
            try:
                if active["settings"].artifact_policy != "selected":
                    _,prediction=pub.evaluate_publication_model(model,data,"train",return_predictions=True)
                    target=Path(active["output"])/"predictions"/"candidates"/(Path(path).stem+"__train.npz")
                    pub.save_prediction_archive_v8(target,data,prediction,"train")
                if isinstance(model,BaseModel) and model.v7_method=="Method7" and bool(model.shape_fitted):
                    for role in ("fit","hold"):
                        clone=copy.copy(data)
                        for part,key in (("users","users"),("items","items"),("ratings_raw","ratings")):
                            setattr(clone,"val_"+part,getattr(data,f"v7_calibration_{role}_{key}"))
                        _,prediction=pub.evaluate_publication_model(model,clone,"val",return_predictions=True)
                        identity=getattr(model,"_v7_artifact_dir",f"{model.v7_method}__split{data.v7_protocol['split_seed']}__seed{model.model_seed}__K{model.k}__NA")
                        target=Path(active["output"])/"predictions"/"calibration"/(Path(identity).name+f"__probability_{role}.npz")
                        pub.save_prediction_archive_v8(target,clone,prediction,"val")
            finally:archive_cache.clear();archive_cache.update(saved_cache)
        return result
    def restore(path, data, device, *, strict_data=True):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        def load_on(target):
            prior=active.get("resident_data")
            if prior is not None and prior is not data:move_data_(prior,"cpu")
            active["resident_data"]=data
            move_data_(data,target)
            if payload.get("method")=="Method10":return method8.load(path,data,target)
            if payload.get('format')=='MBMF-V7':return checkpoints.restore(payload,data,target,strict_data=strict_data)
            if payload.get('format')=='MBMF-Q1-v8' and payload.get('family')=='PaperMBMF' and 'checkpoint_policy' in payload.get('config',{}):
                from .baselines import restore_mbmf
                return restore_mbmf(pub,core,payload,data,target,strict_data)
            return old_restore(path,data,target,strict_data=strict_data)
        try:return load_on(device)
        except torch.cuda.OutOfMemoryError as exc:
            if active['execution'] is None or not active['execution'].cpu_fallback:raise
            exc.__traceback__=None
            import gc
            gc.collect();torch.cuda.empty_cache()
            print('[V7 DEVICE] checkpoint restored on CPU after CUDA OOM',flush=True)
            return load_on(torch.device('cpu'))
    def archive(path, data, pred, split):
        if active["settings"] is not None and active["settings"].artifact_policy == "selected" and active["validation_only"] and split == "train":
            return
        if (archive_cache.get("data") != id(data) or archive_cache.get("split") != split
                or not np.array_equal(pred, archive_cache.get("prediction"))):
            return old_archive(path, data, pred, split)
        u, i, y = pub._split_tensors(data, split)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, pair_id=(u*data.num_items+i).cpu().numpy(),
                            prediction=pred.astype(np.float32), truth=y.cpu().numpy(),
                            **archive_cache.get("extras", {}))
    def job(pub_,core_,cfg,folder,*,args,validation_only=False,extra=None):
        cfg._v7_resume=bool(args.resume)
        if options.v7_isolate_jobs and not getattr(options,"_v7_worker",False):
            import os,subprocess,sys
            from dataclasses import asdict
            from .engine import json_write
            folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
            source=dict(vars(options));source.pop("study",None);source.pop("execution",None)
            specs=active["locked_specs"] or {}
            request=folder/("V7_WORKER_REQUEST_"+str(os.getpid())+".json")
            json_write(request,dict(options=source,study=asdict(options.study),execution=asdict(options.execution),
                args=vars(args),config=asdict(cfg),folder=str(folder.resolve()),validation_only=validation_only,extra=extra,
                locked_specs=[dict(method=m,split_seed=s,spec=v) for (m,s),v in specs.items()]))
            import time
            started=time.time()
            process=subprocess.Popen([sys.executable,"-m","v7.worker",str(request.resolve())],cwd=engine.ROOT)
            code=process.wait()
            json_write(folder/"V7_JOB_PROCESS.json",dict(pid=process.pid,parent_pid=os.getpid(),start_unix=started,end_unix=time.time(),exit_code=code,isolation="sequential_process_per_job"))
            if code:raise RuntimeError(f"Isolated job exited with status {code}; rerun the same command with --resume: {folder}")
            import pandas as pd
            return pd.read_csv(folder/("candidate_validation_runs.csv" if validation_only else "locked_test_runs.csv"))
        from .locking import study_lock
        with study_lock(folder):
            result=old_job(pub_,core_,cfg,folder,args=args,validation_only=validation_only,extra=extra)
            import shutil
            for path in Path(folder).rglob("recovery"):
                if path.is_dir() and not path.is_symlink():shutil.rmtree(path)
            return result

    def self_test():
        pub.scientific_core_immutability_test_v8()
        from .laws import mean_tilt
        levels = torch.arange(1., 6.)
        means = torch.tensor([1.1, 2.3, 3.8, 4.9])
        probabilities = mean_tilt(torch.zeros(4, 5), means, levels)
        if not torch.allclose(probabilities@levels, means, atol=2e-5):
            raise AssertionError("V7 mean-constrained probability self-test failed")
        print(f"[V7 SELF-TEST] {__version__}; frozen legacy scientific core verified", flush=True)
    def progress_eval(progress, source, original, args, kwargs, result):
        current = progress.active
        if (not current or not current.get("training") or len(args) < 3
                or (args[0] is not current.get("model") and args[0] is not getattr(current.get("model"),"inner",None)) or args[2] != "val"):
            return old_progress_eval(progress, source, original, args, kwargs, result)
        current["val_ticks"] += 1
        interval=max(1,int(current['eval_every']))
        tick=current['val_ticks']
        epoch = (args[0]._v7_eval_epoch if hasattr(args[0], "_v7_eval_epoch") else
                 tick if interval==1 else 1 if tick==1 else (tick-1)*interval)
        metrics = result[0] if isinstance(result, tuple) else result
        train_rmse = ""
        if progress.epoch_train_rmse:
            training_metrics = original(args[0], args[1], "train")
            train_rmse = (training_metrics if isinstance(training_metrics,dict) else training_metrics[0])["RMSE"]
            args[0]._v7_last_train_rmse=(epoch,float(train_rmse))
        with progress.log_path.open("a", newline="", encoding="utf8") as handle:
            writer = csv.writer(handle)
            if handle.tell() == 0:
                writer.writerow(["stage", "method", "split_seed", "K", "model_seed", "init", "epoch", "train_RMSE", "val_RMSE", "source"])
            writer.writerow([progress.stage, current["method"], current.get("split_seed"),
                             current["k"], current["seed"], current["init"], epoch, train_rmse,
                             metrics["RMSE"], "V7-mean-selection" if hasattr(args[0], "_v7_eval_epoch") else source])
        progress.n_epoch_records += 1
        if current["val_ticks"] == 1 or current["val_ticks"] % 5 == 0:
            progress.emit(f"EPOCH {epoch} valRMSE={metrics['RMSE']:.5f}")
        return result
    try:
        # Original AutoRec uses rating 3 for test users/items absent in TRAIN.
        # This affects only cold pairs; it does not alter the fitted objective.
        original_autorec = pub.ItemAutoRec
        class PaperAutoRec(original_autorec):
            def predict_raw_pairs(self, data, users, items):
                point = super().predict_raw_pairs(data, users, items)
                key = (id(data), str(users.device), len(data.train_users))
                cached = getattr(self, "_v7_cold_counts", None)
                if cached is None or cached[0] != key:
                    cached = (key,torch.bincount(data.train_users,minlength=data.num_users),
                              torch.bincount(data.train_items,minlength=data.num_items))
                    self._v7_cold_counts = cached
                cold = (cached[1][users] == 0) | (cached[2][items] == 0)
                return torch.where(cold, point.new_tensor(3.), point)
        patch(pub,"ItemAutoRec",PaperAutoRec)
        patch(pub,"AEMCNE",AEMCNE)
        patch(pub,"AEMCNEConfig",AEMCConfig)
        patch(engine,"REPORT_PLOTS",not options.no_plots)
        patch(CandidateProgress,"instrument",instrument)
        patch(pub,"make_core_split",split)
        patch(engine, "run_job",job)
        patch(engine, "load_v8", lambda: (core, pub))
        patch(engine, "make_base", make_base)
        patch(engine, "build_jobs", search.build_jobs)
        patch(engine, "run_final", final)
        patch(engine, "run_calibration", calibration)
        patch(CandidateProgress, "on_eval", progress_eval)
        patch(pub, "instantiate_model_v8", inst)
        patch(pub, "train_model_v8", trainer)
        patch(pub, "build_publication_matrix_data_v8", build)
        patch(pub, "run_publication_study_v8", run)
        patch(pub, "evaluate_publication_model", evaluation.evaluator(pub, pub.evaluate_publication_model, archive_cache))
        patch(pub, "save_model_checkpoint_v8", save)
        patch(pub, "load_model_checkpoint_v8", restore)
        patch(pub, "individual_bound_arrays", lambda m,d,u,i:evaluation.structural_bounds(m,d,u,i) if isinstance(m,BaseModel) else old_bounds(m,d,u,i))
        patch(pub, "save_prediction_archive_v8", archive)
        patch(pub, "make_ablation_variants_v8", search.ablations)
        patch(pub, "publication_extension_self_test_v8", self_test)
        patch(core, "parameter_count", lambda m: sum(p.numel() for p in m.parameters()) if isinstance(m, BaseModel) else old_count(m))
        # V7 writes its complete, styled figures to the visible figures/ folder.
        # Do not also emit the old, crowded q1_* versions from frozen V8.
        import mbmf_q1_visuals_v8 as visuals
        for name in ("generate_publication_figures_v8", "generate_ablation_figures_v8"):
            patch(visuals,name,lambda *args,**kwargs:None)
        if options.no_plots:
            for name in ("plot_convergence", "plot_k_rmse", "plot_validation_landscape", "plot_train_keep_sensitivity"):
                patch(engine, name, lambda *args, **kwargs: None)
            import mbmf_q1_visuals_v8 as visuals
            for name in ("generate_publication_figures_v8", "generate_ablation_figures_v8"):
                patch(visuals, name, lambda *args, **kwargs: None)
        yield core, pub
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)


def run_calibration(pub, core, base, dsroot, args, *, locked_scope):
    """Secondary NLL-selected experiment, separate from primary RMSE study."""
    from . import engine
    from v9.selection import select_validation_winners
    import pandas as pd
    if "Method7" not in base.methods:
        return
    if not base.v7.models["Method7"].probability_fit:
        print("[V7] Probability calibration is a separate disjoint-role study; skipped in point-only paper protocol",flush=True)
        return
    path = Path(dsroot)/"search"/"hyperparameter_validation_all.csv"
    table = pd.read_csv(path)
    table = table[table.method == "Method7"].copy()
    criterion = "val_Probability_NLL"
    if criterion not in table:
        raise ValueError("V7 scan must expose its actual ordinal/beta likelihood before calibration selection")
    winners, _ = select_validation_winners(table, len(base.core_config.split_seeds),
                                          len(base.core_config.model_seeds), metric=criterion)
    root = Path(dsroot)/"calibration"
    root.mkdir(parents=True, exist_ok=True)
    winners.to_csv(root/"validation_probability_nll_winners.csv", index=False)
    jobs = search.build_jobs(pub, base, engine.numbers(args.k), args.scan_mode)
    frames = []
    with locked_scope(base, dsroot, winners, jobs) as plan:
        for _, row in winners.iterrows():
            cfg = copy.deepcopy(plan[str(row.scan_job)])
            cfg.methods, cfg.core_config.split_seeds = ("Method7",), (int(row.split_seed),)
            cfg.core_config.latent_dims = (int(row.K),)
            if not args.dry_run:
                frames.append(engine.run_job(pub, core, cfg, root/f"split_{int(row.split_seed)}", args=args,
                                             extra=dict(type="V7-probability-NLL-selected", selection_metric=criterion)))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(root/"calibration_locked_test_runs.csv", index=False)
    engine.json_write(root/"CALIBRATION_PROTOCOL.json", dict(selection_metric=criterion,
                       test_used_for_selection=False, separate_from_RMSE_primary=True,
                       law="ordinal mean-constrained PMF or explicitly endpoint-clipped beta mixture"))
