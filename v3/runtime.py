"""Scoped V3 integration: all monkeypatches are restored, including on failure."""
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
from .models import BaseModel, instantiate
from .training import train_v3
from .protocol import validation_roles, data_signature
from .selection import prepare_locked_plan
from . import checkpoints, evaluation, search


@contextmanager
def installed(engine, options):
    old_load, old_base = engine.load_v8, engine.make_base
    core, pub = old_load()
    saved = []
    active = {"settings": None, "output": None, "locked_specs": None}
    archive_cache = {}
    def patch(obj, name, value):
        saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)
    Config = make_dataclass("PublicationExperimentConfigV3", [("v3", StudyConfig, field(default_factory=StudyConfig))],
                            bases=(pub.PublicationExperimentConfigV8,))
    def make_base(_pub, args, dataset):
        old = old_base(pub, args, dataset)
        cfg = Config(**{f.name: getattr(old, f.name) for f in fields(old)})
        cfg.v3 = copy.deepcopy(options.study)
        for method, m in cfg.v3.models.items():
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
                m.posterior_probes, m.posterior_cg_steps = 2, 6
                m.angular_reference_pairs = 512
            m.validate()
        cfg.auto_generate_plots = not options.no_plots
        return cfg
    old_inst, old_train = pub.instantiate_model_v8, pub.train_model_v8
    old_build, old_run = pub.build_publication_matrix_data_v8, pub.run_publication_study_v8
    old_save, old_restore = pub.save_model_checkpoint_v8, pub.load_model_checkpoint_v8
    old_bounds, old_count = pub.individual_bound_arrays, core.parameter_count
    old_archive = pub.save_prediction_archive_v8
    old_final = engine.run_final
    from v9.progress import CandidateProgress
    old_progress_eval = CandidateProgress.on_eval
    def inst(method, data, cfg, k, seed, device, init_mode=None):
        if method not in cfg.v3.models:
            model, model_cfg = old_inst(method, data, cfg, k, seed, device, init_mode)
        else:
            model_cfg = replace(cfg.v3.models[method], latent_dim=int(k))
            model = instantiate(data, model_cfg, seed, device)
        if active["locked_specs"] is not None:
            split_seed = data.v3_protocol["split_seed"]
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
    def trainer(method, model, data, config, study):
        if isinstance(model, BaseModel):
            result = train_v3(pub, model, data, config, study)
            # The legacy histories artifact only retains this field. Attach
            # phase summaries to its final row as well as the full checkpoint.
            if result["history"]:
                for key in ("v3_routing", "v3_probability"):
                    if key in result:
                        result["history"][-1][key] = result[key]
            return result
        return old_train(method, model, data, config, study)
    def build(split, spec, data_cfg, device):
        data, backend = old_build(split, spec, data_cfg, device)
        if getattr(data, "train_timestamp_status", None) == "complete":
            # The legacy attachment casts Unix timestamps to float32 before
            # centering. Keep their precision in V3, without editing V8 code.
            import pandas as pd
            tr = split[split["split"] == "train"]
            data.train_timestamps = torch.as_tensor(
                pd.to_numeric(tr["timestamp"], errors="raise").to_numpy(dtype=np.float64), device=device)
        if active["settings"] is None:
            raise RuntimeError("V3 data construction requires an active study")
        return validation_roles(data, active["settings"], data_cfg.split_seed, active["output"]), backend
    def run(cfg, *args, **kwargs):
        if cfg.v3.implementation_sha256 != implementation_signature():
            raise ValueError("V3 source changed since configuration creation")
        previous = dict(active)
        active.update(settings=cfg.v3, output=cfg.output_dir)
        try:
            result = old_run(cfg, *args, **kwargs)
            if cfg.v3.implementation_sha256 != implementation_signature():
                raise RuntimeError("V3 source changed during this experiment; results are not a reproducible completed job")
            return result
        finally:
            active.update(previous)
    def save(path, model, **kwargs):
        return checkpoints.save(path, model, **kwargs) if isinstance(model, BaseModel) else old_save(path, model, **kwargs)
    def restore(path, data, device, *, strict_data=True):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") == "MBMF-V3":
            return checkpoints.restore(payload, data, device, strict_data=strict_data)
        return old_restore(path, data, device, strict_data=strict_data)
    def archive(path, data, pred, split):
        if (archive_cache.get("data") != id(data) or archive_cache.get("split") != split
                or not np.array_equal(pred, archive_cache.get("prediction"))):
            return old_archive(path, data, pred, split)
        u, i, y = pub._split_tensors(data, split)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, pair_id=(u*data.num_items+i).cpu().numpy(),
                            prediction=pred.astype(np.float32), truth=y.cpu().numpy(),
                            **archive_cache.get("extras", {}))
    def self_test():
        pub.scientific_core_immutability_test_v8()
        from .laws import mean_tilt
        levels = torch.arange(1., 6.)
        means = torch.tensor([1.1, 2.3, 3.8, 4.9])
        probabilities = mean_tilt(torch.zeros(4, 5), means, levels)
        if not torch.allclose(probabilities@levels, means, atol=2e-5):
            raise AssertionError("V3 mean-constrained probability self-test failed")
        print(f"[V3 SELF-TEST] {__version__}; frozen legacy scientific core verified", flush=True)
    def progress_eval(progress, source, original, args, kwargs, result):
        current = progress.active
        if (not current or not current.get("training") or len(args) < 3
                or args[0] is not current.get("model") or args[2] != "val"
                or not isinstance(args[0], BaseModel)):
            return old_progress_eval(progress, source, original, args, kwargs, result)
        current["val_ticks"] += 1
        epoch = args[0]._v3_eval_epoch
        metrics = result[0] if isinstance(result, tuple) else result
        train_rmse = ""
        if progress.epoch_train_rmse:
            train_rmse = original(args[0], args[1], "train")["RMSE"]
        with progress.log_path.open("a", newline="", encoding="utf8") as handle:
            writer = csv.writer(handle)
            if handle.tell() == 0:
                writer.writerow(["stage", "method", "split_seed", "K", "model_seed", "init", "epoch", "train_RMSE", "val_RMSE", "source"])
            writer.writerow([progress.stage, current["method"], current.get("split_seed"),
                             current["k"], current["seed"], current["init"], epoch, train_rmse,
                             metrics["RMSE"], "V3-mean-selection"])
        progress.n_epoch_records += 1
        if current["val_ticks"] == 1 or current["val_ticks"] % 5 == 0:
            progress.emit(f"EPOCH {epoch} valRMSE={metrics['RMSE']:.5f}")
        return result
    try:
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
        patch(pub, "individual_bound_arrays", lambda m, d, u, i: None if isinstance(m, BaseModel) else old_bounds(m, d, u, i))
        patch(pub, "save_prediction_archive_v8", archive)
        patch(pub, "make_ablation_variants_v8", search.ablations)
        patch(pub, "publication_extension_self_test_v8", self_test)
        patch(core, "parameter_count", lambda m: sum(p.numel() for p in m.parameters()) if isinstance(m, BaseModel) else old_count(m))
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
    from v9 import engine
    from v9.selection import select_validation_winners
    import pandas as pd
    if "Method7" not in base.methods:
        return
    path = Path(dsroot)/"search"/"hyperparameter_validation_all.csv"
    table = pd.read_csv(path)
    table = table[table.method == "Method7"].copy()
    criterion = "val_Probability_NLL"
    if criterion not in table:
        raise ValueError("V3 scan must expose its actual ordinal/beta likelihood before calibration selection")
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
                                             extra=dict(type="V3-probability-NLL-selected", selection_metric=criterion)))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(root/"calibration_locked_test_runs.csv", index=False)
    engine.json_write(root/"CALIBRATION_PROTOCOL.json", dict(selection_metric=criterion,
                       test_used_for_selection=False, separate_from_RMSE_primary=True,
                       law="ordinal mean-constrained PMF or explicitly endpoint-clipped beta mixture"))
