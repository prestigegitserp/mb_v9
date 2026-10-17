"""Numerical, isolation and integration tests for the seven research replacements."""
from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from v3.config import ModelConfig, StudyConfig
from v3.history import masked_targets
from v3.models import instantiate, Method7
from v3.numerics import (conjugate_gradient, spectral_filter, sphere_exp, unit,
                         signed_moment_projection, gaussian_bilinear_moments)
from v3.laws import mean_tilt, mixture_variance, ordinal_nll
from v3.protocol import validation_roles, data_signature
from v3.transport import AngularTransport
from v3.training import train_v3, optimizer_groups, fit_probability
from v3 import checkpoints


def tiny_data(continuous=False):
    gen = torch.Generator().manual_seed(608)
    pair = torch.randperm(12*16, generator=gen)
    u, i = pair//16, pair % 16
    y = (1+(u % 3)+(i % 3)).float().clamp(1, 5)
    if continuous:
        y = 1+.7*(y-1)+.123
    data = SimpleNamespace(num_users=12, num_items=16, rating_min=1., rating_max=5.,
                           rating_step=1., midpoint=3., half_range=2.)
    for name, sl in (("train", slice(0, 128)), ("val", slice(128, 168)), ("test", slice(168, None))):
        for part, arr in (("users", u[sl]), ("items", i[sl]), ("ratings_raw", y[sl])):
            setattr(data, f"{name}_{part}", arr.clone())
        for part in ("users", "items"):
            setattr(data, f"{name}_{part}_np", getattr(data, f"{name}_{part}").numpy())
        setattr(data, f"{name}_ratings_np", y[sl].numpy())
    data.train_timestamps = torch.arange(128, dtype=torch.float64)+1e9
    data.user_train_mean_raw = torch.stack([data.train_ratings_raw[data.train_users == user].mean() for user in range(12)])
    return data


def config(method):
    return ModelConfig(method=method, latent_dim=4, hidden_dim=8, max_epochs=3, eval_every=1,
                       val_patience=2, steps_per_epoch=1, warmup_steps=2, router_expert_epochs=2,
                       router_epochs=3, router_replicates=2, shape_epochs=3, shape_patience=2,
                       posterior_probes=3, posterior_cg_steps=12, graph_probes=3, cg_steps=20,
                       angular_reference_pairs=128)


def training_pub():
    def evaluate(model, data, split, **kwargs):
        with torch.no_grad():
            model.eval()
            prediction = model.predict_raw_pairs(data, getattr(data, split+"_users"), getattr(data, split+"_items"))
            return {"RMSE": float((prediction-getattr(data, split+"_ratings_raw")).square().mean().sqrt())}
    return SimpleNamespace(MetricConfig=lambda **kw: SimpleNamespace(**kw), evaluate_publication_model=evaluate)


@pytest.mark.parametrize("method", [f"Method{i}" for i in range(1, 8)])
def test_training_checkpoint_catalog_and_no_test_read(method, tmp_path):
    data = validation_roles(tiny_data(), StudyConfig())
    # Poison TEST completely. Training and all fitting stages must ignore it.
    data.test_ratings_raw[:] = float("nan")
    model = instantiate(data, config(method), seed=17)
    result = train_v3(training_pub(), model, data, model.cfg, None)
    assert result["best_epoch"] >= 1
    assert np.isfinite(result["best_val_RMSE"])
    with torch.no_grad():
        prediction = model.predict_raw_pairs(data, data.val_users, data.val_items)
    assert torch.isfinite(prediction).all()
    assert prediction.min() >= 1-1e-5 and prediction.max() <= 5+1e-5
    path = tmp_path/f"{method}.pt"
    checkpoints.save(path, model, config=model.cfg, data=data, train_info=result)
    restored, payload = checkpoints.load(path, data, torch.device("cpu"))
    with torch.no_grad():
        actual = restored.predict_raw_pairs(data, data.val_users, data.val_items)
    torch.testing.assert_close(prediction, actual, atol=2e-6, rtol=1e-6)
    catalog = restored.score_catalog(data, torch.tensor([0, 2]), chunk_size=5)
    torch.testing.assert_close(catalog[1], restored.predict_raw_pairs(data, torch.full((16,), 2), torch.arange(16)))
    bad = copy.copy(data)
    bad.train_ratings_raw = data.train_ratings_raw.clone()
    bad.train_ratings_raw[0] += .1
    with pytest.raises(ValueError, match="train_sha256"):
        checkpoints.load(path, bad, torch.device("cpu"))
    if method == "Method2":
        assert result["v3_routing"]["folds"] == 3
        assert result["v3_routing"]["oof_pairs"] == len(data.train_users)
    if method == "Method7":
        dist = restored.distribution(data, data.val_users, data.val_items)
        torch.testing.assert_close(dist["probabilities"]@restored.levels, actual, atol=2e-5, rtol=1e-6)
        assert result["v3_probability"]["mean_parameters_unchanged"]


@pytest.mark.parametrize("method", [f"Method{i}" for i in range(1, 8)])
def test_cavity_features_and_inference_ignore_target_labels(method):
    data = tiny_data()
    model = instantiate(data, config(method), seed=3)
    model.eval()
    include = torch.ones(128, dtype=torch.bool)
    include[[0, 2, 7, 20]] = False
    h = model.history(include)
    s = model.prepare(h)
    query_u, query_i = data.train_users[~include], data.train_items[~include]
    with torch.no_grad():
        before = model.predict(h, s, query_u, query_i)[0].clone()
    data.train_ratings_raw[~include] = 1000  # Must not enter priors, moments or covariance.
    data.train_timestamps[~include] = -1e12
    h2 = model.history(include)
    s2 = model.prepare(h2)
    torch.testing.assert_close(h.user_features, h2.user_features)
    torch.testing.assert_close(h.item_amplitude, h2.item_amplitude)
    with torch.no_grad():
        after = model.predict(h2, s2, query_u, query_i)[0]
    torch.testing.assert_close(before, after, atol=2e-6, rtol=1e-6)


def test_duplicate_pair_mask_and_validation_role_isolation():
    data = tiny_data()
    data.train_users = torch.cat([data.train_users, data.train_users[:5]])
    data.train_items = torch.cat([data.train_items, data.train_items[:5]])
    include, query = masked_targets(data, .5, torch.Generator().manual_seed(8))
    pair = data.train_users*16+data.train_items
    assert not torch.isin(pair[include], pair[query]).any()
    roles = validation_roles(data, StudyConfig())
    sets = [set((getattr(roles, p+"users")*16+getattr(roles, p+"items")).tolist()) for p in
            ("val_", "v3_calibration_fit_", "v3_calibration_hold_")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sum(map(len, sets)) == len(data.val_users)
    altered = copy.deepcopy(data)
    altered.val_ratings_raw *= -1
    assert validation_roles(altered, StudyConfig()).v3_role_signature == roles.v3_role_signature
    assert roles.test_users is data.test_users and roles.train_users is data.train_users


def test_cg_spectral_geometry_and_zero_information():
    gen = torch.Generator().manual_seed(4)
    matrix = torch.randn(9, 9, generator=gen, dtype=torch.double)
    matrix = matrix.T@matrix+.5*torch.eye(9)
    rhs = torch.randn(9, generator=gen, dtype=torch.double)
    solve, residual = conjugate_gradient(lambda x: matrix@x, rhs, 30, 1e-10)
    torch.testing.assert_close(solve, torch.linalg.solve(matrix, rhs), atol=1e-8, rtol=1e-8)
    assert residual < 1e-8
    zero, _ = conjugate_gradient(lambda x: matrix@x, torch.zeros_like(rhs))
    assert not zero.any()
    q, v = unit(torch.randn(4, 5)), torch.randn(4, 5)
    filtered = spectral_filter(torch.zeros(4, 5, 5), v, 1.)
    torch.testing.assert_close(sphere_exp(q, filtered), q)
    torch.testing.assert_close(sphere_exp(q, v).norm(dim=-1), torch.ones(4), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("step", [None, 1., .5])
def test_signed_moment_polygon(step):
    gen = torch.Generator().manual_seed(33)
    base = 1+4*torch.rand(500, generator=gen)
    proposal = (5*torch.rand(500, 2, generator=gen)).requires_grad_()
    actual = signed_moment_projection(proposal, base, 1, 5, step)
    assert (actual >= -1e-5).all()
    # Continuous feasibility upper edge.
    assert (actual[:, 0]/(5-base)+actual[:, 1]/(base-1) <= 1+2e-4).all()
    if step is not None:
        lower = 1+((base-1)/step).floor()*step
        upper = 1+((base-1)/step).ceil()*step
        assert (actual[:, 0]/(upper-base).clamp_min(1e-6)
                +actual[:, 1]/(base-lower).clamp_min(1e-6) >= 1-2e-4).all()
    actual.square().sum().backward()
    assert torch.isfinite(proposal.grad).all()


@pytest.mark.parametrize("correlation", [0., .4, 1.])
def test_joint_gaussian_moments_against_monte_carlo(correlation):
    torch.manual_seed(31)
    mu, mi = torch.randn(1, 3, dtype=torch.double), torch.randn(1, 3, dtype=torch.double)
    lu, li = .2*torch.randn(1, 3, 2, dtype=torch.double), .2*torch.randn(1, 3, 2, dtype=torch.double)
    mean, variance, _ = gaussian_bilinear_moments(mu, mi, lu, li, correlation)
    a, b = torch.randn(140000, 2, dtype=torch.double), torch.randn(140000, 2, dtype=torch.double)
    z = mu+a@lu[0].T
    d = mi+(correlation*a+(1-correlation**2)**.5*b)@li[0].T
    score = (z*d).sum(-1)
    assert abs(float(score.mean()-mean)) < .012
    assert abs(float(score.var(unbiased=False)-variance)) < .015


def test_m6_spline_constraints_and_actual_feasible_learning_gradient():
    model = AngularTransport(8)
    values = model.projected_values()
    mass = values.diff(dim=-1)
    torch.testing.assert_close(mass@model.matrix.T, model.target, atol=3e-7, rtol=1e-6)
    assert mass.min() >= -1e-7 and (mass <= model.cap+1e-7).all()
    v = torch.linspace(.001, .999, 1000)
    p = torch.full_like(v, .4)
    q = model.quantile(v, p, values)
    assert q.diff().min() >= -1e-6
    assert abs(float(q.mean())-.4) < .002
    assert abs(float(model.quantile(torch.tensor([.5]), torch.tensor([.4]), values))-.4) < 1e-6
    (q*torch.sin(v*7)).mean().backward()
    assert torch.isfinite(model.mass.grad).all() and model.mass.grad.norm() > .01
    # A feasible first-order shape update preserves every equality constraint.
    assert (model.mass.grad@model.matrix.T).abs().max() < 2e-6


def test_m6_extreme_baseline_closure_is_continuous_bounded_and_mean_anchored():
    model = AngularTransport(4)
    values = model.projected_values()
    v = torch.linspace(0., 1., 20001)
    for center in (.001, .01999, .02001, .97999, .98001, .999):
        p = torch.full_like(v, center)
        quantile = model.quantile(v, p, values)
        assert quantile.min() >= -1e-6 and quantile.max() <= 1+1e-6
        assert quantile.diff().min() >= -1e-6
        assert abs(float(torch.trapezoid(quantile, v))-center) < 8e-5
        middle = model.quantile(torch.tensor([.5]), torch.tensor([center]), values)
        assert abs(float(middle)-center) < 2e-6
    probe = torch.linspace(0, 1, 137)
    a = model.quantile(probe, torch.full_like(probe, .02-1e-7), values)
    b = model.quantile(probe, torch.full_like(probe, .02+1e-7), values)
    assert (a-b).abs().max() < 1e-4


def test_mean_tilt_implicit_gradient_and_full_shape_dimension():
    torch.manual_seed(54)
    logits = torch.randn(3, 5, dtype=torch.double, requires_grad=True)
    means = torch.tensor([1.15, 2.8, 4.8], dtype=torch.double, requires_grad=True)
    levels = torch.arange(1., 6., dtype=torch.double)
    assert torch.autograd.gradcheck(lambda a, b: mean_tilt(a, b, levels), (logits, means), atol=1e-5, rtol=1e-4)
    probabilities = mean_tilt(logits, means, levels)
    torch.testing.assert_close(probabilities@levels, means, atol=1e-10, rtol=1e-10)
    jac = torch.autograd.functional.jacobian(lambda a: mean_tilt(a, means, levels), logits)[1, :, 1, :]
    assert int(torch.linalg.matrix_rank(jac, atol=1e-8)) == len(levels)-2
    grad = torch.autograd.grad((probabilities*levels.square()).sum(), logits)[0]
    assert (grad@levels).abs().max() < 1e-10
    with pytest.raises(ValueError, match="Continuous"):
        ordinal_nll(probabilities, torch.tensor([1.1, 2., 5.]), levels)


def test_m7_exact_total_variance_and_shape_gradient_isolation():
    data = validation_roles(tiny_data(), StudyConfig())
    model = instantiate(data, config("Method7"))
    model.eval()
    model.shape_heads.requires_grad_(True)
    h, s = model.eval_state()
    loss = model.probability_loss(h, s, data.val_users, data.val_items, data.val_ratings_raw)
    loss.backward()
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith("shape_heads."))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.shape_heads.parameters())
    dist = model.distribution(data, data.val_users, data.val_items)
    mean = dist["probabilities"]@model.levels
    exact = (dist["probabilities"]*(model.levels-mean[:, None]).square()).sum(-1)
    torch.testing.assert_close(dist["variance"], exact, atol=3e-6, rtol=1e-5)
    assert (dist["variance"] <= (5-mean)*(mean-1)+1e-5).all()


@pytest.mark.parametrize("full", [True, False])
def test_continuous_fallback_is_not_silently_quantized(full):
    data = validation_roles(tiny_data(continuous=True), StudyConfig())
    model = instantiate(data, replace(config("Method7"), shape_full=full))
    model.eval()
    assert not model.ordinal
    before = model.predict_raw_pairs(data, data.val_users, data.val_items).detach()
    info = fit_probability(model, data)
    after = model.predict_raw_pairs(data, data.val_users, data.val_items).detach()
    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)
    assert "beta" in info["law"]
    assert torch.isfinite(model.distribution(data, data.val_users, data.val_items)["variance"]).all()


@pytest.mark.parametrize("center", [-1000., 1000.])
def test_continuous_probability_keeps_the_mean_even_with_saturated_bias(center):
    data = tiny_data(continuous=True)
    model = instantiate(data, config("Method7"))
    with torch.no_grad():
        for member in model.members:
            member.bias.center.fill_(center)
            member.trust.fill_(-1000.)
    model.eval()
    h, state = model.eval_state()
    means, _, beta = model.probability_members(h, state, data.val_users, data.val_items)
    for j, (a, b, mix) in enumerate(beta):
        component_means = 1+4*a/(a+b)
        torch.testing.assert_close(component_means, means[j, :, None].expand_as(component_means), atol=5e-7, rtol=1e-6)
    assert means.min() > 1 and means.max() < 5


def test_m3_radius_block_line_search_and_optimizer_decay_budget():
    data = tiny_data()
    model = instantiate(data, config("Method3"))
    model.refresh_radius()
    assert model.last_radius_objective_after <= model.last_radius_objective_before+1e-6
    assert model.delta.abs().max() <= model.cfg.log_radius_bound
    groups, decay = optimizer_groups(model, 100)
    assert abs((1-model.cfg.lr*decay)**100-(1-model.cfg.encoder_decay_budget)) < 2e-5
    assert all(group["weight_decay"] == 0 for group in groups if group["name"] != "encoder")
    assert all(p is not model.delta for group in groups for p in group["params"])


@pytest.mark.parametrize("method", [f"Method{i}" for i in range(1, 8)])
def test_all_seven_learn_fixed_masked_objective_with_finite_gradients(method):
    data = tiny_data()
    cfg = replace(config(method), lr=.01, direction_lr=.01, bias_lr=.01,
                  feature_dropout=0., gradient_surgery=False)
    model = instantiate(data, cfg, 6)
    groups, _ = optimizer_groups(model, 20)
    optimizer = torch.optim.AdamW(groups)
    include, query = masked_targets(data, .25, torch.Generator().manual_seed(19))
    u, i, y = data.train_users[query], data.train_items[query], data.train_ratings_raw[query]
    losses = []
    for _ in range(20):
        model.train()
        h = model.history(include)
        state = model.prepare(h)
        prediction, auxiliary, _ = model.loss_for_targets(h, state, u, i, y)
        loss = prediction+auxiliary
        losses.append(float(loss.detach()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 5.)
        optimizer.step()
    assert min(losses[-3:]) < .95*losses[0], (method, losses)


def test_seed_reproducibility_is_independent_of_method_order():
    data = validation_roles(tiny_data(), StudyConfig())
    first = instantiate(data, config("Method7"), 12)
    second = instantiate(data, config("Method7"), 12)
    torch.manual_seed(873)
    train_v3(training_pub(), first, data, first.cfg, None)
    torch.manual_seed(42)
    train_v3(training_pub(), second, data, second.cfg, None)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)


def test_scoped_runtime_restoration_search_and_fingerprint(tmp_path):
    from v9 import engine
    from v9.selection import fingerprint_config
    from v3.runtime import installed
    options = SimpleNamespace(study=StudyConfig(), max_epochs=2, steps_per_epoch=1, no_plots=True)
    old = engine.make_base, engine.load_v8, engine.build_jobs
    with pytest.raises(RuntimeError, match="deliberate"):
        with installed(engine, options) as (core, pub):
            args = SimpleNamespace(data_root=str(tmp_path), backend="auto", split_seeds="2026",
                                   model_seeds="1,2", k="4", methods="Method1,Method7", eval_every=1, quick=False)
            cfg = engine.make_base(pub, args, "synthetic")
            before = fingerprint_config(cfg)
            cfg.v3.models["Method1"].information_ridge *= 2
            assert fingerprint_config(cfg) != before
            jobs = engine.build_jobs(pub, cfg, (4,), "hybrid")
            assert jobs and all(job.validation_only for job in jobs.values())
            assert all(not tag.startswith("angle") for tag in jobs)
            assert len({fingerprint_config(job) for job in jobs.values()}) == len(jobs)
            variants = pub.make_ablation_variants_v8(cfg)
            assert variants and all(job.methods[0] in cfg.methods for _, job in variants)
            raise RuntimeError("deliberate")
    assert (engine.make_base, engine.load_v8, engine.build_jobs) == old


def test_default_entry_is_v3_and_v2_entry_remains_explicit():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert "from v3.entry import run" in (root/"run_v9.py").read_text()
    assert "from v2.entry import run" in (root/"run_v2.py").read_text()


@pytest.mark.parametrize("method", [f"Method{i}" for i in range(1, 8)])
def test_dense_and_sparse_data_backends_predict_equivalently(method):
    import pandas as pd
    from v9.engine import load_v8
    core, pub = load_v8()
    source = tiny_data()
    frames = []
    for split in ("train", "val", "test"):
        frames.append(pd.DataFrame(dict(userId=getattr(source, split+"_users").numpy(),
                                        movieId=getattr(source, split+"_items").numpy(),
                                        rating=getattr(source, split+"_ratings_raw").numpy(), split=split)))
    frame = pd.concat(frames, ignore_index=True)
    predictions = []
    for backend in ("dense", "sparse"):
        dc = pub.PublicationDataConfig(dataset="ml-100k", force_backend=backend)
        data, _ = pub.build_publication_matrix_data_v8(frame, pub.DATASET_SPECS["ml-100k"], dc, torch.device("cpu"))
        model = instantiate(data, config(method), 37)
        model.eval()
        with torch.no_grad():
            predictions.append(model.predict_raw_pairs(data, data.val_users, data.val_items))
    torch.testing.assert_close(*predictions, atol=3e-6, rtol=1e-6)


@pytest.mark.parametrize("role", ["val_ratings_raw", "v3_calibration_fit_ratings", "v3_calibration_hold_ratings"])
def test_strict_checkpoint_rejects_changed_selection_or_calibration_labels(role, tmp_path):
    data = validation_roles(tiny_data(), StudyConfig())
    model = instantiate(data, config("Method7"))
    path = tmp_path/"strict.pt"
    checkpoints.save(path, model, config=model.cfg, data=data)
    altered = copy.deepcopy(data)
    getattr(altered, role)[0] += .1
    with pytest.raises(ValueError, match="checkpoint/data mismatch"):
        checkpoints.load(path, altered, torch.device("cpu"))


def test_strict_checkpoint_rejects_source_drift(tmp_path):
    data = validation_roles(tiny_data(), StudyConfig())
    model = instantiate(data, config("Method1"))
    path = tmp_path/"source.pt"
    checkpoints.save(path, model, config=model.cfg, data=data)
    payload = torch.load(path, weights_only=False)
    payload["implementation_sha256"] = "stale-source"
    with pytest.raises(ValueError, match="implementation SHA"):
        checkpoints.restore(payload, data, torch.device("cpu"))


@pytest.mark.parametrize("pvalue", [.0001, .01, .99, .9999])
def test_analytic_transport_extreme_baselines_have_finite_backward(pvalue):
    v = torch.linspace(0., 1., 401, requires_grad=True)
    p = torch.full_like(v, pvalue, requires_grad=True)
    value = AngularTransport.power_quantile(v, p)
    value.sum().backward()
    assert torch.isfinite(v.grad).all() and torch.isfinite(p.grad).all()
    assert value.min() >= 0 and value.max() <= 1


@pytest.mark.parametrize("slope", [-30000., 30000.])
def test_mean_tilt_large_affine_gauge_keeps_the_prescribed_mean(slope):
    levels = torch.arange(1., 6., dtype=torch.double)
    logits = torch.tensor([[2., -.3, .8, 1.1, -.2]], dtype=torch.double)
    means = torch.tensor([2.7], dtype=torch.double)
    reference = mean_tilt(logits, means, levels)
    actual = mean_tilt(logits+slope*levels+99999, means, levels)
    torch.testing.assert_close(actual@levels, means, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(actual, reference, atol=1e-9, rtol=1e-9)


def test_mean_tilt_brackets_large_nonlinear_logits():
    levels = torch.arange(1., 6., dtype=torch.double)
    logits = torch.tensor([[50000., -9000., 40000., 20000., -20000.]], dtype=torch.double)
    means = torch.tensor([4.35], dtype=torch.double)
    probability = mean_tilt(logits, means, levels)
    torch.testing.assert_close(probability@levels, means, atol=2e-8, rtol=1e-9)


@pytest.mark.parametrize("overrides", [{"lr": float("nan")}, {"lr": float("inf")},
    {"max_epochs": 2.5}, {"max_epochs": True}, {"angular_grid": 1},
    {"lr": ".001"}, {"learned_transport": "false"}])
def test_invalid_configuration_fails_before_training(overrides):
    with pytest.raises(ValueError):
        ModelConfig(**overrides).validate()


def _archived_scan_example(tmp_path):
    import json
    import pandas as pd
    from dataclasses import asdict, field, make_dataclass
    from v9.engine import load_v8
    core, pub = load_v8()
    Config = make_dataclass("FrozenExample", [("v3", StudyConfig, field(default_factory=StudyConfig))],
                            bases=(pub.PublicationExperimentConfigV8,))
    base = Config()
    base.data.dataset = "synthetic"
    base.methods = ("Method7",)
    base.core_config.model_seeds, base.core_config.split_seeds = (1,), (2026,)
    base.core_config.latent_dims = (4,)
    candidate = copy.deepcopy(base)
    candidate.save_predictions = candidate.save_selected_checkpoints = False
    candidate.validation_only = True
    folder = tmp_path/"search"/"jobs"/"default__Method7"
    folder.mkdir(parents=True)
    (folder/"config.json").write_text(json.dumps(asdict(candidate)))
    data = validation_roles(tiny_data(), base.v3)
    (folder/"V3_PROTOCOL_2026.json").write_text(json.dumps(data.v3_protocol))
    key = pub._candidate_config_key_v8("Method7", replace(base.v3.models["Method7"], latent_dim=4), "NA")
    winners = pd.DataFrame([dict(dataset="synthetic", method="Method7", split_seed=2026,
                                scan_job="default__Method7", config_key=key, K=4, init_mode="NA")])
    return base, candidate, winners


def test_final_restores_artifact_retention_from_search_settings(tmp_path):
    from v3.selection import prepare_locked_plan
    base, candidate, winners = _archived_scan_example(tmp_path)
    plan, specifications = prepare_locked_plan(base, tmp_path, winners, {"default__Method7": candidate})
    final = plan["default__Method7"]
    assert final.save_predictions and final.save_selected_checkpoints and not final.validation_only
    assert not candidate.save_predictions and candidate.validation_only
    assert ("Method7", 2026) in specifications
    assert "train_sha256" in specifications[("Method7", 2026)]["data_signature"]


@pytest.mark.parametrize("changed", ["model", "role", "data", "seeds", "source"])
def test_locked_refit_rejects_changed_scan_configuration(tmp_path, changed):
    from v3.selection import prepare_locked_plan
    base, candidate, winners = _archived_scan_example(tmp_path)
    if changed == "model":
        candidate.v3.models["Method7"].lr *= 2
    elif changed == "role":
        candidate.v3.role_seed += 1
    elif changed == "data":
        candidate.data.train_fraction -= .1
    elif changed == "seeds":
        candidate.core_config.model_seeds = (1, 2)
    else:
        candidate.v3.implementation_sha256 = "stale-source"
    with pytest.raises(ValueError, match="Frozen scan mismatch"):
        prepare_locked_plan(base, tmp_path, winners, {"default__Method7": candidate})


def test_radius_budget_counts_extra_full_training_block_updates():
    data = tiny_data()
    model = instantiate(data, config("Method3"), 6)
    info = train_v3(training_pub(), model, data, model.cfg, None)
    assert info["radius_block_updates"] >= len(info["history"])
    assert info["radius_block_targets_seen"] == info["radius_block_updates"]*len(data.train_users)


@pytest.mark.parametrize("changed", ["data", "winner_key"])
def test_locked_runtime_rechecks_actual_data_and_selected_config(tmp_path, monkeypatch, changed):
    from v9 import engine
    from v3.runtime import installed
    base, candidate, winners = _archived_scan_example(tmp_path)
    def refit(pub, core, base, root, args, scan_results):
        cfg = scan_results[0]["default__Method7"]
        data = validation_roles(tiny_data(), base.v3)
        if changed == "data":
            data.train_ratings_raw[0] += .25  # Even stale cached metadata must not hide a change.
        return pub.instantiate_model_v8("Method7", data, cfg, 4, 1, torch.device("cpu"))
    monkeypatch.setattr(engine, "run_final", refit)
    if changed == "winner_key":
        winners.loc[0, "config_key"] = '{"method":"different-model"}'
    options = SimpleNamespace(study=StudyConfig(), max_epochs=None, steps_per_epoch=None, no_plots=True)
    args = SimpleNamespace(k="4", scan_mode="one-factor")
    with installed(engine, options) as (core, pub):
        with pytest.raises(ValueError, match="Locked refit"):
            engine.run_final(pub, core, base, tmp_path, args,
                             ({"default__Method7": candidate}, winners))
