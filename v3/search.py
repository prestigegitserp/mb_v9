"""Search and mechanism ablations that actually change V3 computations."""
from __future__ import annotations
import copy
import itertools


def build_jobs(pub, base, ks, mode="hybrid"):
    if mode not in ("one-factor", "hybrid", "joint"):
        raise ValueError(mode)
    jobs, identities = {}, set()
    from v9.selection import fingerprint_config
    def insert(tag, cfg):
        cfg.validation_only = True
        cfg.methods = tuple(m for m in cfg.methods if m in base.methods
                            and pub.method_supported_on_dataset_v8(m, base.data.dataset))
        if not cfg.methods:
            return
        identity = fingerprint_config(cfg)
        if identity not in identities:
            jobs[tag] = cfg
            identities.add(identity)
    for method in base.methods:
        cfg = copy.deepcopy(base)
        cfg.methods = (method,)
        insert(f"default__{method}", cfg)
    # Baseline tuning remains the existing baseline search, with no proposed
    # method accidentally routed through legacy angle/loss hyperparameters.
    from v9.search import build_jobs as baseline_jobs
    baseline = copy.deepcopy(base)
    baseline.methods = tuple(m for m in base.methods if m not in base.v3.models)
    if baseline.methods:
        for tag, cfg in baseline_jobs(pub, baseline, ks, mode).items():
            insert("baseline__"+tag, cfg)
    knobs = {
        "Method1": {"information_ridge": (.1, 1., 10.), "reconstruction_weight": (.01, .05, .2)},
        "Method2": {"router_ridge": (.005, .02, .08), "consistency_weight": (0., .02, .1)},
        "Method3": {"radius_precision": (.5, 2., 8.), "radius_noise": (.15, .25, .4)},
        "Method4": {"auxiliary_weight": (.05, .2, .5), "prior_interior_median": (.4, .6, .8)},
        "Method5": {"posterior_precision": (5., 20., 80.), "posterior_correlation_init": (.01, .05, .2)},
        "Method6": {"angular_reference_weight": (0., .5, .8), "radius_temperature": (1., 2., 4.)},
        "Method7": {"count_tau": (3., 10., 30.), "feature_dropout": (0., .15, .3)},
    }
    for method in base.methods:
        if method not in base.v3.models:
            continue
        def variant(tag, values):
            cfg = copy.deepcopy(base)
            cfg.methods = (method,)
            for key, value in values.items():
                setattr(cfg.v3.models[method], key, value)
            cfg.v3.validate()
            insert(tag+"__"+method, cfg)
        for factor in (.5, 2.):
            original = base.v3.models[method]
            variant(f"lr_factor={factor}", dict(lr=original.lr*factor,
                    direction_lr=original.direction_lr*factor, bias_lr=original.bias_lr*factor))
        if mode in ("one-factor", "hybrid"):
            for key, values in knobs[method].items():
                for value in values:
                    variant(f"{key}={value}", {key: value})
        if mode in ("joint", "hybrid"):
            names = tuple(knobs[method])
            for values in itertools.product(*(knobs[method][name][::2] for name in names)):
                variant("joint_"+"_".join(f"{k}={v}" for k, v in zip(names, values)), dict(zip(names, values)))
            for hidden in (32, 128):
                variant(f"hidden={hidden}", dict(hidden_dim=hidden))
    return jobs


def ablations(base):
    changes = {
        "Method1": {"no_information_filter": dict(information_filter=False),
                    "no_gradient_surgery": dict(gradient_surgery=False),
                    "no_steering": dict(steering_enabled=False),
                    "fixed_radii": dict(radius_enabled=False)},
        "Method2": {"fixed_gate": dict(router_enabled=False),
                    "count_only_gate": dict(router_counts_only=True),
                    "no_within_view_consistency": dict(consistency_weight=0.)},
        "Method3": {"no_graph_uncertainty": dict(graph_uncertainty=False),
                    "strong_radius_prior": dict(radius_precision=20.)},
        "Method4": {"single_signed_head": dict(signed_moments=False),
                    "no_moment_polygon": dict(project_moments=False)},
        "Method5": {"independent_posterior": dict(joint_posterior=False),
                    "zero_cross_covariance": dict(posterior_cross_covariance=False),
                    "no_reconstruction": dict(reconstruction_weight=0.)},
        "Method6": {"analytic_transport": dict(learned_transport=False),
                    "no_transport": dict(transport_enabled=False),
                    "isotropic_reference": dict(angular_reference_weight=0.),
                    "no_reference_balance": dict(balance_transport=False),
                    "degree_only_reference": dict(conditional_reference=False),
                    "fixed_radii": dict(radius_enabled=False),
                    "no_decoder_coupling": dict(decoder_coupling=False)},
        "Method7": {"one_mean_member": dict(ensemble_members=1),
                    "restricted_shape": dict(shape_full=False),
                    "fixed_radii": dict(radius_enabled=False)},
    }
    result = []
    for method in base.methods:
        if method not in changes:
            continue
        for tag, values in [("full", {})]+list(changes[method].items()):
            cfg = copy.deepcopy(base)
            cfg.methods = (method,)
            for key, value in values.items():
                setattr(cfg.v3.models[method], key, value)
            result.append((method+"__"+tag, cfg))
    return result
