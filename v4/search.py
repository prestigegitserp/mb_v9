"""Search and mechanism ablations that actually change V4 computations."""
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
    baseline.methods = tuple(m for m in base.methods if m not in base.v4.models)
    baseline_plan=list(baseline_jobs(pub,baseline,ks,mode).items()) if baseline.methods else []
    knobs = {
        "Method1": {"information_ridge": (.1, 1., 10.), "cavity_strength": (.1, .5, 1.)},
        "Method2": {"router_ridge": (.005, .02, .08), "router_context_penalty": (0., .25, 1.), "consistency_weight": (0., .02, .1)},
        "Method3": {"radius_precision": (.5, 2., 8.), "radius_noise": (.15, .25, .4)},
        "Method4": {"moment_dispersion_weight": (.02, .1, .3), "prior_interior_median": (.4, .6, .8)},
        "Method5": {"posterior_precision": (5., 20., 80.), "posterior_relaxation": (.1, .25, .5)},
        "Method6": {"transport_moment_weight": (0., .02, .1), "radius_temperature": (1., 2., 4.)},
        "Method7": {"ensemble_mean_weight": (0., .25, .5), "feature_dropout": (0., .15, .3)},
    }
    for method in base.methods:
        if method not in base.v4.models:
            continue
        def variant(tag, values):
            cfg = copy.deepcopy(base)
            cfg.methods = (method,)
            for key, value in values.items():
                setattr(cfg.v4.models[method], key, value)
            # Preserve the declared loss/geometry coupling when searching its
            # shared dispersion coefficient; an independent Euclidean control
            # is exposed by the mechanism ablations below.
            if method == "Method4" and "moment_dispersion_weight" in values:
                current=base.v4.models[method]
                if current.moment_projection_weight == current.moment_dispersion_weight:
                    cfg.v4.models[method].moment_projection_weight=values["moment_dispersion_weight"]
            cfg.v4.validate()
            insert(tag+"__"+method, cfg)
        for factor in (.5, 2.):
            original = base.v4.models[method]
            variant(f"lr_factor={factor}", dict(lr=original.lr*factor,
                    direction_lr=original.direction_lr*factor, bias_lr=original.bias_lr*factor))
        for jitter in (0., .15):
            if jitter < min(base.v4.models[method].mask_fraction, 1-base.v4.models[method].mask_fraction):
                variant(f"mask_jitter={jitter}",dict(mask_fraction_jitter=jitter))
        for readout in ("geometric","coupled","decoder"):
            variant("readout="+readout,dict(readout_mode=readout))
        for penalty in (0.,10.,100.,1000.):
            variant(f'collaborative_l2={penalty}',dict(collaborative_l2=penalty))
        if method=="Method4":
            for link,scale in (("linear",.25),("linear",1.),("tanh",.25),("tanh",1.)):
                variant(f"moment_transfer={link},{scale}",dict(moment_refinement_link=link,moment_refinement_scale=scale))
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
    for method,attribute,names,values in (
        ("AutoRec","autorec",("lambda_l2",),(.001,.01,.1,1.,10.,100.,1000.)),
        ("AEMC-NE","aemcne",("lambda_main","lambda_element"),(.01,.1,1.,10.,50.,100.,200.,500.))):
        if method not in base.methods:continue
        for value in values:
            cfg=copy.deepcopy(base);cfg.methods=(method,)
            for name in names:setattr(getattr(cfg,attribute),name,value)
            insert(f"paper_regularization={value}__{method}",cfg)
    if "AEMC-NE" in base.methods and mode in ("joint","hybrid"):
        for main,element in itertools.product((.01,.1,1.,10.,50.,100.,200.,500.),repeat=2):
            if main==element:continue
            cfg=copy.deepcopy(base);cfg.methods=("AEMC-NE",)
            cfg.aemcne.lambda_main,cfg.aemcne.lambda_element=main,element
            insert(f"paper_joint_regularization={main},{element}__AEMC-NE",cfg)
    for tag,cfg in baseline_plan:insert("baseline__"+tag,cfg)
    if base.v4.protocol=='common' and base.v4.mbmf_checkpoint_policy=='validation':
        for method in ('MBMF-F','MBMF-C','MBMF-N'):
            if method not in base.methods:continue
            for lr in (.001,.003,.01):
                cfg=copy.deepcopy(base);cfg.methods=(method,)
                cfg.v4.mbmf_optimizer='adam';cfg.v4.mbmf_adam_lr=lr
                insert(f'optimizer=adam,lr={lr}__{method}',cfg)
    if "AutoRec" in base.methods:
        for lr in (.001,.01,.1):
            cfg=copy.deepcopy(base);cfg.methods=("AutoRec",)
            cfg.autorec.optimizer,cfg.autorec.lr="rprop",lr
            insert(f"rprop_lr={lr}__AutoRec",cfg)
        if base.data.dataset=='ml-100k':
            # Carry the measured, near-matched-capacity controls into the main
            # scan, rather than leaving stronger baselines in a side study.
            for hidden,penalty in itertools.product((146,265),(10.,100.)):
                cfg=copy.deepcopy(base);cfg.methods=("AutoRec",)
                cfg.autorec.hidden_dim,cfg.autorec.lambda_l2=hidden,penalty
                cfg.autorec.optimizer,cfg.autorec.lr="rprop",.01
                insert(f"capacity_hidden={hidden},l2={penalty}__AutoRec",cfg)
    limit=base.v4.search_jobs_per_method
    if limit:
        # A bounded exploratory grid must still expose strong competitors.
        # Do not let insertion order spend every AutoRec slot on weak L2 values.
        chosen={method:set() for method in base.methods}
        def priority(method,tag,cfg):
            if tag==f'default__{method}':return 0
            if method=='AutoRec' and cfg.autorec.lambda_l2==100. and cfg.autorec.optimizer=='rprop':return 1
            if method=='AEMC-NE' and cfg.aemcne.lambda_main==cfg.aemcne.lambda_element==100.:return 1
            if method.startswith('MBMF-') and cfg.v4.mbmf_optimizer=='adam':return 1 if cfg.v4.mbmf_adam_lr==.01 else 2
            if method in cfg.v4.models and tag.startswith('collaborative_l2=1000.'):return 1
            if method in cfg.v4.models and tag.startswith('readout=coupled'):return 2
            return 3
        for method in base.methods:
            candidates=[(index,tag,cfg) for index,(tag,cfg) in enumerate(jobs.items()) if method in cfg.methods]
            candidates.sort(key=lambda item:(priority(method,item[1],item[2]),item[0]))
            chosen[method]={tag for _,tag,_ in candidates[:limit]}
        retained={}
        for tag,cfg in jobs.items():
            applicable=tuple(m for m in cfg.methods if tag in chosen[m])
            if applicable:
                cfg.methods=applicable;retained[tag]=cfg
        return retained
    return jobs


def ablations(base):
    changes = {
        "Method1": {"no_cavity":dict(tangent_cavity=False), "no_information_filter": dict(information_filter=False),
                    "no_gradient_surgery": dict(gradient_surgery=False),
                    "no_steering": dict(steering_enabled=False),
                    "fixed_radii": dict(radius_enabled=False)},
        "Method2": {"coarse_conditional_risk":dict(router_fine_benefit=False), "observed_curvature":dict(router_curvature=False), "legacy_router":dict(router_curvature=False,router_context_only=False), "fixed_gate": dict(router_enabled=False),
                    "count_only_gate": dict(router_counts_only=True),
                    "no_within_view_consistency": dict(consistency_weight=0.)},
        "Method3": {"global_radius":dict(context_radius=False), "no_context_directions":dict(graph_context_directions=False), "no_graph_uncertainty": dict(graph_uncertainty=False),
                    "strong_radius_prior": dict(radius_precision=20.)},
        "Method4": {"euclidean_moment_metric":dict(moment_projection_mode="metric",moment_projection_weight=1.), "weighted_moment_metric":dict(moment_projection_mode="metric",moment_projection_weight=base.v4.models["Method4"].moment_projection_weight), "no_moment_refinement":dict(moment_refinement=False), "legacy_moment_loss":dict(orthogonal_moment_loss=False), "single_signed_head": dict(signed_moments=False),
                    "no_moment_polygon": dict(project_moments=False)},
        "Method5": {"plugin_mean":dict(posterior_mean_steps=0), "legacy_solver":dict(posterior_preconditioned=False,posterior_cg_steps=12), "independent_posterior": dict(joint_posterior=False),
                    "zero_cross_covariance": dict(posterior_cross_covariance=False),
                    "no_reconstruction": dict(reconstruction_weight=0.)},
        "Method6": {"hard_balance":dict(hard_transport_balance=True), "legacy_reference":dict(hierarchical_reference=False), "no_moment_prior":dict(transport_moment_weight=0.), "analytic_transport": dict(learned_transport=False),
                    "no_transport": dict(transport_enabled=False),
                    "isotropic_reference": dict(angular_reference_weight=0.),
                    "degree_only_reference": dict(conditional_reference=False),
                    "fixed_radii": dict(radius_enabled=False),
                    "no_decoder_coupling": dict(decoder_coupling=False)},
        "Method7": {"statistics_only":dict(identity_history=False), "nll_only_shape":dict(probability_rps_weight=0.), "no_ensemble_mean_loss":dict(ensemble_mean_weight=0.), "one_mean_member": dict(ensemble_members=1),
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
                setattr(cfg.v4.models[method], key, value)
            # Cutting a parent representation also removes its dependent readout;
            # disclose the resulting config in the ablation provenance table.
            if (method=="Method7" and not cfg.v4.models[method].identity_history) or (method=="Method3" and not cfg.v4.models[method].context_radius):
                cfg.v4.models[method].readout_mode="geometric"
            result.append((method+"__"+tag, cfg))
        current=base.v4.models[method]
        for tag,overrides in (("geometry_only",dict(readout_mode="geometric")),
                              ("decoder_only_null",dict(readout_mode="decoder")),
                              ("fixed_readout_gate",dict(readout_mode="coupled",readout_count_adaptive=False)),
                              ("no_collaborative_l2",dict(collaborative_l2=0.))):
            cfg=copy.deepcopy(base);cfg.methods=(method,)
            for key,value in overrides.items():setattr(cfg.v4.models[method],key,value)
            if vars(cfg.v4.models[method])!=vars(current):result.append((method+"__"+tag,cfg))
        if method=='Method4' and (current.moment_refinement_link!='tanh' or current.moment_refinement_scale!=.25):
            for tag,penalty in (('legacy_moment_readout',current.collaborative_l2),('legacy_readout_no_l2',0.)):
                cfg=copy.deepcopy(base);cfg.methods=(method,)
                cfg.v4.models[method].moment_refinement_link='tanh';cfg.v4.models[method].moment_refinement_scale=.25
                cfg.v4.models[method].collaborative_l2=penalty
                if tag=='legacy_readout_no_l2' and current.collaborative_l2==0.:continue
                result.append((method+'__'+tag,cfg))
    return result
