"""Search and mechanism ablations that actually change V7 computations."""
from __future__ import annotations
import copy
import itertools
from dataclasses import fields, replace


def build_jobs(pub, base, ks, mode="hybrid"):
    """Curated development or comprehensive search, never historical test selection."""
    from .config import ModelConfig
    from .lineage import historical_records
    from v9.selection import fingerprint_config
    jobs, seen = {}, set()
    limit = base.v7.search_jobs_per_method

    def insert(tag, cfg):
        cfg.methods = tuple(m for m in cfg.methods if m in base.methods and
                            pub.method_supported_on_dataset_v8(m, base.data.dataset))
        cfg.validation_only = True
        if not cfg.methods:
            return
        identity = fingerprint_config(cfg)
        if identity not in seen:
            if tag in jobs:
                raise ValueError("Ambiguous search job tag: " + tag)
            jobs[tag] = cfg
            seen.add(identity)

    def model_trial(method, tag, model):
        c = copy.deepcopy(base)
        c.methods = (method,)
        c.v7.models[method] = model.validate()
        insert(tag + "__" + method, c)

    import random
    from .losses import coefficients,trial_values
    joint_rng=random.Random(base.v7.search_seed)
    records = historical_records()
    allowed = {f.name for f in fields(ModelConfig)}
    for method in base.methods:
        if method not in base.v7.models:
            c = copy.deepcopy(base); c.methods = (method,); insert("default__" + method, c)
            continue
        original = base.v7.models[method]
        model_trial(method, "default", copy.deepcopy(original))
        record = records[method]
        control = ModelConfig(**{k: v for k, v in record["config"].items() if k in allowed})
        control.lineage = record["source_namespace"]
        control.response_shrinkage = False
        control.probability_fit = original.probability_fit
        # The actual archived architecture is preserved; CLI budgets remain
        # matched between controls and updated candidates, including smoke.
        for key in ("max_epochs", "steps_per_epoch", "eval_every", "router_expert_epochs",
                    "router_epochs", "shape_epochs", "graph_probes", "cg_steps",
                    "posterior_probes", "posterior_cg_steps"):
            setattr(control, key, getattr(original, key))
        model_trial(method, "historical_" + control.lineage, control)
        modern = replace(original, lineage="v7", response_shrinkage=True,
                         prediction_loss_units="mse", full_history_weight=0.)
        model_trial(method, "modern_mechanism", modern)
        model_trial(method, "gated_context", replace(modern,history_signal="gated"))
        model_trial(method, "mixed_context", replace(modern,history_signal="mixed"))
        # Joint coefficient draws are prioritized before exhaustive enumeration.
        for trial in range(4):
            values={name:joint_rng.choice(trial_values(modern,name)) for name in coefficients(modern)}
            model_trial(method,"loss_joint_"+str(trial),replace(modern,**values))
        model_trial(method, "raw_history_signal", replace(modern,history_signal="raw"))
        model_trial(method, "compact_regularized", replace(modern, hidden_dim=128,
                    collaborative_l2=100., mechanism_auxiliary_weight=.25,
                    wide_context_readout=True,readout_mode="residual"))
        model_trial(method, "wide_regularized", replace(modern, hidden_dim=500,
                    collaborative_l2=100., mechanism_auxiliary_weight=.25,
                    wide_context_readout=True,readout_mode="residual"))
        model_trial(method, "full_history_control", replace(modern, full_history_weight=.25,
                    collaborative_l2=10.))
        model_trial(method, "geometric_control", replace(modern, readout_mode="geometric",
                    hidden_dim=32 if method in ("Method3","Method4") else 128,
                    wide_context_readout=False, collaborative_l2=0.))
        model_trial(method, "lower_lr", replace(modern, lr=modern.lr*.5,
                    direction_lr=modern.direction_lr*.5,bias_lr=modern.bias_lr*.5))
        if method == "Method1":
            model_trial(method,"cavity_regularized",replace(modern,cavity_strength=.1,
                        information_ridge=10.,mechanism_auxiliary_weight=.1))
        if method == "Method2":
            model_trial(method,"router_regularized",replace(modern,router_ridge=.08,
                        router_context_penalty=1.,consistency_weight=.1))
        if method == "Method3":
            model_trial(method,"radius_regularized",replace(modern,radius_precision=8.,
                        radius_noise=.4,mechanism_auxiliary_weight=.25))
        if method == "Method4":
            model_trial(method,"signed_moments",replace(modern,moment_projection_mode="mean-priority",
                        moment_refinement_link="tanh",moment_refinement_scale=.25,
                        moment_projection_weight=.1,moment_dispersion_weight=.1))
        if method == "Method5":
            model_trial(method,"posterior_regularized",replace(modern,posterior_precision=80.,
                        posterior_relaxation=.1,posterior_preconditioned=True))
        if method == "Method6":
            model_trial(method,"transport_regularized",replace(modern,transport_moment_weight=.1,
                        context_strength=.5,angular_reference_weight=.5))
        if method == "Method7":
            model_trial(method,"ensemble_regularized",replace(modern,hidden_dim=128,
                        collaborative_l2=100.,feature_dropout=.30,ensemble_mean_weight=.5))

    if "Method8" in base.methods:
        for name,value in (("lr",.0003),("lr",.003),("hidden_dim",64),("hidden_dim",256)):
            c=copy.deepcopy(base);c.methods=("Method8",);c.v7.method8[name]=value
            insert(f"original_optimizer_{name}={value}__Method8",c)
    modern_base = copy.deepcopy(base)
    modern_base.v7.search_jobs_per_method = 0
    modern_base.methods=tuple(m for m in modern_base.methods if m!="Method8")
    modern_base.aemcne.warm_start_epochs = 0
    for cfg in modern_base.v7.models.values():
        cfg.lineage = "v7"
    full = (_modern_grid(pub, modern_base, ks, mode)
            if base.v7.search_profile == "comprehensive" else {})
    # Put useful baseline diversity before comprehensive enumeration so a
    # bounded search never spends its budget on weak L2 settings alone.
    for method, attr in (("MF","biasmf"),("BiasMF","biasmf")):
        if method not in base.methods: continue
        for lr, decay in ((.003,1e-4),(.01,1e-4),(.003,.001),(.01,.001),(.001,.001),(.003,.01)):
            c=copy.deepcopy(base);c.methods=(method,)
            c.biasmf.lr,c.biasmf.weight_decay=lr,decay
            insert(f"regularized_lr={lr},wd={decay}__{method}",c)
    for method in ("MBMF-F","MBMF-C","MBMF-N"):
        if method not in base.methods:continue
        for lr in (.003,.01,.001):
            c=copy.deepcopy(base);c.methods=(method,)
            c.v7.mbmf_optimizer="adam";c.v7.mbmf_adam_lr=lr
            insert(f"adam_lr={lr}__{method}",c)
    if "AutoRec" in base.methods:
        for penalty, hidden, optimizer, lr in ((10.,500,"rprop",.01),(1000.,500,"rprop",.01),
                (100.,265,"rprop",.01),(10.,265,"rprop",.01),(100.,500,"adam",.001),
                (100.,500,"rprop",.001)):
            c=copy.deepcopy(base);c.methods=("AutoRec",)
            c.autorec.lambda_l2,c.autorec.hidden_dim=penalty,hidden
            c.autorec.optimizer,c.autorec.lr=optimizer,lr
            insert(f"strong_l2={penalty},h={hidden},{optimizer},lr={lr}__AutoRec",c)
    if "AEMC-NE" in base.methods:
        for skip,main_activation,element_activation,initialization in (
                (.8,"tanh","tanh","torch"),(0.,"sigmoid","relu","identity-transfer"),
                (.8,"sigmoid","tanh","torch")):
            c=copy.deepcopy(base);c.methods=("AEMC-NE",)
            c.aemcne.element_skip=skip
            c.aemcne.main_activation,c.aemcne.element_activation=main_activation,element_activation
            c.aemcne.initialization=initialization;c.aemcne.warm_start_epochs=0
            insert(f"official_informed_skip={skip},{main_activation},{element_activation}__AEMC-NE",c)
        for main, element, lr, warm in ((100.,100.,.0003,0),(10.,10.,.001,0),
                (1.,1.,.001,0),(100.,1.,.001,0),(100.,100.,.0003,100),
                (10.,1.,.0003,100),(100.,10.,.0001,100)):
            c=copy.deepcopy(base);c.methods=("AEMC-NE",)
            c.aemcne.lambda_main,c.aemcne.lambda_element,c.aemcne.lr=main,element,lr
            c.aemcne.warm_start_epochs=min(warm,c.aemcne.max_epochs)
            c.aemcne.initialization="identity-transfer";c.aemcne.element_activation="relu"
            c.aemcne.element_skip=0.
            insert(f"strong_penalties={main},{element},lr={lr},warm={warm}__AEMC-NE",c)
    if base.v7.search_profile == "comprehensive":
        if "AEMC-NE" in base.methods:
            for main,element in itertools.product((.01,.1,1.,10.,50.,100.,200.,500.),repeat=2):
                c=copy.deepcopy(base);c.methods=("AEMC-NE",)
                c.aemcne.lambda_main,c.aemcne.lambda_element=main,element
                c.aemcne.element_skip=.8;c.aemcne.main_activation=c.aemcne.element_activation="tanh"
                c.aemcne.initialization="torch";c.aemcne.warm_start_epochs=0
                insert(f"official_informed_penalties={main},{element}__AEMC-NE",c)
        if "AEMC-NE" in base.methods:
            for main, element in itertools.product(("sigmoid","relu","tanh"),repeat=2):
                for penalty in (1.,100.):
                    c=copy.deepcopy(base);c.methods=("AEMC-NE",)
                    c.aemcne.main_activation,c.aemcne.element_activation=main,element
                    c.aemcne.initialization="xavier";c.aemcne.warm_start_epochs=0
                    c.aemcne.lambda_main=c.aemcne.lambda_element=penalty
                    insert(f"activation_audit={main},{element},l2{penalty}__AEMC-NE",c)
        for tag,cfg in full.items():
            cfg.v7.search_jobs_per_method=limit
            insert("comprehensive__"+tag,cfg)
    counts={m:0 for m in base.methods};retained={}
    mechanism_prefixes=("cavity_regularized__","router_regularized__","radius_regularized__",
                        "signed_moments__","posterior_regularized__","transport_regularized__",
                        "ensemble_regularized__")
    def order(item):
        tag=item[0]
        return (0 if tag.startswith("default__") else 1 if tag.startswith("historical_")
                else 2 if tag.startswith("gated_context") else 3 if tag.startswith("loss_joint_")
                else 4 if tag.startswith(mechanism_prefixes) else 5)
    for tag,cfg in sorted(jobs.items(),key=order):
        applicable=tuple(m for m in cfg.methods if not limit or counts[m]<limit)
        if applicable:
            cfg.methods=applicable;retained[tag]=cfg
            for method in applicable:counts[method]+=1
    return retained


def _modern_grid(pub, base, ks, mode="hybrid"):
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
    # The paper chose the two AEMC penalties independently. A tied-lambda grid
    # is an explicitly smaller search and must not masquerade as reproduction.
    if 'AEMC-NE' in base.methods:
        penalties=(.01,.1,1.,10.,50.,100.,200.,500.)
        for main,element in itertools.product(penalties,penalties):
            c=copy.deepcopy(base);c.methods=('AEMC-NE',)
            c.aemcne.lambda_main,c.aemcne.lambda_element=main,element
            insert(f'paper_penalties={main},{element}__AEMC-NE',c)
    # Baseline tuning remains the existing baseline search, with no proposed
    # method accidentally routed through legacy angle/loss hyperparameters.
    from v9.search import build_jobs as baseline_jobs
    baseline = copy.deepcopy(base)
    baseline.methods = tuple(m for m in base.methods if m not in base.v7.models)
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
        if method not in base.v7.models:
            continue
        def variant(tag, values):
            cfg = copy.deepcopy(base)
            cfg.methods = (method,)
            for key, value in values.items():
                setattr(cfg.v7.models[method], key, value)
            # Preserve the declared loss/geometry coupling when searching its
            # shared dispersion coefficient; an independent Euclidean control
            # is exposed by the mechanism ablations below.
            if method == "Method4" and "moment_dispersion_weight" in values:
                current=base.v7.models[method]
                if current.moment_projection_weight == current.moment_dispersion_weight:
                    cfg.v7.models[method].moment_projection_weight=values["moment_dispersion_weight"]
            cfg.v7.validate()
            insert(tag+"__"+method, cfg)
        for factor in (.5, 2.):
            original = base.v7.models[method]
            variant(f"lr_factor={factor}", dict(lr=original.lr*factor,
                    direction_lr=original.direction_lr*factor, bias_lr=original.bias_lr*factor))
        for jitter in (0., .15):
            if jitter < min(base.v7.models[method].mask_fraction, 1-base.v7.models[method].mask_fraction):
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
    # Audit activation/initialization/batch assumptions rather than labelling
    # the old ReLU/full-batch choices as verified author code.
    if "AEMC-NE" in base.methods:
        for activation,initialization,batch in itertools.product(("relu","sigmoid","tanh"),("torch","identity-transfer"),(0,64)):
            for penalty in (1.,100.):
                c=copy.deepcopy(base);c.methods=("AEMC-NE",)
                c.aemcne.main_activation=activation;c.aemcne.element_activation="relu"
                c.aemcne.initialization=initialization;c.aemcne.item_batch_size=batch
                c.aemcne.warm_start_epochs=0
                c.aemcne.lambda_main=c.aemcne.lambda_element=penalty
                c.aemcne.validate()
                insert(f"aemc_control={activation},{initialization},batch{batch},l2{penalty}__AEMC-NE",c)
    # The strong path avoids forcing every decoder through the geometric K.
    # Its shared encoder still feeds each distinct geometric mechanism.
    from .config import ModelConfig
    from v5.config import default_models as v5_controls
    legacy=v5_controls()
    for method in base.methods:
        if method not in base.v7.models:continue
        c=copy.deepcopy(base);c.methods=(method,)
        c.v7.models[method]=ModelConfig(**vars(legacy[method]))
        c.v7.models[method].probability_fit=base.v7.models[method].probability_fit
        c.v7.models[method].max_epochs=base.v7.models[method].max_epochs
        c.v7.models[method].steps_per_epoch=base.v7.models[method].steps_per_epoch
        insert("v5_parameter_control_in_v7__"+method,c)
        for hidden,penalty,weight in itertools.product((128,500),(0.,1.,10.,100.),(0.,.25)):
            c=copy.deepcopy(base);c.methods=(method,);m=c.v7.models[method]
            m.hidden_dim=hidden;m.history_activation="sigmoid";m.history_normalization="none"
            m.history_pooling_power=0.;m.history_degree_gate=False;m.wide_context_readout=True
            m.readout_mode="residual";m.readout_weight=.25;m.readout_count_adaptive=False
            m.collaborative_l2=penalty;m.full_history_weight=weight;m.validate()
            insert(f"wide_residual={hidden},l2{penalty},hybrid{weight}__"+method,c)
        c=copy.deepcopy(base);c.methods=(method,);m=c.v7.models[method]
        m.readout_mode="decoder";m.wide_context_readout=True
        m.mechanism_auxiliary_weight=0.;m.consistency_weight=0.;m.code_prior=0.;m.direction_prior=0.
        insert("wide_decoder_only_null__"+method,c)
    for tag,cfg in baseline_plan:insert("baseline__"+tag,cfg)
    if base.v7.mbmf_checkpoint_policy=='validation':
        for method in ('MBMF-F','MBMF-C','MBMF-N'):
            if method not in base.methods:continue
            for lr in (.001,.003,.01):
                cfg=copy.deepcopy(base);cfg.methods=(method,)
                cfg.v7.mbmf_optimizer='adam';cfg.v7.mbmf_adam_lr=lr
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
    limit=base.v7.search_jobs_per_method
    if limit:
        # A bounded exploratory grid must still expose strong competitors.
        # Do not let insertion order spend every AutoRec slot on weak L2 values.
        chosen={method:set() for method in base.methods}
        def priority(method,tag,cfg):
            if tag==f'default__{method}':return 0
            if method=="AEMC-NE" and tag.startswith("aemc_control=sigmoid,identity-transfer,batch64,l2100"):return .5
            if method in cfg.v7.models and tag.startswith("wide_residual=500,l210.0,hybrid0.25"):return .5
            if method=='AutoRec' and cfg.autorec.lambda_l2==100. and cfg.autorec.optimizer=='rprop':return 1
            if method=='AEMC-NE' and cfg.aemcne.lambda_main==cfg.aemcne.lambda_element==100.:return 1
            if method.startswith('MBMF-') and cfg.v7.mbmf_optimizer=='adam':return 1 if cfg.v7.mbmf_adam_lr==.01 else 2
            if method in cfg.v7.models and tag.startswith('collaborative_l2=1000.'):return 1
            if method in cfg.v7.models and tag.startswith('readout=coupled'):return 2
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
        "Method4": {"euclidean_moment_metric":dict(moment_projection_mode="metric",moment_projection_weight=1.), "weighted_moment_metric":dict(moment_projection_mode="metric",moment_projection_weight=base.v7.models["Method4"].moment_projection_weight), "no_moment_refinement":dict(moment_refinement=False), "legacy_moment_loss":dict(orthogonal_moment_loss=False), "single_signed_head": dict(signed_moments=False),
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
    additions={
        'Method1':{'degree_normalized_cavity_information':dict(cavity_degree_normalization=True)},
        'Method2':{'no_support_shrinkage':dict(router_support_shrinkage=False)},
        'Method3':{'no_radius_line_search':dict(radius_line_search=False)},
        'Method4':{'symmetric_amplitude_prior':dict(moment_context_prior=False)},
        'Method5':{'degree_scaled_posterior_precision':dict(posterior_degree_precision=True)},
        'Method6':{'no_transport_support_shrinkage':dict(transport_support_shrinkage=False)},
        'Method7':{'full_bootstrap_variance':dict(ensemble_bootstrap_temperature=1.)}}
    for method in changes:
        changes[method].update(additions[method])
        changes[method]["no_response_correction"] = dict(response_shrinkage=False)
        gated=base.v7.models[method].history_degree_gate
        changes[method]['no_history_degree_gate' if gated else 'degree_gated_history']=dict(history_degree_gate=not gated)
        changes[method]['no_history_residual_skip']=dict(history_residual_skip=False)
        if base.v7.models[method].readout_mode!='geometric':
            changes[method]['no_wide_decoder']=dict(wide_context_readout=False)
        if base.v7.models[method].history_signal=='gated':
            changes[method]['residual_only_signal']=dict(history_signal='residual')
            changes[method]['raw_only_signal']=dict(history_signal='raw')
        changes[method]['no_full_history_term']=dict(full_history_weight=0.)
        if method!='Method7':changes[method]['no_mechanism_auxiliary']=dict(mechanism_auxiliary_weight=0.)
        if base.v7.models[method].readout_mode=='residual':
            changes[method]['no_geometric_residual']=dict(readout_weight=0.)
            changes[method]['wide_decoder_training_null']=dict(readout_mode='decoder',mechanism_auxiliary_weight=0.,consistency_weight=0.,code_prior=0.,direction_prior=0.)
        changes[method]['layernorm_encoder_control']=dict(history_normalization='layernorm',history_activation='gelu',history_pooling_power=.5)
    result = []
    for method in base.methods:
        if method not in changes:
            continue
        for tag, values in [("full", {})]+list(changes[method].items()):
            if method=="Method7" and not base.v7.models[method].probability_fit and values and set(values)<={
                    "probability_rps_weight","shape_full","shape_epochs","shape_lr","shape_patience"}:
                continue
            cfg = copy.deepcopy(base)
            cfg.methods = (method,)
            for key, value in values.items():
                setattr(cfg.v7.models[method], key, value)
            # Cutting a parent representation also removes its dependent readout;
            # disclose the resulting config in the ablation provenance table.
            if (method=="Method7" and not cfg.v7.models[method].identity_history) or (method=="Method3" and not cfg.v7.models[method].context_radius):
                cfg.v7.models[method].readout_mode="geometric"
            if tag!='full' and vars(cfg.v7.models[method])==vars(base.v7.models[method]):continue
            result.append((method+"__"+tag, cfg))
        current=base.v7.models[method]
        for tag,overrides in (("geometry_only",dict(readout_mode="geometric")),
                              ("decoder_only_null",dict(readout_mode="decoder")),
                              ("fixed_readout_gate",dict(readout_mode="coupled",readout_count_adaptive=False)),
                              ("no_collaborative_l2",dict(collaborative_l2=0.))):
            cfg=copy.deepcopy(base);cfg.methods=(method,)
            for key,value in overrides.items():setattr(cfg.v7.models[method],key,value)
            if vars(cfg.v7.models[method])!=vars(current):result.append((method+"__"+tag,cfg))
        if method=='Method4' and (current.moment_refinement_link!='tanh' or current.moment_refinement_scale!=.25):
            for tag,penalty in (('legacy_moment_readout',current.collaborative_l2),('legacy_readout_no_l2',0.)):
                cfg=copy.deepcopy(base);cfg.methods=(method,)
                cfg.v7.models[method].moment_refinement_link='tanh';cfg.v7.models[method].moment_refinement_scale=.25
                cfg.v7.models[method].collaborative_l2=penalty
                if tag=='legacy_readout_no_l2' and current.collaborative_l2==0.:continue
                result.append((method+'__'+tag,cfg))
    # An absent legacy setting must never be reported as a real ablation.
    from .lineage import legacy_config
    import json
    effective=[]
    for tag,cfg in result:
        method=cfg.methods[0];before=base.v7.models[method];after=cfg.v7.models[method]
        if before.lineage != "v7":
            if tag.endswith("__full") or before.response_shrinkage != after.response_shrinkage:
                effective.append((tag,cfg));continue
            old=json.dumps(vars(legacy_config(before)),sort_keys=True)
            new=json.dumps(vars(legacy_config(after)),sort_keys=True)
            if old==new:continue
        effective.append((tag,cfg))
    return effective
