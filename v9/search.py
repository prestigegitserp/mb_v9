"""Build V9 hyperparameter trials. No test metadata is accepted for selection."""
from __future__ import annotations
import copy, itertools

# Data sensitivity is explicitly NOT a hyperparameter choice. Test and validation
# population changes make across-job val_RMSE incomparable.
FORBIDDEN_SEARCH_PREFIXES=('train_fraction=','synthetic_density=','synthetic_missing=')

def build_jobs(pub,base,ks,mode='hybrid'):
    if mode not in ('one-factor','hybrid','joint'):
        raise ValueError(mode)
    jobs={}
    def insert(tag,cfg):
        allowed=set(base.methods)
        applicable=tuple(x for x in cfg.methods if x in allowed and pub.method_supported_on_dataset_v8(x,base.data.dataset))
        if not applicable:return
        c=copy.deepcopy(cfg);c.methods=applicable;c.validation_only=True
        c.core_config.latent_dims=tuple(c.core_config.latent_dims)
        if (c.data.train_fraction!=base.data.train_fraction or c.data.val_fraction!=base.data.val_fraction or c.data.test_fraction!=base.data.test_fraction or c.data.synthetic_density!=base.data.synthetic_density):
            raise AssertionError('A data protocol variant reached hyperparameter search')
        if tag in jobs:raise ValueError(f'duplicate job tag {tag}')
        jobs[tag]=c
    # Baseline inclusion is necessary for a meaningful validation selection.
    for method in base.methods:
        c=copy.deepcopy(base);c.methods=(method,)
        insert(f'default__{method}',c)
    plan=pub.HyperparameterScanPlanV8(latent_dims=tuple(ks))
    if mode in ('one-factor','hybrid'):
        for tag,c in pub.make_hyperparameter_scan_jobs_v8(base,plan):
            if tag.startswith(FORBIDDEN_SEARCH_PREFIXES):continue
            insert(tag,c)
    if mode in ('joint','hybrid'):
        # Additional budgeted capacity and optimizer sweeps. Each method is tuned
        # with the same validation-only rule; the budget is written to the plan.
        for i in range(1,8):
            model=f'Method{i}'
            orig=getattr(base.core_config,f'method{i}')
            for factor in (.3333333333333333,1.0,3.0):
                c=copy.deepcopy(base);c.methods=(model,)
                getattr(c.core_config,f'method{i}').lr=float(orig.lr)*factor
                insert(f'lr__M{i}__factor={factor:g}',c)
            for wd in (0.0,1e-6,1e-4):
                c=copy.deepcopy(base);c.methods=(model,)
                getattr(c.core_config,f'method{i}').weight_decay=wd
                insert(f'wd__M{i}__value={wd:g}',c)
        for i in (1,5,6,7):
            orig=getattr(base.core_config,f'method{i}')
            for size in sorted({64,128,256,int(orig.hidden_dim)}):
                c=copy.deepcopy(base);c.methods=(f'Method{i}',)
                getattr(c.core_config,f'method{i}').hidden_dim=size
                insert(f'hidden__M{i}__dim={size}',c)
        for i in (1,2,3,4,5,6):
            for init in ('legacy_uniform_2pi','canonical_uniform','isotropic_gaussian'):
                c=copy.deepcopy(base);c.methods=(f'Method{i}',)
                getattr(c.core_config,f'method{i}').angle_init=init
                insert(f'angle__M{i}__init={init}',c)
        for lr_factor in (.3333333333333333,1.0,3.0):
            c=copy.deepcopy(base);c.methods=('BiasMF','MF')
            c.biasmf.lr=base.biasmf.lr*lr_factor
            insert(f'lr__BiasMF_MF__factor={lr_factor:g}',c)
        for wd in (0.,1e-4,1e-3):
            c=copy.deepcopy(base);c.methods=('BiasMF','MF')
            c.biasmf.weight_decay=wd
            insert(f'wd__BiasMF_MF__value={wd:g}',c)
        for opt,lr in [('rprop',.001),('adam',.0005),('adam',.001),('adam',.002)]:
            c=copy.deepcopy(base);c.methods=('AutoRec',)
            c.autorec.optimizer=opt;c.autorec.lr=lr
            insert(f'optim__AutoRec__{opt}__lr={lr:g}',c)
        for factor in (.3333333333333333,1.0,3.0):
            c=copy.deepcopy(base);c.methods=('AEMC-NE',)
            c.aemcne.lr=base.aemcne.lr*factor
            insert(f'lr__AEMC__factor={factor:g}',c)
        for alpha,beta in itertools.product(plan.method1_alpha,plan.method1_beta):
            c=copy.deepcopy(base);c.methods=('Method1',)
            c.core_config.method1.alpha_ae=alpha;c.core_config.method1.beta_mbmf=beta
            insert(f'joint__M1__alpha={alpha}__beta={beta}',c)
        for gamma,cons in itertools.product(plan.method2_gamma,plan.method2_consistency):
            c=copy.deepcopy(base);c.methods=('Method2',)
            c.core_config.method2.fusion_weight_user_view=gamma
            c.core_config.method2.consistency_weight=cons
            insert(f'joint__M2__gamma={gamma}__cons={cons}',c)
        for gamma in plan.method2_gamma:
            c=copy.deepcopy(base);c.methods=('Method2',)
            c.core_config.method2.fusion_weight_user_view=gamma
            c.core_config.method2.consistency_weight=gamma*(1-gamma)
            insert(f'coupled__M2__gamma={gamma}',c)
        for scale,delta in itertools.product(plan.method3_log_scale,plan.method3_delta_l2):
            c=copy.deepcopy(base);c.methods=('Method3',)
            c.core_config.method3.max_abs_log_scale=scale;c.core_config.method3.delta_l2=delta
            insert(f'joint__M3__scale={scale}__delta={delta}',c)
        for shift,delta in itertools.product(plan.method4_logit_shift,plan.method3_delta_l2):
            c=copy.deepcopy(base);c.methods=('Method4',)
            c.core_config.method4.max_abs_logit_shift=shift;c.core_config.method4.delta_l2=delta
            insert(f'joint__M4__shift={shift}__delta={delta}',c)
        for gamma,rec in itertools.product(plan.method5_fusion,(0.0,0.1,1.0)):
            c=copy.deepcopy(base);c.methods=('Method5',)
            c.core_config.method5.fusion_weight_user_view=gamma
            c.core_config.method5.user_reconstruction_weight=rec
            c.core_config.method5.item_reconstruction_weight=rec
            insert(f'joint__M5__gamma={gamma}__rec={rec}',c)
        for temp,rec in itertools.product(plan.method6_temperature,(0.0,0.1,1.0)):
            c=copy.deepcopy(base);c.methods=('Method6',)
            c.core_config.method6.radius_temperature=temp
            c.core_config.method6.user_reconstruction_weight=rec
            c.core_config.method6.item_reconstruction_weight=rec
            insert(f'joint__M6__temp={temp}__rec={rec}',c)
        for context,tau in itertools.product(plan.method7_context_strength,plan.method7_tau_init):
            c=copy.deepcopy(base);c.methods=('Method7',)
            c.core_config.method7.context_strength=context;c.core_config.method7.tau_init=tau
            insert(f'joint__M7__context={context}__tau={tau}',c)
        # Unlike one-factor scan, joint mode explicitly includes all four K values.
        for k in ks:
            c=copy.deepcopy(base);c.methods=tuple(m for m in base.methods if m not in ('AutoRec','AEMC-NE'))
            c.core_config.latent_dims=(k,)
            insert(f'joint__K={k}',c)
    return jobs

def candidate_count(cfg):
    nk=len(cfg.core_config.latent_dims); ns=len(cfg.core_config.split_seeds)*len(cfg.core_config.model_seeds)
    return sum(ns*(1 if m in ('AutoRec','AEMC-NE') else nk)*(len(cfg.core_config.paper.init_modes) if m in ('MBMF-F','MBMF-C','MBMF-N') else 1)
               for m in cfg.methods if pub_method_supported(m,cfg))

def pub_method_supported(m,cfg):
    # build_jobs strips inapplicable methods. No public global needed here.
    return True
