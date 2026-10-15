"""Safe V2 CLI: build on V9 protocol, no changes to V9 main or SHA-frozen V7."""
from __future__ import annotations
import argparse,copy,hashlib,json,pathlib,sys
from . import __version__
from .adapters import PROFILES, adapt, apply_profile
from .training import train_v2
from .calibration import calibrate_method7,nll_on_disjoint_selection_half
from .checkpoints import install_checkpoint_hooks


def run(argv=None):
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('--v2-profile',choices=tuple(PROFILES),default='balanced')
    ap.add_argument('--v2-train-chunk',type=int,default=125000)
    ap.add_argument('--v2-calibration',choices=('split','none'),default='split',
                    help='Validation-only variance fitting on half of validation; NLL is measured on the other half')
    v2args,rest=ap.parse_known_args(argv)
    if v2args.v2_train_chunk<1:ap.error('--v2-train-chunk must be positive')
    if '--output' not in rest:rest+=['--output','./results_v2']
    from v9 import engine
    old_load=engine.load_v8
    old_base=engine.make_base
    old_fingerprint=engine.fingerprint_config
    old_build_jobs=engine.build_jobs
    def build_v2_jobs(pub,base,ks,mode='hybrid'):
        # Angle-coordinate initialization has no meaning once V2 uses Cartesian
        # directions; do not burn budget on duplicate hyperparameter candidates.
        all_jobs=old_build_jobs(pub,base,ks,mode)
        return {k:v for k,v in all_jobs.items() if not k.startswith('angle__M')}
    engine.build_jobs=build_v2_jobs
    # A V2 study must never resume a V1 study with identical configuration.
    engine.fingerprint_config=lambda cfg: hashlib.sha256(
        (old_fingerprint(cfg)+':'+__version__+':'+v2args.v2_profile+':cal='+v2args.v2_calibration).encode('utf8')).hexdigest()

    def base_v2(pub,args,dataset):
        c=apply_profile(old_base(pub,args,dataset),v2args.v2_profile)
        if args.quick or args.eval_every is not None:
            interval=1 if args.quick else args.eval_every
            for i in range(1,8):getattr(c.core_config,f'method{i}').eval_every=interval
        return c
    engine.make_base=base_v2

    def load_v2():
        core,pub=old_load()
        if getattr(pub,'_v2_runtime_installed',False):return core,pub
        # Preserve Method2 consistency as an independently searchable V2 knob.
        # The V7 preset algebraically couples it to gamma and would silently
        # invalidate candidate labels in a V2 hyperparameter scan.
        if not getattr(core,'_v2_safe_preset_installed',False):
            original_preset=core.apply_q1_safe_preset
            def v2_safe_preset(experiment):
                requested=float(experiment.method2.consistency_weight)
                result=original_preset(experiment)
                experiment.method2.consistency_weight=requested
                return result
            core.apply_q1_safe_preset=v2_safe_preset
            core._v2_safe_preset_installed=True
        install_checkpoint_hooks(core,pub,v2args.v2_profile)
        old_inst=pub.instantiate_model_v8
        old_train=pub.train_model_v8
        old_bound=pub.individual_bound_arrays
        old_eval=pub.evaluate_publication_model

        def inst(method,data,cfg,k,model_seed,device,init_mode=None):
            # Perform the actual hyperparameter config changes BEFORE constructor;
            # V1 models and baseline models stay untouched.
            if method.startswith('Method') and method[6:].isdigit():
                i=int(method[6:])
                if 1<=i<=7:
                    cfg=copy.deepcopy(cfg)
                    cfg.core_config=getattr(cfg,'core_config')
                    if i==6 and float(cfg.core_config.method6.radius_temperature)<=0:
                        cfg.core_config.method6.radius_temperature=max(.5*(int(k)**.5),1e-3)
                    # Defensive: objective changes are reflected in returned mcfg.
            m,mcfg=old_inst(method,data,cfg,k,model_seed,device,init_mode)
            m=adapt(m,method,data,v2args.v2_profile,core,pub)
            if method=='Method2':
                # V7 safe preset may overwrite the V2 consistency search value.
                m.cfg.consistency_weight=float(cfg.core_config.method2.consistency_weight)
            if method=='Method7':m.v2_calibration_policy=v2args.v2_calibration
            return m,mcfg

        def trainer(method,model,data,mcfg,pcfg):
            if getattr(model,'v2_method',None) in {f'Method{i}' for i in range(1,7)}:
                return train_v2(pub,model,data,mcfg,chunk_size=v2args.v2_train_chunk)
            info=old_train(method,model,data,mcfg,pcfg)
            if method=='Method7' and v2args.v2_calibration=='split':
                calibration=calibrate_method7(model,data)
                info['v2_variance_calibration']=calibration
                print('[V2 CALIBRATION]',calibration,flush=True)
            return info

        def pub_eval(model,data,split,*args,**kw):
            out=old_eval(model,data,split,*args,**kw)
            if (split=='val' and getattr(model,'v2_method',None)=='Method7'
                    and v2args.v2_calibration=='split'):
                extras=nll_on_disjoint_selection_half(model,data)
                if extras is not None:
                    if isinstance(out,tuple):
                        metrics,pred=out;metrics.update(extras);return metrics,pred
                    out.update(extras)
            return out

        def individual_bounds(model,data,u,i):
            # Old MBMF radius-bound metrics are invalid after introducing an
            # independent bias center and asymmetric radius. Report NA instead.
            if hasattr(model,'v2_method'):return None
            return old_bound(model,data,u,i)

        pub.instantiate_model_v8=inst
        pub.train_model_v8=trainer
        pub.individual_bound_arrays=individual_bounds
        pub.evaluate_publication_model=pub_eval
        pub._v2_runtime_installed=True
        return core,pub

    engine.load_v8=load_v2
    return engine.main(rest)
