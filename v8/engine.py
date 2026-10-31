"""Independent V8 orchestration with validation locking and complete reports."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, pathlib, re, sys, time, traceback, tempfile
from dataclasses import asdict

from . import __version__
from .audit import audit_selection
from .figures import plot_convergence,plot_k_rmse,plot_validation_landscape,plot_train_keep_sensitivity,plot_ablation_effects
from v9.progress import CandidateProgress
from v9.search import build_jobs,candidate_count
from v9.selection import select_validation_winners
from .recovery import scientific_fingerprint as fingerprint_config

ROOT=pathlib.Path(__file__).resolve().parents[1]
VENDOR=ROOT/'vendor_v8'
REQUIRED_SHA='6da726cb155ab6ad291f5f608febc37f42d1c25ac3924910f56917908c1f124c'
REPORT_PLOTS=True
DATASETS=('synthetic','jester','bookcrossing','ml-100k','ml-1m','ml-10m')

def load_v8():
    v8=VENDOR/'mbmf_q1_final_v7.py'
    if not v8.exists():raise RuntimeError('Run: python bootstrap_v8.py --from-v8 <path_to_fixed_v8>')
    normalized=hashlib.sha256(v8.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    if normalized!=REQUIRED_SHA:raise RuntimeError(f'V8 scientific core SHA mismatch: {normalized}')
    if str(VENDOR) not in sys.path:sys.path.insert(0,str(VENDOR))
    import mbmf_q1_final_v7 as core
    import mbmf_q1_infrastructure_v8 as pub
    return core,pub

def numbers(s):return tuple(int(x.strip()) for x in str(s).split(',') if x.strip())

def basename(s):return re.sub('[^a-zA-Z0-9_.=-]','_',str(s))

def eligible_blend(args,base):
    if not getattr(args,'include_blend',False):return None
    methods=tuple(m.strip() for m in args.blend_methods.split(','))
    if len(methods)!=2 or len(set(methods))!=2:raise ValueError('Two distinct blend methods required')
    return methods if set(methods)<=set(base.methods) else None

def make_base(pub,args,dataset):
    cfg=pub.PublicationExperimentConfigV8()
    cfg.methods=tuple(cfg.methods)+("Method8",)
    cfg.data.dataset=dataset;cfg.data.data_root=args.data_root
    cfg.data.force_backend=args.backend
    cfg.core_config.split_seeds=numbers(args.split_seeds)
    cfg.core_config.model_seeds=numbers(args.model_seeds)
    cfg.core_config.latent_dims=numbers(args.k)
    for label,values in (("split-seeds",cfg.core_config.split_seeds),("model-seeds",cfg.core_config.model_seeds),("k",cfg.core_config.latent_dims)):
        if len(values)!=len(set(values)):raise ValueError("Repeated values in --"+label+" are not independent repetitions")
    if args.methods:
        allowed=set(cfg.methods);chosen=tuple(x.strip() for x in args.methods.split(',') if x.strip())
        if set(chosen)-allowed:raise ValueError(f'Unknown methods: {set(chosen)-allowed}')
        cfg.methods=chosen
    cfg.methods=tuple(m for m in cfg.methods if pub.method_supported_on_dataset_v8(m,dataset))
    if not cfg.methods:raise ValueError('No eligible methods for dataset '+dataset)
    if args.eval_every is not None:
        if args.eval_every<1:raise ValueError('--eval-every must be >=1')
        cfg.core_config.paper.eval_every=args.eval_every
        for i in range(1,8):getattr(cfg.core_config,f'method{i}').eval_every=args.eval_every
        cfg.autorec.eval_every=cfg.aemcne.eval_every=cfg.biasmf.eval_every=args.eval_every
    if args.quick:
        cfg.core_config.split_seeds=cfg.core_config.split_seeds[:1]
        cfg.core_config.model_seeds=cfg.core_config.model_seeds[:1]
        cfg.core_config.latent_dims=cfg.core_config.latent_dims[:1]
        cfg.core_config.paper.init_modes=('isotropic_gaussian',)
        cfg.core_config.paper.max_epochs=min(cfg.core_config.paper.max_epochs,3)
        cfg.core_config.paper.eval_every=1
        for i in range(1,8):
            c=getattr(cfg.core_config,f'method{i}');c.max_epochs=min(c.max_epochs,3);c.eval_every=1;c.val_patience=min(c.val_patience,2)
        for c in (cfg.autorec,cfg.aemcne,cfg.biasmf):c.max_epochs=min(c.max_epochs,3);c.eval_every=1
    if not cfg.core_config.split_seeds or not cfg.core_config.model_seeds or not cfg.core_config.latent_dims:
        raise ValueError('Nonempty split, model-seed, K arrays required')
    return cfg

def json_write(path,obj):
    p=pathlib.Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(obj,ensure_ascii=False,indent=2,default=str).encode('utf8')
    # Unique staging plus fsync prevents an old RUNNING write from surviving a
    # successful completion on filesystems that delay replacement visibility.
    for _ in range(3):
        with tempfile.NamedTemporaryFile(dir=p.parent,prefix='.'+p.name+'.',suffix='.tmp',delete=False) as handle:
            tmp=pathlib.Path(handle.name);handle.write(payload);handle.flush();os.fsync(handle.fileno())
        try:
            os.replace(tmp,p)
            if p.read_bytes()==payload:return
        finally:tmp.unlink(missing_ok=True)
    raise OSError('Atomic JSON write failed read-back verification: '+str(p))

def run_job(pub,core,cfg,folder,*,args,validation_only=False,extra=None):
    from .retention import skip_trial_report
    folder=pathlib.Path(folder);folder.mkdir(parents=True,exist_ok=True)
    cfg=copy.deepcopy(cfg);cfg.output_dir=str(folder)
    cfg.validation_only=bool(validation_only)
    identity=fingerprint_config(cfg)
    statefile=folder/'V8_JOB_STATUS.json'
    required=folder/('candidate_validation_runs.csv' if validation_only else 'locked_test_runs.csv')
    if args.resume and statefile.is_file():
        state=json.loads(statefile.read_text(encoding='utf8'))
        if state.get('fingerprint')!=identity:raise ValueError(f'Existing run with DIFFERENT config: {folder}; choose a new --output')
        if state.get('training_complete', state.get('status')=='COMPLETED') and required.is_file():
            expected=state.get('results_sha256')
            if expected and hashlib.sha256(required.read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Result integrity mismatch: {required}')
            print(f'[RESUME] completed: {folder}',flush=True)
            import pandas as pd
            data=pd.read_csv(required)
            from .reporting import generate_report
            if not validation_only:
                from .reporting import summarise_locked_runs
                summarise_locked_runs(data).to_csv(folder/'locked_test_summary.csv',index=False)
            if not skip_trial_report(cfg, validation_only, extra):
                generate_report(folder,data,validation_only=validation_only,plots=cfg.auto_generate_plots)
            state.update(status='COMPLETED', training_complete=True, end_unix=time.time())
            json_write(statefile,state)
            return data
    else:
        if required.is_file() or (folder/'config.json').is_file():
            raise ValueError(f'Existing untracked/previous study at {folder}; use --resume or unique output')
    info={'status':'RUNNING','version':__version__,'command_arguments':vars(args),'fingerprint':identity,'start_unix':time.time(),'validation_only':validation_only,'extra':extra or {},'candidate_count_expected':candidate_count(cfg)}
    json_write(statefile,info)
    epoch_file = folder/'V8_EPOCH_METRICS.csv'
    if epoch_file.exists() and not args.resume:
        attempts = folder/'attempts';attempts.mkdir(exist_ok=True)
        epoch_file.replace(attempts/f'epoch_metrics_before_{time.time_ns()}.csv')
    progress=CandidateProgress(candidate_count(cfg),str(extra or folder.name),folder/'V8_EPOCH_METRICS.csv',epoch_train_rmse=args.epoch_train_rmse)
    try:
        with progress.instrument(pub,core):
            candidates,locked=pub.run_publication_study_v8(cfg)
        if progress.done!=progress.total:
            raise RuntimeError(f'Candidate count mismatch: {progress.done}/{progress.total}')
        if epoch_file.is_file():
            import pandas as pd
            ticks=pd.read_csv(epoch_file);keys=[c for c in ("stage","method","split_seed","K","model_seed","init","epoch") if c in ticks]
            ticks.drop_duplicates(keys,keep="last").to_csv(epoch_file,index=False)
        data=candidates if validation_only else locked
        if validation_only and any(x.startswith('test_') for x in data.columns):
            raise AssertionError('LEAK: validation_only produced test metrics')
        from .reporting import generate_report
        if not validation_only:
            from .reporting import summarise_locked_runs
            summarise_locked_runs(data).to_csv(folder/'locked_test_summary.csv',index=False)
        info.update(status='REPORT_PENDING',training_complete=True,candidate_count_actual=progress.done,
                    results_sha256=hashlib.sha256(required.read_bytes()).hexdigest())
        json_write(statefile,info)
        if not skip_trial_report(cfg, validation_only, extra):
            generate_report(folder,data,validation_only=validation_only,plots=cfg.auto_generate_plots)
        info.update(status='COMPLETED',end_unix=time.time(),candidate_count_actual=progress.done,epoch_events=progress.n_epoch_records)
        json_write(statefile,info)
        return data
    except BaseException as exc:
        info.update(status='REPORT_FAILED' if info.get('training_complete') else 'FAILED',end_unix=time.time(),error=repr(exc),traceback=traceback.format_exc())
        json_write(statefile,info)
        raise

def scan_dataset(pub,core,base,dsroot,args):
    import pandas as pd
    target=pathlib.Path(dsroot)/'search';target.mkdir(parents=True,exist_ok=True)
    if any((target/'final').glob('*')):pass
    plan=build_jobs(pub,base,numbers(args.k),mode=args.scan_mode)
    # Validation candidates retain train/validation predictions and reports.
    # Checkpoint retention can be reduced explicitly; test is never evaluated here.
    for config in plan.values():
        config.save_predictions=True
        config.save_candidate_checkpoints=base.save_candidate_checkpoints
        config.save_selected_checkpoints=False
    summary=[];all_tables=[]
    declaration={'mode':args.scan_mode,'jobs':[{ 'tag':tag,'methods':list(cfg.methods),'candidates':candidate_count(cfg)} for tag,cfg in plan.items()],
        'base_configuration_fingerprint':fingerprint_config(base),
        'search_jobs_per_method_limit':base.v8.search_jobs_per_method,'grid_truncated':bool(base.v8.search_jobs_per_method),
        'excluded_from_search':['train_fraction','synthetic_density','dataset-sensitive tests'],
        'selection_rule':'mean val_RMSE across identical split/model seeds; never test metrics'}
    plan_file=target/'V8_SEARCH_PLAN.json'
    if plan_file.is_file() and json.loads(plan_file.read_text())!=declaration:
        raise ValueError('Different search plan at existing output; choose a new --output')
    json_write(plan_file,declaration)
    if args.dry_run:
        print(f'[PLAN] {base.data.dataset}: {len(plan)} search jobs, {sum(candidate_count(c) for c in plan.values()):,} candidate trainings');return None
    for index,(tag,cfg) in enumerate(plan.items(),1):
        print(f'\n[SEARCH JOB {index}/{len(plan)}] {tag}; candidates={candidate_count(cfg)}',flush=True)
        path=target/'jobs'/basename(tag)
        data=run_job(pub,core,cfg,path,args=args,validation_only=True,extra={'type':'hyper','tag':tag,'dataset':base.data.dataset})
        cohorts={str(seed):json.loads((path/f'V8_PROTOCOL_{seed}.json').read_text())['data_signature']
                 for seed in base.core_config.split_seeds}
        cohort_file=target/'V8_SEARCH_COHORTS.json'
        if cohort_file.is_file():
            if json.loads(cohort_file.read_text())!=cohorts:
                raise ValueError('Search data cohort mismatch; input data/roles changed at an existing study')
        else:
            json_write(cohort_file,cohorts)
        data=data.copy();data['scan_job']=tag;data['artifact_root']=str(path.resolve());all_tables.append(data)
        from .retention import retain_leaders
        retain_leaders(target,pd.concat(all_tables,ignore_index=True,sort=False),base)
    if not all_tables:raise RuntimeError('No validation candidates')
    table=pd.concat(all_tables,ignore_index=True,sort=False)
    if any(str(c).startswith('test_') for c in table.columns):raise AssertionError('Test leak')
    table.to_csv(target/'hyperparameter_validation_all.csv',index=False)
    winners,ranked=select_validation_winners(table,len(base.core_config.split_seeds),len(base.core_config.model_seeds))
    winners.to_csv(target/'validation_winners.csv',index=False)
    json_write(target/'best_configurations.json', {
        'selected_on':'validation only','test_used':False,
        'winners':[{'dataset':str(r.dataset),'method':str(r.method),'split_seed':int(r.split_seed),
                    'scan_job':str(r.scan_job),'config':json.loads(r.config_key)}
                   for _,r in winners.iterrows()]})
    ranked.to_csv(target/'validation_ranked_configs.csv',index=False)
    if REPORT_PLOTS:plot_validation_landscape(target/'validation_ranked_configs.csv',target/'figures')
    json_write(target/'validation_selection_manifest.json',{'rule':'FOR EACH OUTER SPLIT: min mean val RMSE across matching model seeds, tie val std, K, tag, config_key','test_seen':False,
      'selected_split_configs':[{'method':str(r.method),'split_seed':int(r.split_seed),'scan_job':str(r.scan_job),'K':None if pd.isna(r.K) else int(r.K),'init_mode':str(r.init_mode),'val_mean':float(r.val_mean),'n_model_seeds':int(r.n)} for _,r in winners.iterrows()]})
    from .search_report import generate_search_report
    generate_search_report(target,table,winners,ranked,plots=base.auto_generate_plots)
    components=eligible_blend(args,base)
    if components:
        from .ensemble import fit_validation_blend
        fit_validation_blend(target,table,winners,components)
    elif args.include_blend:
        json_write(target/'ensemble'/'BLEND_STATUS.json',dict(status='SKIPPED',reason='Predeclared components are not both eligible/requested on this dataset'))
    return plan,winners

def run_final(pub,core,base,dsroot,args,scan_results=None):
    import pandas as pd
    root=pathlib.Path(dsroot);searchroot=root/'search';
    if scan_results is None:
        f=searchroot/'validation_winners.csv'
        if not f.exists():raise RuntimeError(f'No validation winners at {f}; run search first')
        winners=pd.read_csv(f);plan=build_jobs(pub,base,numbers(args.k),args.scan_mode)
    else:plan,winners=scan_results
    complete=[]
    for _,r in winners.iterrows():
        method=str(r.method);tag=str(r.scan_job);split_seed=int(r.split_seed)
        if tag not in plan:raise ValueError(f'Unknown winner tag; scan mode changed? {tag}')
        cfg=copy.deepcopy(plan[tag]);cfg.methods=(method,);cfg.validation_only=False
        cfg.core_config.split_seeds=(split_seed,)
        if pd.notna(r.K):cfg.core_config.latent_dims=(int(r.K),)
        else:cfg.core_config.latent_dims=(int(base.core_config.latent_dims[0]),)
        if method in ('MBMF-F','MBMF-C','MBMF-N'):
            cfg.core_config.paper.init_modes=(str(r.init_mode),)
        if args.dry_run:
            print(f'[PLAN FINAL] {method} split={split_seed}: job={tag} K={cfg.core_config.latent_dims} init={cfg.core_config.paper.init_modes if method.startswith("MBMF-") else "NA"}')
            continue
        target=root/'final'/method/f'split_{split_seed}'
        print(f'\n[LOCKED TEST] {base.data.dataset} / {method} / split={split_seed}: {tag} K={cfg.core_config.latent_dims}',flush=True)
        locked=run_job(pub,core,cfg,target,args=args,extra={'type':'final','selected_by':'validation','job':tag})
        locked.insert(0,'selected_search_job',tag);complete.append(locked)
    if complete:
        combined=pd.concat(complete,ignore_index=True,sort=False)
        combined.to_csv(root/'final_locked_test_all.csv',index=False)
        from .reporting import summarise_locked_runs
        summarise_locked_runs(combined).to_csv(root/'final_locked_test_summary.csv',index=False)
        try:
            sm=combined.groupby(['dataset','method','split_seed'],as_index=False).test_RMSE.mean()
            if sm[['dataset','split_seed']].drop_duplicates().shape[0]<2:
                raise ValueError('One outer split: descriptive results only, no across-split statistical claim')
            pub.pairwise_wilcoxon_holm(sm,metric='test_RMSE',group_cols=('dataset','split_seed')).to_csv(root/'final_wilcoxon_holm.csv',index=False)
            ranks,stat=pub.friedman_nemenyi_summary(sm,metric='test_RMSE',block_cols=('dataset','split_seed'))
            ranks.to_csv(root/'final_friedman_ranks.csv',index=False)
            pd.DataFrame([stat]).to_csv(root/'final_friedman_stats.csv',index=False)
            json_write(root/'final_statistical_status.json',dict(status="EXPLORATORY_OVERLAPPING_RESPLITS",
                independent_samples_assumed=False,confirmatory_significance_claim=False,
                note="Wilcoxon/Friedman tables are exploratory; overlapping random splits do not establish independent repetitions"))
        except Exception as exc:
            import warnings
            warnings.warn(f'V8 final statistical comparison skipped: {exc}')
            json_write(root/'final_statistical_status.json',{'status':'SKIPPED','reason':str(exc)})
        json_write(root/'final_selection_provenance.json',{'selected_on':'validation ONLY','test_used_for_selection':False,'selected_method_splits':winners[['method','split_seed']].to_dict(orient='records'),'scientific_core_sha256':REQUIRED_SHA})
        from .reporting import generate_report
        from .fidelity import assess_final
        assess_final(root,combined,base.v8.protocol)
        generate_report(root,combined,combined=True,plots=base.auto_generate_plots)
        components=eligible_blend(args,base)
        if components:
            from .ensemble import apply_locked_blend
            apply_locked_blend(root,combined,components,plots=base.auto_generate_plots)
        return combined

def run_calibration(pub,core,base,dsroot,args):
    # Method7 alternate validation objective. NEVER mix accuracy-optimized and
    # calibration-optimized scores in the same unlabelled primary comparison.
    import pandas as pd
    if 'Method7' not in base.methods:
        print('[CALIBRATION] Method7 not selected; skipped');return
    root=pathlib.Path(dsroot);valcsv=root/'search'/'hyperparameter_validation_all.csv'
    if not valcsv.exists():raise FileNotFoundError(f'Run scan before calibration: {valcsv}')
    d=pd.read_csv(valcsv);d=d[d.method=='Method7'].copy()
    criterion='val_Probability_NLL'
    if criterion not in d.columns:raise ValueError('Method7 validation likelihood unavailable')
    d=d[d[criterion].notna()]
    winners,_=select_validation_winners(d,len(base.core_config.split_seeds),len(base.core_config.model_seeds),metric=criterion)
    out=root/'calibration';out.mkdir(parents=True,exist_ok=True)
    winners.to_csv(out/'validation_nll_winners.csv',index=False)
    jobs=build_jobs(pub,base,numbers(args.k),args.scan_mode);frames=[]
    for _,r in winners.iterrows():
        c=copy.deepcopy(jobs[str(r.scan_job)]);c.methods=('Method7',)
        c.core_config.split_seeds=(int(r.split_seed),)
        c.core_config.latent_dims=(int(r.K),)
        if args.dry_run:continue
        locked=run_job(pub,core,c,out/f'split_{int(r.split_seed)}',args=args,extra={'type':'method7-val-nll-selected','selection_metric':criterion})
        frames.append(locked)
    if frames:
        x=pd.concat(frames,ignore_index=True);x.to_csv(out/'calibration_locked_test_runs.csv',index=False)
        pub.summarise_publication_runs(x).to_csv(out/'calibration_locked_test_summary.csv',index=False)
        json_write(out/'CALIBRATION_PROTOCOL.json',{'selection_metric':criterion,'test_used_in_selection':False,'reported_separately_from_rmse_tuned_main':True})


def run_kstudy(pub,core,base,dsroot,args):
    import pandas as pd
    root=pathlib.Path(dsroot)/'kstudy';root.mkdir(parents=True,exist_ok=True)
    from .studies import frozen_configurations,anchor_protocol
    configurations=frozen_configurations(pub,base,dsroot,args)
    all_runs=[]
    for k in numbers(args.k):
        print(f'\n[FIXED-K TEST SENSITIVITY] dataset={base.data.dataset} K={k}; TEST IS DESCRIPTIVE ONLY',flush=True)
        pieces=[]
        for tag,original in configurations:
            c=copy.deepcopy(original);c.methods=tuple(m for m in c.methods if m not in ('AutoRec','AEMC-NE'));c.core_config.latent_dims=(k,)
            if not c.methods:continue
            if args.dry_run:continue
            pieces.append(run_job(pub,core,c,root/f'K_{k}'/tag,args=args,extra={'type':'fixed-K-descriptive','K':k,'source':tag}))
        if not pieces:continue
        rows=pd.concat(pieces,ignore_index=True,sort=False)
        rows.to_csv(root/f'K_{k}'/'locked_test_runs.csv',index=False)
        if rows.K.isna().any() or not rows.K.eq(k).all():raise AssertionError('K-study mismatch')
        all_runs.append(rows)
    if all_runs:
        pd.concat(all_runs,ignore_index=True).to_csv(root/'all_fixed_k_locked_runs.csv',index=False)
        plot_k_rmse(root,root/'figures')
    json_write(root/'K_STUDY_PROTOCOL.json',anchor_protocol(configurations,kind='fixed-K sensitivity',test_role='descriptive; never select K on these test curves'))

def run_ablation(pub,core,base,dsroot,args):
    import pandas as pd
    root=pathlib.Path(dsroot)/'ablations';root.mkdir(parents=True,exist_ok=True)
    from .studies import frozen_configurations,anchor_protocol
    configurations=frozen_configurations(pub,base,dsroot,args)
    variants=[(tag,cfg,source) for source,original in configurations for tag,cfg in pub.make_ablation_variants_v8(original)]
    frames=[]
    for i,(tag,cfg,source) in enumerate(variants,1):
        print(f'\n[ABLATION {i}/{len(variants)}] {base.data.dataset}: {tag}',flush=True)
        if args.dry_run:
            print(f'  candidates={candidate_count(cfg)}');continue
        result=run_job(pub,core,cfg,root/tag/source,args=args,extra={'type':'ablation','tag':tag,'anchor':source})
        result=result.copy();result.insert(0,'ablation',tag);frames.append(result)
    if frames:
        combined=pd.concat(frames,ignore_index=True,sort=False)
        combined.to_csv(root/'ablation_locked_test_runs.csv',index=False)
        pub.summarise_publication_runs(combined).to_csv(root/'ablation_summary.csv',index=False)
        if REPORT_PLOTS:plot_ablation_effects(combined,root/'figures')
    json_write(root/'ABLATION_PROTOCOL.json',anchor_protocol(configurations,kind='frozen-winner mechanism ablation',
        K='same as full winner in each split',reselection=False,budget='same declared epochs/steps; actual early-stopping work is reported'))

def run_ablation_fixed_k(pub,core,base,dsroot,args):
    # Strictly matched latent dimension for mechanism isolation. Not a hyperparameter tuner.
    import pandas as pd
    fixed_k=int(args.ablation_fixed_k)
    if fixed_k<=0:raise ValueError('--ablation-fixed-k must be positive')
    base=copy.deepcopy(base);base.core_config.latent_dims=(fixed_k,)
    root=pathlib.Path(dsroot)/f'ablations_matched_K_{fixed_k}';root.mkdir(parents=True,exist_ok=True)
    from .studies import frozen_configurations,anchor_protocol
    # Rebuild/verify the scan using its original K grid BEFORE changing K.
    scan_base=copy.deepcopy(base);scan_base.core_config.latent_dims=numbers(args.k)
    configurations=frozen_configurations(pub,scan_base,dsroot,args)
    frames=[];variants=[]
    for source,original in configurations:
        original=copy.deepcopy(original);original.core_config.latent_dims=(fixed_k,)
        variants.extend((tag,cfg,source) for tag,cfg in pub.make_ablation_variants_v8(original))
    for n,(tag,cfg,source) in enumerate(variants,1):
        print(f'[MATCHED-K ABLATION {n}/{len(variants)}] {base.data.dataset} {tag}: K={fixed_k}',flush=True)
        if args.dry_run:continue
        r=run_job(pub,core,cfg,root/tag/source,args=args,extra={'type':'matched-K-ablation','tag':tag,'K':fixed_k,'anchor':source})
        if not r.K.eq(fixed_k).all():raise AssertionError(f'Ablation K mismatch: {tag}')
        x=r.copy();x.insert(0,'ablation',tag);frames.append(x)
    if frames:
        combined=pd.concat(frames,ignore_index=True,sort=False)
        combined.to_csv(root/'ablation_locked_test_runs.csv',index=False)
        pub.summarise_publication_runs(combined).to_csv(root/'ablation_summary.csv',index=False)
        if REPORT_PLOTS:plot_ablation_effects(combined,root/'figures')
        json_write(root/'MATCHED_K_PROTOCOL.json',anchor_protocol(configurations,kind='matched-K ablation',K=fixed_k,rule='freeze validation-selected settings, replace K equally in full and all cuts; actual parameter counts and work remain reported'))


def subset_training_keep(split,keep,seed):
    """Remove ONLY training edges, preserving original val/test and train user/item support."""
    import numpy as np
    if not 0<keep<=1:raise ValueError('keep must be (0,1]')
    train=split[split['split']=='train'];other=split[split['split']!='train']
    if keep==1:return split
    rng=np.random.default_rng(int(seed)+1771)
    ids=list(train.index);rng.shuffle(ids)
    user_counts=train.userId.value_counts().to_dict();item_counts=train.movieId.value_counts().to_dict()
    target=int(round(len(train)*keep));remove_limit=max(0,len(train)-target)
    drop=[]
    for idx in ids:
        if len(drop)>=remove_limit:break
        u=int(train.at[idx,'userId']);v=int(train.at[idx,'movieId'])
        if user_counts.get(u,0)>1 and item_counts.get(v,0)>1:
            user_counts[u]-=1;item_counts[v]-=1;drop.append(idx)
    reduced=split.drop(index=drop).copy()
    assert len(reduced[reduced['split']!='train'])==len(other)
    return reduced

def run_sparsity(pub,core,base,dsroot,args):
    import pandas as pd
    from .studies import frozen_configurations,anchor_protocol
    root=pathlib.Path(dsroot)/'sparsity';root.mkdir(parents=True,exist_ok=True)
    rates=tuple(float(x) for x in args.keep_rates.split(','))
    if not rates or any(not 0<keep<=1 for keep in rates) or len(set(rates))!=len(rates):raise ValueError('Unique keep rates in (0,1] required')
    configurations=frozen_configurations(pub,base,dsroot,args)
    old_split=pub.make_core_split
    for keep in rates:
        folder=root/f'keep_{keep:g}'
        def patched(raw,spec,dc):
            return subset_training_keep(old_split(raw,spec,dc),keep,dc.split_seed)
        print(f'\n[TRAIN-ONLY SPARSITY] {base.data.dataset}: keep={keep:g}; held-out val/test fixed',flush=True)
        if args.dry_run:continue
        try:
            pub.make_core_split=patched
            frames=[]
            for tag,original in configurations:
                cfg=copy.deepcopy(original)
                if cfg.v8.analysis_anchor:
                    cfg.v8.analysis_anchor['data_signature'].pop('train_sha256',None)
                rows=run_job(pub,core,cfg,folder/tag,args=args,extra={'type':'sparsity-heldout-fixed','train_keep_requested':keep,'anchor':tag}).copy()
                counts={}
                for seed in rows.split_seed.unique():
                    assignment=pd.read_csv(folder/tag/f'split_assignments_{int(seed)}.csv')
                    counts[int(seed)]=int((assignment.split=='train').sum())
                rows['train_pairs']=rows.split_seed.map(counts)
                if cfg.v8.analysis_anchor:
                    reference=pathlib.Path(dsroot)/'search'/'jobs'/basename(cfg.v8.analysis_anchor['scan_job'])/f"split_assignments_{cfg.v8.analysis_anchor['split_seed']}.csv"
                    original=pd.read_csv(reference);rows['base_train_pairs']=int((original.split=='train').sum())
                    rows['train_keep_realized']=rows.train_pairs/rows.base_train_pairs
                frames.append(rows)
            pd.concat(frames,ignore_index=True,sort=False).to_csv(folder/'locked_test_runs.csv',index=False)
        finally:
            pub.make_core_split=old_split
    # Verify identical heldouts across fractions; no equality expected for TRAIN.
    if args.dry_run:return
    for seed in base.core_config.split_seeds:
        heldouts=[]
        for keep in rates:
            for f in sorted((root/f'keep_{keep:g}').rglob(f'split_assignments_{seed}.csv')):
                d=pd.read_csv(f);d=d[d['split']!='train']
                heldouts.append(set(map(tuple,d[['userId','movieId','rating','split']].itertuples(index=False,name=None))))
        if not heldouts:raise ValueError('No sparsity holdout assignments')
        if len({frozenset(x) for x in heldouts})!=1:raise AssertionError(f'Holdout changed during training sparsity sweep: split {seed}')
    if REPORT_PLOTS:plot_train_keep_sensitivity(root,root/'figures')
    json_write(root/'SPARSITY_PROTOCOL.json',anchor_protocol(configurations,kind='train-only thinning',heldout_sets_unchanged=True,keep_rates=rates,
        meaning='fraction of original TRAIN edges retained subject to user/item support; freeze full-data-selected hyperparameters; do not retune on TEST'))

def run_synthetic_density(pub,core,base,dsroot,args):
    if base.data.dataset!='synthetic':return
    root=pathlib.Path(dsroot)/'synthetic_density'
    for density in tuple(float(x) for x in args.density_rates.split(',')):
        if not 0<density<=1:raise ValueError('Synthetic density must be (0,1]')
        cfg=copy.deepcopy(base);cfg.data.synthetic_density=density
        print(f'[SYNTHETIC DENSITY] density={density:g}; underlying sampled observation/held-out population can differ')
        if not args.dry_run:run_job(pub,core,cfg,root/f'density_{density:g}',args=args,extra={'type':'synthetic-density-descriptive','density':density})
    json_write(root/'DENSITY_PROTOCOL.json',{'density_values':args.density_rates,'note':'Data-generating observations change; do NOT use for hyperparameter selection, do NOT claim fixed test sets'})

def run_convergence(pub,core,base,dsroot,args):
    # Separate validation-only convergence study; never evaluates Test.
    from .studies import frozen_configurations,anchor_protocol
    configurations=frozen_configurations(pub,base,dsroot,args)
    opts=copy.copy(args);opts.epoch_train_rmse=True
    root=pathlib.Path(dsroot)/'convergence'
    print(f'[CONVERGENCE] {base.data.dataset}: validation only, original checkpoint schedules; train RMSE added on those ticks')
    if not args.dry_run:
        for tag,original in configurations:
            cfg=copy.deepcopy(original);cfg.validation_only=True
            # Altering eval_every also alters patience measured in eval ticks.
            # Preserve the winner's checkpoint schedule in this diagnostic run.
            run_job(pub,core,cfg,root/tag,args=opts,validation_only=True,extra={'type':'val-only-convergence','checkpoint_schedule':'unchanged from anchor'})
            if REPORT_PLOTS:plot_convergence(root/tag/'V8_EPOCH_METRICS.csv',root/tag/'figures')
        json_write(root/'CONVERGENCE_PROTOCOL.json',anchor_protocol(configurations,kind='validation-only convergence',checkpoint_schedule='unchanged; train metrics are observational'))

def run_temporal(pub,core,base,dsroot,args):
    if base.data.dataset not in ('ml-100k','ml-1m','ml-10m'):return
    c=copy.deepcopy(base);c.data.split_mode='temporal_per_user'
    c.v8.protocol='common'
    print(f'[TEMPORAL SPLIT] {base.data.dataset}; its own validation-only K/init selection')
    if not args.dry_run:
        target=pathlib.Path(dsroot)/'temporal'
        result=scan_dataset(pub,core,c,target,args)
        run_final(pub,core,c,target,args,result)

def plan_for_ds(pub,base,args):
    jobs=build_jobs(pub,base,numbers(args.k),args.scan_mode)
    abl=pub.make_ablation_variants_v8(base)
    print(f'\n{base.data.dataset}: {len(base.methods)} supported methods | scan jobs={len(jobs)} ({sum(candidate_count(x) for x in jobs.values()):,} candidate trainings) | ablations={len(abl)} ({sum(candidate_count(x) for _,x in abl):,})')
    for k in numbers(args.k):
        c=copy.deepcopy(base);c.methods=tuple(m for m in base.methods if m not in ('AutoRec','AEMC-NE'));c.core_config.latent_dims=(k,)
        print(f'  K={k}: {candidate_count(c):,} candidate trainings')
    print('  NOTE: validation, final, ablation and K-study are distinct stages; no test used in tuning')

def main(argv=None):
    ap=argparse.ArgumentParser(description='MBMF V8 — eight research models, validation locking and complete reports')
    ap.add_argument('command',choices=['plan','all','scan','final','test','main','ablation','ablation-fixed','kstudy','calibration','sparsity','density','temporal','convergence','audit','curves','self-test','report','ensemble','lossstudy'])
    ap.add_argument('--loss-methods')
    ap.add_argument('--loss-parameters')
    ap.add_argument('--datasets',default=','.join(DATASETS),help='Comma-separated datasets; default: all six')
    ap.add_argument('--data-root',default='./datasets')
    ap.add_argument('--output',default='./results_v9')
    ap.add_argument('--split-seeds',default='2026,2027,2028,2029,2030')
    ap.add_argument('--model-seeds',default='11,22,33,44,55')
    ap.add_argument('--k',default='5,10,20,50,64')
    ap.add_argument('--methods',default=None)
    ap.add_argument('--scan-mode',choices=['one-factor','joint','hybrid'],default='hybrid')
    ap.add_argument('--backend',choices=['auto','dense','sparse'],default='auto')
    ap.add_argument('--keep-rates',default='0.3,0.5,0.7,1.0')
    ap.add_argument('--density-rates',default='0.05,0.10,0.20,0.40,0.60')
    ap.add_argument('--ablation-fixed-k',type=int,default=20,help='Matched-K mechanism comparison; secondary analysis')
    ap.add_argument('--include-temporal',action='store_true',help='Also run MovieLens chronological per-user split in all')
    ap.add_argument('--include-convergence',action='store_true',help='Full train/val RMSE epoch-tick study; expensive; no test measurements')
    ap.add_argument('--eval-every',type=int,default=None)
    ap.add_argument('--epoch-train-rmse',action=argparse.BooleanOptionalAction,default=True,help='Extra train evaluation every validation tick; computationally expensive')
    ap.add_argument('--resume',action='store_true',help='Reuse finished candidates; recover mean-phase optimizers at safe epoch boundaries')
    ap.add_argument('--dry-run',action='store_true',help='Plan without model training')
    ap.add_argument('--quick',action='store_true',help='Smoke ONLY: first split/seed/K, 3 epochs; never Q1 final')
    ap.add_argument('--scan-only',action='store_true',help='Stop after validation search; default scan also runs frozen final evaluation')
    ap.add_argument('--analysis-source',choices=['winners','defaults'],default='winners',help='Secondary ablation/K/sparsity/convergence studies inherit frozen scan winners by default')
    ap.add_argument('--include-blend',action='store_true',help='Freeze and evaluate an explicitly labelled two-model ensemble null alongside original methods')
    ap.add_argument('--blend-methods',default='AutoRec,Method4',help='Two predeclared ensemble components; coefficient uses validation only')
    a=ap.parse_args(argv)
    if a.command=='test':a.command='final'
    if a.scan_only and a.command!='scan':ap.error('--scan-only is only valid with scan')
    datasets=tuple(x.strip() for x in a.datasets.split(',') if x.strip())
    if set(datasets)-set(DATASETS):ap.error(f'Unknown datasets {set(datasets)-set(DATASETS)}')
    if a.quick and a.command in ('all','final','main'):
        print('WARNING: QUICK RESULTS ARE NOT SUITABLE FOR PUBLICATION',flush=True)
    a.data_root=str(pathlib.Path(a.data_root).resolve())
    root=pathlib.Path(a.output).resolve()
    a.output=str(root)
    if root==ROOT:raise ValueError('Choose a results folder, not the repository root')
    root.mkdir(parents=True,exist_ok=True);(root/'.gitignore').write_text('*\n')
    core,pub=load_v8()
    if a.command=='self-test':
        pub.publication_extension_self_test_v8();return
    if a.command=='curves':
        for ds in datasets:
            for p in (root/ds).rglob('V8_EPOCH_METRICS.csv'):
                plot_convergence(p,p.parent/'figures')
        return
    if a.command=='report':
        import pandas as pd
        from .reporting import generate_report
        for ds in datasets:
            from .lossstudy import rebuild as rebuild_loss
            rebuild_loss(root/ds/'lossstudy',plots=REPORT_PLOTS)
            for file in sorted((root/ds).rglob('candidate_validation_runs.csv')):
                if (file.parent/'V8_AGGREGATE_ONLY.json').exists():continue
                if any(parent.name=='final' and (parent.parent/'final_locked_test_all.csv').is_file()
                       for parent in file.parents):
                    continue # Complete raw child archives feed the combined dossier below.
                search_parent=file.parent.parent.parent
                if file.parent.parent.name=='jobs' and (search_parent/'V8_ARTIFACT_RETENTION.json').exists():
                    continue
                locked=file.parent/'locked_test_runs.csv';chosen=locked if locked.exists() else file
                if locked.exists():
                    from .reporting import summarise_locked_runs
                    summarise_locked_runs(pd.read_csv(locked)).to_csv(file.parent/'locked_test_summary.csv',index=False)
                generate_report(file.parent,pd.read_csv(chosen),validation_only=not locked.exists(),plots=REPORT_PLOTS)
            for combined in sorted((root/ds).rglob('final_locked_test_all.csv')):
                from .reporting import summarise_locked_runs
                summarise_locked_runs(pd.read_csv(combined)).to_csv(combined.parent/'final_locked_test_summary.csv',index=False)
                generate_report(combined.parent,pd.read_csv(combined),combined=True,plots=REPORT_PLOTS)
            for search_table in sorted((root/ds).rglob('hyperparameter_validation_all.csv')):
                search=search_table.parent
                from .search_report import generate_search_report
                generate_search_report(search,pd.read_csv(search/'hyperparameter_validation_all.csv'),pd.read_csv(search/'validation_winners.csv'),pd.read_csv(search/'validation_ranked_configs.csv'),plots=REPORT_PLOTS)
                if REPORT_PLOTS:plot_validation_landscape(search/'validation_ranked_configs.csv',search/'figures')
            ensemble=root/ds/'ensemble'/'ensemble_locked_test_runs.csv'
            if ensemble.is_file():generate_report(ensemble.parent,pd.read_csv(ensemble),plots=REPORT_PLOTS)
            if REPORT_PLOTS:
                for folder in (root/ds).glob('ablations*'):
                    file=folder/'ablation_locked_test_runs.csv'
                    if file.is_file():plot_ablation_effects(pd.read_csv(file),folder/'figures')
                plot_k_rmse(root/ds/'kstudy',root/ds/'kstudy'/'figures')
                plot_train_keep_sensitivity(root/ds/'sparsity',root/ds/'sparsity'/'figures')
            from .readiness import build_checklist
            build_checklist(root/ds)
            from .secondary_report import rebuild_secondary_reports
            rebuild_secondary_reports(root/ds,plots=REPORT_PLOTS)
        return
    for ds in datasets:
        base=make_base(pub,a,ds);dsroot=root/ds
        if a.command not in ('audit','plan') and not a.dry_run:
            from .recovery import require_dataset_inputs
            require_dataset_inputs(base.data)
        if a.command not in ('audit','plan') and not a.dry_run:
            json_write(dsroot/'V8_STUDY_MODE.json',{'quick':a.quick,'mode':'smoke' if a.quick else 'research','command':a.command})
        if a.command=='plan':plan_for_ds(pub,base,a);continue
        if a.command=='audit':
            result=audit_selection(dsroot)
            print(ds,json.dumps(result,ensure_ascii=False))
            if result['status']!='PASS':raise RuntimeError('V8 artifact audit failed: '+'; '.join(result['errors']))
            continue
        if a.dry_run:
            plan_for_ds(pub,base,a);continue
        pub.publication_extension_self_test_v8()
        if a.command=='ensemble':
            import pandas as pd
            from .ensemble import apply_locked_blend
            apply_locked_blend(dsroot,pd.read_csv(dsroot/'final_locked_test_all.csv'),tuple(m.strip() for m in a.blend_methods.split(',')),plots=base.auto_generate_plots)
        if a.command=='main':run_job(pub,core,base,dsroot/'main',args=a,extra={'type':'main-default'})
        if a.command in ('scan','all'):
            scan=scan_dataset(pub,core,base,dsroot,a)
        else:scan=None
        if a.command in ('final','all') or (a.command=='scan' and not a.scan_only):run_final(pub,core,base,dsroot,a,scan)
        if a.command in ('lossstudy','all'):
            from .lossstudy import run as run_loss_study
            run_loss_study(pub,core,base,dsroot,a)
        if a.command in ('calibration','all'):run_calibration(pub,core,base,dsroot,a)
        if a.command in ('ablation','all'):run_ablation(pub,core,base,dsroot,a)
        if a.command in ('ablation-fixed','all'):run_ablation_fixed_k(pub,core,base,dsroot,a)
        if a.command in ('kstudy','all'):run_kstudy(pub,core,base,dsroot,a)
        if a.command in ('sparsity','all'):run_sparsity(pub,core,base,dsroot,a)
        if a.command in ('density','all'):run_synthetic_density(pub,core,base,dsroot,a)
        if a.command=='temporal' or (a.command=='all' and a.include_temporal):run_temporal(pub,core,base,dsroot,a)
        if a.command=='convergence' or (a.command=='all' and a.include_convergence):run_convergence(pub,core,base,dsroot,a)
        if a.command=='all':
            for f in (dsroot/'final').rglob('V8_EPOCH_METRICS.csv'):
                plot_convergence(f,f.parent/'figures')
            print('[POST-RUN AUDIT]',ds, json.dumps(audit_selection(dsroot),ensure_ascii=False),flush=True)
        from .secondary_report import rebuild_secondary_reports
        rebuild_secondary_reports(dsroot,plots=REPORT_PLOTS)
        from .readiness import build_checklist
        build_checklist(dsroot)
    print(f'V8 completed requested stages. Root={root}',flush=True)

if __name__=='__main__':main()
