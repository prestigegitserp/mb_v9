"""V9 orchestration around the frozen V8 scientific runtime.

No scientific model, loss function, trainer or test-label selection code is modified.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, pathlib, re, sys, time, traceback
from dataclasses import asdict

from . import __version__
from .audit import audit_selection
from .figures import plot_convergence,plot_k_rmse,plot_validation_landscape,plot_train_keep_sensitivity
from .progress import CandidateProgress
from .search import build_jobs,candidate_count
from .selection import fingerprint_config,select_validation_winners

ROOT=pathlib.Path(__file__).resolve().parents[1]
VENDOR=ROOT/'vendor_v8'
REQUIRED_SHA='6da726cb155ab6ad291f5f608febc37f42d1c25ac3924910f56917908c1f124c'
DATASETS=('synthetic','jester','bookcrossing','ml-100k','ml-1m','ml-10m')

def load_v8():
    v7=VENDOR/'mbmf_q1_final_v7.py'
    if not v7.exists():raise RuntimeError('Run: python bootstrap_v8.py --from-v8 <path_to_fixed_v8>')
    normalized=hashlib.sha256(v7.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    if normalized!=REQUIRED_SHA:raise RuntimeError(f'V7 scientific core SHA mismatch: {normalized}')
    if str(VENDOR) not in sys.path:sys.path.insert(0,str(VENDOR))
    import mbmf_q1_final_v7 as core
    import mbmf_q1_infrastructure_v8 as pub
    return core,pub

def numbers(s):return tuple(int(x.strip()) for x in str(s).split(',') if x.strip())

def basename(s):return re.sub('[^a-zA-Z0-9_.=-]','_',str(s))

def make_base(pub,args,dataset):
    cfg=pub.PublicationExperimentConfigV8()
    cfg.data.dataset=dataset;cfg.data.data_root=args.data_root
    cfg.data.force_backend=args.backend
    cfg.core_config.split_seeds=numbers(args.split_seeds)
    cfg.core_config.model_seeds=numbers(args.model_seeds)
    cfg.core_config.latent_dims=numbers(args.k)
    if args.methods:
        allowed=set(cfg.methods);chosen=tuple(x.strip() for x in args.methods.split(',') if x.strip())
        if set(chosen)-allowed:raise ValueError(f'Unknown methods: {set(chosen)-allowed}')
        cfg.methods=chosen
    cfg.methods=tuple(m for m in cfg.methods if pub.method_supported_on_dataset_v8(m,dataset))
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
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf8');tmp.replace(p)

def run_job(pub,core,cfg,folder,*,args,validation_only=False,extra=None):
    folder=pathlib.Path(folder);folder.mkdir(parents=True,exist_ok=True)
    cfg=copy.deepcopy(cfg);cfg.output_dir=str(folder)
    cfg.validation_only=bool(validation_only)
    identity=fingerprint_config(cfg)
    statefile=folder/'V9_JOB_STATUS.json'
    required=folder/('candidate_validation_runs.csv' if validation_only else 'locked_test_runs.csv')
    if args.resume and statefile.is_file():
        state=json.loads(statefile.read_text(encoding='utf8'))
        if state.get('fingerprint')!=identity:raise ValueError(f'Existing run with DIFFERENT config: {folder}; choose a new --output')
        if state.get('status')=='COMPLETED' and required.is_file():
            print(f'[RESUME] completed: {folder}',flush=True)
            import pandas as pd
            return pd.read_csv(required)
    else:
        if required.is_file() or (folder/'config.json').is_file():
            raise ValueError(f'Existing untracked/previous study at {folder}; use --resume or unique output')
    info={'status':'RUNNING','v9':__version__,'fingerprint':identity,'start_unix':time.time(),'validation_only':validation_only,'extra':extra or {},'candidate_count_expected':candidate_count(cfg)}
    json_write(statefile,info)
    progress=CandidateProgress(candidate_count(cfg),str(extra or folder.name),folder/'V9_EPOCH_METRICS.csv',epoch_train_rmse=args.epoch_train_rmse)
    try:
        with progress.instrument(pub,core):
            candidates,locked=pub.run_publication_study_v8(cfg)
        if progress.done!=progress.total:
            raise RuntimeError(f'Candidate count mismatch: {progress.done}/{progress.total}')
        data=candidates if validation_only else locked
        if validation_only and any(x.startswith('test_') for x in data.columns):
            raise AssertionError('LEAK: validation_only produced test metrics')
        info.update(status='COMPLETED',end_unix=time.time(),candidate_count_actual=progress.done,epoch_events=progress.n_epoch_records)
        json_write(statefile,info)
        return data
    except BaseException as exc:
        info.update(status='FAILED',end_unix=time.time(),error=repr(exc),traceback=traceback.format_exc())
        json_write(statefile,info)
        raise

def scan_dataset(pub,core,base,dsroot,args):
    import pandas as pd
    target=pathlib.Path(dsroot)/'search';target.mkdir(parents=True,exist_ok=True)
    if any((target/'final').glob('*')):pass
    plan=build_jobs(pub,base,numbers(args.k),mode=args.scan_mode)
    # Hyperparameter search needs numerical train/validation metrics, not millions
    # of per-candidate prediction archives (which can exhaust disk on ML-10M).
    # K-study/final/ablation runs retain their prediction archives.
    for config in plan.values():
        config.save_predictions=False
        config.save_selected_checkpoints=False
    summary=[];all_tables=[]
    json_write(target/'V9_SEARCH_PLAN.json',{'mode':args.scan_mode,'jobs':[{ 'tag':tag,'methods':list(cfg.methods),'candidates':candidate_count(cfg)} for tag,cfg in plan.items()],
        'excluded_from_search':['train_fraction','synthetic_density','dataset-sensitive tests'],
        'selection_rule':'mean val_RMSE across identical split/model seeds; never test metrics'})
    if args.dry_run:
        print(f'[PLAN] {base.data.dataset}: {len(plan)} search jobs, {sum(candidate_count(c) for c in plan.values()):,} candidate trainings');return None
    for index,(tag,cfg) in enumerate(plan.items(),1):
        print(f'\n[SEARCH JOB {index}/{len(plan)}] {tag}; candidates={candidate_count(cfg)}',flush=True)
        path=target/'jobs'/basename(tag)
        data=run_job(pub,core,cfg,path,args=args,validation_only=True,extra={'type':'hyper','tag':tag,'dataset':base.data.dataset})
        data=data.copy();data['scan_job']=tag;all_tables.append(data)
    if not all_tables:raise RuntimeError('No validation candidates')
    table=pd.concat(all_tables,ignore_index=True,sort=False)
    if any(str(c).startswith('test_') for c in table.columns):raise AssertionError('Test leak')
    table.to_csv(target/'hyperparameter_validation_all.csv',index=False)
    winners,ranked=select_validation_winners(table,len(base.core_config.split_seeds),len(base.core_config.model_seeds))
    winners.to_csv(target/'validation_winners.csv',index=False)
    ranked.to_csv(target/'validation_ranked_configs.csv',index=False)
    plot_validation_landscape(target/'validation_ranked_configs.csv',target/'figures')
    json_write(target/'validation_selection_manifest.json',{'rule':'FOR EACH OUTER SPLIT: min mean val RMSE across matching model seeds, tie val std, K, tag, config_key','test_seen':False,
      'selected_split_configs':[{'method':str(r.method),'split_seed':int(r.split_seed),'scan_job':str(r.scan_job),'K':None if pd.isna(r.K) else int(r.K),'init_mode':str(r.init_mode),'val_mean':float(r.val_mean),'n_model_seeds':int(r.n)} for _,r in winners.iterrows()]})
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
        pub.summarise_publication_runs(combined).to_csv(root/'final_locked_test_summary.csv',index=False)
        # All final methods used the same locked held-out interactions per outer split.
        from mbmf_q1_visuals_v8 import generate_publication_figures_v8
        generate_publication_figures_v8(combined,root/'final_report')
        try:
            sm=combined.groupby(['dataset','method','split_seed'],as_index=False).test_RMSE.mean()
            pub.pairwise_wilcoxon_holm(sm,metric='test_RMSE',group_cols=('dataset','split_seed')).to_csv(root/'final_wilcoxon_holm.csv',index=False)
            ranks,stat=pub.friedman_nemenyi_summary(sm,metric='test_RMSE',block_cols=('dataset','split_seed'))
            ranks.to_csv(root/'final_friedman_ranks.csv',index=False)
            pd.DataFrame([stat]).to_csv(root/'final_friedman_stats.csv',index=False)
        except Exception as exc:
            import warnings
            warnings.warn(f'V9 final statistical comparison skipped: {exc}')
        json_write(root/'final_selection_provenance.json',{'selected_on':'validation ONLY','test_used_for_selection':False,'selected_method_splits':winners[['method','split_seed']].to_dict(orient='records'),'scientific_core_sha256':REQUIRED_SHA})
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
    criterion='val_Gaussian_NLL'
    if criterion not in d.columns:raise ValueError('Method7 validation Gaussian NLL unavailable')
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
    methods=tuple(m for m in base.methods if m not in ('AutoRec','AEMC-NE'))
    all_runs=[]
    for k in numbers(args.k):
        c=copy.deepcopy(base);c.methods=methods;c.core_config.latent_dims=(k,)
        print(f'\n[FIXED-K TEST SENSITIVITY] dataset={base.data.dataset} K={k}; TEST IS DESCRIPTIVE ONLY',flush=True)
        if args.dry_run:
            print(f'  candidates={candidate_count(c)}');continue
        rows=run_job(pub,core,c,root/f'K_{k}',args=args,extra={'type':'fixed-K-descriptive','K':k})
        if rows.K.isna().any() or not rows.K.eq(k).all():raise AssertionError('K-study mismatch')
        all_runs.append(rows)
    if all_runs:
        pd.concat(all_runs,ignore_index=True).to_csv(root/'all_fixed_k_locked_runs.csv',index=False)
        plot_k_rmse(root,root/'figures')

def run_ablation(pub,core,base,dsroot,args):
    import pandas as pd
    root=pathlib.Path(dsroot)/'ablations';root.mkdir(parents=True,exist_ok=True)
    variants=pub.make_ablation_variants_v8(base)
    frames=[]
    for i,(tag,cfg) in enumerate(variants,1):
        print(f'\n[ABLATION {i}/{len(variants)}] {base.data.dataset}: {tag}',flush=True)
        if args.dry_run:
            print(f'  candidates={candidate_count(cfg)}');continue
        result=run_job(pub,core,cfg,root/tag,args=args,extra={'type':'ablation','tag':tag})
        result=result.copy();result.insert(0,'ablation',tag);frames.append(result)
    if frames:
        combined=pd.concat(frames,ignore_index=True,sort=False)
        combined.to_csv(root/'ablation_locked_test_runs.csv',index=False)
        pub.summarise_publication_runs(combined).to_csv(root/'ablation_summary.csv',index=False)
        from mbmf_q1_visuals_v8 import generate_ablation_figures_v8
        generate_ablation_figures_v8(combined,root)

def run_ablation_fixed_k(pub,core,base,dsroot,args):
    # Strictly matched latent dimension for mechanism isolation. Not a hyperparameter tuner.
    import pandas as pd
    fixed_k=int(args.ablation_fixed_k)
    if fixed_k<=0:raise ValueError('--ablation-fixed-k must be positive')
    base=copy.deepcopy(base);base.core_config.latent_dims=(fixed_k,)
    root=pathlib.Path(dsroot)/f'ablations_matched_K_{fixed_k}';root.mkdir(parents=True,exist_ok=True)
    frames=[];variants=pub.make_ablation_variants_v8(base)
    for n,(tag,cfg) in enumerate(variants,1):
        print(f'[MATCHED-K ABLATION {n}/{len(variants)}] {base.data.dataset} {tag}: K={fixed_k}',flush=True)
        if args.dry_run:continue
        r=run_job(pub,core,cfg,root/tag,args=args,extra={'type':'matched-K-ablation','tag':tag,'K':fixed_k})
        if not r.K.eq(fixed_k).all():raise AssertionError(f'Ablation K mismatch: {tag}')
        x=r.copy();x.insert(0,'ablation',tag);frames.append(x)
    if frames:
        combined=pd.concat(frames,ignore_index=True,sort=False)
        combined.to_csv(root/'ablation_locked_test_runs.csv',index=False)
        pub.summarise_publication_runs(combined).to_csv(root/'ablation_summary.csv',index=False)
        from mbmf_q1_visuals_v8 import generate_ablation_figures_v8
        generate_ablation_figures_v8(combined,root)
        json_write(root/'MATCHED_K_PROTOCOL.json',{'K':fixed_k,'rule':'same K for full and every ablation; independent of test; settings are Q1-safe primary defaults'})


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
    root=pathlib.Path(dsroot)/'sparsity';root.mkdir(parents=True,exist_ok=True)
    rates=tuple(float(x) for x in args.keep_rates.split(','))
    old_split=pub.make_core_split
    for keep in rates:
        cfg=copy.deepcopy(base)
        folder=root/f'keep_{keep:g}'
        def patched(raw,spec,dc):
            return subset_training_keep(old_split(raw,spec,dc),keep,dc.split_seed)
        print(f'\n[TRAIN-ONLY SPARSITY] {base.data.dataset}: keep={keep:g}; held-out val/test fixed',flush=True)
        if args.dry_run:continue
        try:
            pub.make_core_split=patched
            run_job(pub,core,cfg,folder,args=args,extra={'type':'sparsity-heldout-fixed','train_keep_requested':keep})
        finally:
            pub.make_core_split=old_split
    # Verify identical heldouts across fractions; no equality expected for TRAIN.
    if args.dry_run:return
    for seed in base.core_config.split_seeds:
        heldouts=[]
        for keep in rates:
            f=root/f'keep_{keep:g}'/f'split_assignments_{seed}.csv'
            d=pd.read_csv(f);d=d[d['split']!='train']
            heldouts.append(set(map(tuple,d[['userId','movieId','rating','split']].itertuples(index=False,name=None))))
        if len({frozenset(x) for x in heldouts})!=1:raise AssertionError(f'Holdout changed during training sparsity sweep: split {seed}')
    plot_train_keep_sensitivity(root,root/'figures')
    json_write(root/'SPARSITY_PROTOCOL.json',{'heldout_sets_unchanged':True,'keep_rates':rates,'meaning':'percentage of BASE TRAIN eligible edges retained, subject to user/item support; never train fraction of all observations'})

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
    cfg=copy.deepcopy(base);cfg.validation_only=True
    interval=args.eval_every if args.eval_every is not None else 5
    cfg.core_config.paper.eval_every=interval
    for i in range(1,8):getattr(cfg.core_config,f'method{i}').eval_every=interval
    for c in (cfg.autorec,cfg.aemcne,cfg.biasmf):c.eval_every=interval
    opts=copy.copy(args);opts.epoch_train_rmse=True
    root=pathlib.Path(dsroot)/'convergence'
    print(f'[CONVERGENCE] {base.data.dataset}: validation only, train RMSE on evaluation ticks every {interval} epochs')
    if not args.dry_run:
        run_job(pub,core,cfg,root,args=opts,validation_only=True,extra={'type':'val-only-convergence','train_val_each_tick':interval})
        plot_convergence(root/'V9_EPOCH_METRICS.csv',root/'figures')

def run_temporal(pub,core,base,dsroot,args):
    if base.data.dataset not in ('ml-100k','ml-1m','ml-10m'):return
    c=copy.deepcopy(base);c.data.split_mode='temporal_per_user'
    print(f'[TEMPORAL SPLIT] {base.data.dataset}; its own validation-only K/init selection')
    if not args.dry_run:run_job(pub,core,c,pathlib.Path(dsroot)/'temporal',args=args,extra={'type':'temporal-per-user-secondary-study'})

def plan_for_ds(pub,base,args):
    jobs=build_jobs(pub,base,numbers(args.k),args.scan_mode)
    abl=pub.make_ablation_variants_v8(base)
    print(f'\n{base.data.dataset}: {len(base.methods)} supported methods | scan jobs={len(jobs)} ({sum(candidate_count(x) for x in jobs.values()):,} candidate trainings) | ablations={len(abl)} ({sum(candidate_count(x) for _,x in abl):,})')
    for k in numbers(args.k):
        c=copy.deepcopy(base);c.methods=tuple(m for m in base.methods if m not in ('AutoRec','AEMC-NE'));c.core_config.latent_dims=(k,)
        print(f'  K={k}: {candidate_count(c):,} candidate trainings')
    print('  NOTE: validation, final, ablation and K-study are distinct stages; no test used in tuning')

def main(argv=None):
    ap=argparse.ArgumentParser(description='MBMF V9 — fixed scientific core, visible progress and validation-first selection')
    ap.add_argument('command',choices=['plan','all','scan','final','main','ablation','ablation-fixed','kstudy','calibration','sparsity','density','temporal','convergence','audit','curves','self-test'])
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
    ap.add_argument('--epoch-train-rmse',action='store_true',help='Extra train evaluation every validation tick; computationally expensive')
    ap.add_argument('--resume',action='store_true',help='Skip completed jobs only if config fingerprint and result files match')
    ap.add_argument('--dry-run',action='store_true',help='Plan without model training')
    ap.add_argument('--quick',action='store_true',help='Smoke ONLY: first split/seed/K, 3 epochs; never Q1 final')
    a=ap.parse_args(argv)
    datasets=tuple(x.strip() for x in a.datasets.split(',') if x.strip())
    if set(datasets)-set(DATASETS):ap.error(f'Unknown datasets {set(datasets)-set(DATASETS)}')
    if a.quick and a.command in ('all','final','main'):
        print('WARNING: QUICK RESULTS ARE NOT SUITABLE FOR PUBLICATION',flush=True)
    root=pathlib.Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    core,pub=load_v8()
    if a.command=='self-test':
        pub.publication_extension_self_test_v8();return
    if a.command=='curves':
        for ds in datasets:
            for p in (root/ds).rglob('V9_EPOCH_METRICS.csv'):
                plot_convergence(p,p.parent/'figures')
        return
    for ds in datasets:
        base=make_base(pub,a,ds);dsroot=root/ds
        if a.command=='plan':plan_for_ds(pub,base,a);continue
        if a.command=='audit':
            result=audit_selection(dsroot)
            print(ds,json.dumps(result,ensure_ascii=False))
            continue
        if a.dry_run:
            plan_for_ds(pub,base,a);continue
        pub.publication_extension_self_test_v8()
        if a.command=='main':run_job(pub,core,base,dsroot/'main',args=a,extra={'type':'main-default'})
        if a.command in ('scan','all'):
            scan=scan_dataset(pub,core,base,dsroot,a)
        else:scan=None
        if a.command in ('final','all'):run_final(pub,core,base,dsroot,a,scan)
        if a.command in ('calibration','all'):run_calibration(pub,core,base,dsroot,a)
        if a.command in ('ablation','all'):run_ablation(pub,core,base,dsroot,a)
        if a.command in ('ablation-fixed','all'):run_ablation_fixed_k(pub,core,base,dsroot,a)
        if a.command in ('kstudy','all'):run_kstudy(pub,core,base,dsroot,a)
        if a.command in ('sparsity','all'):run_sparsity(pub,core,base,dsroot,a)
        if a.command in ('density','all'):run_synthetic_density(pub,core,base,dsroot,a)
        if a.command=='temporal' or (a.command=='all' and a.include_temporal):run_temporal(pub,core,base,dsroot,a)
        if a.command=='convergence' or (a.command=='all' and a.include_convergence):run_convergence(pub,core,base,dsroot,a)
        if a.command=='all':
            for f in (dsroot/'final').rglob('V9_EPOCH_METRICS.csv'):
                plot_convergence(f,f.parent/'figures')
            print('[POST-RUN AUDIT]',ds, json.dumps(audit_selection(dsroot),ensure_ascii=False),flush=True)
    print(f'V9 completed requested stages. Root={root}',flush=True)

if __name__=='__main__':main()
