"""Lossless split-role storage and immutable shared CSV artifacts (V8 only)."""
from collections import OrderedDict
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
import uuid
import numpy as np
import pandas as pd


class OneEntryCache(dict):
    def __setitem__(self,key,value):
        self.clear()
        super().__setitem__(key,value)


def artifact_key(cfg,kind):
    from .recovery import input_digest
    if not hasattr(cfg,'_v8_input_digest'):cfg._v8_input_digest=input_digest(cfg.data)
    identity=dict(data=asdict(cfg.data),input=cfg._v8_input_digest,kind=kind,
        source=cfg.v8.implementation_sha256,pandas=pd.__version__,roles={
            key:getattr(cfg.v8,key) for key in ('role_seed','split_validation_roles',
                'validation_selection_fraction','validation_calibration_fraction')})
    identity['data'].pop('data_root',None)
    return hashlib.sha256(json.dumps(identity,sort_keys=True,default=str).encode()).hexdigest()


def _cache_root(cfg):
    root=Path(getattr(cfg,'_v8_shared_data_root',Path(cfg.output_dir)/'_shared_data'))
    root.mkdir(parents=True,exist_ok=True)
    return root


def _link(cache,target,cfg):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink():raise ValueError('Refusing shared artifact through a symbolic link')
    if target.exists() and os.path.samefile(cache,target):return target
    if target.exists():target.chmod(0o644);target.unlink()
    try:os.link(cache,target);mode='hardlink'
    except OSError:shutil.copyfile(cache,target);mode='copy-fallback'
    cache.chmod(0o444);target.chmod(0o444)
    record=dict(target=str(target),cache=str(cache),mode=mode,bytes=cache.stat().st_size)
    with (Path(cfg.output_dir)/'V8_SHARED_ARTIFACTS.jsonl').open('a',encoding='utf8') as handle:
        handle.write(json.dumps(record)+'\n')
    return target


def shared_file(cfg,target,kind,builder):
    cache=_cache_root(cfg)/(artifact_key(cfg,kind)+'.csv')
    if not cache.is_file():
        temporary=cache.with_name('.'+cache.name+'.'+uuid.uuid4().hex)
        try:builder(temporary);temporary.replace(cache)
        finally:temporary.unlink(missing_ok=True)
    return _link(cache,target,cfg)


def shared_csv(frame,target,cfg,kind):
    return shared_file(cfg,target,kind,lambda path:frame.to_csv(path,index=False))


def shared_export(pub,data,target,cfg,split_seed,out):
    return shared_file(cfg,target,'external_'+str(split_seed),
        lambda path:pub.export_split_for_external(data,path,id_map_dir=out/'id_maps',dataset_name=cfg.data.dataset))


def shared_load(pub,cfg,out):
    cache=_cache_root(cfg)/artifact_key(cfg,'id_maps')
    names=[cfg.data.dataset+'_user_id_map.csv',cfg.data.dataset+'_item_id_map.csv']
    if all((cache/name).is_file() for name in names):
        raw,spec=pub.load_publication_dataset(cfg.data,save_mappings_to=None)
    else:
        cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            raw,spec=pub.load_publication_dataset(cfg.data,save_mappings_to=temporary)
            for name in names:
                target=cache/name
                if not target.exists():shutil.copyfile(Path(temporary)/name,target)
    for name in names:_link(cache/name,out/'id_maps'/name,cfg)
    return raw,spec


class SplitFrames:
    """Keep raw data once and one int8 role per interaction, not S full frames."""
    names=np.array(['train','val','test'],dtype=object)
    def __init__(self,raw,cfg,pub,spec):
        self.raw=raw;self.cfg=cfg;self.pub=pub;self.spec=spec
        self.paths=OrderedDict();self.dtypes={};self.original_execution=cfg.execution
        total=int(cfg.execution.ssd_budget_gb*1024**3)
        needed=(len(raw)+4096)*len(cfg.core_config.split_seeds)
        self.budget=min(total//4,needed) if cfg.execution.storage!='resident' else 0
        self.used=0;self.peak_used=0;self.regenerations=0;self.evictions=0
        if cfg.execution.cache_dir:Path(cfg.execution.cache_dir).mkdir(parents=True,exist_ok=True)
        self.root=Path(tempfile.mkdtemp(prefix='v8_split_roles_',dir=cfg.execution.cache_dir))
        cfg.execution=replace(cfg.execution,ssd_budget_gb=(total-self.budget)/1024**3)
        cfg._v8_split_frames=self

    def __setitem__(self,seed,frame):
        if list(frame.columns)!=list(self.raw.columns)+['split'] or not frame.index.equals(self.raw.index):
            raise ValueError('Split changed raw row order/columns; refusing a lossy cache')
        if any(not frame[name].equals(self.raw[name]) for name in self.raw):
            raise ValueError('Split changed raw values; refusing a lossy cache')
        codes=pd.Categorical(frame['split'],categories=list(self.names)).codes
        if (codes<0).any():raise ValueError('Unknown split role')
        self.dtypes[int(seed)]=frame['split'].dtype
        if int(seed) in self.paths:
            old,size=self.paths.pop(int(seed));old.unlink();self.used-=size
        required=codes.nbytes+4096
        if required>self.budget:return
        while self.paths and self.used+required>self.budget:
            _,(old,size)=self.paths.popitem(last=False);old.unlink();self.used-=size;self.evictions+=1
        path=self.root/(str(seed)+'.npy');np.save(path,codes,allow_pickle=False)
        self.paths[int(seed)]=(path,required);self.used+=required;self.peak_used=max(self.peak_used,self.used)

    def __getitem__(self,seed):
        seed=int(seed)
        if seed not in self.dtypes:raise KeyError(seed)
        if seed in self.paths:
            path,size=self.paths.pop(seed);self.paths[seed]=(path,size)
            codes=np.load(path,mmap_mode='r',allow_pickle=False)
            try:
                frame=self.raw.copy()
                frame['split']=pd.Series(self.names[codes],index=frame.index,dtype=self.dtypes[seed])
                return frame
            finally:codes._mmap.close()
        import copy
        dc=copy.deepcopy(self.cfg.data);dc.split_seed=seed
        frame=self.pub.make_core_split(self.raw,self.spec,dc)
        self.regenerations+=1;self[seed]=frame
        return frame

    def close(self):
        record=dict(policy='shared raw dataframe + int8 split roles; one final tensor-data cache entry',
            cached_role_vectors=len(self.paths),role_cache_reserved_bytes=self.budget,
            role_cache_peak_upper_bound_bytes=self.peak_used,evictions=self.evictions,regenerations=self.regenerations,
            saved_tensor_ssd_budget_gb=self.cfg.execution.ssd_budget_gb,
            total_requested_temporary_ssd_budget_gb=self.original_execution.ssd_budget_gb,
            raw_values_and_row_order_preserved=True)
        try:(Path(self.cfg.output_dir)/'V8_SPLIT_CACHE.json').write_text(json.dumps(record,indent=2))
        finally:
            self.cfg.execution=self.original_execution
            shutil.rmtree(self.root)
            self.paths.clear();self.raw=None
            del self.cfg._v8_split_frames
