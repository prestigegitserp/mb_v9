"""Strict V2 checkpoint roundtrip without touching the frozen V8 checkpoint module.

V8 performs TEST only after validation selection by *reloading* the selected
candidate. A V2 model must be re-adapted BEFORE loading its expanded state_dict.
"""
from __future__ import annotations
import torch
from . import __version__
from .adapters import adapt


def install_checkpoint_hooks(core,pub,profile):
    if getattr(pub,'_v2_checkpoint_hooks_installed',False):return
    original_save=pub.save_model_checkpoint_v8
    original_load=pub.load_model_checkpoint_v8

    def save(path,model,*,config,data,train_info=None,extra=None):
        ex=dict(extra or {})
        if hasattr(model,'v2_method'):
            ex['v2_architecture_version']=__version__
            ex['v2_profile']=profile
            ex['method']=model.v2_method
        return original_save(path,model,config=config,data=data,train_info=train_info,extra=ex)

    def load(path,data,device,*,strict_data=True):
        ck=torch.load(path,map_location='cpu',weights_only=False)
        extra=ck.get('extra',{})
        version=extra.get('v2_architecture_version')
        if version is None:
            return original_load(path,data,device,strict_data=strict_data)
        if version!=__version__:
            raise ValueError(f'V2 checkpoint architecture version mismatch: {version} vs {__version__}')
        if extra.get('v2_profile')!=profile:
            raise ValueError('V2 checkpoint profile mismatch; refusing mixed-profile predictions')
        if ck.get('format')!='MBMF-Q1-v8' or ck.get('frozen_v7_core_sha256')!=pub.FROZEN_V7_CORE_SHA256:
            raise ValueError('V2 checkpoint format or frozen V7 scientific SHA mismatch')
        if strict_data:
            new=pub.data_signature(data);old=ck['data_signature']
            for key in ('num_users','num_items','rating_min','rating_max','train_sha256'):
                if str(new[key])!=str(old[key]):
                    raise ValueError(f'V2 checkpoint/data mismatch for {key}')
        method=extra.get('method')
        if method not in {f'Method{i}' for i in range(1,8)}:
            raise ValueError('V2 checkpoint missing a valid replacement method identity')
        cfg_payload=ck['config']
        family=ck['family']
        if method in ('Method1','Method2','Method3','Method4'):
            idx=int(method[-1]);config_cls=getattr(core,f'Method{idx}Config')
            classes={
                1:('ThesisMethod1','ScalableThesisMethod1'),
                2:('ThesisMethod2','ScalableThesisMethod2'),
                3:('ProposedMethod3','ScalableProposedMethod3'),
                4:('ProposedMethod4','ScalableProposedMethod4'),
            }
            dense_name,sparse_name=classes[idx]
            cls_name=sparse_name if family==sparse_name else dense_name
            if family not in (dense_name,sparse_name):
                raise ValueError(f'V2 model family mismatch {method}: {family}')
            model_cls=getattr(pub if cls_name==sparse_name else core,cls_name)
            cfg=config_cls(**cfg_payload)
            model=model_cls(data,cfg,device)
        elif method in ('Method5','Method6'):
            if family!='DecoderCoupledDualViewMBMF':
                raise ValueError('V2 dual-view family mismatch')
            cfg=(core.Method5Config if method=='Method5' else core.Method6Config)(**cfg_payload)
            model=core.DecoderCoupledDualViewMBMF(data,cfg,device)
        else:
            if family!='ProposedMethod7':raise ValueError('V2 Method7 family mismatch')
            cfg=core.Method7Config(**cfg_payload)
            model=core.ProposedMethod7(data,cfg,device)
        model=adapt(model,method,data,profile,core,pub)
        model.load_state_dict(ck['state_dict'],strict=True)
        model.to(device);model.eval()
        return model,ck

    pub.save_model_checkpoint_v8=save
    pub.load_model_checkpoint_v8=load
    pub._v2_checkpoint_hooks_installed=True
