"""Backward-compatible V2 model *instances* layered over immutable V7/V8 classes.

The runtime swaps instance-bound methods after the original constructor has run.
Consequently isinstance checks, publication metrics and PyTorch state_dict behavior
continue to work, while V1 files / SHA checks stay identical.
"""
from __future__ import annotations
import copy, math, types
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import unit, cosine_pairs, bounded_residual, monotone_base, radial_gain

# Fixed research defaults are *hypotheses*, not empirical optima.
PROFILES = {
    'regularized': {'reconstruction': .025, 'prediction': 5., 'decay': 1e-4,
                    'gain_floor': .15, 'warmup': 20, 'hidden_scale': .5},
    'balanced': {'reconstruction': .08, 'prediction': 5., 'decay': 2e-5,
                 'gain_floor': .15, 'warmup': 20, 'hidden_scale': 1.},
    'capacity': {'reconstruction': .08, 'prediction': 5., 'decay': 1e-4,
                 'gain_floor': .15, 'warmup': 20, 'hidden_scale': 2.},
}


def apply_profile(cfg, name):
    """Modify only configuration fields supported by frozen V7 dataclasses.

    Change in parameter count is explicit and subject to validation selection;
    increasing layers by default would worsen overfitting rather than cure it.
    """
    if name not in PROFILES:raise ValueError(f'Unknown V2 profile {name}')
    opt=PROFILES[name]
    cfg=copy.deepcopy(cfg)
    for i in range(1, 8):
        c=getattr(cfg.core_config,f'method{i}')
        if i<=6:c.angle_init='isotropic_gaussian'
        if i in (1,5,6,7):
            c.hidden_dim=max(16,min(512,int(round(c.hidden_dim*opt['hidden_scale']))))
        c.weight_decay=max(float(c.weight_decay),opt['decay'])
        caps={1:450,2:450,3:500,4:2000,5:450,6:650,7:650}
        c.max_epochs=min(c.max_epochs,caps[i])
        c.eval_every={1:10,2:10,3:10,4:20,5:10,6:10,7:5}[i]
        c.val_patience={1:12,2:12,3:15,4:15,5:12,6:15,7:15}[i]
        if i==1:
            c.alpha_ae=opt['reconstruction'];c.beta_mbmf=1.
        if i==2:
            c.consistency_weight=.04
        if i==3:
            c.alpha_ae=opt['reconstruction'];c.beta_mbmf=1.;c.delta_l2=.01
        if i==4:
            c.alpha_ae=opt['reconstruction'];c.beta_mbmf=1.;c.delta_l2=.01
        if i in (5,6):
            c.prediction_weight=opt['prediction']
            c.user_reconstruction_weight=opt['reconstruction']
            c.item_reconstruction_weight=opt['reconstruction']
            c.branch_consistency_weight=.02
            c.l2_code_weight=1e-5
        if i==6:
            # Absolute temperature is set by instantiate_v2 after K is selected.
            c.radius_temperature=0.
        if i==7:
            c.feature_dropout=.15 if name!='regularized' else .25
            c.nll_weight=.75;c.huber_weight=.25
    return cfg


def _init_dirs(m,nu,ni,k,device,two=False):
    def param(n):return nn.Parameter(unit(torch.randn(n,k,device=device)))
    m.register_parameter('v2_u_dir',param(nu))
    m.register_parameter('v2_i_dir',param(ni))
    if two:
        m.register_parameter('v2_u_dir_alt',param(nu))
        m.register_parameter('v2_i_dir_alt',param(ni))


def _init_bias(m,data,device):
    rmin,rmax=float(data.rating_min),float(data.rating_max)
    mean=float(data.train_ratings_raw.mean().item())
    t=min(max((mean-rmin)/(rmax-rmin),1e-4),1-1e-4)
    m.register_parameter('v2_base_logit',nn.Parameter(torch.tensor(math.log(t/(1-t)),device=device)))
    m.v2_user_bias=nn.Embedding(data.num_users,1,device=device)
    m.v2_item_bias=nn.Embedding(data.num_items,1,device=device)
    nn.init.zeros_(m.v2_user_bias.weight);nn.init.zeros_(m.v2_item_bias.weight)
    m.register_parameter('v2_gain_logit',nn.Parameter(torch.tensor(0.,device=device)))
    m.register_buffer('v2_user_counts',torch.bincount(data.train_users,minlength=data.num_users).float())
    m.register_buffer('v2_item_counts',torch.bincount(data.train_items,minlength=data.num_items).float())


def _base(m,data,u,i):
    # Count shrinkage limits memorization of rare-item and rare-user biases.
    uw=m.v2_user_counts[u]/(m.v2_user_counts[u]+10.)
    iw=m.v2_item_counts[i]/(m.v2_item_counts[i]+10.)
    logits=(m.v2_base_logit
            +uw*m.v2_user_bias(u).squeeze(-1)
            +iw*m.v2_item_bias(i).squeeze(-1))
    return float(data.rating_min)+(float(data.rating_max)-float(data.rating_min))*torch.sigmoid(logits)


def _predict(m,data,u,i,score,gain):
    base=_base(m,data,u,i)
    return bounded_residual(base,score.clamp(-1,1),gain.clamp(0,1),
                            float(data.rating_min),float(data.rating_max))


def _base_penalty(m):
    return 1e-4*(m.v2_user_bias.weight.square().mean()+m.v2_item_bias.weight.square().mean())


def _radius_terms(m,data,u,i):
    method=m.v2_method
    if method=='Method1':
        zu,zi,ru,ri,_,_=m.encoded(data)
        return (ru[u,0],ri[i,0],zu,zi)
    if method=='Method3':
        zu,zi,du,di,ru,ri,_,_=m.encoded(data)
        return (ru[u,0],ri[i,0],zu,zi)
    if method=='Method4':
        zu,zi,du,di,ru,ri,_,_=m.encoded(data)
        return (ru[u,0],ri[i,0],zu,zi)
    raise KeyError(method)


def _predict_single(self,data,u,i):
    ru,ri,_,_=_radius_terms(self,data,u,i)
    if self.v2_method=='Method3' and self.v2_phase.item()>0.5:
        # Stage 1: train directions and bias before allowing log-radius correction.
        ru=self.rW0[u,0].detach()
        ri=self.rH0[i,0].detach()
    cos=cosine_pairs(self.v2_u_dir,self.v2_i_dir,u,i)
    gain=radial_gain(ru,ri,data.half_range,self.v2_gain_logit,self.v2_gain_floor)
    return _predict(self,data,u,i,cos,gain)


def _loss_single(self,data):
    u,i,y=data.train_users,data.train_items,data.train_ratings_raw
    state=self.encoded(data)
    zu,zi=state[:2]
    ru,ri=state[-4],state[-3]
    if self.v2_method=='Method3' and self.v2_phase.item()>0.5:
        ru=self.rW0.detach();ri=self.rH0.detach()
    score=cosine_pairs(self.v2_u_dir,self.v2_i_dir,u,i)
    gain=radial_gain(ru[u,0],ri[i,0],data.half_range,self.v2_gain_logit,self.v2_gain_floor)
    pred=_predict(self,data,u,i,score,gain)
    width=float(data.rating_max-data.rating_min)
    lp=((pred-y)/width).square().mean()
    rec=torch.zeros_like(lp);anchor=torch.zeros_like(lp)
    if self.v2_method=='Method1':
        target=y-data.midpoint
        rec=(self.decoder_user.selected(zu,u,i)-target).square().mean()/width**2
        rec=rec+(self.decoder_item.selected(zi,i,u)-target).square().mean()/width**2
    elif self.v2_method in ('Method3','Method4'):
        d1,d2=state[2],state[3]
        target=y-data.midpoint
        rec=(self.user_branch.selected_reconstruction(zu,u,i)-target).square().mean()/width**2
        rec=rec+(self.item_branch.selected_reconstruction(zi,i,u)-target).square().mean()/width**2
        anchor=d1.square().mean()+d2.square().mean()
    return (float(self.cfg.beta_mbmf)*lp+float(self.cfg.alpha_ae)*rec+float(getattr(self.cfg,'delta_l2',0.))*anchor+_base_penalty(self),
           {'L_pred':float(lp.detach()),'L_reconstruction':float(rec.detach()),'L_radius_anchor':float(anchor.detach())})


def _two_view_parts(self,data,u,i):
    a,b,c,d,_,_,_,_=self.factors(data)
    s1=cosine_pairs(self.v2_u_dir,self.v2_i_dir,u,i)
    s2=cosine_pairs(self.v2_u_dir_alt,self.v2_i_dir_alt,u,i)
    # Pair-dependent expert gate: sparse user histories shift weight toward item-view.
    log_ratio=torch.log1p(self.v2_user_counts[u])-torch.log1p(self.v2_item_counts[i])
    g=torch.sigmoid(self.v2_mix_logit+0.15*log_ratio)
    g1=radial_gain(a[u,0],b[i,0],data.half_range,self.v2_gain_logit,self.v2_gain_floor)
    g2=radial_gain(c[i,0],d[u,0],data.half_range,self.v2_gain_logit,self.v2_gain_floor)
    return s1,s2,g,g1,g2


def _predict_two(self,data,u,i):
    s1,s2,g,g1,g2=_two_view_parts(self,data,u,i)
    # Both views are themselves range-safe. Their convex fusion is range-safe.
    p1=_predict(self,data,u,i,s1,g1)
    p2=_predict(self,data,u,i,s2,g2)
    return g*p1+(1.-g)*p2


def _loss_two(self,data):
    u,i,y=data.train_users,data.train_items,data.train_ratings_raw
    s1,s2,g,g1,g2=_two_view_parts(self,data,u,i)
    p1=_predict(self,data,u,i,s1,g1);p2=_predict(self,data,u,i,s2,g2)
    pred=g*p1+(1.-g)*p2;width=float(data.rating_max-data.rating_min)
    main=((pred-y)/width).square().mean()
    aux=0.5*(((p1-y)/width).square().mean()+((p2-y)/width).square().mean())
    agreement=((p1-p2)/width).square().mean()
    # DO NOT use lambda=gamma(1-gamma), which makes the loss separable.
    total=main+.2*aux+float(self.cfg.consistency_weight)*agreement+_base_penalty(self)
    return total,{'L_pred':float(main.detach()),'L_experts':float(aux.detach()),'L_consistency':float(agreement.detach()),'mean_mix_gate':float(g.mean().detach())}


def _predict_views_v2(self,data,u,i):
    state=self._all_state(data)
    zu,zi,du,di,ru_u,ru_i,ri_i,ri_u,*_=state
    s1=cosine_pairs(self.v2_u_dir,self.v2_i_dir,u,i)
    s2=cosine_pairs(self.v2_u_dir_alt,self.v2_i_dir_alt,u,i)
    g1=radial_gain(ru_u[u,0],ru_i[i,0],data.half_range,self.v2_gain_logit,self.v2_gain_floor)
    g2=radial_gain(ri_u[u,0],ri_i[i,0],data.half_range,self.v2_gain_logit,self.v2_gain_floor)
    p1=_predict(self,data,u,i,s1,g1);p2=_predict(self,data,u,i,s2,g2)
    # Learn gate independent from prediction magnitude.
    gamma=torch.sigmoid(self.v2_mix_logit)
    if self.cfg.branch_mode=='user': p=p1
    elif self.cfg.branch_mode=='item': p=p2
    else: p=gamma*p1+(1-gamma)*p2
    return p1,p2,p,state


def _diagnostics_v2(self,data,u,i):
    with torch.no_grad():
        p=self.predict_raw_pairs(data,u,i)
        below=F.relu(float(data.rating_min)-p)
        above=F.relu(p-float(data.rating_max))
        return {'v2_max_global_range_violation':float((below+above).max()) if len(p) else 0.,
                'v2_prediction_std':float(p.std(unbiased=False)) if len(p)>1 else 0.,
                'v2_center_logit':float(self.v2_base_logit.detach()),
                'v2_gain':float(torch.sigmoid(self.v2_gain_logit).detach())}


def _loss_56(self,data):
    """Predictor-first objective; a single-branch ablation cannot train the absent view.

    Old V7 reconstruction losses share columns with angular predictor magnitudes.
    V2 uses separately learned counterpart radii; decoder norms are only weak priors.
    All weights are interpretable on a [0,1] normalized rating scale.
    """
    u,i,y=data.train_users,data.train_items,data.train_ratings_raw
    p1,p2,pred,state=self.predict_views(data,u,i)
    zu,zi,du,di=state[:4]
    width=float(data.rating_max-data.rating_min)
    target=(y-float(data.rating_min))/width
    lp=((pred-y)/width).square().mean()
    zero=lp.new_zeros(())
    active_u=self.cfg.branch_mode in ('both','user')
    active_i=self.cfg.branch_mode in ('both','item')
    lu=zero
    if active_u:
        rec_u=(zu[u]*du[i]).sum(-1)+self.user_view_item_bias(i).squeeze(-1)
        lu=(rec_u-target).square().mean()
    li=zero
    if active_i:
        rec_i=(zi[i]*di[u]).sum(-1)+self.item_view_user_bias(u).squeeze(-1)
        li=(rec_i-target).square().mean()
    lc=((p1-p2)/width).square().mean() if self.cfg.branch_mode=='both' else zero
    cap=math.sqrt(float(data.half_range))
    align=zero
    if active_u:
        rb=self._radius(du,data).detach()
        lb=2.*cap*torch.sigmoid(self.v2_item_radius_logits.weight)
        align=align+((lb-rb)/cap).square().mean()
    if active_i:
        rd=self._radius(di,data).detach()
        ld=2.*cap*torch.sigmoid(self.v2_user_radius_logits.weight)
        align=align+((ld-rd)/cap).square().mean()
    total=(float(self.cfg.prediction_weight)*lp
           +float(self.cfg.user_reconstruction_weight)*lu
           +float(self.cfg.item_reconstruction_weight)*li
           +float(self.cfg.branch_consistency_weight)*lc
           +(.002 if self.v2_method=='Method5' else .005)*align
           +_base_penalty(self))
    return total, {'L_pred':float(lp.detach()),'L_user_reconstruction':float(lu.detach()),
                   'L_item_reconstruction':float(li.detach()),
                   'L_branch_consistency':float(lc.detach()),
                   'L_decoder_radius_alignment':float(align.detach())}


def _forward7(self,data,users,items,return_aux=False):
    # Preserve original architecture's uncertainty head and feature pipeline,
    # but decouple expert selection from radius-dependent prediction amplitude.
    cfg=self.cfg
    ud,us,ur=self._entity_state(users,self.user_features,self.user_counts,self.user_direction,
                                self.user_encoder,self.user_radius,self.user_radius_prior,self.log_tau_user)
    idr,ins,ir=self._entity_state(items,self.item_features,self.item_counts,self.item_direction,
                                  self.item_encoder,self.item_radius,self.item_radius_prior,self.log_tau_item)
    cosine=(ud*idr).sum(-1).clamp(-1,1)
    bias=torch.tanh(self.global_bias+self.user_bias(users).squeeze(-1)+self.item_bias(items).squeeze(-1)) if cfg.use_bias_expert else torch.zeros_like(cosine)
    gate=(torch.sigmoid(self.v2_expert_logit+self.v2_expert_user(users).squeeze(-1)+self.v2_expert_item(items).squeeze(-1))
          if cfg.use_bias_expert else torch.ones_like(cosine))
    score=(1.-gate)*bias+gate*cosine
    # Expert gate no longer equals us*ins; both experts keep trainable gradients.
    pred=self.midpoint+self.half_range*score
    if not return_aux:return pred
    if cfg.heteroscedastic:
        vl=self.log_variance_prior+self.user_logvariance(self.user_features[users]).squeeze(-1)+self.item_logvariance(self.item_features[items]).squeeze(-1)
    else:vl=self.log_variance_prior.expand_as(pred)
    # Separate geometry radius reliability from expert choice: low reliability
    # broadens predictive intervals; it no longer silences the bias expert.
    radial_reliability=(us*ins).clamp(0.,1.)
    var=F.softplus(vl)*torch.exp(2*self.v2_var_logscale)*(2.-radial_reliability)+cfg.min_variance
    return pred,{'variance_normalized':var,'variance_raw':var*(self.half_range**2),
                 'gate':gate,'amplitude_gate':us*ins,'user_scale':us,'item_scale':ins,
                 'user_reliability':ur,'item_reliability':ir,'cosine':cosine,'bias':bias}


def _loss7(self,data,u,i,y):
    total,parts=self.v2_old_loss_pairs(data,u,i,y)
    # Regularize the shared gate prior, without a costly SECOND network forward.
    g=torch.sigmoid(self.v2_expert_logit)
    reg=.005*(g-0.5).square()+1e-5*(self.v2_expert_user.weight.square().mean()+self.v2_expert_item.weight.square().mean())
    parts['L_gate_balance']=float(reg.detach())
    return total+reg,parts




def _attach_fast_factors(model,method,data,core,pub):
    """Avoid wasted legacy spherical-coordinate products on every V2 forward.

    Dense and sparse encoder paths retain their V7/V8 semantics and train-only
    observedness masks; only Cartesian factors are assembled differently.
    """
    if core is None or pub is None:return
    sparse=isinstance(data,pub.SparseMatrixData)
    if method in ('Method1','Method3','Method4'):
        def encoded(self,d):
            if sparse:
                uv,um,iv,im=pub.sparse_encoder_matrices(d,self.cfg.model_space)
                if method=='Method1':
                    zu=pub._sparse_ae_encode(self.encoder_user,uv,um,d.num_items,self.cfg.input_mode)
                    zi=pub._sparse_ae_encode(self.encoder_item,iv,im,d.num_users,self.cfg.input_mode)
                else:
                    zu=pub._sparse_residual_encode(self.user_branch,uv,um,d.num_items,self.cfg.input_mode)
                    zi=pub._sparse_residual_encode(self.item_branch,iv,im,d.num_users,self.cfg.input_mode)
            else:
                xu,xi=core.build_encoder_inputs(d,self.cfg.input_mode,self.cfg.model_space)
                if method=='Method1':
                    zu=self.encoder_user(xu);zi=self.encoder_item(xi)
                else:
                    zu=self.user_branch.encode(xu);zi=self.item_branch.encode(xi)
            if method=='Method1':
                bs=core.model_pair_bound_scale(d,self.cfg.model_space)
                ru=core.latent_radius(zu,self.cfg.radius_mode,bs,self.cfg.radius_eps)
                ri=core.latent_radius(zi,self.cfg.radius_mode,bs,self.cfg.radius_eps)
            elif method=='Method3':
                du=core._bounded_log_correction(self.user_branch.raw_log_correction(zu),self.cfg)
                di=core._bounded_log_correction(self.item_branch.raw_log_correction(zi),self.cfg)
                ru=self.rW0*torch.exp(du);ri=self.rH0*torch.exp(di)
            else:
                du=core._bounded_shift(self.user_branch.raw_log_correction(zu),
                                        self.cfg.max_abs_logit_shift,self.cfg.residual_transform)
                di=core._bounded_shift(self.item_branch.raw_log_correction(zi),
                                        self.cfg.max_abs_logit_shift,self.cfg.residual_transform)
                ru=self.radius_cap*torch.sigmoid(self.logitW0+du)
                ri=self.radius_cap*torch.sigmoid(self.logitH0+di)
            W=ru*unit(self.v2_u_dir);H=ri*unit(self.v2_i_dir)
            if method=='Method1':return zu,zi,ru,ri,W,H
            return zu,zi,du,di,ru,ri,W,H
        model.encoded=types.MethodType(encoded,model)
    elif method=='Method2':
        def factors(self,d):
            if sparse:
                uv,um,iv,im=pub.sparse_encoder_matrices(d,self.cfg.model_space)
                hu=pub._sparse_two_stage(self.user_features,uv,um,d.num_items,self.cfg.input_mode)
                hi=pub._sparse_two_stage(self.item_features,iv,im,d.num_users,self.cfg.input_mode)
            else:
                xu,xi=core.build_encoder_inputs(d,self.cfg.input_mode,self.cfg.model_space)
                hu=self.user_features(xu);hi=self.item_features(xi)
            bs=core.model_pair_bound_scale(d,self.cfg.model_space)
            a=core.latent_radius(hu,self.cfg.radius_mode,bs,self.cfg.radius_eps)
            b=self.item_counterpart();c=core.latent_radius(hi,self.cfg.radius_mode,bs,self.cfg.radius_eps)
            dd=self.user_counterpart()
            return (a,b,c,dd,a*unit(self.v2_u_dir),b*unit(self.v2_i_dir),
                    c*unit(self.v2_i_dir_alt),dd*unit(self.v2_u_dir_alt))
        model.factors=types.MethodType(factors,model)
    elif method in ('Method5','Method6'):
        def state(self,d):
            zu,zi,du,di=self.codes(d)
            a=self._radius(zu,d);raw_b=self._radius(du,d)
            c=self._radius(zi,d);raw_d=self._radius(di,d)
            # Decoder columns remain a weak prior, NOT a hard constraint
            # on the prediction magnitudes. Learned heads are independent.
            eta=torch.sigmoid(self.v2_radius_coupling_logit)
            cap=math.sqrt(float(d.half_range))
            learned_b=2.*cap*torch.sigmoid(self.v2_item_radius_logits.weight)
            learned_d=2.*cap*torch.sigmoid(self.v2_user_radius_logits.weight)
            b=(1.-eta)*learned_b+eta*raw_b
            dd=(1.-eta)*learned_d+eta*raw_d
            return zu,zi,du,di,a,b,c,dd,unit(self.v2_u_dir),unit(self.v2_i_dir),unit(self.v2_u_dir_alt),unit(self.v2_i_dir_alt)
        model._all_state=types.MethodType(state,model)


def adapt(model,method,data,profile='balanced',core=None,pub=None):
    """Inject independent V2 modules into a V8 model *instance*; avoid V1 edits."""
    if method not in {f'Method{i}' for i in range(1,8)}:return model
    if hasattr(model,'v2_method'):raise RuntimeError('V2 double adaptation')
    if profile not in PROFILES:raise ValueError(profile)
    device=data.train_ratings_raw.device
    k=model.cfg.latent_dim
    model.v2_method=method;model.v2_epoch=0
    # Phase is a CHECKPOINT BUFFER; without this a post-load Method3 can
    # silently evaluate a different predictor than validation selected.
    model.register_buffer('v2_phase',torch.tensor(1.,device=device))
    model.v2_warmup=PROFILES[profile]['warmup']
    model.v2_gain_floor=PROFILES[profile]['gain_floor']
    model.v2_rec_weight=float(getattr(model.cfg,'alpha_ae',PROFILES[profile]['reconstruction']))
    model.v2_anchor_weight=.01
    if method in ('Method1','Method2','Method3','Method4','Method5','Method6'):
        _init_dirs(model,data.num_users,data.num_items,k,device,method in ('Method2','Method5','Method6'))
        for name in ('Phi','Theta','Phi_user','Theta_user','Phi_item','Theta_item','uv_user_angles','uv_item_angles','iv_user_angles','iv_item_angles'):
            old=getattr(model,name,None)
            if isinstance(old,nn.Parameter):old.requires_grad_(False)
        _init_bias(model,data,device)
        _attach_fast_factors(model,method,data,core,pub)
        model.diagnostics=types.MethodType(_diagnostics_v2,model)
        if method in ('Method1','Method3','Method4'):
            model.predict_raw_pairs=types.MethodType(_predict_single,model)
            model.loss=types.MethodType(_loss_single,model)
        elif method=='Method2':
            g=min(max(float(model.cfg.fusion_weight_user_view),1e-4),1-1e-4)
            model.register_parameter('v2_mix_logit',nn.Parameter(torch.tensor(math.log(g/(1-g)),device=device)))
            model.predict_raw_pairs=types.MethodType(_predict_two,model)
            model.loss=types.MethodType(_loss_two,model)
        else:
            g=min(max(float(model.cfg.fusion_weight_user_view),1e-4),1-1e-4)
            model.register_parameter('v2_mix_logit',nn.Parameter(torch.tensor(math.log(g/(1-g)),device=device)))
            model.v2_user_radius_logits=nn.Embedding(data.num_users,1,device=device)
            model.v2_item_radius_logits=nn.Embedding(data.num_items,1,device=device)
            nn.init.zeros_(model.v2_user_radius_logits.weight)
            nn.init.zeros_(model.v2_item_radius_logits.weight)
            model.register_parameter('v2_radius_coupling_logit',nn.Parameter(torch.tensor(math.log(1/3),device=device)))
            model.predict_views=types.MethodType(_predict_views_v2,model)
            model.loss=types.MethodType(_loss_56,model)
    else:
        model.register_parameter('v2_expert_logit',nn.Parameter(torch.tensor(0.,device=device)))
        model.register_parameter('v2_var_logscale',nn.Parameter(torch.tensor(.1,device=device)))
        model.v2_expert_user=nn.Embedding(data.num_users,1,device=device)
        model.v2_expert_item=nn.Embedding(data.num_items,1,device=device)
        nn.init.zeros_(model.v2_expert_user.weight)
        nn.init.zeros_(model.v2_expert_item.weight)
        model.v2_old_loss_pairs=model.loss_pairs
        model.forward_pairs=types.MethodType(_forward7,model)
        model.loss_pairs=types.MethodType(_loss7,model)
    return model
