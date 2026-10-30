"""Registry of actual objective coefficients, with explicit conditional activity."""
from dataclasses import fields

COMMON = ('bias_prior','code_prior','direction_prior','collaborative_l2',
          'mechanism_auxiliary_weight','full_history_weight','encoder_decay_budget')
SPECIFIC = {
    'Method1': ('reconstruction_weight','auxiliary_gradient_ratio'),
    'Method2': ('auxiliary_weight','consistency_weight','router_ridge','router_context_penalty'),
    'Method3': ('reconstruction_weight','radius_precision','radius_noise'),
    'Method4': ('reconstruction_weight','auxiliary_weight','moment_dispersion_weight','radius_precision'),
    'Method5': ('reconstruction_weight','posterior_precision','posterior_noise'),
    'Method6': ('reconstruction_weight','transport_moment_weight'),
    'Method7': ('ensemble_mean_weight','probability_rps_weight'),
}


def coefficients(config):
    """Return executable knobs, excluding settings ignored by historical code."""
    method=config.method
    names=COMMON+SPECIFIC[method]
    if config.lineage!='v7':
        from .lineage import legacy_config
        allowed={f.name for f in fields(legacy_config(config))}
        names=tuple(n for n in names if n in allowed)
    if method=='Method4': names=tuple(n for n in names if n!='bias_prior') # fixed after warmup
    if method=='Method4':
        inactive='auxiliary_weight' if config.orthogonal_moment_loss else 'moment_dispersion_weight'
        names=tuple(n for n in names if n!=inactive)
    if method=='Method1' and not config.gradient_surgery:
        names=tuple(n for n in names if n!='auxiliary_gradient_ratio')
    if method=='Method7':
        names=tuple(n for n in names if n not in ('mechanism_auxiliary_weight','direction_prior'))
        if config.readout_mode=='geometric':names=tuple(n for n in names if n!='code_prior')
    if method=='Method7' and not config.probability_fit:
        names=tuple(n for n in names if n!='probability_rps_weight')
    return dict.fromkeys(names)


def trial_values(config, name):
    value=getattr(config,name)
    if name=='encoder_decay_budget':return tuple(sorted(set((0.,value*.25,value,min(value*4,.95)))))
    if name=='mechanism_auxiliary_weight':return (0.,.1,.25,1.)
    if name=='router_ridge':return tuple(sorted(set((value*.25,value,value*4))))
    if name in ('radius_noise','posterior_noise'):
        return tuple(sorted(set((value*.5,value,value*2))))
    if name in ('radius_precision','posterior_precision'):
        return tuple(sorted(set((max(value*.25,1e-5),value,value*4))))
    if name in ('full_history_weight','ensemble_mean_weight'):
        return (0.,.25,.5,.75)
    return tuple(sorted(set((0.,value*.25,value,value*4 if value else .1))))


def coefficient_kind(name):
    if name=='encoder_decay_budget':return 'optimizer shrinkage budget'
    if name=='auxiliary_gradient_ratio':return 'gradient-surgery budget'
    if name=='probability_rps_weight':return 'probability-fit objective coefficient'
    if name in ('radius_precision','posterior_precision','radius_noise','posterior_noise'):
        return 'prior precision / likelihood scale'
    return 'objective coefficient'


def coefficient_overrides(config,values):
    """Preserve an explicitly tied metric/dispersion pair in Method4."""
    changes=dict(values)
    tied=(config.method=='Method4' and config.moment_projection_mode=='metric'
          and config.orthogonal_moment_loss
          and abs(config.moment_projection_weight-config.moment_dispersion_weight)<1e-12)
    if tied and 'moment_dispersion_weight' in changes:
        weight=changes['moment_dispersion_weight']
        if weight>0:changes['moment_projection_weight']=weight
        else:changes['moment_projection_mode']='mean-priority'
    return changes


def registry_rows(study):
    rows=[]
    for method,config in study.models.items():
        for name in coefficients(config):
            kind=coefficient_kind(name)
            rows.append(dict(method=method,lineage=config.lineage,parameter=name,value=getattr(config,name),
                             kind=kind,condition='active only if its parent mechanism/representation is enabled'))
    rows.append(dict(method='Method8',lineage='uploaded Method10',parameter='observed_frobenius',value=1.,
                     kind='immutable original objective',condition='not changed; no synthetic auxiliary loss'))
    return rows
