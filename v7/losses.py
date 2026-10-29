"""Registry of actual objective coefficients, with explicit conditional activity."""
from dataclasses import fields

COMMON = ('bias_prior','code_prior','direction_prior','collaborative_l2',
          'mechanism_auxiliary_weight','full_history_weight')
SPECIFIC = {
    'Method1': ('reconstruction_weight',),
    'Method2': ('auxiliary_weight','consistency_weight','router_ridge','router_context_penalty'),
    'Method3': ('reconstruction_weight','radius_precision','radius_noise'),
    'Method4': ('reconstruction_weight','auxiliary_weight','moment_dispersion_weight'),
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
    if method=='Method7':names=tuple(n for n in names if n!='mechanism_auxiliary_weight')
    if method=='Method7' and not config.probability_fit:
        names=tuple(n for n in names if n!='probability_rps_weight')
    return dict.fromkeys(names)


def trial_values(config, name):
    value=getattr(config,name)
    if name=='mechanism_auxiliary_weight':return (0.,.1,.25,1.)
    if name=='router_ridge':return tuple(sorted(set((value*.25,value,value*4))))
    if name in ('radius_noise','posterior_noise'):
        return tuple(sorted(set((value*.5,value,value*2))))
    if name in ('radius_precision','posterior_precision'):
        return tuple(sorted(set((max(value*.25,1e-5),value,value*4))))
    if name in ('full_history_weight','ensemble_mean_weight'):
        return (0.,.25,.5,.75)
    return tuple(sorted(set((0.,value*.25,value,value*4 if value else .1))))


def registry_rows(study):
    rows=[]
    for method,config in study.models.items():
        for name in coefficients(config):
            kind='prior precision / likelihood scale' if name in ('radius_precision','posterior_precision','radius_noise','posterior_noise') else 'objective coefficient'
            rows.append(dict(method=method,lineage=config.lineage,parameter=name,value=getattr(config,name),
                             kind=kind,condition='active only if its parent mechanism/representation is enabled'))
    rows.append(dict(method='Method8',lineage='uploaded Method10',parameter='observed_frobenius',value=1.,
                     kind='immutable original objective',condition='not changed; no synthetic auxiliary loss'))
    return rows
