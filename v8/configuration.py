"""Canonical identities retain effective MBMF initialization and all settings."""
import json


def canonical_candidate_key(original, method, config, init_mode):
    value=json.loads(original(method,config,init_mode))
    if method.startswith('MBMF-'):
        allowed=tuple(getattr(config,'init_modes',()))
        if init_mode not in allowed:
            raise ValueError('Effective MBMF initialization is absent from its declared modes')
        # The containing scan grid is not a parameter of this fitted candidate.
        # Both its init_mode identity and one-element effective list remain.
        value['config']['init_modes']=[init_mode]
    return json.dumps(value,sort_keys=True)
