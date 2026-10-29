"""Atomic, content-addressed candidate and optimizer recovery (V7 only)."""
from dataclasses import asdict, is_dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import hashlib
import json
import os
import random
import tempfile
import time
import numpy as np
import torch

_recovered_time=ContextVar('v7_recovered_training_time',default=None)


@contextmanager
def recovered_training_time():
    """Collect restored, disjoint phases once, including completed OOF experts."""
    phases={};token=_recovered_time.set(phases)
    try:yield phases
    finally:_recovered_time.reset(token)


def cpu_tree(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(cpu_tree(v) for v in obj)
    return obj


def atomic_save(payload, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(cpu_tree(payload), handle); handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def tensor_digest(values):
    digest = hashlib.sha256()
    for t in values:
        t = t.detach().cpu().contiguous()
        digest.update(str((str(t.dtype), tuple(t.shape))).encode())
        digest.update(t.numpy().tobytes())
    return digest.hexdigest()


def candidate_identity(model, data, config):
    from .config import implementation_signature
    spec = dict(config=asdict(config) if is_dataclass(config) else vars(config),
                seed=model._v7_model_seed, source=implementation_signature(),
                triples=tensor_digest([getattr(data, role+'_'+part) for role in ('train','val')
                        for part in ('users','items','ratings_raw')]))
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()


def candidate_path(model):
    folder = getattr(model, '_v7_artifact_dir', None)
    return Path(folder)/'recovery'/'candidate_complete.pt' if folder else None


def load_candidate(model, identity):
    path = candidate_path(model)
    if not getattr(model, '_v7_resume', False) or not path or not path.is_file():
        return None
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['identity'] != identity:
        raise ValueError('Candidate recovery identity mismatch: '+str(path))
    model.load_state_dict(payload['model']); model.eval()
    if hasattr(model, 'clear_cache'): model.clear_cache()
    info = payload['info']; info['v7_resume'] = dict(level='completed_candidate', reused=True)
    print('[RESUME] reused completed candidate: '+str(path.parent.parent), flush=True)
    return info


def save_candidate(model, identity, info):
    path = candidate_path(model)
    if path:
        atomic_save(dict(identity=identity, model=model.state_dict(), info=info), path)


def rng_state(generator=None):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                generator=generator.get_state() if generator is not None else None)


def restore_rng(state, generator=None):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_initialized(): torch.cuda.set_rng_state_all(state['cuda'])
    if generator is not None and state['generator'] is not None: generator.set_state(state['generator'])


class EpochRecovery:
    """Save at safe epoch boundaries; optimizer and mask RNG are part of state."""
    def __init__(self, model, phase='mean', config=None):
        folder = getattr(model, '_v7_artifact_dir', None)
        self.path = Path(folder)/'recovery'/(phase+'.pt') if folder else None
        self.started=time.perf_counter();self.elapsed_before=0.;self.model=model
        self.enabled = bool(getattr(model, '_v7_resume', False))
        self.interval = int(getattr(model, '_v7_recovery_interval', 5))
        self.config=asdict(config or model.cfg) if is_dataclass(config or model.cfg) else vars(config or model.cfg)
        self.payload = None
        if self.path and self.enabled and self.path.is_file():
            self.payload = torch.load(self.path, map_location='cpu', weights_only=False)
            self.elapsed_before=self.payload.get('elapsed_seconds',0.)
            model._v7_prior_training_seconds=self.elapsed_before
            phases=_recovered_time.get()
            if phases is not None:phases[str(self.path)]=self.elapsed_before
            if self.payload['config'] != self.config:
                raise ValueError('Epoch recovery configuration mismatch')
            model.load_state_dict(self.payload['model'])
            for name, p in model.named_parameters():
                p.requires_grad_(self.payload['requires_grad'][name])
            model._step = self.payload['local'].get('step_count', 0)

    def restore(self, optimizer, generator):
        if self.payload is None: return None
        if optimizer is not None:optimizer.load_state_dict(self.payload['optimizer'])
        restore_rng(self.payload['rng'], generator)
        print('[RESUME] optimizer recovery at epoch '+str(self.payload['local']['epoch']), flush=True)
        return self.payload['local']

    def save(self, model, optimizer, generator, state, force=False):
        if not self.path or (not force and state['epoch'] % self.interval): return
        # AEMC constructs its main recovery before warm-start. If that warm
        # phase was itself resumed, carry its OLD time into future main saves;
        # current-attempt warm work is already included by this timer.
        warm_root=self.path.parent.parent/'warm_start'
        warm_prior=(sum(seconds for name,seconds in (_recovered_time.get() or {}).items()
                        if warm_root in Path(name).parents) if self.path.name=='baseline.pt' else 0.)
        atomic_save(dict(config=self.config, model=model.state_dict(), optimizer=optimizer.state_dict() if optimizer is not None else None,
                         rng=rng_state(generator), requires_grad={n:p.requires_grad for n,p in model.named_parameters()},
                         local=state,elapsed_seconds=self.elapsed_before+warm_prior+time.perf_counter()-self.started), self.path)


def scientific_fingerprint(config):
    """Placement and output paths are execution choices, not a new experiment."""
    values = asdict(config)
    for key in ('output_dir','execution','auto_generate_plots','save_predictions',
                'save_candidate_checkpoints','save_selected_checkpoints'):
        values.pop(key, None)
    values['data'].pop('data_root', None)
    values['input_content'] = input_digest(config.data)
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


def input_digest(config):
    """Hash dataset input bytes, excluding generated outputs and cache files."""
    root = Path(config.data_root)/config.dataset
    if config.dataset=='ml-10m' and (Path(config.data_root)/'ml-10M100K/ratings.dat').exists():root=Path(config.data_root)/'ml-10M100K'
    if config.dataset == 'synthetic': return 'generated_from_serialized_configuration'
    candidates = {'ml-100k': ('u.data',), 'ml-1m': ('ratings.dat',),
                  'ml-10m': ('ratings.dat',), 'bookcrossing': ('BX-Book-Ratings.csv','BX-Book-Ratings.csv.txt','Ratings.csv')}
    names = candidates.get(config.dataset)
    files = sorted(p for p in root.rglob('*') if p.is_file() and
                   (p.name in names if names else p.suffix.lower() in ('.xls','.xlsx','.csv','.dat')))
    if not files: return 'missing:'+config.dataset
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(4*1024*1024), b''): digest.update(block)
    return digest.hexdigest()
