"""Integration only: uploaded Method10 is publicly named Method8, unchanged."""
from dataclasses import replace
from pathlib import Path
import hashlib
import torch
from vendor_method10.method10 import Method10, Method10Config, train_method10, save_method10, load_method10

SOURCE_SHA256 = 'c76754f899c31a99a99eb99bfb8291b97e40d7eeda463f58ed47d6ae86e19c77'


def verify_source():
    path = Path(__file__).resolve().parents[1]/'vendor_method10/method10.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError('The uploaded Method10 source was modified')


def create(data, values, k, seed, device):
    verify_source()
    config = replace(Method10Config(**values), latent_dim=int(k))
    device = torch.device(device)
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        model = Method10(data, config, device)
    return model, config


def train(model, data):
    result = train_method10(model, data, verbose=True)
    result.update(source_method='Method10', public_alias='Method8', source_sha256=SOURCE_SHA256,
                  original_objective_preserved=True, recovery_capability='completed_candidate_only')
    return result


def save(path, model, *, data, train_info=None, **kwargs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    import json
    # Normalize metadata scalar subclasses; keep the original weights-only loader.
    clean=json.loads(json.dumps(train_info or {},default=str))
    save_method10(path, model, data, clean)
    return Path(path)


def load(path, data, device):
    return load_method10(path, data, device)
