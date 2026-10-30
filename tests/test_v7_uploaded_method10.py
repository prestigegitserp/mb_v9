"""Scientific contracts of Method10, independent of the frozen model code."""
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vendor_method10.method10 import (Method10, Method10Config, backward_observed_objective,
                         bounded_spherical, evaluate_method10, load_method10,
                         observed_fusion_loss, save_method10, train_method10)


@pytest.fixture
def data():
    torch.set_num_threads(1)
    users, items = torch.meshgrid(torch.arange(6), torch.arange(7), indexing="ij")
    users, items = users.flatten(), items.flatten()
    ratings = ((2 * users + 3 * items) % 5 + 1).float()
    d = SimpleNamespace(num_users=6, num_items=7, rating_min=1., rating_max=5.)
    for name, mask in (("train", items < 5), ("val", items == 5), ("test", items == 6)):
        for key, values in (("users", users), ("items", items), ("ratings_raw", ratings)):
            setattr(d, f"{name}_{key}", values[mask])
    return d


def make_model(data, **kwargs):
    torch.manual_seed(31)
    cfg = Method10Config(hidden_dim=12, latent_dim=4, decoder_dim=8, chunk_size=7,
                         max_epochs=3, **kwargs)
    return Method10(data, cfg)


def test_pair_predictions_equal_transposed_full_reconstruction(data):
    model = make_model(data)
    ru, ri = model.reconstruct_views()
    assert ru.shape == (6, 7) and ri.shape == (7, 6)
    u, i = torch.tensor([0, 2, 5]), torch.tensor([6, 1, 3])
    pu, pi = model.predict_views_pairs(u, i)
    torch.testing.assert_close(pu, ru[u, i])
    torch.testing.assert_close(pi, ri[i, u])
    torch.testing.assert_close(model()[u, i], (pu + pi) / 2)


def test_spherical_dot_preserves_actual_directions():
    h = torch.tensor([[2., -3., 1.]], dtype=torch.float64)
    w = torch.tensor([[-1., 4., 2.]], dtype=torch.float64)
    cap, temperature = 2., 1.3
    hb = bounded_spherical(h, cap, temperature)
    wb = bounded_spherical(w, cap, temperature)
    cosine = (h * w).sum() / (h.norm() * w.norm())
    expected = cap**2 * torch.tanh(h.norm()/temperature) * torch.tanh(w.norm()/temperature) * cosine
    torch.testing.assert_close((hb * wb).sum(), expected)
    torch.testing.assert_close(hb / hb.norm(), h / h.norm())
    torch.testing.assert_close(wb / wb.norm(), w / w.norm())


def test_zero_vectors_have_finite_nonzero_jacobian():
    x = torch.zeros((2, 4), dtype=torch.float64, requires_grad=True)
    bounded_spherical(x, 2., 0.5).sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 4.))
    assert torch.autograd.gradcheck(lambda v: bounded_spherical(v, 2., 0.5), (x,))


@pytest.mark.parametrize("bounds", [(1., 5.), (-10., 10.), (0., 1.)])
def test_both_views_and_fusion_are_bounded_without_clipping(data, bounds):
    old_lo, old_hi = data.rating_min, data.rating_max
    data.rating_min, data.rating_max = bounds
    data.train_ratings_raw = bounds[0] + (data.train_ratings_raw-old_lo) / (old_hi-old_lo) * (bounds[1]-bounds[0])
    model = make_model(data)
    with torch.no_grad():
        for ae in (model.user_ae, model.item_ae):
            ae.output_weight.mul_(1e6)
            ae.decoder_hidden.weight.mul_(1e6)
            ae.decoder_hidden.bias.fill_(1e6)
    for prediction in (*model.reconstruct_views(), model()):
        assert torch.isfinite(prediction).all()
        assert prediction.min() >= bounds[0] - 1e-5
        assert prediction.max() <= bounds[1] + 1e-5
    for factors in model.factors():
        assert factors.norm(dim=1).max() <= model.radius_cap + 1e-5


def test_fuse_before_loss_not_average_of_losses():
    # Opposite view errors cancel in the requested fused objective.
    target, pu, pi = torch.tensor([3., 2.]), torch.tensor([4., 3.]), torch.tensor([2., 1.])
    assert observed_fusion_loss(pu, pi, target).item() == 0.
    assert ((pu-target).square().mean() + (pi-target).square().mean()).item() / 2 == 1.


def test_histories_only_contain_training_ratings(data):
    model = make_model(data)
    expected = torch.zeros(6, 7)
    expected[data.train_users, data.train_items] = (data.train_ratings_raw-3)/2
    torch.testing.assert_close(model.user_history.to_dense(), expected)
    torch.testing.assert_close(model.item_history.to_dense(), expected.T)
    altered = copy.deepcopy(data)
    altered.val_ratings_raw.fill_(999.)
    altered.test_ratings_raw.fill_(-999.)
    other = make_model(altered)
    torch.testing.assert_close(other(), model())


@pytest.mark.parametrize("reduction", ["frobenius", "squared_frobenius", "mse"])
def test_chunked_gradient_matches_dense_masked_objective(data, reduction):
    # This checks both masking and exact full-norm gradient across chunk sizes.
    a = make_model(data, loss_reduction=reduction).double()
    b = make_model(data, loss_reduction=reduction).double()
    target = data.train_ratings_raw.double()
    dense_target = torch.full((6, 7), 1e9, dtype=torch.float64)
    mask = torch.zeros((6, 7), dtype=torch.bool)
    mask[data.train_users, data.train_items] = True
    dense_target[mask] = target
    ru, ri = a.reconstruct_views()
    loss = observed_fusion_loss(ru[mask], ri.T[mask], dense_target[mask], reduction)
    loss.backward()
    chunk_loss = backward_observed_objective(b, data.train_users, data.train_items, target)
    assert chunk_loss == pytest.approx(loss.item(), rel=1e-10)
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-9, atol=1e-10)
    for ae in (b.user_ae, b.item_ae):
        assert ae.input_layer.weight.grad.abs().sum() > 0
        assert ae.output_weight.grad.abs().sum() > 0


def test_learning_decreases_training_objective(data):
    model = make_model(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    first = evaluate_method10(model, data.train_users, data.train_items, data.train_ratings_raw)["RMSE"]
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        backward_observed_objective(model, data.train_users, data.train_items, data.train_ratings_raw)
        optimizer.step()
    last = evaluate_method10(model, data.train_users, data.train_items, data.train_ratings_raw)["RMSE"]
    assert last < first * 0.7


def test_checkpoint_roundtrip_and_wrong_history_rejected(data, tmp_path):
    # The trainer must work with no test tensors at all.
    training_data = SimpleNamespace(**{k: v for k, v in vars(data).items() if not k.startswith("test_")})
    model = make_model(data)
    info = train_method10(model, training_data, verbose=False)
    assert info["best_val_RMSE"] == pytest.approx(evaluate_method10(
        model, data.val_users, data.val_items, data.val_ratings_raw)["RMSE"])
    path = tmp_path / "model.pt"
    save_method10(path, model, data, info)
    restored, payload = load_method10(path, data)
    torch.testing.assert_close(restored(), model())
    changed = copy.deepcopy(data)
    changed.train_ratings_raw[0] += 0.1
    with pytest.raises(ValueError, match="history"):
        load_method10(path, changed)


def test_duplicate_train_pairs_and_invalid_config_rejected(data):
    data.train_users[1] = data.train_users[0]
    data.train_items[1] = data.train_items[0]
    with pytest.raises(ValueError, match="duplicate"):
        make_model(data)
    with pytest.raises(ValueError, match="radius_temperature"):
        Method10Config(radius_temperature=0.)

