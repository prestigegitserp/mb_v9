"""Validation-only finite response correction; raw risk remains reportable.

p_a = clip(b + a * (p - b), L, U). The finite grid contains a=1, which
reproduces the uncorrected predictor. This guarantees only non-increasing
selection error on this validation population, never lower test error.
"""
import torch

SCALES = (0., .25, .5, .75, 1., 1.25, 1.5, 2.)


def baseline(model, history, users, items):
    if getattr(model, "inner", None) is not None:
        return model.inner.base(history, users, items)
    if model.bias is not None:
        return model.base(history, users, items)
    return torch.stack([m.bias(users, items, history.user_counts, history.item_counts)
                        for m in model.members]).mean(0)


def transform(prediction, base, scale, lo, hi):
    if float(scale) == 1.:
        return prediction
    return (base + scale * (prediction - base)).clamp(lo, hi)


@torch.no_grad()
def fit_response(model, data):
    if not model.cfg.response_shrinkage:
        return dict(status="DISABLED", selected_scale=1.)
    model.eval()
    h, state = model.eval_state()
    # Training history and held-out validation are the only observations read.
    u, i, target = data.val_users, data.val_items, data.val_ratings_raw
    sums = {a: 0. for a in SCALES}
    for start in range(0, len(u), model.cfg.batch_size):
        sl = slice(start, start + model.cfg.batch_size)
        raw = model.predict(h, state, u[sl], i[sl])[0]
        b = baseline(model, h, u[sl], i[sl])
        for a in SCALES:
            sums[a] += float((transform(raw, b, a, model.lo, model.hi) - target[sl]).double().square().sum())
    if not len(u) or not all(torch.isfinite(torch.tensor(v)) for v in sums.values()):
        raise FloatingPointError("No finite response-selection population")
    score, selected = min((value / len(u), a) for a, value in sums.items())
    model.response_scale.fill_(selected)
    model.response_fitted.fill_(True)
    return dict(status="VALIDATION_SELECTED", selected_scale=selected,
                degenerates_to_bias_only=(selected == 0.),
                raw_val_RMSE=(sums[1.] / len(u)) ** .5,
                corrected_val_RMSE=score ** .5, selection_pairs=len(u),
                criterion="final point prediction validation SSE",
                grid=list(SCALES), test_used=False,
                test_improvement_guaranteed=False)
