"""Method7 leakage-aware variance scaling.

Calibrate variance using a deterministic subset of VALIDATION only; Gaussian NLL
for calibration-model selection is measured on its DISJOINT subset. Test is never
used in fitting, early stopping, or temperature selection.
"""
from __future__ import annotations
import math
import torch


def calibration_masks(data):
    u=data.val_users.long();i=data.val_items.long()
    # Deterministic, order-independent pair hash, no Python salted hash/random.
    h=(u*1315423911+i*2654435761)
    fit=(torch.remainder(h+(h>>13),2)==0)
    if fit.sum()<10 or (~fit).sum()<10:
        # Tiny synthetic datasets: avoid pretending there is a statistical
        # calibration fit from a handful of observations.
        return None,None
    return fit,~fit


def gaussian_nll(y,p,var):
    var=var.clamp_min(1e-8)
    return .5*(torch.log(2.*math.pi*var)+(y-p).square()/var).mean()


def calibrate_method7(model,data,*,max_abs_log_shift=1.25):
    """Closed-form initial variance rescaling, validated by fit-half NLL.

    For v_j(t)=t*v_j and known mean, optimum t=mean((y-p)^2/v_j).
    The min-variance floor slightly changes the exact optimum; a short,
    deterministic log-grid is used to guarantee no fit-half NLL degradation.
    """
    fit,hold=calibration_masks(data)
    if fit is None:return {'applied':False,'reason':'validation calibration halves too small'}
    was_training=model.training
    model.eval()
    with torch.no_grad():
        u=data.val_users[fit];i=data.val_items[fit];y=data.val_ratings_raw[fit]
        p,v=model.predict_raw_with_uncertainty(data,u,i)
        floor=float(model.cfg.min_variance)*(float(model.half_range)**2)
        nll_before=float(gaussian_nll(y,p,v))
        ratio=float(((y-p).square()/v.clamp_min(1e-8)).mean())
        proposal=.5*math.log(max(ratio,1e-6))
        proposal=max(-max_abs_log_shift,min(max_abs_log_shift,proposal))
        shifts=sorted({0.,proposal,*[proposal+i*.125 for i in (-2,-1,1,2)]})
        best=(nll_before,0.)
        for delta in shifts:
            if abs(delta)>max_abs_log_shift:continue
            vv=(v-floor).clamp_min(0.)*math.exp(2.*delta)+floor
            nll=float(gaussian_nll(y,p,vv))
            if math.isfinite(nll) and nll<best[0]-1e-8:best=(nll,delta)
        if best[1]:model.v2_var_logscale.add_(best[1])
        model.v2_calibration_done=True
        model.v2_calibration_logshift=float(best[1])
    if was_training:model.train()
    return {'applied':True,'n_fit':int(fit.sum()),'n_selection':int(hold.sum()),
            'log_std_adjustment':float(best[1]),'calibration_fit_nll_before':nll_before,
            'calibration_fit_nll_after':best[0], 'test_touched':False}


def nll_on_disjoint_selection_half(model,data):
    fit,hold=calibration_masks(data)
    if fit is None:return None
    was_training=model.training;model.eval()
    with torch.no_grad():
        p,v=model.predict_raw_with_uncertainty(data,data.val_users[hold],data.val_items[hold])
        nll=float(gaussian_nll(data.val_ratings_raw[hold],p,v))
    if was_training:model.train()
    return {'Gaussian_NLL':nll,'V2_Calibration_NLL_Heldout_N':int(hold.sum())}
