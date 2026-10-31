"""Joint geometry/response validation calibration; no TEST improvement claim.

The old response grid and the unmodified predictor are included. On the convex
interval 0<=a<=1 the exact least-squares optimum is also evaluated. Extrapolation
uses the original discrete grid and is disabled for probabilistic Method7.
"""
import math
import torch

SCALES = (0., .25, .5, .75, 1., 1.25, 1.5, 2.)
GEOMETRY_SCALES = (0., .25, .5, .75, 1.)


def baseline(model, history, users, items):
    if getattr(model,'inner',None) is not None:
        return model.inner.base(history,users,items)
    from .readout import point_baseline
    return point_baseline(model,history,users,items)


def transform(prediction, base, scale, lo, hi):
    return (base+scale*(prediction-base)).clamp(lo,hi)


@torch.no_grad()
def fit_response(model,data):
    if not model.cfg.response_shrinkage:
        return dict(status='DISABLED',selected_scale=1.,selected_geometry_scale=1.,test_used=False)
    from .models import Method7
    convex_only=isinstance(model,Method7)
    geometry_active=(not hasattr(model,'inner') and model.cfg.readout_mode!='decoder'
                     and model.cfg.calibrate_geometry)
    scales=tuple(a for a in SCALES if not convex_only or a<=1.)
    geometry=GEOMETRY_SCALES if geometry_active else (1.,)
    model.eval()
    model.response_fitted.fill_(False)
    model.response_scale.fill_(1.)
    model.geometry_scale.fill_(1.)
    model.clear_cache()
    h,state=model.eval_state()
    u,i,target=data.val_users,data.val_items,data.val_ratings_raw
    if not len(u):raise ValueError('Empty response-selection population')
    candidates=[]
    raw_mse=None
    for q in geometry:
        model.geometry_scale.fill_(q)
        raw_parts,base_parts=[],[]
        for start in range(0,len(u),model.cfg.batch_size):
            sl=slice(start,start+model.cfg.batch_size)
            raw_parts.append(model.predict(h,state,u[sl],i[sl])[0].double().cpu())
            base_parts.append(baseline(model,h,u[sl],i[sl]).double().cpu())
        raw,b=torch.cat(raw_parts),torch.cat(base_parts)
        y=target.double().cpu()
        if not torch.isfinite(raw).all() or not torch.isfinite(b).all() or not torch.isfinite(y).all():
            raise FloatingPointError('Nonfinite response-selection population')
        if q==1.:raw_mse=float((raw-y).square().mean())
        values=list(scales)
        if model.cfg.continuous_response:
            d=raw-b
            denominator=float(d.square().sum())
            if denominator>1e-20:
                values.append(min(1.,max(0.,float((d*(y-b)).sum())/denominator)))
        for a in sorted(set(values)):
            mse=float((transform(raw,b,a,model.lo,model.hi)-y).square().mean())
            candidates.append(dict(response_scale=a,geometry_scale=q,val_RMSE=math.sqrt(mse)))
    chosen=min(candidates,key=lambda r:(r['val_RMSE'],abs(r['response_scale']-1.),abs(r['geometry_scale']-1.)))
    model.response_scale.fill_(chosen['response_scale'])
    model.geometry_scale.fill_(chosen['geometry_scale'])
    model.response_fitted.fill_(True)
    model.clear_cache()
    return dict(status='VALIDATION_SELECTED',selected_scale=chosen['response_scale'],
        selected_geometry_scale=chosen['geometry_scale'],raw_val_RMSE=math.sqrt(raw_mse),
        corrected_val_RMSE=chosen['val_RMSE'],selection_pairs=len(u),
        degenerates_to_bias_only=chosen['response_scale']==0.,geometry_attenuated=chosen['geometry_scale']<1.,
        convex_response_only=convex_only,continuous_convex_candidate=model.cfg.continuous_response,
        criterion='joint geometry/response validation SSE',grid=list(scales),candidates=candidates,
        test_used=False,test_improvement_guaranteed=False)
