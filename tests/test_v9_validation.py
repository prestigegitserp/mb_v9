import pathlib,sys,pytest
import pandas as pd
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from v9.selection import select_validation_winners
from v9.engine import subset_training_keep


def rows():
    records=[]
    for meth in ('Method1','Method2'):
        for job in ('default','improved'):
            for split_seed in (2026,2027):
                for model_seed in (11,22):
                    records.append(dict(dataset='ml-100k',method=meth,scan_job=job,config_key=f'{meth}:{job}',
                        val_RMSE=.8 if job=='improved' else .9,K=20,init_mode='NA',split_seed=split_seed,model_seed=model_seed))
    return pd.DataFrame(records)

def test_selection_validation_only():
    winners,ranked=select_validation_winners(rows(),2,2)
    assert len(winners)==4
    assert set(winners.scan_job)=={'improved'}

def test_selection_refuses_test_leak():
    r=rows();r['test_RMSE']=.1
    with pytest.raises(ValueError,match='TEST LEAK'):select_validation_winners(r,2,2)

def test_selection_refuses_missing_seed():
    with pytest.raises(ValueError,match='replicate|coverage|Incomplete'):
        select_validation_winners(rows().iloc[:-1],2,2)

def test_selection_refuses_nonfinite():
    r=rows();r.loc[0,'val_RMSE']=float('nan')
    with pytest.raises(ValueError,match='Non-finite'):select_validation_winners(r,2,2)

def test_train_keep_does_not_change_holdout():
    r=pd.DataFrame([
      (1,11,3.,'train'),(1,12,4.,'train'),(2,11,5.,'train'),(2,12,4.,'train'),
      (1,13,3.,'val'),(2,13,4.,'test')
    ],columns=['userId','movieId','rating','split'])
    a=subset_training_keep(r,.5,seed=2026)
    assert set(map(tuple,a[a.split!='train'].itertuples(index=False,name=None)))==set(map(tuple,r[r.split!='train'].itertuples(index=False,name=None)))
    assert set(a[a.split=='train'].userId)==set(r[r.split=='train'].userId)
    assert set(a[a.split=='train'].movieId)==set(r[r.split=='train'].movieId)


def test_selection_independent_per_outer_split():
    r=rows()
    # For split 2027, 'default' wins; in 2026, 'improved' wins.
    r.loc[(r.split_seed==2027)&(r.scan_job=='default'),'val_RMSE']=0.60
    r.loc[(r.split_seed==2027)&(r.scan_job=='improved'),'val_RMSE']=0.80
    winners,_=select_validation_winners(r,2,2)
    assert set(winners[winners.split_seed==2026].scan_job)=={'improved'}
    assert set(winners[winners.split_seed==2027].scan_job)=={'default'}

def test_method7_calibration_uses_validation_nll_not_test():
    r=rows();r=r[r.method=='Method1'].copy()
    r['val_Gaussian_NLL']=r.scan_job.map({'default':.2,'improved':.6})
    r['test_Gaussian_NLL']=r.scan_job.map({'default':.9,'improved':.1})
    with pytest.raises(ValueError,match='TEST LEAK'):
        select_validation_winners(r,2,2,metric='val_Gaussian_NLL')
    r=r.drop(columns='test_Gaussian_NLL')
    winners,_=select_validation_winners(r,2,2,metric='val_Gaussian_NLL')
    assert set(winners.scan_job)=={'default'}
    with pytest.raises(ValueError,match='VALIDATION'):
        select_validation_winners(r,2,2,metric='train_RMSE')
