"""Descriptive uncertainty diagnostics; no settings are selected from TEST."""
from pathlib import Path
import numpy as np
import pandas as pd


def export_diagnostics(report,primary,plots=False,smoke=False):
    report=Path(report);tables=report/'tables'
    risks=[];diversity=[]
    for path in sorted((tables/'per_pair').glob('*__'+primary+'.csv.gz')):
        frame=pd.read_csv(path)
        if 'variance' not in frame:continue
        method=path.name.split('__',1)[0]
        error=frame.squared_error.to_numpy()
        ordering={'predictive_variance':frame.variance.to_numpy()}
        if 'user_train_count' in frame and 'item_train_count' in frame:
            ordering['inverse_history_support']=1/(1+np.minimum(frame.user_train_count,frame.item_train_count).to_numpy())
        for name,uncertainty in ordering.items():
            order=np.argsort(uncertainty,kind='stable')
            for coverage in np.linspace(.1,1.,19):
                n=max(1,int(np.floor(len(frame)*coverage)))
                risks.append(dict(method=method,partition=primary,source=path.name,
                    uncertainty=name,coverage=n/len(frame),retained_pairs=n,
                    total_pairs=len(frame),RMSE=np.sqrt(error[order[:n]].mean()),
                    descriptive_only=True,threshold_selected_for_deployment=False))
        members=[c for c in frame if c.startswith('member_means_')]
        if len(members)>1:
            residual=frame[members].to_numpy()-frame.truth.to_numpy()[:,None]
            corr=np.corrcoef(residual,rowvar=False)
            for i in range(len(members)):
                for j in range(i+1,len(members)):
                    diversity.append(dict(method=method,partition=primary,source=path.name,
                        member_a=i,member_b=j,residual_correlation=corr[i,j],
                        member_a_RMSE=np.sqrt((residual[:,i]**2).mean()),
                        member_b_RMSE=np.sqrt((residual[:,j]**2).mean()),
                        mean_between_variance=frame.between_variance.mean() if 'between_variance' in frame else np.nan,
                        mean_total_variance=frame.variance.mean(),
                        between_fraction=frame.between_variance.mean()/frame.variance.mean() if 'between_variance' in frame else np.nan))
    risk=pd.DataFrame(risks,columns=['method','partition','source','uncertainty','coverage','retained_pairs','total_pairs','RMSE','descriptive_only','threshold_selected_for_deployment'])
    risk.to_csv(tables/'risk_coverage.csv',index=False)
    pd.DataFrame(diversity,columns=['method','partition','source','member_a','member_b','residual_correlation','member_a_RMSE','member_b_RMSE','mean_between_variance','mean_total_variance','between_fraction']).to_csv(tables/'ensemble_diversity.csv',index=False)
    (report/'UNCERTAINTY_SCOPE.md').write_text(
        '# Uncertainty diagnostics\n\n'
        'Risk–coverage is descriptive at changing retained coverage; its lower RMSE is not the full TEST result. '
        'Deployment thresholds must be chosen on validation. Member disagreement is model-dependent, not proven epistemic uncertainty. '
        'Structural bounds constrain predictions and are not rating confidence intervals. '
        'Ordinal equal-tail interval coverage can exceed nominal because ratings are discrete.\n')
    if plots and not risk.empty:
        from .figures import _style,_save
        with _style() as plt:
            fig,ax=plt.subplots(figsize=(6.4,3.8),layout='constrained')
            for (method,kind),group in risk.groupby(['method','uncertainty']):
                summary=group.groupby('coverage',as_index=False).RMSE.mean().sort_values('coverage')
                ax.plot(summary.coverage,summary.RMSE,marker='o',ms=3,label=method+' / '+kind)
            ax.set(xlabel='Retained evaluation coverage',ylabel=primary.upper()+' RMSE on retained pairs',
                   title='Selective prediction · descriptive, changing coverage')
            ax.grid();ax.legend(fontsize=7)
            _save(fig,report/'figures','risk_coverage',smoke)
