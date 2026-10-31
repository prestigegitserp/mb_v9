from pathlib import Path
import pandas as pd
from v8.reporting import _render


def test_actual_test_figure_exports_png_pdf_and_svg(tmp_path):
    runs=pd.DataFrame(dict(dataset=['ml-100k']*3,method=['AutoRec','Method1','Method7'],
        split_seed=[2026]*3,model_seed=[1]*3,K=[20]*3,init_mode=['NA']*3,
        test_RMSE=[.9,.91,.905],test_MAE=[.7,.71,.705]))
    empty=pd.DataFrame()
    _render(tmp_path,runs,'test',empty,empty,empty,empty,empty,smoke=False)
    for ext in ('png','pdf','svg'):
        assert (tmp_path/('rank_test_RMSE.'+ext)).stat().st_size>100


def test_six_job_search_preserves_incumbent_and_transport_range(tmp_path):
    from types import SimpleNamespace
    from v8.config import StudyConfig
    from v8.execution import ExecutionConfig
    from v8.runtime import installed
    from v8 import engine,search
    options=SimpleNamespace(study=StudyConfig(search_jobs_per_method=6),execution=ExecutionConfig(),
        max_epochs=120,baseline_max_epochs=200,steps_per_epoch=2,no_plots=True,
        retain_candidates='metrics',v8_recovery_interval=5,v8_isolate_jobs=False)
    args=SimpleNamespace(data_root=str(tmp_path),backend='auto',split_seeds='2026',model_seeds='1',k='20',
        methods='Method3,Method6',eval_every=None,quick=False,resume=True)
    with installed(engine,options) as (_,pub):
        base=engine.make_base(pub,args,'ml-100k')
        jobs=search.build_jobs(pub,base,(20,))
        for method in ('Method3','Method6'):
            assert 'v7_incumbent__'+method in jobs
            assert sum(method in cfg.methods for cfg in jobs.values())==6
        values={cfg.v8.models['Method6'].transport_moment_weight for cfg in jobs.values() if 'Method6' in cfg.methods}
        assert {0.,.02,.1}<=values
