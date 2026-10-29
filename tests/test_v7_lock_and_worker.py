"""Real process locking, crash release and paths outside the repository."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
pytest.importorskip('filelock')
from v7.locking import study_lock


def environment():
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join(str(Path(p).resolve()) for p in env.get('PYTHONPATH','').split(os.pathsep) if p)
    return env


def test_simultaneous_writer_rejected_without_sending_signals(tmp_path,monkeypatch):
    def no_signal(*args,**kwargs):raise AssertionError('A lock must never signal a process')
    monkeypatch.setattr(os,'kill',no_signal)
    with study_lock(tmp_path):
        with pytest.raises(RuntimeError,match='Another process'):
            with study_lock(tmp_path):pass
    with study_lock(tmp_path):pass


def test_os_releases_lock_after_abrupt_process_exit(tmp_path):
    root=Path(__file__).resolve().parents[1]
    script='from v7.locking import study_lock;import os,sys\nwith study_lock(sys.argv[1]):os._exit(23)'
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path)],cwd=root,env=environment(),capture_output=True,text=True)
    assert result.returncode==23,result.stderr
    with study_lock(tmp_path):pass


def test_worker_resolves_relative_paths_from_invoking_directory(tmp_path):
    pytest.importorskip('xlsxwriter')
    from test_v7_data_cache import raw_data
    root=Path(__file__).resolve().parents[1]
    folder=tmp_path/'datasets/ml-100k';folder.mkdir(parents=True)
    raw_data()[['userId','movieId','rating','timestamp']].to_csv(folder/'u.data',sep='\t',index=False,header=False)
    command=[sys.executable,str(root/'run_v7.py'),'main','--datasets','ml-100k','--data-root','./datasets',
        '--methods','MF,Method8','--split-seeds','2026','--model-seeds','1','--k','4','--quick',
        '--v7-no-plots','--output','./results']
    result=subprocess.run(command,cwd=tmp_path,env=environment(),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    target=tmp_path/'results/ml-100k/main'
    assert json.loads((target/'dataset_manifest.json').read_text())['n_interactions']==192
    assert not list(target.rglob('recovery'))
    request=next(target.glob('V7_WORKER_REQUEST_*.json'))
    assert Path(json.loads(request.read_text())['config']['data']['data_root']).is_absolute()


def test_duplicate_seeds_are_not_counted_as_independent_repetitions(tmp_path):
    from v7.entry import run
    with pytest.raises(ValueError,match='Repeated values'):
        run(['plan','--datasets','synthetic','--model-seeds','1,1','--split-seeds','2026','--k','4','--output',str(tmp_path)])


def test_releasing_already_disposed_model_is_safe():
    import torch
    from v7.lifecycle import release_model
    model=torch.nn.Linear(3,2);release_model(model);release_model(model)
