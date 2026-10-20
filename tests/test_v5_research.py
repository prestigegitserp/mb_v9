"""V5 scientific invariants, exact storage, exports and protocol boundaries."""
import copy
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from test_v3_research import tiny_data, training_pub, config as v3_config
from v5.config import ModelConfig,StudyConfig
from v5.models import instantiate
from v5.protocol import validation_roles
from v5.training import train_v5
from v5.checkpoints import save,load
from v5.execution import ExecutionConfig,SavedTensorStore
from v5.bounds import structural_bounds

METHODS=[f'Method{i}' for i in range(1,8)]


def config(method):
    return ModelConfig(**{**vars(v3_config(method)),'posterior_cg_steps':64,'posterior_solver_tolerance':1e-4})


@pytest.mark.parametrize('method',METHODS)
def test_v5_training_poisoned_test_checkpoint_bounds_and_mean_law(method,tmp_path):
    torch.set_num_threads(1)
    data=validation_roles(tiny_data(),StudyConfig());data.test_ratings_raw[:]=torch.nan
    model=instantiate(data,config(method),17);model._v5_artifact_dir=tmp_path/'trace'
    info=train_v5(training_pub(),model,data,model.cfg,None)
    assert np.isfinite(info['best_val_RMSE'])
    model.eval()
    with torch.no_grad():prediction=model.predict_raw_pairs(data,data.val_users,data.val_items)
    lo,hi=structural_bounds(model,data,data.val_users,data.val_items)
    assert np.all(prediction.numpy()>=lo-2e-5) and np.all(prediction.numpy()<=hi+2e-5)
    path=tmp_path/'model.pt';save(path,model,config=model.cfg,data=data,train_info=info)
    restored,_=load(path,data,torch.device('cpu'))
    with torch.no_grad():other=restored.predict_raw_pairs(data,data.val_users,data.val_items)
    torch.testing.assert_close(prediction,other,atol=0,rtol=0)
    if method=='Method7':
        distribution=restored.distribution(data,data.val_users,data.val_items)
        torch.testing.assert_close(distribution['probabilities']@restored.levels,prediction,atol=2e-5,rtol=1e-6)
        assert info['v5_probability']['mean_parameters_unchanged']


@pytest.mark.parametrize('method',METHODS)
def test_v5_query_labels_and_timestamps_cannot_enter_context(method):
    torch.set_num_threads(1);data=tiny_data();model=instantiate(data,config(method),19).eval()
    keep=torch.arange(len(data.train_users))%3!=0
    def prediction():
        with torch.no_grad():
            h=model.history(keep);s=model.prepare(h)
            return model.predict(h,s,data.train_users[~keep],data.train_items[~keep])[0]
    before=prediction();data.train_ratings_raw[~keep]=1000;data.train_timestamps[~keep]=-1e12
    torch.testing.assert_close(before,prediction(),atol=0,rtol=0)


@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('policy',['host','ssd'])
def test_v5_storage_keeps_all_model_gradients_and_adam_state_exact(method,policy,tmp_path):
    torch.set_num_threads(1);states=[]
    for storage in ['resident',policy]:
        torch.manual_seed(11);data=tiny_data();model=instantiate(data,config(method),31)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        keep=torch.arange(128)%4!=0;query=~keep
        with SavedTensorStore(ExecutionConfig(storage=storage,cache_dir=str(tmp_path))):
            for _ in range(2):
                model.train();h=model.history(keep);s=model.prepare(h)
                lp,la,_=model.loss_for_targets(h,s,data.train_users[query],data.train_items[query],data.train_ratings_raw[query])
                opt.zero_grad(set_to_none=True);(lp+la).backward();opt.step();model.clear_cache()
        states.append((copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict())))
    def equal(a,b):
        if torch.is_tensor(a):torch.testing.assert_close(a,b,atol=0,rtol=0)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equal(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            for x,y in zip(a,b):equal(x,y)
        else:assert a==b
    equal(*states);assert not list(tmp_path.glob('v5_graph_*'))


@pytest.mark.parametrize('dtype',[torch.float64,torch.float32,torch.float16,torch.bfloat16,torch.int64,torch.bool])
def test_raw_ssd_roundtrip_dtype_views_and_multiple_unpack(dtype,tmp_path):
    original=torch.arange(24).reshape(4,6).to(dtype).T
    with SavedTensorStore(ExecutionConfig(storage='ssd',cache_dir=str(tmp_path))) as store:
        handle=store.pack(original)
        torch.testing.assert_close(store.unpack(handle),original,atol=0,rtol=0)
        torch.testing.assert_close(store.unpack(handle),original,atol=0,rtol=0)
        original.zero_()
        assert store.disk_bytes==24*torch.tensor([],dtype=dtype).element_size()


def test_ssd_budget_fails_explicitly_and_cleans_up(tmp_path):
    with pytest.raises(MemoryError,match='budget'):
        with SavedTensorStore(ExecutionConfig(storage='ssd',cache_dir=str(tmp_path),ssd_budget_gb=1e-12)) as store:
            store.pack(torch.ones(12))
    assert not list(tmp_path.glob('v5_graph_*'))


def test_cpu_auto_spills_saved_graph_when_ram_reserve_is_reached(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from v5 import execution
    monkeypatch.setattr(execution.psutil,'virtual_memory',lambda:SimpleNamespace(available=20,total=1000))
    tensor=torch.arange(12,dtype=torch.float32).reshape(3,4)
    with SavedTensorStore(ExecutionConfig(storage='auto',cache_dir=str(tmp_path))) as store:
        handle=store.pack(tensor)
        assert handle.path is not None and store.disk_bytes==tensor.numel()*tensor.element_size()
        torch.testing.assert_close(store.unpack(handle),tensor,atol=0,rtol=0)
    assert not list(tmp_path.glob('v5_graph_*'))


def test_degree_gated_encoder_has_no_cold_start_signal():
    from v5.history import HistoryEncoder
    encoder=HistoryEncoder(4,8,4)
    value=encoder(torch.tensor([0]),torch.tensor([1]),torch.tensor([.7]),torch.tensor([1.,0.]),2)
    torch.testing.assert_close(value[1],torch.zeros(4),atol=0,rtol=0)


def test_excel_is_complete_numeric_and_formula_safe(tmp_path):
    pytest.importorskip('xlsxwriter');pytest.importorskip('openpyxl')
    from v5.export import export_workbook
    tables=tmp_path/'tables';tables.mkdir()
    data=pd.DataFrame({'method':['Method1','=1+2'],'value':[.123456789012345,2.], 'pair_id':['1234567890123456789','12']})
    data.to_csv(tables/'metrics.csv',index=False)
    for name in ('first','second'):pd.DataFrame(columns=['status','reason']).to_csv(tables/(name+'.csv'),index=False)
    manifest=export_workbook(tmp_path)
    sheet=next(x['sheets'][0] for x in manifest['sources'] if x['source']=='tables/metrics.csv')
    import openpyxl
    book=openpyxl.load_workbook(tmp_path/'research_tables.xlsx',read_only=True,data_only=False)
    rows=list(book[sheet].values)
    assert len(rows)==3 and rows[2][0]=='=1+2' and rows[1][2]=='1234567890123456789'
    assert abs(rows[1][1]-data.value.iloc[0])<1e-15
    assert book[sheet]['A3'].data_type=='s'


def test_interrupted_table_export_keeps_previous_complete_file(tmp_path,monkeypatch):
    from v5.reporting import _write
    file=tmp_path/'pairs.csv.gz';_write(pd.DataFrame({'x':[1.,2.]}),file)
    before=file.read_bytes()
    def interrupted(self,path,**kwargs):
        Path(path).write_bytes(b'incomplete gzip');raise OSError('simulated interruption')
    monkeypatch.setattr(pd.DataFrame,'to_csv',interrupted)
    with pytest.raises(OSError,match='simulated interruption'):_write(pd.DataFrame({'x':[3.]}),file)
    assert file.read_bytes()==before and pd.read_csv(file).x.tolist()==[1.,2.]
    assert not list(tmp_path.glob('*.partial'))


def test_device_selection_is_explicit(monkeypatch):
    from v5.resources import resolve_device,move_data_
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    device,reason=resolve_device('auto');assert str(device)=='cpu' and 'interpreter' in reason
    with pytest.raises(RuntimeError,match='unavailable'):resolve_device('cuda:0')
    data=tiny_data();before=data.train_ratings_raw.clone();move_data_(data,'cpu')
    torch.testing.assert_close(before,data.train_ratings_raw,atol=0,rtol=0)


def test_nonfinite_gradient_is_classified_without_applying_an_update():
    from v5.training import _clip_gradients
    parameter=torch.nn.Parameter(torch.ones(2));before=parameter.detach().clone()
    parameter.grad=torch.tensor([float('nan'),1.])
    with pytest.raises(FloatingPointError,match='optimizer update was not applied'):
        _clip_gradients([parameter],1.)
    torch.testing.assert_close(parameter,before,atol=0,rtol=0)


def test_cpu_replay_rolls_back_failed_candidate_and_rng(monkeypatch):
    """Simulate CUDA failure control flow; this is not CUDA hardware parity."""
    from types import SimpleNamespace
    from v5 import resources
    from v5.execution import capture_rng
    class Marked(torch.nn.Linear):
        def parameters(self,recurse=True):
            if getattr(self,'mock_cuda',False):return iter([SimpleNamespace(device=torch.device('cuda:0'),grad=None)])
            return super().parameters(recurse)
    torch.manual_seed(39);model=Marked(3,2);model.mock_cuda=True
    initial={k:v.clone() for k,v in model.state_dict().items()}
    state=torch.get_rng_state();expected=torch.rand(4);torch.set_rng_state(state)
    data=tiny_data();calls=[];replayed=[]
    def runner(current,callback,rebuild,config,on_replay=None):
        calls.append(str(next(current.parameters()).device))
        if len(calls)==1:
            with torch.no_grad():current.weight.add_(100)
            torch.rand(17)
            raise torch.cuda.OutOfMemoryError('simulated partial candidate')
        return callback()
    def callback():
        for key,value in initial.items():torch.testing.assert_close(model.state_dict()[key],value,atol=0,rtol=0)
        torch.testing.assert_close(torch.rand(4),expected,atol=0,rtol=0)
        return {'v5_execution':{}}
    monkeypatch.setattr(resources,'run_training',runner)
    monkeypatch.setattr(resources,'placement',lambda *a:{'actual_device':'cuda:0'})
    monkeypatch.setattr(resources,'capture_rng',lambda device:capture_rng('cpu'))
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    result=resources.execute_candidate(model,data,callback,lambda device=None:Marked(3,2),ExecutionConfig(),lambda:replayed.append(True))
    assert calls==['cuda:0','cpu'] and replayed==[True]
    assert result['v5_execution']['placement']['fallback']['replay']=='entire_candidate_from_initial_state'


def test_sparse_probability_metric_does_not_make_an_unbounded_figure(tmp_path):
    from v5.publication import render_clean
    from PIL import Image
    (tmp_path/'tables').mkdir()
    runs=pd.DataFrame(dict(method=['Method1','Method7'],split_seed=[1,1],test_RMSE=[1.,.9],test_Probability_NLL=[np.nan,1.2]))
    render_clean(tmp_path,runs,'test')
    with Image.open(tmp_path/'figures/rank_test_Probability_NLL.png') as im:
        assert im.width<5000 and im.height<5000


def test_publication_gallery_excludes_incomplete_temporary_images(tmp_path):
    pytest.importorskip('xlsxwriter')
    from v5.publication import finish_report
    report=tmp_path/'publication';(report/'figures').mkdir(parents=True);(report/'tables').mkdir()
    (report/'figures/.interrupted.partial.png').write_bytes(b'incomplete')
    runs=pd.DataFrame(columns=['method','split_seed','test_RMSE'])
    finish_report(tmp_path,report,runs,'test',plots=True,smoke=False)
    assert not (tmp_path/'figures/.interrupted.partial.png').exists()
    assert json.loads((tmp_path/'V5_PAPER_OUTPUTS.json').read_text())['figures']==[]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires real CUDA hardware; CPU parity does not prove CUDA parity')
def test_v5_cuda_raw_ssd_adam_parity(tmp_path):
    results=[]
    for policy in ('resident','host','ssd'):
        torch.manual_seed(9);model=torch.nn.Linear(4,2).cuda();opt=torch.optim.Adam(model.parameters(),lr=.003)
        x=torch.arange(24,device='cuda',dtype=torch.float32).reshape(6,4)/24
        with SavedTensorStore(ExecutionConfig(storage=policy,cache_dir=str(tmp_path))):
            for _ in range(3):
                opt.zero_grad(set_to_none=True);model(x).square().mean().backward();opt.step()
        results.append({k:v.detach().cpu() for k,v in model.state_dict().items()})
    for state in results[1:]:
        for key in state:torch.testing.assert_close(state[key],results[0][key],atol=0,rtol=0)
