"""Verify delivered checkpoints on selection validation only; never train or test."""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import numpy as np,pandas as pd,torch
    from v9.engine import load_v8
    from v4.config import StudyConfig,implementation_signature
    from v4.protocol import validation_roles
    from v4 import checkpoints
    from v4.baselines import restore_mbmf
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment',type=Path,required=True)
    parser.add_argument('--checkpoints',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();core,pub=load_v8();torch.set_num_threads(1)
    frame=pd.read_csv(args.assignment)
    data,_=pub.build_publication_matrix_data_v8(frame,pub.DATASET_SPECS['ml-100k'],pub.PublicationDataConfig(dataset='ml-100k',force_backend='dense'),torch.device('cpu'))
    data.train_timestamps=torch.tensor(frame.loc[frame.split=='train','timestamp'].to_numpy(),dtype=torch.float64)
    data.test_ratings_raw[:]=torch.nan;data=validation_roles(data,StudyConfig())
    rows=[]
    for path in sorted(args.checkpoints.glob('*.pt')):
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload.get('format')=='MBMF-V4':model,_=checkpoints.restore(payload,data,torch.device('cpu'))
        elif payload.get('family')=='PaperMBMF' and 'checkpoint_policy' in payload['config']:
            model,_=restore_mbmf(pub,core,payload,data,torch.device('cpu'))
        else:model,_=pub.load_model_checkpoint_v8(path,data,torch.device('cpu'))
        model.eval()
        with torch.no_grad():pred=(model.predict_raw_pairs(data.val_users,data.val_items) if isinstance(model,core.PaperMBMF)
                                  else model.predict_raw_pairs(data,data.val_users,data.val_items)).numpy()
        archive_path=path.with_suffix('.validation.npz')
        with np.load(archive_path,allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['pair_id'],(data.val_users*data.num_items+data.val_items).numpy())
            np.testing.assert_array_equal(archive['truth'],data.val_ratings_raw.numpy())
            np.testing.assert_allclose(pred,archive['prediction'],atol=2e-6,rtol=1e-6)
            difference=float(np.max(np.abs(pred-archive['prediction'])))
        rows.append(dict(method=path.stem,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         validation_RMSE=float(np.mean((pred.astype(float)-data.val_ratings_raw.numpy())**2)**.5),
                         validation_pairs=len(pred),max_abs_prediction_difference=difference))
        del model,payload
    result=dict(status='PASS',release_implementation_sha256=implementation_signature(),checkpoints=rows,
                optimizer_steps_run=0,new_real_test_evaluated=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
