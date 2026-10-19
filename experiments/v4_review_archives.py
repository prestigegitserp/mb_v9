"""Recompute historical supplied archives; never select a new model using TEST."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_archive(path):
    with np.load(path, allow_pickle=False) as data:
        pair = data['pair_id'].astype(np.int64)
        truth, prediction = [data[k].astype(np.float64) for k in ('truth','prediction')]
    if pair.ndim != 1 or pair.shape != truth.shape or pair.shape != prediction.shape:
        raise ValueError('Invalid prediction shape: '+str(path))
    if len(np.unique(pair)) != len(pair) or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('Duplicate/nonfinite observations: '+str(path))
    return pd.DataFrame(dict(pair_id=pair, truth=truth, prediction=prediction)).set_index('pair_id').sort_index()


def metrics(frame):
    error = frame.prediction-frame.truth
    clipped = frame.prediction.clip(1.,5.)-frame.truth
    return dict(n=len(frame), RMSE=float(np.sqrt(np.mean(error**2))), MAE=float(np.mean(abs(error))),
                clipped_RMSE=float(np.sqrt(np.mean(clipped**2))),
                global_OOB_rate=float(((frame.prediction < 1)|(frame.prediction > 5)).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames, rows, checks = {}, [], []
    sources = {v: args.inputs/f'predictions_{name}'/'predictions'
               for v,name in [('v1','v1'),('v2','v9_v2'),('v3','v9_v3')]}
    for version, folder in sources.items():
        for path in sorted(folder.glob('*.npz')):
            parts = path.stem.split('__')
            method, partition = parts[1], parts[-1]
            frame = read_archive(path)
            init = parts[-2]
            rows.append(dict(version=version, method=method, partition=partition,
                             init=init, source=path.relative_to(args.inputs).as_posix(),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **metrics(frame)))
            frames[(version,method,partition,init)] = frame
    table = pd.DataFrame(rows)
    table.to_csv(args.output/'historical_metrics.csv', index=False)
    table[table.partition == 'test'].pivot(index='method',columns='version',values='RMSE').to_csv(
        args.output/'historical_test_RMSE.csv')
    canonical = pd.read_csv(args.inputs/'figures_v9_v3'/'split_assignments_2026.csv')
    ni = int(canonical.movieId.max())+1
    canonical['pair_id'] = canonical.userId*ni+canonical.movieId
    expected = canonical.set_index('pair_id')
    for identity, frame in frames.items():
        population = expected.loc[frame.index]
        if not np.array_equal(population.rating.to_numpy(), frame.truth.to_numpy()):
            raise ValueError('Truth disagrees with supplied fixed split: '+str(identity))
        if not population.split.eq(identity[2]).all():
            raise ValueError('Partition disagrees with supplied fixed split: '+str(identity))
    for method in sorted(table.method.unique()):
        a = [k for k in frames if k[:3] == ('v2',method,'test')]
        b = [k for k in frames if k[:3] == ('v3',method,'test')]
        checks.append(dict(method=method, v2_v3_test_identical=bool(frames[a[0]].equals(frames[b[0]]))))
    common = None
    valframes = {}
    # Follow each archive's locked winning initialization, rather than choosing
    # an initialization with hindsight from its test error.
    for version in sources:
        for method in sorted(table.method.unique()):
            test_key = next(k for k in frames if k[:3] == (version,method,'test'))
            key = (version,method,'val',test_key[-1])
            valframes[(version,method)] = frames[key]
            common = frames[key].index if common is None else common.intersection(frames[key].index)
    aligned = [dict(version=v,method=m,**metrics(f.loc[common])) for (v,m),f in valframes.items()]
    pd.DataFrame(aligned).to_csv(args.output/'historical_common_validation.csv',index=False)
    pd.DataFrame(checks).to_csv(args.output/'historical_prediction_identity.csv',index=False)
    status = dict(archives_verified=len(rows), unique_pairs=len(canonical),
                  truth_and_partitions_match_assignment=True, common_validation_pairs=len(common),
                  note='Historical v2/v3 baselines with identical predictions are reused results, not independent replications.',
                  missing_attachment='figures_v1.rar', new_test_evaluated=False)
    (args.output/'ARCHIVE_VERIFICATION.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2))


if __name__ == '__main__':
    main()
