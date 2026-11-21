"""Recompute primary comparisons from completed native TPO prediction shards."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory
import tarfile

import numpy as np

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths


def macro_f1(histogram):
    support, predicted, tp = histogram
    denom = support + predicted
    return np.divide(2 * tp, denom, where=denom > 0, out=np.zeros_like(tp, dtype=float)).mean()


def read(path, digest=None):
    path = Path(path)
    if digest is not None and file_hash(path) != digest:
        raise ValueError(f'Input changed: {path}')
    return json.loads(path.read_bytes())


def paired(dataset, targets, candidates, files, interval):
    gold = np.array([candidates.index(row['gold']) for row in targets])
    groups, cluster = np.unique([t['group'] for t in targets], return_inverse=True)
    ids = [row['id'] for row in targets]
    width, count = len(candidates), len(gold)
    if len(ids) != len(set(ids)) or len(groups) < 2:
        raise ValueError('Unique recordings and at least two clusters are required')
    results = {}
    for mode, arms in files.items():
        if len(arms) != 2:
            raise ValueError('Expected one baseline and one candidate')
        choice = mode == 'choice'
        distributions, hits, hashes = [], [], []
        for arm in arms:
            predictions = {}
            policy = arm['metric']['policy_id']
            for path in arm['paths']:
                receipt = read(path)
                if receipt['policy_id'] != policy:
                    raise ValueError('Prediction policy differs from independently scored endpoint')
                hashes.append(file_hash(path))
                for row in receipt['records']:
                    if row['id'] in predictions:
                        raise ValueError('Duplicate prediction')
                    predictions[row['id']] = row['logp' if choice else 'prob']
            if set(ids) != set(predictions):
                raise ValueError('Prediction roster differs from frozen targets')
            values = np.asarray([predictions[uid] for uid in ids])
            if values.shape != (count, width + choice) or not np.isfinite(values).all():
                raise ValueError('Prediction vector shape or value differs')
            if (choice and (np.any(values > 0) or
                            not np.allclose(np.exp(values).sum(1), 1, atol=1e-5, rtol=0))
                    or not choice and np.any((values < 0) | (values > 1))):
                raise ValueError('Invalid predictions')
            winner = values.argmax(1)
            correct = winner == gold
            valid = winner < width
            hist = np.zeros((len(groups), 3, width), dtype=np.int64)
            np.add.at(hist[:, 0], (cluster, gold), 1)
            np.add.at(hist[:, 1], (cluster[valid], winner[valid]), 1)
            np.add.at(hist[:, 2], (cluster[correct], winner[correct]), 1)
            observed = arm['metric']
            accuracy_key = 'top1_accuracy' if choice else 'top1_event_accuracy'
            if (observed['queries'] != count or
                    not np.isclose(macro_f1(hist.sum(0)), observed['top1_macro_f1'], atol=1e-10, rtol=0) or
                    not np.isclose(correct.mean(), observed[accuracy_key], atol=1e-10, rtol=0) or
                    choice and not np.isclose((~valid).mean(), observed['false_none_rate'], atol=1e-10, rtol=0)):
                raise ValueError('Recomputed metrics differ from independent scoring')
            distributions.append(hist)
            hits.append(correct)
        if arms[0]['metric']['policy_id'] == arms[1]['metric']['policy_id']:
            raise ValueError('Expected distinct policies')
        baseline, target = hits
        rng = np.random.default_rng(interval['seed'])
        differences = []
        for _ in range(interval['samples']):
            draw = rng.integers(len(groups), size=len(groups))
            weights = np.bincount(draw, minlength=len(groups))
            left, right = [macro_f1(np.tensordot(weights, hist, axes=1)) for hist in distributions]
            differences.append(right - left)
        results[mode] = dict(
            queries=count, clusters=len(groups), interval=interval,
            baseline=arms[0]['metric']['policy_id'], endpoint=arms[1]['metric']['policy_id'],
            both_correct=int((baseline & target).sum()), fixed=int((~baseline & target).sum()),
            regressed=int((baseline & ~target).sum()), both_wrong=int((~baseline & ~target).sum()),
            macro_f1_delta_ci95=np.quantile(differences, [.025, .975]).tolist(), prediction_sha256=hashes)
    return dataset, results


def score(config, staging):
    datasets = config['datasets']
    interval = config['interval']
    if (interval['method'] != 'paired_cluster_percentile_bootstrap'
            or interval['confidence'] != .95 or interval['samples'] < 100):
        raise ValueError('Expected a predeclared paired percentile interval')
    cases = {}
    transfers = []
    seal_root = Path(config['seal']['root'])
    seal = read(seal_root / 'MANIFEST.json', config['seal']['sha256'])
    for dataset in datasets:
        def sealed(name):
            return read(seal_root / name, seal['files'][name])
        targets = sealed(f'scoring/{dataset}/targets.json')
        roster = sealed(f'inference/{dataset}/rows.json')
        candidates = sealed(f'inference/{dataset}/panel.json')['candidate_ids']
        if (len(targets) != seal['datasets'][dataset]['recordings'] or
                any(any(t[k] != r[k] for k in ('id', 'audio_sha256'))
                    for t, r in zip(targets, roster, strict=True))):
            raise ValueError('Sealed inference and scoring rosters differ')
        cases[dataset] = (dataset, targets, candidates, {'noul': [], 'choice': []}, config['interval'])
    for index, arm in enumerate(config['arms']):
        binding = read(arm['operations']['path'], arm['operations']['sha256'])
        operation = binding['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
        entry = read(arm['entry']['path'], arm['entry']['sha256'])['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
        terminal = read(arm['terminal'])
        scored = read(arm['score']['path'], arm['score']['sha256'])
        metrics = scored.get('datasets', scored)
        if operation['host'] != config['host'] or terminal != {
                'exit_code': 0, 'status': 'declared_stages_finished'}:
            raise ValueError('Successful evaluation on the bound host is required')
        for dataset in datasets:
            if entry['endpoints'][dataset]['cache']['seal'] != config['seal']:
                raise ValueError('Evaluation seal differs')
            for mode, ref in arm['readouts'].items():
                suffix = '-choices.json' if mode == 'choice' else '.json'
                relative = [str(Path(Path(p).name) / (ref['arm'] + suffix)) for p in entry['shards'][dataset]]
                names = [staging / str(index) / name for name in relative]
                values = metrics[dataset]['choices'] if mode == 'choice' else metrics[dataset]
                metric = values[ref['arm']]
                if metric['policy_id'] != ref['policy_id']:
                    raise ValueError('Scoring receipt policy differs')
                cases[dataset][3][mode].append(dict(paths=names, metric=metric))
                transfers.append((operation['remote_root'], relative, names))

    def transfer(job):
        remote, relative, names = job
        command = shlex.join(['tar', '-C', remote, '-cf', '-', '--', *relative])
        with TemporaryDirectory(dir=staging) as tmp:
            archive = Path(tmp) / 'shards.tar'
            with archive.open('wb') as stream:
                subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                config['host'], command], stdout=stream, check=True,
                               timeout=config['transfer_timeout'])
            with tarfile.open(archive) as bundle:
                members = bundle.getmembers()
                if [m.name for m in members] != relative or not all(m.isfile() for m in members):
                    raise ValueError('Prediction archive membership differs')
                for member, dest in zip(members, names, strict=True):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(bundle.extractfile(member).read())

    with ThreadPoolExecutor(config['io_workers']) as pool:
        list(pool.map(transfer, transfers))
    # Spawned workers import NumPy after inheriting these limits.
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    with ProcessPoolExecutor(config['cpu_workers'], mp_context=get_context('spawn')) as pool:
        results = dict(pool.map(score_case, cases.values()))
    return dict(schema='jev-tpo-native-pairs-v1', seal_sha256=config['seal']['sha256'],
                datasets=results, model_calls=0, npu_executed=False)


def score_case(args):
    return paired(*args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('local')
    args = parser.parse_args()
    config, _ = load_config(args.config, args.local)
    output = Path(config['output'])
    check_paths({'paired scoring': output})
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        commit_json(output, score(config, Path(temp)))
