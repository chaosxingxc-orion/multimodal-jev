"""Archive-only clean TEST reference uncertainty; no model or selection access."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.reports.score_slurp_context_test_pair import reconstruct_targets


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def cluster_interval(records, *, seed=317, replicates=2000):
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Unique complete reference records required')
    groups = {}
    for row in records:
        if (type(row['correct']) is not bool or type(row['source_group']) is not str
                or not row['source_group']):
            raise ValueError('Boolean correctness and source identity required')
        n, correct = groups.setdefault(row['source_group'], (0, 0))
        groups[row['source_group']] = (n + 1, correct + row['correct'])
    names = sorted(groups)
    if len(names) < 2:
        raise ValueError('At least two source clusters required')
    counts = np.asarray([groups[k][0] for k in names], dtype=np.int64)
    correct = np.asarray([groups[k][1] for k in names], dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(0, len(names), size=(replicates, len(names)), dtype=np.int64)
    samples = np.concatenate([correct[d].sum(1) / counts[d].sum(1)
                              for d in np.array_split(draws, 20)])
    return dict(recordings=int(counts.sum()), correct=int(correct.sum()),
                accuracy=float(correct.sum()/counts.sum()), source_groups=len(names),
                cluster_sizes=dict(Counter(int(n) for n in counts)),
                percentile95=np.quantile(samples, [.025, .975], method='linear').tolist(),
                bootstrap=dict(unit='source_group', seed=seed, replicates=replicates, rng='PCG64',
                               estimand='recording_weighted_accuracy', quantile_method='linear',
                               draws_sha256=hashlib.sha256(draws.astype('<i8', copy=False).tobytes()).hexdigest()))


def run(contract_path):
    if sha(contract_path) != '25c3ff09668a3e3c10570efa3f3e1651059ae5cc92098731d58d3ae2258e7825':
        raise ValueError('Pre-existing frozen clean TEST scoring contract required')
    contract = read(contract_path)
    if contract['optimization_allowed'] is not False:
        raise ValueError('TEST scoring only')
    artifacts = contract['artifacts']
    sources = {a['path']: a['sha256'] for a in artifacts.values()}
    for path, expected in sources.items():
        if sha(path) != expected:
            raise ValueError('Scoring artifact changed: '+path)
    result = read(artifacts['reference_result']['path'])
    for path, expected in result['pins'].items():
        if sha(path) != expected:
            raise ValueError('Archived reference dependency changed: '+path)
        sources[path] = expected
    sources[str(contract_path)] = sha(contract_path)
    for path in (Path(__file__), Path('tools/reports/score_slurp_context_test_pair.py'),
                 Path('tools/jev/rescore_slurp_clean_native.py')):
        sources[str(path)] = sha(path)
    def get(name):return artifacts[name]['path']
    inputs = rows(get('input_rows'))
    task = read(get('tasks'))['intent']
    truth = reconstruct_targets(rows(get('native_requests')), rows(get('native_targets')),
                                rows(get('clean_metadata')), read(get('excluded')),
                                read(get('population')), inputs, task)
    preds = rows(get('reference_predictions'))
    indexed = {'slurp:'+p['file']: p for p in preds}
    if len(indexed) != len(preds) or set(indexed) != set(truth):
        raise ValueError('Complete distinct reference predictions required')
    records = []
    group_truth = {}
    invalid = 0
    for row in inputs:
        uid = row['id']; gold = truth[uid]; pred = indexed[uid]
        if group_truth.setdefault(row['source_group'], gold) != gold:
            raise ValueError('Conflicting source-group gold')
        invalid += pred['scenario'] == '__invalid_native_output__'
        records.append(dict(id=uid, source_group=row['source_group'], gold=gold,
                            correct=[pred['scenario'], pred['action']] == task['candidate_pairs'][gold]))
    stats = cluster_interval(records)
    receipt = read(get('reference_receipt'))
    if (stats['recordings'] != 12991 or stats['source_groups'] != 2947
            or stats['correct'] != receipt['joint_correct'] or invalid != receipt['malformed']
            or result['optimization_allowed'] is not False):
        raise ValueError('Archive aggregate does not reproduce')
    return dict(schema='slurp-omni-clean-reference-cluster-v1', sources=sources,
                reference=stats, malformed_in_full_denominator=invalid,
                records_sha256=hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                protocol_timing='Supplementary descriptive interval after historical reference scores were known; before Jev TEST outputs. Not a preregistered primary test.',
                limits=['Retained historical native composite generation, not a new inference or a Jev interface.',
                        'Clean exclusion subset, not official full TEST. Invalid outputs remain wrong.',
                        'Source-utterance clusters do not address speaker correlation across utterances, training seeds, multiplicity or model selection.',
                        'No Jev TEST results, paired improvement, probability calibration or matched-budget algorithm claim.',
                        'No changes to prompts, thresholds, training or queued scoring contracts.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.contract)
    with Path(args.output).open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result['reference'], indent=2))
