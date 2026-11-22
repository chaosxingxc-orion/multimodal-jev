"""Independently audit complete scoring panels and batch admission on CPU."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json


def read(path):
    return json.loads(path.read_bytes())


def predictions(result, expected, labels, max_width):
    rows = result['records']
    if [r['id'] for r in rows] != expected or any(len(r['prob']) != len(labels) for r in rows):
        raise ValueError('recording IDs or card counts differ from the sealed panel')
    if any(type(p) not in (int,float) or not math.isfinite(p) or not 0 <= p <= 1
           for row in rows for p in row['prob']):
        raise ValueError('invalid membership probability')
    counts = {int(k):v for k,v in result['batch_sizes'].items()}
    if (not counts or max(counts) > max_width or min(counts) < 1 or
            any(type(v) is not int or v <= 0 for v in counts.values()) or
            sum(k*v for k,v in counts.items()) != len(rows)*len(labels) or
            not math.isfinite(result['seconds']) or result['seconds'] <= 0 or
            type(result['peak_bytes']) is not int or result['peak_bytes'] <= 0):
        raise ValueError('inconsistent batch, timing or memory counters')
    return dict((r['id'], r['prob']) for r in rows), counts


def compare(panel, reference, observed, atol):
    labels = panel['labels']
    groups = {label:[] for label in labels}
    deltas = []
    winners = 0
    for row in panel['rows']:
        uid = row['id']
        groups[row['intent']].append(uid)
        r, o = reference[uid], observed[uid]
        deltas.extend(abs(x-y) for x,y in zip(r,o,strict=True))
        winners += r.index(max(r)) != o.index(max(o))

    def confusion(values):
        means = [[statistics.fmean(values[uid][i] for uid in groups[label])
                  for i in range(len(labels))] for label in labels]
        # A tie has the same sampling mass; compare the rank groups in both maps.
        return [
            rank_groups({j: (means[i][j] + means[j][i])/2 for j in range(len(labels)) if j != i})
            for i in range(len(labels))
        ]

    ranks_a, ranks_b = confusion(reference), confusion(observed)
    changed = sum(a != b for a,b in zip(ranks_a,ranks_b,strict=True))
    return dict(max_probability_abs=max(deltas), mean_probability_abs=statistics.fmean(deltas),
                probabilities_over_atol=sum(d > atol for d in deltas),
                top1_disagreements=winners, class_rank_disagreements=changed)


def rank_groups(scores):
    ordered = {}
    for idx, score in sorted(scores.items(), key=lambda x: (-x[1],x[0])):
        ordered.setdefault(score, []).append(idx)
    return list(ordered.values())


def measure_width(members, width, panel, reference, atol, name='', settings=None):
    suffix = f'-{name}' if name else ''
    raw = [read(path/f'batch-{width}{suffix}.json') for path,_ in members]
    ooms = [idx for idx,x in enumerate(raw) if x.get('oom')]
    if ooms:
        return dict(passed=False, oom_ordinals=ooms)
    labels = panel['labels']
    pairs = len(panel['rows'])*len(labels)
    observed, counts = {}, Counter()
    for result, (_,ids) in zip(raw,members,strict=True):
        p, stats = predictions(result, ids, labels, width)
        observed.update(p)
        counts.update(stats)
    metrics = compare(panel, reference, observed, atol)
    full_batches = [x['batch_sizes'].get(str(width), 0) for x in raw]
    passed = all(full_batches) and all(metrics[k]==0 for k in (
        'probabilities_over_atol', 'top1_disagreements', 'class_rank_disagreements'))
    slowest = max(raw, key=lambda x: x['seconds'])
    seconds = slowest['seconds']
    summary = dict(passed=bool(passed), seconds=seconds, pairs_per_second=pairs/seconds,
                   full_batches=full_batches, batch_sizes=dict(counts),
                   peak_bytes=max(x['peak_bytes'] for x in raw), **metrics)
    if settings is not None:
        if any((t['workers'],t['ahead']) != tuple(settings)
               or not 0 < t['first_batch'] <= t['input_wait'] <= x['seconds']
               or not 0 < t['forward'] <= x['seconds']
               or t['input_wait']+t['forward'] > x['seconds']
               for x in raw for t in (x['input_timing'],)):
            raise ValueError('streaming concurrency or interval accounting differs')
        summary['slowest_worker_input_timing'] = slowest['input_timing']
    return summary


def audit(root, entry, operations):
    c = read(entry/'score.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    panel = read(entry/'panel.json')
    cid = read(operations/'score-launch.json')['container_id']
    capture = read(operations/'score-inspect.json')
    receipt = read(root/'receipt.json')
    if (capture['Id'] != cid or capture['State']['Status'] != 'exited' or
            capture['State']['ExitCode'] != 0 or capture['State']['OOMKilled'] or
            read(operations/'score-wait.json') != {'exit_code':0} or
            read(operations/'score-collected.json') != receipt or
            receipt['devices'] != c['devices'] or len(receipt['workers']) != len(c['devices']) or
            file_hash(entry/'panel.json') != c['panel']['sha256'] or
            receipt['panel_sha256'] != c['panel']['sha256']):
        raise ValueError('no complete device release or panel identity')
    labels = panel['labels']
    count = len(c['devices'])
    members, reference, old_seconds = [], {}, []
    for i,w in enumerate(receipt['workers']):
        path = root/f'worker-{i}'
        ids = [r['id'] for r in panel['rows'][i::count]]
        if w['device'] != c['devices'][i] or w['rows'] != len(ids):
            raise ValueError('worker identity or roster differs')
        baseline = read(path/'baseline.json')
        p, _ = predictions(baseline, ids, labels, read(entry/'training.json')['epoch_mining']['forward_rows'])
        reference.update(p)
        members.append((path, ids))
        old_seconds.append(baseline['seconds'])
    total = len(panel['rows'])
    pairs = total*len(labels)
    results = {}
    streaming = c.get('streaming')
    arms = tuple(streaming['arms'].items()) if streaming else (('', None),)
    for width in c['widths']:
        for name, settings in arms:
            result = measure_width(members, width, panel, reference, c['probability_atol'],
                                   name, settings)
            if 'oom_ordinals' in result:
                result['oom_devices'] = [c['devices'][i] for i in result.pop('oom_ordinals')]
            key = f'{width}-{name}' if name else str(width)
            results[key] = result
    return dict(container_id=cid, policy_id=panel['source_policy_id'], panel_sha256=c['panel']['sha256'],
                cards_sha256=c['cards']['sha256'], probability_atol=c['probability_atol'],
                width_results=results, recordings=total, pairs=pairs,
                baseline_seconds=max(old_seconds), baseline_pairs_per_second=pairs/max(old_seconds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'entry', 'operations', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    report = audit(args.raw, args.entry, args.operations)
    commit_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
