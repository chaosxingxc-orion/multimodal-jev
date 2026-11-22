"""Independently audit complete scoring panels and batch admission on CPU."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import map_cpu, map_io
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


def validate_block(task):
    from tools.reports.card_scoring_benchmark import read_block

    path, lineage, ids, labels, width, settings = task
    payload = read_block(path, lineage)
    predictions(payload, ids, labels, width)
    validate_timing(payload, settings)


def validate_timing(result, settings):
    timing = result['input_timing']
    if ((timing['workers'], timing['ahead']) != tuple(settings)
            or not 0 < timing['first_batch'] <= result['seconds']
            or not 0 < timing['input_wait'] <= result['seconds']
            or not 0 < timing['forward'] <= result['seconds']
            or timing['input_wait']+timing['forward'] > result['seconds']):
        raise ValueError('scoring input concurrency or timing differs')


def audit_formal(root, entry, c, panel, receipt, cid):
    from tools.reports.card_scoring_benchmark import scoring_lineage

    formal = c['formal']
    full = read(entry/'full.json')
    identities = dict(full_sha256=c['full']['sha256'], policy_id=formal['policy_id'],
                      policy_sha256=c['policy_sha256'])
    if (any(receipt[k] != v for k, v in identities.items())
            or file_hash(entry/'full.json') != c['full']['sha256']
            or file_hash(entry/'training.json') != c['training']['sha256']
            or full['labels'] != panel['labels'] or full['population'] != panel['population']
            or full['source_training_sha256'] != c['training']['sha256']):
        raise ValueError('scoring input or endpoint identity differs')

    labels = panel['labels']
    roster = [row['id'] for row in full['rows']]
    if len(roster) != panel['population'] or len(set(roster)) != len(roster):
        raise ValueError('full TRAIN roster differs')
    width, = c['widths']
    baseline_width = read(entry/'training.json')['epoch_mining']['forward_rows']
    replicas = len(c['devices'])
    size = formal['recordings_per_block']

    def inspect_worker(ordinal):
        worker = receipt['workers'][ordinal]
        directory = root/f'worker-{ordinal}'
        ids = roster[ordinal::replicas]
        lineage = scoring_lineage(c, ordinal)
        if (worker['device'] != c['devices'][ordinal] or worker['rows'] != len(ids)
                or worker['lineage'] != lineage or read(directory/'lineage.json') != {'sha256': lineage}
                or read(directory/'result.json') != worker):
            raise ValueError('worker identity or full TRAIN roster differs')
        panel_ids = [row['id'] for row in panel['rows'][ordinal::replicas]]
        baseline = read(directory/'baseline.json')
        candidate = read(directory/f'batch-{width}.json')
        settings = (c['io_workers'], c['prefetch_audio'])
        validate_timing(baseline, settings)
        validate_timing(candidate, settings)
        ref, _ = predictions(baseline, panel_ids, labels, baseline_width)
        observed, counts = predictions(candidate, panel_ids, labels, width)
        if ref != observed or not counts.get(width):
            raise ValueError('endpoint panel probabilities differ or candidate had no full batch')
        offsets = range(0, len(ids), size)
        names = [f'block-{index:06d}.json' for index in range(len(offsets))]
        if ([block['name'] for block in worker['blocks']] != names
                or sorted(p.name for p in directory.glob('block-*.json')) != names):
            raise ValueError('missing, extra or out of order scoring blocks')

        tasks, reused_records = [], 0
        for offset, block in zip(offsets, worker['blocks'], strict=True):
            path = directory/block['name']
            if file_hash(path) != block['sha256'] or type(block['reused']) is not bool:
                raise ValueError('scoring block changed after commit')
            tasks.append((path, lineage, ids[offset:offset+size], labels, width, settings))
            if block['reused']:
                reused_records += min(size, len(ids)-offset)
        seconds = worker['scoring_seconds']
        if not 0 < seconds <= receipt['worker_wall_seconds'] < math.inf:
            raise ValueError('invalid independent scoring wall intervals')
        return tasks, dict(reused_records=reused_records, seconds=seconds,
                           result_sha256=file_hash(directory/'result.json'))

    results, _ = map_io(inspect_worker, range(replicas), workers=c['io_workers'])
    map_cpu(validate_block, (task for tasks, _ in results for task in tasks),
            workers=c['cpu_workers'], min_parallel=c['audit_min_parallel_blocks'])
    workers = [worker for _, worker in results]
    recordings = len(roster)
    reused = sum(worker['reused_records'] for worker in workers)
    pairs = recordings*len(labels)
    seconds = max(worker['seconds'] for worker in workers)
    return dict(passed=True, container_id=cid, **identities, recordings=recordings, pairs=pairs,
                labels=labels, replicas=replicas, batch_rows=width, io_workers=c['io_workers'],
                prefetch_audio=c['prefetch_audio'], updates=formal['updates'],
                panel_pairs=len(panel['rows'])*len(labels), scoring_seconds=seconds,
                worker_wall_seconds=receipt['worker_wall_seconds'], reused_recordings=reused,
                new_pairs_per_second=(recordings-reused)*len(labels)/seconds,
                workers=workers, container_exited=True)


def audit(root, entry, operations):
    c = read(entry/'score.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    panel = read(entry/'panel.json')
    launch = read(operations/'score-launch.json')
    cid = launch['container_id']
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
    if 'formal' in c or c.get('capture_host_memory'):
        from tools.reports.context_macro_probe_operations import inspect_runtime

        from tools.reports.native_seal_operations import verify_memory

        inspect_runtime(capture, launch, terminal=True)
        verify_memory(capture, launch['argv'])
        admission = read(operations/'score-admission.json')
        if (c['devices'] != admission['idle_devices']
                or admission['entry_pins']['score.json'] != file_hash(entry/'score.json')
                or c['source']['sha256'] != admission['entry_pins']['source/FILES.json']
                or c['cards']['sha256'] != file_hash(entry/'cards.json')):
            raise ValueError('scoring admission or frozen configuration changed')
        if 'formal' in c:
            return dict(audit_formal(root, entry, c, panel, receipt, cid),
                        memory_limit_bytes=capture['HostConfig']['Memory'])
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
    result = dict(container_id=cid, policy_id=panel['source_policy_id'], panel_sha256=c['panel']['sha256'],
                cards_sha256=c['cards']['sha256'], probability_atol=c['probability_atol'],
                width_results=results, recordings=total, pairs=pairs,
                baseline_seconds=max(old_seconds), baseline_pairs_per_second=pairs/max(old_seconds))
    if c.get('capture_host_memory'):
        memory = receipt['host_memory']
        limit = capture['HostConfig']['Memory']
        if any(int(m['memory.max']) != limit for m in memory.values()):
            raise ValueError('cgroup memory limit differs from Docker')
        result.update(memory_limit_bytes=limit, host_memory=memory)
    return result


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
