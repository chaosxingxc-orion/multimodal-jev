"""Independently audit complete scoring panels and batch admission on CPU."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from multimodal_jev.card_scoring_settings import scoring_settings
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
    return [row['prob'] for row in payload['records']]


def validate_timing(result, settings):
    timing = result['input_timing']
    if ((timing['workers'], timing['ahead']) != tuple(settings)
            or not 0 < timing['first_batch'] <= result['seconds']
            or not 0 < timing['input_wait'] <= result['seconds']
            or not 0 < timing['forward'] <= result['seconds']
            or timing['input_wait']+timing['forward'] > result['seconds']):
        raise ValueError('scoring input concurrency or timing differs')


def audit_admission(root, c, panel, baseline_width):
    from tools.reports.card_scoring_benchmark import concurrency, replica_path, scoring_lineage

    width, = c['widths']
    replicas = len(c['devices'])
    count = concurrency(c)
    settings = (c['io_workers'], c['prefetch_audio'])

    def inspect_panel(ordinal):
        directory = root/f'worker-{ordinal}'
        ids = [row['id'] for row in panel['rows'][ordinal::replicas]]
        ref_path = directory/'baseline.json'
        baseline = read(ref_path)
        validate_timing(baseline, settings)
        ref, _ = predictions(baseline, ids, panel['labels'], baseline_width)
        lineage = scoring_lineage(c, ordinal)
        if read(directory/'lineage.json') != {'sha256': lineage}:
            raise ValueError('scoring panel lineage differs')
        markers = []
        for lane in range(count):
            lane_dir = replica_path(root, ordinal, lane) if count > 1 else directory
            candidate_path = lane_dir/f'batch-{width}.json'
            marker = dict(lineage=lineage, candidate_sha256=file_hash(candidate_path))
            if lane == 0:
                marker['baseline_sha256'] = file_hash(ref_path)
            if read(lane_dir/'panel-ready.json') != marker:
                raise ValueError('scoring panel bytes or lineage differ')
            candidate = read(candidate_path)
            validate_timing(candidate, settings)
            observed, counts = predictions(candidate, ids, panel['labels'], width)
            if ref != observed or not counts.get(width):
                raise ValueError(f'device {c["devices"][ordinal]}, replica {lane}: '
                                 'endpoint probabilities differ or no full batch')
            markers.append(marker)
        marker = markers if count > 1 else markers[0]
        shard = None
        if 'full_shards' in c:
            fsdp_path = root/f'fsdp-{ordinal}'/'reference.json'
            result = read(fsdp_path)
            if result['policy_id'] != c['formal']['policy_id'] or result['rank'] != ordinal:
                raise ValueError('FSDP reference policy or rank differs')
            reference, _ = predictions(result, ids, panel['labels'], baseline_width)
            shard = dict(reference=reference, observed=ref, sha256=file_hash(fsdp_path))
        return marker, shard

    results, _ = map_io(inspect_panel, range(replicas), workers=c['io_workers'])
    workers, fsdp = zip(*results, strict=True)
    gate = dict(passed=True, comparison='independent-replica-batch-width', devices=c['devices'],
                policy_id=c['formal']['policy_id'], checkpoint_sha256=c['formal']['checkpoint_sha256'],
                baseline_rows=baseline_width, batch_rows=width,
                pairs=len(panel['rows'])*len(panel['labels']), workers=list(workers))
    if 'full_shards' in c:
        reference, observed = {}, {}
        for shard in fsdp:
            reference.update(shard['reference'])
            observed.update(shard['observed'])
        parity = compare(panel, reference, observed, c['probability_atol'])
        if any(parity[key] for key in ('probabilities_over_atol', 'top1_disagreements',
                                       'class_rank_disagreements')):
            raise ValueError(f'FSDP reference parity failed: {parity}')
        gate.update(comparison='fsdp-and-independent-replica-batch-width',
                    fsdp=dict(atol=c['probability_atol'], worker_sha256=[row['sha256'] for row in fsdp],
                              **parity))
    return gate


def audit_pilot(root, c, full):
    from tools.reports.card_scoring_benchmark import (
        concurrency, concurrent_offsets, replica_path, scoring_lineage, scoring_offsets,
    )

    count = concurrency(c)
    size = c['formal']['recordings_per_block']
    settings = (c['io_workers'], c['prefetch_audio'])
    width, = c['widths']
    tasks, starts, finishes, sources = [], [], [], []
    new = 0
    slots = set()
    for i in range(len(c['devices'])):
        rows = full['rows'][i::len(c['devices'])]
        offsets = scoring_offsets(c, len(rows))
        for lane in range(count):
            path = replica_path(root, i, lane)/'pilot.json'
            r = read(path)
            expected, _ = concurrent_offsets(c, offsets, lane, len(rows))
            entries = r['blocks']
            if len(entries) != len(expected):
                raise ValueError('incomplete concurrent TRAIN pilot')
            made = 0
            for off, block in zip(expected, entries, strict=True):
                name = f'block-{off//size:06d}.json'
                key = (i, name)
                batch = rows[off:off+size]
                file = path.parent.parent/name
                if (key in slots or block['name'] != name or file_hash(file) != block['sha256']
                        or type(block['reused']) is not bool):
                    raise ValueError('concurrent TRAIN pilot block identity differs')
                slots.add(key)
                if not block['reused']:
                    made += len(batch)
                sources.extend(row['synthetic'] for row in batch)
                tasks.append((file, scoring_lineage(c, i), [row['id'] for row in batch],
                              full['labels'], width, settings))
            if made:
                if not 0 < r['started'] < r['finished'] < math.inf:
                    raise ValueError('invalid measured pilot interval')
                starts.append(r['started'])
                finishes.append(r['finished'])
                new += made
            elif r['started'] is not None or r['finished'] is not None:
                raise ValueError('reused pilot cannot contribute new timing')
    recordings = len(sources)
    if recordings < c['concurrent_scoring']['pilot_recordings'] or len(set(sources)) != 2:
        raise ValueError('scoring pilot must cover the configured population and both acoustic sources')
    map_cpu(validate_block, tasks, workers=c['cpu_workers'], min_parallel=c['audit_min_parallel_blocks'])
    seconds = max(finishes)-min(starts) if new else None
    return dict(recordings=recordings, synthetic=sum(sources), human=recordings-sum(sources),
                pairs=recordings*len(full['labels']), replicas_per_device=count,
                new_recordings=new, seconds=seconds,
                new_pairs_per_second=new*len(full['labels'])/seconds if new == recordings else None)


def audit_replica_results(root, c, ordinal, worker, offsets, ids):
    from tools.reports.card_scoring_benchmark import concurrency, concurrent_offsets, replica_path
    count = concurrency(c)
    size = c['formal']['recordings_per_block']
    paths = [replica_path(root, ordinal, lane)/'result.json' for lane in range(count)]
    if worker['replica_result_sha256'] != [file_hash(path) for path in paths]:
        raise ValueError('concurrent scores differ from replica receipts')
    results = [read(path) for path in paths]
    for lane, row in enumerate(results):
        pilot, rest = concurrent_offsets(c, offsets, lane, len(ids))
        expected = sorted(pilot+rest)
        if (row['device'] != worker['device'] or row['lineage'] != worker['lineage']
                or row['rows'] != sum(min(size, len(ids)-i) for i in expected)
                or [block['name'] for block in row['blocks']] != [f'block-{i//size:06d}.json' for i in expected]
                or not 0 < row['scoring_seconds'] <= worker['scoring_seconds']):
            raise ValueError('concurrent scoring replica has incomplete or foreign work')
    if sorted((b for r in results for b in r['blocks']), key=lambda b: b['name']) != worker['blocks']:
        raise ValueError('concurrent scoring replica coverage differs from the worker receipt')


def audit_roster(entry, c, panel):
    full = read(entry/'full.json')
    if (file_hash(entry/'full.json') != c['full']['sha256']
            or {k: full.get(k) for k in panel if k != 'rows'} != {k: v for k, v in panel.items() if k != 'rows'}
            or len(full['rows']) != panel['population']):
        raise ValueError('sealed full TRAIN roster differs')
    if 'full_shards' in c:
        from tools.reports.card_scoring_benchmark import restore_binding

        rows = full['rows']
        ids = [row['id'] for row in rows]
        common = full['acoustic_common']
        by_id = dict(zip(ids, rows, strict=True))
        if ids != sorted(set(ids)) or len(c['full_shards']) != len(c['devices']) or any(
                row['intent'] != by_id[row['id']]['intent'] or
                row['synthetic'] != by_id[row['id']]['synthetic'] or
                row['binding'] != restore_binding(by_id[row['id']], common) for row in panel['rows']):
            raise ValueError('compact roster differs from panel identities')

        def check(ordinal):
            path = entry/f'full-{ordinal}.json'
            if (file_hash(path) != c['full_shards'][ordinal]['sha256'] or
                    read(path) != dict(acoustic_common=common, rows=rows[ordinal::len(c['devices'])])):
                raise ValueError(f'frozen TRAIN shard {ordinal} differs from the full roster')

        map_io(check, range(len(c['devices'])), workers=c['io_workers'])
    return full


def progress(root, entry):
    """Count only durable, verified blocks, including a partially finished run."""
    c = read(entry/'score.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    from tools.reports.card_scoring_benchmark import scoring_lineage, scoring_offsets

    for name in ('full', 'panel', 'training'):
        if c[name]['sha256'] != file_hash(entry/(name+'.json')):
            raise ValueError(f'frozen scoring {name} differs')
    panel = read(entry/'panel.json')
    full = audit_roster(entry, c, panel)
    path = root/'admission.json'
    counts = dict(admitted=False, blocks=0, recordings=0, pairs=0,
                  total_recordings=full['population'], total_pairs=full['population']*len(full['labels']))
    if not path.is_file():
        return counts
    gate = audit_admission(root, c, panel, scoring_settings(read(entry/'training.json'), c.get('scoring_reference'))['forward_rows'])
    if read(path) != gate:
        raise ValueError('stored scoring admission differs')
    width, = c['widths']
    settings = (c['io_workers'], c['prefetch_audio'])
    size = c['formal']['recordings_per_block']
    replicas = len(c['devices'])
    tasks, workers = [], []
    for ordinal, device in enumerate(c['devices']):
        ids = [row['id'] for row in full['rows'][ordinal::replicas]]
        expected = {f'block-{offset//size:06d}.json': ids[offset:offset+size]
                    for offset in scoring_offsets(c, len(ids))}
        directory = root/f'worker-{ordinal}'
        lineage = scoring_lineage(c, ordinal)
        paths = sorted(directory.glob('block-*.json'))
        rows = 0
        for block in paths:
            if block.name not in expected:
                raise ValueError(f'unexpected scoring block: {block.name}')
            batch = expected[block.name]
            tasks.append((block, lineage, batch, full['labels'], width, settings))
            rows += len(batch)
        workers.append(dict(device=device, blocks=len(paths), recordings=rows, total_recordings=len(ids)))
    map_cpu(validate_block, tasks, workers=c['cpu_workers'], min_parallel=c['audit_min_parallel_blocks'])
    rows = sum(worker['recordings'] for worker in workers)
    return dict(counts, admitted=True, admission_sha256=file_hash(path), blocks=len(tasks),
                recordings=rows, pairs=rows*len(full['labels']), workers=workers)


def audit_formal(root, entry, c, panel, receipt, cid):
    from tools.reports.card_scoring_benchmark import concurrency, scoring_lineage

    formal = c['formal']
    full = audit_roster(entry, c, panel)
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
    baseline_width = scoring_settings(read(entry/'training.json'), c.get('scoring_reference'))['forward_rows']
    gate = root/'admission.json'
    if (receipt['admission_sha256'] != file_hash(gate)
            or read(gate) != audit_admission(root, c, panel, baseline_width)):
        raise ValueError('all-replica admission differs from its committed panel')
    replicas = len(c['devices'])
    size = formal['recordings_per_block']
    if len(receipt['workers']) != replicas:
        raise ValueError('scoring receipt must contain one worker group per device')

    def inspect_worker(ordinal):
        worker = receipt['workers'][ordinal]
        directory = root/f'worker-{ordinal}'
        ids = roster[ordinal::replicas]
        lineage = scoring_lineage(c, ordinal)
        if (worker['device'] != c['devices'][ordinal] or worker['rows'] != len(ids)
                or worker['lineage'] != lineage or read(directory/'lineage.json') != {'sha256': lineage}
                or read(directory/'result.json') != worker):
            raise ValueError('worker identity or full TRAIN roster differs')
        settings = (c['io_workers'], c['prefetch_audio'])
        offsets = range(0, len(ids), size)
        names = [f'block-{index:06d}.json' for index in range(len(offsets))]
        if ([block['name'] for block in worker['blocks']] != names
                or sorted(p.name for p in directory.glob('block-*.json')) != names):
            raise ValueError('missing, extra or out of order scoring blocks')

        if concurrency(c) > 1:
            audit_replica_results(root, c, ordinal, worker, offsets, ids)
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
    tasks = [task for group, _ in results for task in group]
    observations, _ = map_cpu(validate_block, tasks, workers=c['cpu_workers'],
                             min_parallel=c['audit_min_parallel_blocks'])
    gold = {row['id']: row['intent'] for row in full['rows']}
    label_index = {label: i for i, label in enumerate(labels)}
    totals = [[0.] * len(labels) for _ in labels]
    counts = [0] * len(labels)
    for task, values in zip(tasks, observations, strict=True):
        for uid, probs in zip(task[2], values, strict=True):
            i = label_index[gold[uid]]
            counts[i] += 1
            for j, value in enumerate(probs):
                totals[i][j] += value
    if counts != [Counter(gold.values())[label] for label in labels]:
        raise ValueError('class denominator differs from the frozen TRAIN roster')
    import torch
    from multimodal_jev.intent_triplet_training import confusion_distribution
    mining = scoring_settings(read(entry/'training.json'), c.get('scoring_reference'))
    distribution = confusion_distribution(labels, torch.tensor(totals, dtype=torch.float64),
                                           torch.tensor(counts), mining['zipf_exponent'])
    workers = [worker for _, worker in results]
    recordings = len(roster)
    reused = sum(worker['reused_records'] for worker in workers)
    pairs = recordings*len(labels)
    seconds = max(worker['seconds'] for worker in workers)
    extra = {}
    if concurrency(c) > 1:
        pilot = audit_pilot(root, c, full)
        if pilot != read(root/'pilot.json'):
            raise ValueError('stored concurrent TRAIN pilot differs from its blocks')
        extra = dict(replicas_per_device=concurrency(c), pilot=pilot)
    return dict(passed=True, container_id=cid, **identities, recordings=recordings, pairs=pairs, **extra,
                checkpoint_sha256=formal['checkpoint_sha256'], admission_sha256=receipt['admission_sha256'],
                labels=labels, replicas=replicas, batch_rows=width, io_workers=c['io_workers'],
                prefetch_audio=c['prefetch_audio'], updates=formal['updates'],
                panel_pairs=len(panel['rows'])*len(labels), scoring_seconds=seconds,
                worker_wall_seconds=receipt['worker_wall_seconds'], reused_recordings=reused,
                new_pairs_per_second=(recordings-reused)*len(labels)/seconds,
                workers=workers, container_exited=True,
                confusion=dict(labels=labels, counts=counts, probability_sums=totals,
                               distribution=distribution, pairs=pairs))


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
        p, _ = predictions(baseline, ids, labels, scoring_settings(read(entry/'training.json'), c.get('scoring_reference'))['forward_rows'])
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
    parser.add_argument('--progress', action='store_true', help='Audit durable blocks without a terminal receipt')
    for name in ('raw', 'entry'):
        parser.add_argument(name, type=Path)
    for name in ('operations', 'output'):
        parser.add_argument(name, type=Path, nargs='?')
    args = parser.parse_args()
    if args.progress:
        print(json.dumps(progress(args.raw, args.entry)))
        return
    if args.operations is None or args.output is None:
        parser.error('operations and output are required for terminal audit')
    report = audit(args.raw, args.entry, args.operations)
    commit_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
