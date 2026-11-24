"""Publish a single-card TPO view from a completed, audited TRAIN score."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unicodedata

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import map_cpu, map_io
from multimodal_jev.local_checkpoint import commit_file, commit_json, sync_directory
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.audit_card_scoring_benchmark import audit_roster, read, validate_block
from tools.reports.build_tpo_bidirectional_targets import compile_schedule, pinned
from tools.reports.build_tpo_gap_targets import shuffled_batches
from tools.reports.card_scoring_benchmark import pinned as read_pin, scoring_lineage, scoring_offsets
from tools.reports.card_scoring_shards import check_partition


def jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def score_spec(path):
    return read(path)['macros']['NATIVE_SEAL_CONTAINER_CONFIG']


def load_scores(config):
    if 'single_host_score' in config:
        path = Path(config['single_host_score'])
        ops = Path(config['single_host_operations'])
        audit = path/'audit.json'
        from tools.reports.audit_card_scoring_benchmark import audit as audit_score
        verified = audit_score(path/'collected/score', path/'entry', ops)
        saved = read(audit)
        if any(saved[key] != value for key, value in verified.items() if key != 'confusion'):
            raise ValueError('saved score does not match the independently reaudited output')
        config.update(merged=dict(path=str(audit), sha256=file_hash(audit)))
        spec = score_spec(path/'entry/score.json')
        full = read(path/'entry/full.json')
        tasks = []
        for ordinal, worker in enumerate(read(path/'collected/score/receipt.json')['workers']):
            ids = [row['id'] for row in full['rows'][ordinal::len(spec['devices'])]]
            size = spec['formal']['recordings_per_block']
            lineage = scoring_lineage(spec, ordinal)
            for number, block in enumerate(worker['blocks']):
                start = number*size
                tasks.append((path/'collected/score'/f'worker-{ordinal}'/block['name'],
                              lineage, ids[start:start+size], full['labels'], spec['widths'][0],
                              (spec['io_workers'], spec['prefetch_audio'])))
        cpu = dict(config['execution'])
        del cpu['io_workers']
        values, _ = map_cpu(validate_block, tasks, workers=cpu.pop('cpu_workers'), **cpu)
        pairs = {uid: probability for task, rows in zip(tasks, values, strict=True)
                 for uid, probability in zip(task[2], rows, strict=True)}
        if len(pairs) != saved['recordings'] or sum(len(t[2]) for t in tasks) != len(pairs):
            raise ValueError('validated scores did not cover the full population exactly once')
        return spec, full, pairs, path/'entry'
    merged = read_pin(config['merged'])
    specs, receipts = [], []
    for lane in config['lanes']:
        path, ops = Path(lane['package']), Path(lane['operations'])
        score = path/'entry/score.json'
        admission = read_pin(lane['admission'])
        collected = read_pin(lane['collected'])
        audit = read_pin(lane['audit'])
        state = read(ops/'score-inspect.json')
        spec = score_spec(score)
        if (admission['entry_pins']['score.json'] != file_hash(score)
                or state['Id'] != audit['container_id']
                or state['Id'] != read(ops/'score-launch.json')['container_id']
                or state['State']['Status'] != 'exited' or state['State']['ExitCode'] != 0
                or state['State']['OOMKilled'] or audit['passed'] is not True
                or audit['container_exited'] is not True
                or audit['block_range'] != spec['formal']['block_range']
                or collected['full_sha256'] != spec['full']['sha256']
                or collected['policy_id'] != spec['formal']['policy_id']
                or collected['policy_sha256'] != spec['policy_sha256']
                or collected['admission_sha256'] != audit['admission_sha256']):
            raise ValueError('collected score lane is detached from the audited runtime')
        specs.append(spec)
        receipts.append((path, collected, audit))
    if [r[2]['recordings'] for r in receipts] != [x['recordings'] for x in merged['lanes']]:
        raise ValueError('score lanes differ from merged receipt')
    source = specs[0]
    first = receipts[0][0]/'entry'
    full = audit_roster(first, source, read_pin(dict(path=first/'panel.json', sha256=source['panel']['sha256'])))
    if (merged['passed'] is not True or merged['container_exited'] is not True
            or merged['recordings'] != len(full['rows'])
            or merged['pairs'] != len(full['rows'])*len(full['labels'])
            or merged['checkpoint_sha256'] != source['formal']['checkpoint_sha256']
            or merged['updates'] != source['formal']['updates']):
        raise ValueError('full score identity differs from merged audit')
    check_partition(source, specs, len(full['rows']), 0)
    replicas, block_size = len(source['devices']), source['formal']['recordings_per_block']
    tasks = []
    for spec, (path, receipt, _) in zip(specs, receipts, strict=True):
        if (any(spec[k] != source[k] for k in ('full', 'full_shards', 'cards', 'training', 'policy_sha256',
                                            'devices', 'widths', 'streaming', 'io_workers', 'prefetch_audio'))
                or any(spec['formal'][k] != source['formal'][k] for k in
                       ('policy_id', 'checkpoint_sha256', 'image', 'updates', 'recordings_per_block'))):
            raise ValueError('incompatible score lane')
        if len(receipt['workers']) != replicas:
            raise ValueError('incomplete score worker receipt')
        for ordinal, worker in enumerate(receipt['workers']):
            if worker['device'] != spec['devices'][ordinal]:
                raise ValueError('invalid score worker order')
            ids = [r['id'] for r in full['rows'][ordinal::replicas]]
            width, = spec['widths']
            blocks = {b['name']: b['sha256'] for b in worker['blocks']}
            offsets = scoring_offsets(spec, len(ids))
            expected = [f'block-{offset//block_size:06d}.json' for offset in offsets]
            if set(blocks) != set(expected) or len(blocks) != len(worker['blocks']):
                raise ValueError('score receipt coverage differs')
            for offset, name in zip(offsets, expected, strict=True):
                block = path/'collected/score'/f'worker-{ordinal}'/name
                tasks.append((blocks[name], block, scoring_lineage(spec, ordinal),
                              ids[offset:offset+block_size], full['labels'], width,
                              (spec['io_workers'], spec['prefetch_audio'])))
    map_io(lambda t: checked_file(t[1], t[0]), tasks, workers=config['execution']['io_workers'])
    cpu = dict(config['execution'])
    del cpu['io_workers']
    values, _ = map_cpu(validate_block, [t[1:] for t in tasks], workers=cpu.pop('cpu_workers'), **cpu)
    probs = {uid: probability for task, rows in zip(tasks, values, strict=True)
             for uid, probability in zip(task[3], rows, strict=True)}
    if len(probs) != len(full['rows']) or sum(len(t[3]) for t in tasks) != len(probs):
        raise ValueError('score recording coverage differs')
    return source, full, probs, first


def checked_file(path, expected):
    if file_hash(path) != expected:
        raise ValueError(f'frozen input changed: {path}')


def norm_text(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def eligible(config, full):
    real_root = Path(config['release']['path'])
    real_manifest = read_pin(dict(path=real_root/'MANIFEST.json', sha256=config['release']['sha256']))
    real = jsonl(pinned(real_root, real_manifest, 'learning.jsonl'))
    selection = jsonl(pinned(real_root, real_manifest, 'selection.jsonl'))
    synth = read_pin(config['synthetic'])
    calibration = read_pin(config['calibration'])
    native = Path(config['native']['path'])
    native_manifest = read_pin(dict(path=native/'MANIFEST.json', sha256=config['native']['sha256']))
    native_ids = {r['id']: r['source_ids'] for split in ('learning', 'selection')
                  for r in jsonl(pinned(native, native_manifest, split+'.targets.jsonl'))}
    data = Path(config['annotation']['path'])
    checked_file(data, config['annotation']['sha256'])
    wording = {str(row['slurp_id']): norm_text(row['sentence']) for row in jsonl(data)}
    blocked = calibration + selection
    blocked_groups = {r['source_group'] for r in blocked}
    blocked_hashes = {r['audio_sha256'] for r in blocked}
    blocked_ids = {sid for r in blocked for sid in native_ids[r['id']]}
    blocked_text = {wording[sid] for sid in blocked_ids}
    originals = {r['id']: (False, r) for r in real}
    originals.update({r['id']: (True, r) for r in synth})
    if len(originals) != len(real)+len(synth) or len(originals) != len(full['rows']):
        raise ValueError('TRAIN membership changed')
    selected, hash_labels = {}, defaultdict(set)
    excluded = Counter()
    for row in full['rows']:
        uid = row['id']
        synthetic, meta = originals[uid]
        if ({k: row[k] for k in ('id','intent','audio','audio_sha256')}
                != {k: meta[k] for k in ('id','intent','audio','audio_sha256')}
                or row['synthetic'] is not synthetic or meta['role'] != 'learning'
                or meta['official_split'] != 'train' or meta['known_pending_source'] is not False):
            raise ValueError('TRAIN metadata differs from scored audio')
        sha = row['audio_sha256']
        hash_labels[sha].add(row['intent'])
        group = meta['source_group']
        if not group:
            raise ValueError('missing source group')
        if group in blocked_groups or sha in blocked_hashes:
            excluded['heldout_group_or_audio'] += 1
            continue
        if synthetic and (meta['slurp_id'] in blocked_ids or norm_text(meta['annotation_text']) in blocked_text):
            excluded['heldout_synthetic_source_or_text'] += 1
            continue
        selected[uid] = dict(row, group=group, text=(norm_text(meta['annotation_text']) if synthetic else
                                                    wording[native_ids[uid][0]]))
    seen, clean = set(), {}
    # Prefer the native recording over a byte-identical synthesized duplicate.
    for uid, row in sorted(selected.items(), key=lambda pair: (pair[1]['synthetic'], pair[0])):
        sha = row['audio_sha256']
        if len(hash_labels[sha]) != 1:
            excluded['conflicting_gold'] += 1
        elif sha in seen:
            excluded['duplicate_audio'] += 1
        else:
            seen.add(sha)
            clean[uid] = row
    return clean, calibration, dict(excluded)


def select_rows(clean, probs, labels, selection, margin, seed):
    label_index = {label: i for i, label in enumerate(labels)}

    def seeded(uid):
        return hashlib.sha256(f'{seed}:{uid}'.encode()).hexdigest(), uid

    annotated, easy = {}, []
    for uid, row in clean.items():
        gold = label_index[row['intent']]
        prob = probs[uid]
        rival = max((i for i in range(len(labels)) if i != gold), key=prob.__getitem__)
        gap = prob[gold]-prob[rival]
        kind = 'error' if gap < 0 else 'tie' if gap == 0 else 'boundary' if gap < margin else 'easy'
        annotated[uid] = dict(row, gold=gold, competitor=rival, ref_gap=gap, stratum=kind)
        if kind == 'easy':
            easy.append(uid)
    existing = set(easy)
    for kind, metric in (
        ('positive_underconfidence', lambda r, p: 1-p[r['gold']]),
        ('negative_overconfidence', lambda r, p: math.fsum(-math.log1p(-min(p[i], 1-1e-6))
                                                           for i in range(len(p)) if i != r['gold'])),
    ):
        ranked = sorted(existing, key=lambda uid: (-metric(annotated[uid], probs[uid]), seeded(uid)))
        for uid in ranked[:selection['probability_examples_per_type']]:
            existing.remove(uid)
            annotated[uid]['stratum'] = kind
    hard = annotated.keys()-existing
    if not hard or not existing:
        raise ValueError('both hard examples and correctly ranked replay are required')
    boundaries = {(annotated[uid]['gold'], annotated[uid]['competitor']) for uid in hard}
    words = {(annotated[uid]['gold'], annotated[uid]['text']) for uid in hard}

    def priority(uid):
        r = annotated[uid]
        return ((r['gold'], r['text']) not in words,
                (r['gold'], r['competitor']) not in boundaries and (r['competitor'], r['gold']) not in boundaries,
                seeded(uid))

    buckets = defaultdict(list)
    for uid in existing:
        buckets[annotated[uid]['gold']].append(uid)
    for group in buckets.values():
        group.sort(key=priority)
    fraction = Fraction(str(selection['easy_fraction']))
    requested = math.ceil(len(hard)*fraction/(1-fraction))
    target = min(requested, len(existing))
    balanced = [buckets[i][depth] for depth in range(selection['replay_class_priority'])
                for i in sorted(buckets, key=lambda j: seeded(labels[j])) if len(buckets[i]) > depth]
    replay = set(balanced[:target])
    replay.update(sorted(existing-replay, key=priority)[:target-len(replay)])
    result = {}
    for uid in sorted(hard | replay):
        row = annotated[uid].copy()
        del row['text']
        if uid in replay:
            row['stratum'] = 'replay'
        row['index'] = len(result)
        result[uid] = row
    return result, dict(replay_requested=requested, replay_actual=len(replay), replay_shortfall=requested-len(replay))


def build(config):
    dst = Path(config['output'])
    check_paths({'card TPO targets': dst})
    source, full, probs, first = load_scores(config)
    labels = full['labels']
    clean, calibration, excluded = eligible(config, full)
    schedule = config['schedule']
    loss = config['loss']
    if (not 0 < loss['margin'] < 1 or loss['weight'] != 1/len(labels)
            or loss.get('gate') != 'strict_target_top1' or schedule['epochs'] != 1):
        raise ValueError('invalid single-card probability gap or schedule')
    rows, replay = select_rows(clean, probs, labels, config['selection'], loss['margin'], schedule['seed'])
    batches = shuffled_batches(rows, schedule)
    target, batches, counts, audit, ordering = compile_schedule(
        dict(schedule=schedule, reward=config['reward'], updates=len(batches)), rows, batches, list(rows), probs)
    gold = torch.tensor([r['gold'] for r in rows.values()])
    probability = target[0].sigmoid()
    q_true = probability.gather(1, gold[:, None]).squeeze(1)
    q_rival = probability.scatter(1, gold[:, None], float('-inf')).amax(1)
    gate_counts = dict(strict=int((q_true>q_rival).sum()), tied=int((q_true==q_rival).sum()),
                       wrong=int((q_true<q_rival).sum()))
    evidence = dict(eligible=len(clean), excluded=excluded, **replay,
                    strata=dict(Counter(r['stratum'] for r in rows.values())),
                    source_types=dict(Counter('synthetic' if r['synthetic'] else 'real' for r in rows.values())),
                    class_strata={label: dict(Counter(r['stratum'] for r in rows.values() if r['intent']==label))
                                  for label in labels}, source_groups=len({r['group'] for r in rows.values()}),
                    confusion_directions=dict(Counter(f"{r['intent']}/{labels[r['competitor']]}" for r in rows.values())),
                    target_rank=gate_counts)
    manifest = dict(schema='jev-tpo-reward-view-v1', source_policy=source['formal']['policy_id'],
                    source_policy_sha256=source['policy_sha256'],
                    release_sha256=config['release']['sha256'], reference_sha256=config['merged']['sha256'],
                    view_sha256=source['full']['sha256'], candidate_ids=labels,
                    card_input=dict(cards_sha256=source['cards']['sha256'], arm=read(first/'panel.json')['card_arm'],
                                    acoustic_common=full['acoustic_common'],
                                    checkpoint_sha256=source['formal']['checkpoint_sha256']),
                    schedule=schedule, recordings=len(rows), updates=len(batches), batch_real=counts,
                    real_exposures=sum(counts), probability_gap=loss, strata=evidence['strata'],
                    bidirectional_contrast=dict(reward=config['reward'], batch_order_sha256=ordering, audit=audit))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.card-tpo-', dir=dst.parent) as temp:
        stage = Path(temp)/'targets'
        stage.mkdir()
        for name, content in (
            ('rows.json', rows), ('batches.json', batches), ('calibration.json', calibration),
            ('population-audit.json', evidence),
        ):
            commit_json(stage/name, content)
        cards = first/'cards.json'
        checked_file(cards, manifest['card_input']['cards_sha256'])
        commit_file(stage/'cards.json', lambda stream: stream.write(cards.read_bytes()))
        reference = torch.tensor([probs[uid] for uid in rows], dtype=torch.float32)
        commit_file(stage/'targets.pt', lambda stream: torch.save(target, stream))
        commit_file(stage/'reference.pt', lambda stream: torch.save(reference, stream))
        manifest['files'] = {p.name:file_hash(p) for p in sorted(stage.iterdir())}
        commit_json(stage/'MANIFEST.json', manifest)
        if dst.exists():
            checked_file(dst/'MANIFEST.json', file_hash(stage/'MANIFEST.json'))
            for name, sha in manifest['files'].items():
                checked_file(dst/name, sha)
        else:
            stage.rename(dst)
            sync_directory(dst.parent)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('local')
    args = parser.parse_args()
    config, _ = load_config(args.config, args.local)
    torch.set_num_threads(config['cpu_threads'])
    result = build(config)
    print(json.dumps({k:result[k] for k in ('recordings', 'updates', 'strata')}))
