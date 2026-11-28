"""Benchmark single-card source TRAIN scoring using independent NPU replicas."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain, islice
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from multimodal_jev.card_scoring_settings import scoring_settings
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import content_key
from multimodal_jev.local_checkpoint import commit_json


def pinned(entry):
    path = Path(entry['path'])
    if file_hash(path) != entry['sha256']:
        raise ValueError(f'changed input: {path}')
    return json.loads(path.read_bytes())


def acoustic_binding(row, spec):
    return dict(schema='encoded-audio-cache-v1',
                request=dict(spec['acoustic_template']['panel'], audio=row['audio']),
                audio_sha256=row['audio_sha256'], processor_sha256=spec['processor_sha256'],
                preprocessing=None, context_encoding=spec['acoustic_encoding'])


def restore_binding(row, common):
    return dict(common, audio_sha256=row['audio_sha256'],
                request=dict(common['request'], audio=row['audio']))


def make_panel(c, output):
    train, spec = pinned(c['training']), pinned(c['acoustics'])
    synth = pinned(c['synthetic'])
    root = Path(train['release'])
    manifest = pinned(dict(path=root/'MANIFEST.json', sha256=train['mixed']['real_release_sha256']))
    real_path = root/'learning.jsonl'
    if file_hash(real_path) != manifest['files']['learning.jsonl']:
        raise ValueError('TRAIN release membership changed')
    with real_path.open() as stream:
        real = [json.loads(line) for line in stream]
    longest = pinned(c['real_lengths'])
    cards = pinned(c['cards'])
    policy = Path(c['policy'])
    if (set(longest) != {r['id'] for r in real} or len(real)+len(synth) != train['batch_config']['source_queries']
            or spec['full_verified'] != len(real)+len(synth)
            or train['layout_cache']['sha256'] != c['acoustics']['sha256']
            or train['single_card']['sha256'] != c['cards']['sha256']
            or scoring_settings(train, c.get('scoring_reference'))['card_arm'] != c['card_arm']
            or pinned(dict(path=policy/'MANIFEST.json', sha256=c['policy_sha256']))['policy_id']
            != train['source_policy_id']):
        raise ValueError('benchmark must bind to the frozen TRAIN population and source policy')
    labels = cards['labels']
    by_class = {label: [] for label in labels}
    for row in chain(real, synth):
        if row['role'] != 'learning' or row['official_split'] != 'train':
            raise ValueError('benchmark only admits source TRAIN')
        by_class[row['intent']].append(row)
    randomizer = random.Random(c['seed'])
    if len({row['id'] for row in chain(real, synth)}) != len(real)+len(synth):
        raise ValueError('duplicate TRAIN recording')
    sample = [row for label in labels
              for row in randomizer.sample(by_class[label], c['panel_per_class'])]
    sample_ids = {row['id'] for row in sample}
    sample.extend(islice((row for row in sorted(real, key=lambda r: (-longest[r['id']], r['id']))
                         if row['id'] not in sample_ids), c['long_audio_count']))
    randomizer.shuffle(sample)
    synth_ids = {row['id'] for row in synth}
    rows = [dict(id=row['id'], intent=row['intent'], synthetic=row['id'] in synth_ids,
                 binding=acoustic_binding(row, spec)) for row in sample]
    panel = dict(rows=rows, labels=labels, widths=c['widths'], card_arm=c['card_arm'],
                 source_policy_id=train['source_policy_id'], population=len(real)+len(synth),
                 source_cards_sha256=c['cards']['sha256'], source_training_sha256=c['training']['sha256'],
                 cache_proof_sha256=c['acoustics']['sha256'])
    commit_json(output, panel)
    if c.get('formal'):
        full = sorted(chain(real, synth), key=lambda row: row['id'])
        compact = [dict(id=row['id'], intent=row['intent'], synthetic=row['id'] in synth_ids,
                        audio=row['audio'], audio_sha256=row['audio_sha256']) for row in full]
        common = acoustic_binding(dict(audio=None, audio_sha256=None), spec)
        del common['audio_sha256'], common['request']['audio']
        commit_json(output.with_name('full.json'), dict(panel, rows=compact, acoustic_common=common))
        for ordinal in range(c['replicas']):
            commit_json(output.with_name(f'full-{ordinal}.json'),
                        dict(acoustic_common=common, rows=compact[ordinal::c['replicas']]))
    return panel


def load_panel(c):
    panel, cards = pinned(c['panel']), pinned(c['cards'])
    indices = cards['arms'][panel['card_arm']]
    if (panel['widths'] != c['widths'] or panel['labels'] != cards['labels']
            or panel['source_cards_sha256'] != c['cards']['sha256']
            or [cards['catalog'][i]['id'] for i in indices] != panel['labels']):
        raise ValueError('frozen card order or benchmark design changed')
    return panel, [cards['catalog'][i] for i in indices]


def render(inputs, cards, collate, uid, policy):
    import torch

    audio_token = collate['audio_token_id']
    marker = collate['decision_token_id']
    sound = inputs['input_ids'][0]
    sound = sound[sound == audio_token]
    for index, (prefix, suffix) in enumerate(cards):
        ids = torch.cat((prefix, sound, suffix))[None, :]
        pos = (ids[0] == marker).nonzero().T.contiguous()
        if pos.shape != (1, 1) or int(pos[0, 0]) < len(prefix)+len(sound):
            raise ValueError('one marker after the audio required')
        yield dict(request_id=f'{uid}:{index}', policy_id=policy, positions=pos,
                   inputs=dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                               input_features=inputs['input_features'],
                               input_features_mask=inputs['input_features_mask']))


def batches(rows, width, *, group_audio, collate):
    if group_audio:
        # The legacy path consumes all candidates for each recording before advancing.
        for per_audio in rows:
            yield from batches([per_audio], width, group_audio=False, collate=collate)
        return
    # An audio shorter than a CNN chunk retains its own inference window.
    chunk = collate['audio_chunk_size']
    buckets = {}
    for per_audio in rows:
        for item in per_audio:
            features = item['inputs']['input_features_mask']
            key = min(int(features.sum()), chunk)
            group = buckets.setdefault(key, [])
            group.append(item)
            if len(group) == width:
                yield group
                buckets[key] = []
    for group in buckets.values():
        if group:
            yield group


def length_order(compiled, window):
    """Sort only a bounded window while the next recordings are being read."""
    if type(window) is not int or window < 1:
        raise ValueError('positive length grouping window required')
    items = iter(compiled)
    while block := list(islice(items, window)):
        block.sort(key=lambda row: row[1][0]['inputs']['input_features'].shape[-1], reverse=True)
        yield from block


def concurrency(c):
    ramp = c.get('concurrent_scoring')
    if ramp is None:
        return 1
    from tools.reports.card_evaluation_replicas import replica_count
    if (not isinstance(ramp, dict) or set(ramp) != {'replicas_per_device', 'max_batch_audio_chunks',
            'pilot_recordings'} or 'formal' not in c or 'block_range' in c['formal']
            or any(type(ramp[k]) is not int or ramp[k] <= 0
                   for k in ('max_batch_audio_chunks', 'pilot_recordings'))
            or replica_count(ramp['replicas_per_device']) < 2):
        raise ValueError('concurrent TRAIN scoring requires a bounded whole-population plan')
    return ramp['replicas_per_device']


def replica_path(root, ordinal, lane):
    return root/f'worker-{ordinal}'/f'replica-{lane}'


def concurrent_offsets(c, offsets, lane, recordings):
    """Partition immutable shard blocks; spread the pilot over each source shard."""
    count = concurrency(c)
    size = c['formal']['recordings_per_block']
    quota = math.ceil(c['concurrent_scoring']['pilot_recordings'] / (size * len(c['devices']) * count))
    owned = list(offsets[lane::count])
    eligible = [offset for offset in owned if offset+size <= recordings]
    if len(eligible) < quota:
        raise ValueError('insufficient full blocks for the configured scoring pilot')
    pilot = [eligible[((2 * (i * count + lane) + 1) * len(eligible)) // (2 * quota * count)]
             for i in range(quota)]
    pilot_set = set(pilot)
    return pilot, [offset for offset in owned if offset not in pilot_set]


def scoring_lineage(c, ordinal):
    formal = c['formal']
    layout = ('fsdp-probability-parity', c['full_shards'][ordinal]['sha256']) if 'full_shards' in c else ()
    return content_key('full-train-scoring-v2', formal['policy_id'], formal['image'],
                       formal['checkpoint_sha256'],
                       formal['recordings_per_block'], c['full']['sha256'], c['source']['sha256'],
                       c['policy_sha256'], c['cards']['sha256'], c['training']['sha256'],
                       c['devices'], ordinal, c['widths'], c['streaming'], c['io_workers'],
                       c['prefetch_audio'], c['torch_threads'], c['max_model_len'], *layout,
                       *((c['seed'],) if c.get('bind_inference_seed') else ()),
                       *([formal['block_range']] if 'block_range' in formal else []),
                       *([c['concurrent_scoring']] if 'concurrent_scoring' in c else []),
                       *([c['scoring_reference']] if 'scoring_reference' in c else []))


def scoring_offsets(c, rows):
    formal = c['formal']
    offsets = range(0, rows, formal['recordings_per_block'])
    if 'block_range' in formal:
        lower, upper = formal['block_range']
        if type(lower) is not int or lower < 0 or (upper is not None
                and (type(upper) is not int or upper <= lower)):
            raise ValueError('invalid frozen block range')
        offsets = offsets[lower:upper]
        if not offsets:
            raise ValueError('empty frozen block assignment')
    return offsets


def read_block(path, lineage):
    block = json.loads(path.read_bytes())
    keys = {'lineage', 'sha256', 'payload'}
    if (set(block) not in (keys, keys | {'schema'}) or block['lineage'] != lineage
            or ('schema' in block and block['schema'] != 'jev-card-scoring-block-v2')):
        raise ValueError(f'scoring block identity differs: {path.name}')
    payload = block['payload']
    digest = content_key(payload)
    if 'schema' not in block and digest != block['sha256']:
        # The old writer sorted integer keys before JSON converted them to
        # strings. Reproduce that exact digest; never skip the check.
        sizes = payload['batch_sizes']
        if any(not k.isascii() or not k.isdecimal() or str(int(k)) != k for k in sizes):
            raise ValueError(f'noncanonical scoring batch size: {path.name}')
        digest = content_key(dict(payload, batch_sizes={int(k): n for k, n in sizes.items()}))
    if digest != block['sha256']:
        raise ValueError(f'scoring block identity differs: {path.name}')
    return payload


def commit_scoring_block(path, lineage, payload):
    payload = dict(payload, batch_sizes={str(k): v for k, v in payload['batch_sizes'].items()})
    commit_json(path, dict(schema='jev-card-scoring-block-v2', lineage=lineage,
                           sha256=content_key(payload), payload=payload))


def resume_config(c):
    if not c['formal'].get('resume_from'):
        return None
    return pinned(c['formal']['resume_config'])['macros']['NATIVE_SEAL_CONTAINER_CONFIG']


def wait_for_receipts(c, ready, workers, deadline):
    while not all(p.is_file() for p in ready):
        for i, process in enumerate(workers):
            if (code := process.poll()) is not None:
                if code != 0 or not ready[i].is_file():
                    raise RuntimeError(f'scoring worker {i} exited {code} without {ready[i].name}')
        if time.monotonic() >= deadline:
            raise TimeoutError('scoring receipts did not arrive before the configured deadline')
        time.sleep(c['formal']['coordination_seconds'])


def wait_for_admission(c, output, workers, deadline):
    from tools.reports.audit_card_scoring_benchmark import audit_admission

    count = concurrency(c)
    ready = [(replica_path(output, i, lane) if count > 1 else output/f'worker-{i}')/'panel-ready.json'
             for i in range(len(c['devices'])) for lane in range(count)]
    wait_for_receipts(c, ready, workers, deadline)
    admission = audit_admission(output, c, load_panel(c)[0],
                                scoring_settings(pinned(c['training']), c.get('scoring_reference'))['forward_rows'])
    if resume := c['formal'].get('resume_from'):
        previous = Path(resume)
        if any(previous.glob('worker-*/block-*.json')):
            observed = audit_admission(previous, resume_config(c), load_panel(c)[0],
                                       admission['baseline_rows'])
            if (json.loads((previous/'admission.json').read_bytes()) != observed
                    or observed['checkpoint_sha256'] != admission['checkpoint_sha256']):
                raise ValueError('resume requires the original all-replica admission')
    commit_json(output/'admission.json', admission)
    print(json.dumps(dict(event='admitted', **admission)), flush=True)


def worker(c, output, ordinal, lane=None):
    import torch
    from contextlib import nullcontext
    from multimodal_jev.ordered_prefetch import OrderedPrefetch
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache
    from tools.reports.evaluate_tpo_endpoints import start_model

    start = time.monotonic()
    panel, catalog = load_panel(c)
    train = pinned(c['training'])
    pid = panel['source_policy_id']
    initial_policy = pid
    formal = c.get('formal')
    count = concurrency(c)
    if ((count == 1 and lane is not None) or
            (count > 1 and (type(lane) is not int or not 0 <= lane < count))):
        raise ValueError('worker replica differs from frozen scoring topology')
    if formal:
        pid = formal['policy_id']
    devices = c['devices']
    if (train['source_policy_id'] != initial_policy or panel['source_training_sha256'] != c['training']['sha256']
            or file_hash(Path(c['policy'])/'MANIFEST.json') != c['policy_sha256']
            or checkpoint_identity(train['model'])[0] != train['base_id']):
        raise ValueError('baseline weights or training configuration changed')
    model_config = dict(model=train['model'], base_id=train['base_id'], contract=train['source_contract'],
                        snapshot=c['policy'], policy_id=pid, physical_devices=[devices[ordinal]],
                        max_model_len=c['max_model_len'],
                        reference_inference=dict(seed=c['seed'], torch_threads=c['torch_threads']),
                        process_device_scope=dict(kind='same_container_runtime_visibility_v1',
                                                  mounted_physical=devices, visible_ordinal=str(ordinal)))
    output.mkdir()
    provider, model, device, _, collate, handle = start_model(model_config, output)
    cache = {False: ReadOnlyEncodedAudioCache(train['encoded_audio_cache']),
             True: ReadOnlyEncodedAudioCache(c['synthetic_cache'])}
    rows = panel['rows'][ordinal::len(devices)]
    cards = [tuple(torch.tensor(card[k]) for k in ('prefix', 'suffix')) for card in catalog]

    def encode(row):
        inputs, _, _ = cache[row['synthetic']].read(row['binding'])
        return row['id'], list(render(inputs, cards, collate, row['id'], pid))

    streaming = c.get('streaming')
    if not streaming:
        with OrderedPrefetch(encode, rows, workers=c['io_workers'], ahead=c['prefetch_audio']) as queue:
            compiled = list(queue)
        ordered = list(length_order(compiled, len(rows)))
    startup_seconds = time.monotonic()-start

    def measure(width, legacy, *, settings=None, selected=None):
        selected = rows if selected is None else selected
        row_order = {r['id']: i for i, r in enumerate(selected)}
        if streaming:
            workers, ahead = (settings or (c['io_workers'], c['prefetch_audio']))
            stream = OrderedPrefetch(encode, selected, workers=workers, ahead=ahead)
        else:
            stream = nullcontext(compiled if legacy else ordered)
        with torch.inference_mode(), stream as queue:
            tick = time.monotonic()
            torch.npu.reset_peak_memory_stats()
            prob = [[None]*len(cards) for _ in selected]
            calls = Counter()
            records = length_order(queue, streaming['length_window']) if streaming and not legacy else queue
            rows_by_audio = (per_audio for _, per_audio in records)
            if lane is not None and not legacy:
                from tools.reports.evaluate_single_cards import bounded_batches
                groups = iter(bounded_batches(rows_by_audio, width, collate['audio_chunk_size'],
                                               c['concurrent_scoring']['max_batch_audio_chunks']))
            else:
                groups = iter(batches(rows_by_audio, width, group_audio=legacy, collate=collate))
            input_wait, forward, first_batch = 0., 0., None
            while True:
                ready = time.monotonic()
                group = next(groups, None)
                launch = time.monotonic()
                input_wait += launch-ready
                if group is None:
                    break
                if first_batch is None:
                    first_batch = launch-tick
                logits = provider.forward(model, group, device=device, dtype=torch.bfloat16,
                                          policy_id=pid, **collate)
                values = logits.sigmoid().float().flatten().cpu().tolist()
                forward += time.monotonic()-launch
                for item, value in zip(group, values, strict=True):
                    uid, index = item['request_id'].rsplit(':', 1)
                    prob[row_order[uid]][int(index)] = value
                calls[len(group)] += 1
            result = dict(seconds=time.monotonic()-tick, peak_bytes=torch.npu.max_memory_allocated(),
                          batch_sizes=dict(calls), records=[dict(id=r['id'], prob=p)
                                                           for r, p in zip(selected, prob, strict=True)])
            if streaming:
                result['input_timing'] = dict(workers=workers, ahead=ahead, first_batch=first_batch,
                                              input_wait=input_wait, forward=forward)
            return result

    if formal:
        from tools.reports.audit_card_scoring_benchmark import predictions, read, validate_block, validate_timing
        width, = c['widths']
        legacy_width = scoring_settings(train, c.get('scoring_reference'))['forward_rows']
        shared = output.parent if lane is not None else output
        if lane is None or lane == 0:
            commit_json(shared/'baseline.json', measure(legacy_width, True))
        commit_json(output/f'batch-{width}.json', measure(width, False))
        part = pinned(c['full_shards'][ordinal])
        shard = [dict(id=row['id'], intent=row['intent'], synthetic=row['synthetic'],
                      binding=restore_binding(row, part['acoustic_common'])) for row in part['rows']]
        if len(shard) != len(range(ordinal, panel['population'], len(devices))):
            raise ValueError('formal TRAIN shard size differs')
        size = formal['recordings_per_block']
        lineage = scoring_lineage(c, ordinal)
        resumed = Path(formal['resume_from'])/f'worker-{ordinal}' if formal.get('resume_from') else None
        prior_lineage = scoring_lineage(resume_config(c), ordinal) if resumed is not None else None
        offsets = scoring_offsets(c, len(shard))
        pilot, rest = concurrent_offsets(c, offsets, lane, len(shard)) if lane is not None else ([], offsets)
        offsets = [*pilot, *rest]
        names = [f'block-{offset//size:06d}.json' for offset in offsets]
        settings = (c['io_workers'], c['prefetch_audio'])
        saved_names = set()
        if resumed is not None:
            saved_names = {p.name for p in resumed.glob('block-*.json')}
            if saved_names and (read(resumed/'lineage.json') != {'sha256': prior_lineage}
                                or not saved_names <= {f'block-{off//size:06d}.json'
                                    for off in scoring_offsets(c, len(shard))}):
                raise ValueError('resume requires the same frozen shard, inputs and runtime')
            for offset, name in zip(offsets, names, strict=True):
                if name in saved_names:
                    validate_block((resumed/name, prior_lineage, [r['id'] for r in shard[offset:offset+size]],
                                    panel['labels'], width, settings))
        if lane is None or lane == 0:
            commit_json(shared/'lineage.json', dict(sha256=lineage))
        marker = dict(lineage=lineage, candidate_sha256=file_hash(output/f'batch-{width}.json'))
        if lane is None or lane == 0:
            marker['baseline_sha256'] = file_hash(shared/'baseline.json')
        commit_json(output/'panel-ready.json', marker)
        gate = shared.parent/'admission.json'
        deadline = start + c['worker_timeout_seconds']
        while not gate.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError('all-replica scoring admission did not open')
            time.sleep(formal['coordination_seconds'])
        admitted = read(gate)['workers'][ordinal]
        if (admitted if lane is None else admitted[lane]) != marker:
            raise ValueError('scoring gate belongs to a different panel')
        blocks = []
        tick = time.monotonic()
        pilot_start, pilot_end = None, None
        for index, (offset, name) in enumerate(zip(offsets, names, strict=True)):
            batch = shard[offset:offset+size]
            reused = name in saved_names
            if index < len(pilot) and not reused and pilot_start is None:
                pilot_start = time.monotonic()
            if reused:
                block = read_block(resumed/name, prior_lineage)
            else:
                block = measure(width, False, selected=batch)
            predictions(block, [r['id'] for r in batch], panel['labels'], width)
            validate_timing(block, settings)
            path = shared/name
            commit_scoring_block(path, lineage, block)
            blocks.append(dict(name=name, sha256=file_hash(path), reused=reused))
            if index < len(pilot) and not reused:
                pilot_end = time.monotonic()
            if index+1 == len(pilot):
                commit_json(output/'pilot.json', dict(blocks=blocks[:len(pilot)],
                                                     started=pilot_start, finished=pilot_end))
                while not (shared.parent/'pilot.json').is_file():
                    if time.monotonic() >= deadline:
                        raise TimeoutError('concurrent TRAIN pilot did not pass the scoring audit')
                    time.sleep(formal['coordination_seconds'])
            print(json.dumps(dict(event='block_committed', ordinal=ordinal, name=name,
                                  sha256=blocks[-1]['sha256'], recordings=len(batch),
                                  pairs=len(batch)*len(cards), reused=reused)), flush=True)
        commit_json(output/'result.json', dict(device=devices[ordinal],
                    rows=sum(min(size, len(shard)-offset) for offset in offsets),
                    cache_and_model_seconds=startup_seconds, scoring_seconds=time.monotonic()-tick,
                    blocks=sorted(blocks, key=lambda b: b['name']), lineage=lineage))
        return

    for width in c['widths']:
        arms = tuple(streaming['arms'].items()) if streaming else (('', None),)
        for name, settings in arms:
            try:
                result = measure(width, False, settings=settings)
            except RuntimeError as exc:
                if 'out of memory' not in str(exc).lower():
                    raise
                result = dict(oom=True, peak_bytes=torch.npu.max_memory_allocated())
            suffix = f'-{name}' if name else ''
            commit_json(output/f'batch-{width}{suffix}.json', result)
            if result.get('oom'):
                torch.npu.empty_cache()
    legacy_width = scoring_settings(train, c.get('scoring_reference'))['forward_rows']
    commit_json(output/'baseline.json', measure(legacy_width, True))
    commit_json(output/'result.json', dict(device=devices[ordinal], rows=len(rows),
                                          cache_and_model_seconds=startup_seconds))


def merge_replicas(c, output):
    """Preserve physical-shard receipts; independently show every launched replica."""
    from tools.reports.audit_card_scoring_benchmark import read
    count = concurrency(c)
    for ordinal, device in enumerate(c['devices']):
        paths = [replica_path(output, ordinal, lane)/'result.json' for lane in range(count)]
        results = [read(p) for p in paths]
        lineage = scoring_lineage(c, ordinal)
        if any(r['device'] != device or r['lineage'] != lineage for r in results):
            raise ValueError('concurrent scoring result has a different device or lineage')
        blocks = [b for r in results for b in r['blocks']]
        if len({b['name'] for b in blocks}) != len(blocks):
            raise ValueError('concurrent scoring blocks overlap')
        commit_json(output/f'worker-{ordinal}'/'result.json', dict(
            device=device, lineage=lineage, rows=sum(r['rows'] for r in results),
            cache_and_model_seconds=max(r['cache_and_model_seconds'] for r in results),
            scoring_seconds=max(r['scoring_seconds'] for r in results),
            blocks=sorted(blocks, key=lambda b: b['name']), replica_result_sha256=[file_hash(p) for p in paths]))


def host_memory():
    # cgroup counters for host RAM, independent of accelerator memory.
    root = Path('/sys/fs/cgroup')
    result = {}
    for name in ('memory.max', 'memory.current', 'memory.peak', 'memory.swap.current',
                 'memory.events', 'memory.stat', 'memory.pressure'):
        path = root/name
        if path.is_file():
            result[name] = path.read_text().strip()
    return result


def run(c):
    output = Path(c['output'])
    start = time.monotonic()
    deadline = start+c['worker_timeout_seconds']
    output.mkdir()
    capture_memory = 'formal' in c or c.get('capture_host_memory', False)
    memory_before = host_memory() if capture_memory else None
    if 'formal' in c:
        from tools.reports.audit_card_scoring_benchmark import audit_roster

        audit_roster(Path(c['full']['path']).parent, c, load_panel(c)[0])
        args = [sys.executable, '-B', '-m', 'torch.distributed.run', '--nnodes=1', '--node-rank=0',
                '--master-addr', c['formal']['fsdp_master_addr'],
                '--master-port', str(c['formal']['fsdp_master_port']),
                '--nproc-per-node', str(len(c['devices'])), '--max-restarts', '0',
                '-m', __spec__.name, 'reference', c['worker_config'], str(output)]
        subprocess.run(args, check=True, timeout=max(1, deadline-time.monotonic()))
    workers = []
    count = concurrency(c)
    with ThreadPoolExecutor(max_workers=len(c['devices'])*count) as pool:
        try:
            for index, _ in enumerate(c['devices']):
                shared = output/f'worker-{index}'
                if count > 1:
                    shared.mkdir()
                for lane in range(count):
                    path = replica_path(output, index, lane) if count > 1 else shared
                    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(index),
                               OMP_NUM_THREADS=str(c['torch_threads']))
                    scratch = Path(env['TMPDIR'])/path.relative_to(output)
                    scratch.mkdir(parents=True)
                    from tools.reports.card_evaluation_replicas import _CACHE_VARIABLES
                    env.update(dict.fromkeys(_CACHE_VARIABLES, str(scratch)))
                    args = [sys.executable, '-B', '-m', __spec__.name, 'worker',
                            c['worker_config'], str(path), str(index)]
                    if count > 1:
                        args.append(str(lane))
                    workers.append(subprocess.Popen(args, env=env))
            if 'formal' in c:
                wait_for_admission(c, output, workers, deadline)
            if count > 1:
                from tools.reports.audit_card_scoring_benchmark import audit_pilot
                ready = [replica_path(output, i, lane)/'pilot.json'
                         for i in range(len(c['devices'])) for lane in range(count)]
                wait_for_receipts(c, ready, workers, deadline)
                pilot = audit_pilot(output, c, pinned(c['full']))
                commit_json(output/'pilot.json', pilot)
                print(json.dumps(dict(event='scoring_pilot_audited', **pilot)), flush=True)
            exits = {pool.submit(p.wait): ordinal for ordinal, p in enumerate(workers)}
            for done in as_completed(exits, timeout=max(0, deadline-time.monotonic())):
                if code := done.result():
                    raise RuntimeError(f'scoring worker {exits[done]} exited {code}')
        finally:
            active = [p for p in workers if p.poll() is None]
            for p in active:
                p.terminate()
            for p in active:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
    if count > 1:
        merge_replicas(c, output)
    receipt = dict(devices=c['devices'], panel_sha256=c['panel']['sha256'],
                   workers=[json.loads((output/f'worker-{i}'/'result.json').read_bytes())
                            for i in range(len(c['devices']))])
    if capture_memory:
        receipt['host_memory'] = dict(before=memory_before, after=host_memory())
    if 'formal' in c:
        receipt.update(full_sha256=c['full']['sha256'], policy_id=c['formal']['policy_id'],
                       policy_sha256=c['policy_sha256'], worker_wall_seconds=time.monotonic()-start,
                       admission_sha256=file_hash(output/'admission.json'))
    commit_json(output/'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('worker', 'reference'))
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('ordinal', type=int, nargs='?')
    parser.add_argument('lane', type=int, nargs='?')
    args = parser.parse_args()
    raw = json.loads(args.config.read_bytes())
    config = raw['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    if args.action == 'reference':
        from tools.reports.card_scoring_fsdp_reference import reference
        reference(config, args.output)
    else:
        worker(config, args.output, args.ordinal, args.lane)


if __name__ == '__main__':
    main()
