"""Benchmark single-card source TRAIN scoring using independent NPU replicas."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain, islice
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

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
            or train['epoch_mining']['card_arm'] != c['card_arm']
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
        commit_json(output.with_name('full.json'), dict(panel, rows=[
            dict(id=row['id'], intent=row['intent'], synthetic=row['id'] in synth_ids,
                 binding=acoustic_binding(row, spec)) for row in full]))
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
        # The legacy path consumes the 68 candidates for each recording before advancing.
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


def scoring_lineage(c, ordinal):
    formal = c['formal']
    return content_key('full-train-scoring-v1', formal['policy_id'], formal['image'],
                       formal['recordings_per_block'], c['full']['sha256'], c['source']['sha256'],
                       c['policy_sha256'], c['cards']['sha256'], c['training']['sha256'],
                       c['devices'], ordinal, c['widths'], c['streaming'], c['io_workers'],
                       c['prefetch_audio'], c['torch_threads'], c['max_model_len'])


def read_block(path, lineage):
    block = json.loads(path.read_bytes())
    if (block['lineage'] != lineage or
            content_key(block['payload']) != block['sha256']):
        raise ValueError(f'scoring block identity differs: {path.name}')
    return block['payload']


def worker(c, output, ordinal):
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
            groups = iter(batches((per_audio for _, per_audio in records), width,
                                  group_audio=legacy, collate=collate))
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
        from tools.reports.audit_card_scoring_benchmark import predictions
        full = pinned(c['full'])
        if (len(full['rows']) != panel['population'] or full['labels'] != panel['labels']
                or full['cache_proof_sha256'] != panel['cache_proof_sha256']):
            raise ValueError('formal TRAIN roster differs')
        width, = c['widths']
        legacy_width = train['epoch_mining']['forward_rows']
        baseline, candidate = measure(legacy_width, True), measure(width, False)
        ids = [r['id'] for r in rows]
        r, _ = predictions(baseline, ids, panel['labels'], legacy_width)
        p, sizes = predictions(candidate, ids, panel['labels'], width)
        if r != p or not sizes.get(width):
            raise ValueError('endpoint scoring admission requires a full batch and exact probabilities')
        commit_json(output/'baseline.json', baseline)
        commit_json(output/f'batch-{width}.json', candidate)
        shard = full['rows'][ordinal::len(devices)]
        size = formal['recordings_per_block']
        lineage = scoring_lineage(c, ordinal)
        resumed = Path(formal['resume_from'])/f'worker-{ordinal}' if formal.get('resume_from') else None
        if resumed is not None and resumed.exists() and (
                json.loads((resumed/'lineage.json').read_bytes()) != {'sha256': lineage}):
            raise ValueError('resume requires identical frozen inputs and runtime')
        commit_json(output/'lineage.json', dict(sha256=lineage))
        blocks = []
        tick = time.monotonic()
        for i, offset in enumerate(range(0, len(shard), size)):
            batch = shard[offset:offset+size]
            name = f'block-{i:06d}.json'
            saved = resumed/name if resumed else None
            reused = saved is not None and saved.is_file()
            if reused:
                block = read_block(saved, lineage)
            else:
                block = measure(width, False, selected=batch)
            predictions(block, [r['id'] for r in batch], panel['labels'], width)
            commit_json(output/name, dict(lineage=lineage, sha256=content_key(block), payload=block))
            blocks.append(dict(name=name, sha256=file_hash(output/name), reused=reused))
        commit_json(output/'result.json', dict(device=devices[ordinal], rows=len(shard),
                    cache_and_model_seconds=startup_seconds, scoring_seconds=time.monotonic()-tick,
                    blocks=blocks, lineage=lineage))
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
    legacy_width = train['epoch_mining']['forward_rows']
    commit_json(output/'baseline.json', measure(legacy_width, True))
    commit_json(output/'result.json', dict(device=devices[ordinal], rows=len(rows),
                                          cache_and_model_seconds=startup_seconds))


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
    output.mkdir()
    capture_memory = 'formal' in c or c.get('capture_host_memory', False)
    memory_before = host_memory() if capture_memory else None
    workers = []
    with ThreadPoolExecutor(max_workers=len(c['devices'])) as pool:
        try:
            for index, _ in enumerate(c['devices']):
                path = output/f'worker-{index}'
                env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(index),
                           OMP_NUM_THREADS=str(c['torch_threads']))
                scratch = Path(env['TMPDIR'])/path.name
                scratch.mkdir()
                for name in ('TMPDIR', 'HF_HOME', 'XDG_CACHE_HOME', 'ASCEND_WORK_PATH', 'ASCEND_CACHE_PATH'):
                    env[name] = str(scratch)
                args = [sys.executable, '-B', '-m', __spec__.name, 'worker',
                        c['worker_config'], str(path), str(index)]
                workers.append(subprocess.Popen(args, env=env))
            exits = {pool.submit(p.wait): ordinal for ordinal, p in enumerate(workers)}
            for done in as_completed(exits, timeout=c['worker_timeout_seconds']):
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
    receipt = dict(devices=c['devices'], panel_sha256=c['panel']['sha256'],
                   workers=[json.loads((output/f'worker-{i}'/'result.json').read_bytes())
                            for i in range(len(workers))])
    if capture_memory:
        receipt['host_memory'] = dict(before=memory_before, after=host_memory())
    if 'formal' in c:
        receipt.update(full_sha256=c['full']['sha256'], policy_id=c['formal']['policy_id'],
                       policy_sha256=c['policy_sha256'], worker_wall_seconds=time.monotonic()-start)
    commit_json(output/'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('worker',))
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('ordinal', type=int)
    args = parser.parse_args()
    raw = json.loads(args.config.read_bytes())
    worker(raw['macros']['NATIVE_SEAL_CONTAINER_CONFIG'], args.output, args.ordinal)


if __name__ == '__main__':
    main()
