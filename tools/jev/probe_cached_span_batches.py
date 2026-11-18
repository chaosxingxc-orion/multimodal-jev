#!/usr/bin/env python3
"""Compare serial preparation with CPU cache/prefetch on frozen schedule batches.

No artificial NPU delay: these timings measure input service, not overlapped
training. Independent fresh encodings are checked outside the measured windows.
"""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def run(config, output):
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.cached_span_preparation import CachedSpanPreparer, bind_instruction_slot
    from multimodal_jev.cached_instruction_batches import CachedInstructionBatches
    from multimodal_jev.encoded_audio_cache import EncodedAudioCache
    from multimodal_jev.ordered_prefetch import OrderedPrefetch
    from multimodal_jev.slurp_official_training import load, example
    from multimodal_jev.slurp_training import content_hash
    from multimodal_jev.span_contract import encode_span_request, span_labels
    from multimodal_jev.span_request_identity import tensor_identity

    if list(Path('/dev').glob('davinci[0-9]*')):
        raise ValueError('CPU-only probe required')
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('pinned file changed: '+path)
    torch.set_num_threads(1)
    rows, task, _ = load(config['release'], config['release_sha256'], 'learning', purpose='optimize')
    population = {row['id']: row for row in rows}
    schedule = json.loads(Path(config['epoch_file']).read_text())['batches']
    factory = lambda: Qwen3ASRProcessor.from_pretrained(config['model'], local_files_only=True)
    processor = factory()
    preparer = CachedSpanPreparer(factory, EncodedAudioCache(output/'cache'),
                                 processor_sha256=content_hash(config['pins']))
    results = []
    for batch_index in config['batch_indices']:
        slots = schedule[batch_index]
        if len(slots) != 128:
            raise ValueError('complete global128 batch required')
        compiled = []
        for slot in slots:
            if (slot['view'], slot['variant']) != ('direct', 'native'):
                raise ValueError('fixed direct task required')
            row = population[slot['id']]
            request, target, _ = example(row, task, slot['view'], slot['variant'])
            compiled.append((request, target, row['audio_sha256']))
        jobs = [(request, digest) for request, _, digest in compiled]
        start = time.monotonic()
        baseline = []
        for request, _, digest in compiled:
            if file_hash(request.audio) != digest:
                raise ValueError('source audio changed')
            audio, receipt = load_audio(request.audio, return_receipt=True)
            inputs, positions = encode_span_request(processor, request, audio)
            candidates = [m['token_ids'] for m in span_labels(processor.tokenizer, request.questions)]
            baseline.append((inputs, positions, receipt, candidates))
        serial_seconds = time.monotonic()-start
        start = time.monotonic()
        cold = [preparer(job) for job in jobs]
        cache_build_seconds = time.monotonic()-start
        timings = []
        fingerprints = []
        for mode in ('warm_serial', 'warm_prefetch2', 'warm_prefetch2', 'warm_serial'):
            start = time.monotonic()
            if mode == 'warm_serial':
                prepared = [preparer(job) for job in jobs]
            else:
                with OrderedPrefetch(preparer, jobs, workers=2, ahead=8) as stream:
                    prepared = list(stream)
            timings.append(dict(mode=mode, seconds=time.monotonic()-start,
                                hits=sum(item.cache_hit for item in prepared)))
            if not all(item.cache_hit for item in prepared):
                raise ValueError('warm cache missed')
            for index, (item, reference) in enumerate(zip(prepared, baseline, strict=True)):
                inputs, positions, receipt, candidates = reference
                if (item.inputs.keys() != inputs.keys() or
                        any(not torch.equal(item.inputs[k], v) for k, v in inputs.items()) or
                        not torch.equal(item.positions, positions) or item.audio_receipt != receipt or
                        item.candidates != candidates):
                    raise ValueError('fresh versus cached input differs')
                request, target, _ = compiled[index]
                bound = bind_instruction_slot(item, request, target, slots[index], epoch=0,
                    batch_index=batch_index, index=index, policy_id='current-probe-policy')
                if (bound['targets'].tolist() != [target['targets']] or
                        bound['request_weight'] != slots[index]['weight'] or
                        bound['policy_id'] != 'current-probe-policy'):
                    raise ValueError('late supervision binding changed')
            if not fingerprints:
                fingerprints = [dict(id=slot['id'], padding=slot['padding'], weight=slot['weight'],
                    inputs={k:tensor_identity(v) for k, v in ref[0].items()},
                    positions=tensor_identity(ref[1]), receipt_sha256=content_hash(ref[2]),
                    candidates=ref[3]) for slot, ref in zip(slots, baseline, strict=True)]
        record = dict(batch_index=batch_index, rows=len(slots),
            real_rows=sum(not slot['padding'] for slot in slots), serial_prepare_seconds=serial_seconds,
            cold_cache_build_seconds=cache_build_seconds, cold_hits=sum(item.cache_hit for item in cold),
            warm=timings, fingerprints=fingerprints)
        # Real task compiler and consumer integration, including a nonzero
        # restored cursor at the tail. Timed separately from cache microbench.
        ordered_schedule = [(0, index, value) for index, value in enumerate(schedule)]
        with CachedInstructionBatches(ordered_schedule, population, task, preparer,
                start_update=batch_index, stop_update=batch_index+1, context_limit=4096) as pipeline:
            start = time.monotonic()
            items, metadata = pipeline.consume(batch_index, 'pipeline-current-policy')
            record['batch_pipeline_seconds'] = time.monotonic()-start
            for index, (item, meta, ref) in enumerate(zip(items, metadata, baseline, strict=True)):
                request, target, _ = compiled[index]
                if (any(not torch.equal(item['inputs'][k], v) for k, v in ref[0].items()) or
                        not torch.equal(item['positions'], ref[1]) or item['candidates'] != ref[3] or
                        item['targets'].tolist() != [target['targets']] or
                        item['supervised'].tolist() != [target['supervised']] or
                        item['span_weights'].tolist() != [target['span_weights']] or
                        item['request_weight'] != slots[index]['weight'] or
                        item['real_request'] != (not slots[index]['padding']) or
                        item['policy_id'] != 'pipeline-current-policy' or
                        meta['audio_receipt'] != ref[2] or
                        meta['input_sha256'] != content_hash(request.model_input())):
                    raise ValueError('batch pipeline changed original input or supervision')
            if len({item['request_id'] for item in items}) != 128:
                raise ValueError('padding slot identity duplicated')
            record['batch_pipeline_exact'] = True
        results.append(record)
        atomic_json(output/f'batch-{batch_index}.json', record)
        atomic_json(output/'progress.json', dict(completed=len(results)))
    files = sorted((output/'cache').glob('*.pt'))
    return dict(passed=True, batches=results, cache_entries=len(files),
        cache_bytes=sum(p.stat().st_size for p in files), training_dispatch_allowed=False,
        scope='Two full scheduled CPU input batches including zero-weight tail; no NPU overlap, '
              'training speedup, full-population throughput or restored training claimed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    try:
        atomic_json(args.output/'result.json', run(json.loads(args.config.read_text()), args.output))
        atomic_json(args.output/'terminal.json', dict(exit_code=0, finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time()))
        raise
