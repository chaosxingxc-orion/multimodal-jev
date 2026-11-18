"""CPU-only clean TEST encoding and independent cache receipt verification.

Uses the frozen input roster and processor; never opens target annotations,
loads model weights, selects a policy, or changes a training dependency.
"""
import argparse
import json
import os
import time
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_decision_contract import render_context_request
from multimodal_jev.slurp_training import content_hash


def reconstruct(row, definitions):
    if (set(row) != {'id', 'audio', 'audio_sha256', 'source_group',
                    'request_sha256', 'rendered_sha256'}
            or row['id'] != 'slurp:' + Path(row['audio']).name):
        raise ValueError('canonical label-free input required')
    request = compile_candidates(row['audio'], **definitions)
    system, text = render_context_request(request)
    if (request.version != 'jev-context-exclusive_event-v1'
            or len(request.questions) != 60
            or content_hash(request.model_input()) != row['request_sha256']
            or content_hash(dict(system=system, text=text)) != row['rendered_sha256']):
        raise ValueError('frozen request/render identity differs')
    return request


def authenticate(config):
    for name, digest in config['pins'].items():
        if file_hash(name) != digest:
            raise ValueError('source/input dependency changed: ' + name)
    root = Path(config['inputs'])
    if file_hash(root / 'MANIFEST.json') != config['input_manifest_sha256']:
        raise ValueError('input manifest changed')
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    if (manifest['schema'] != 'slurp-clean-context-test-inputs-v1'
            or manifest['recordings'] != 12991 or manifest['source_groups'] != 2947
            or manifest['optimization_allowed'] is not False
            or manifest['target_labels_opened'] is not False
            or set(manifest['files']) != {'inputs.jsonl', 'definitions.json'}):
        raise ValueError('complete accepted clean TEST input manifest required')
    for name, digest in manifest['files'].items():
        if file_hash(root / name) != digest:
            raise ValueError('input file changed')
    definitions = json.loads((root / 'definitions.json').read_text())
    if (set(definitions) != {'candidates', 'requirements', 'semantics'}
            or definitions['semantics'] != 'exclusive_event'
            or [c['id'] for c in definitions['candidates']] != manifest['candidate_ids']
            or len(set(manifest['candidate_ids'])) != 60):
        raise ValueError('unchanged complete TRAIN-defined candidate menu required')
    with (root / 'inputs.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    if (len(rows) != 12991 or len({r['id'] for r in rows}) != 12991
            or len({r['source_group'] for r in rows}) != 2947):
        raise ValueError('full unique accepted population required')
    # Validate every request before any processor execution; no source_files
    # annotation paths in the provenance manifest are opened here.
    for row in rows:
        reconstruct(row, definitions)
    return rows, definitions


def runtime(config, out):
    import torch
    # Resolve transformers' lazy module in the main thread before prefetch.
    # Concurrent first access from worker-local factories can race on import.
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.cached_context_preparation import (
        CachedContextPreparer, processor_cache_identity)
    from multimodal_jev.encoded_audio_cache import EncodedAudioCache
    if list(Path('/dev').glob('davinci[0-9]*')):
        raise ValueError('CPU-only preflight requires no NPU devices')
    torch.set_num_threads(1)
    def factory():
        proc = Qwen3ASRProcessor.from_pretrained(
            Path(config['source_policy']) / 'processor', local_files_only=True)
        vocabulary = proc.tokenizer.get_vocab()
        proc.tokenizer.get_vocab = lambda: dict(vocabulary)
        return proc
    identity = processor_cache_identity(config['source_policy'])
    cache = EncodedAudioCache(out / 'cache')
    prepare = CachedContextPreparer(factory, cache, processor_sha256=identity,
        decision_token_id=151706, audio_token_id=151676, max_tokens=4096)
    return cache, prepare, identity


def run(config, out):
    from multimodal_jev.encoded_audio_cache import key_for
    from multimodal_jev.local_checkpoint import commit_json
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.ordered_prefetch import OrderedPrefetch
    from multimodal_jev.span_request_identity import tensor_identity
    out = Path(out)
    require_training_storage(out)
    rows, definitions = authenticate(config)
    cache, prepare, identity = runtime(config, out)
    def encode(row):
        request = reconstruct(row, definitions)
        value = prepare((request, row['audio_sha256']))
        key = key_for(prepare.identity(request, row['audio_sha256']))
        return dict(id=row['id'], audio_sha256=row['audio_sha256'],
            request_sha256=row['request_sha256'], rendered_sha256=row['rendered_sha256'],
            cache_key=key, cache_sha256=file_hash(cache.root / (key + '.pt')),
            tensors={k: tensor_identity(v) for k, v in
                     dict(value.inputs, positions=value.positions).items()},
            positions=value.positions.tolist(), tokens=value.inputs['input_ids'].shape[-1])
    start = time.monotonic()
    maximum = 0
    with (out / 'inputs.jsonl').open('x') as stream:
        with OrderedPrefetch(encode, rows, workers=2, ahead=4) as prepared:
            for index, row in enumerate(prepared, 1):
                stream.write(json.dumps(row, separators=(',', ':')) + '\n')
                maximum = max(maximum, row['tokens'])
                if index % 256 == 0:
                    stream.flush()
                    atomic_json(out / 'progress.json', dict(completed=index,
                        total=len(rows), max_tokens=maximum, seconds=time.monotonic()-start))
        stream.flush()
        os.fsync(stream.fileno())
    result = dict(schema='slurp-clean-context-test-processor-v1', recordings=len(rows),
        decisions=60, max_tokens=maximum, context_limit=4096, processor_sha256=identity,
        cache_root=str(cache.root), input_manifest_sha256=config['input_manifest_sha256'],
        receipts_sha256=file_hash(out / 'inputs.jsonl'), seconds=time.monotonic()-start,
        scope='Full audio hash/decode/processor/marker check; no model forward or TEST score.')
    commit_json(out / 'MANIFEST.json', result)
    return result


def verify(config, out):
    import torch
    from multimodal_jev.encoded_audio_cache import key_for
    from multimodal_jev.span_request_identity import tensor_identity
    out = Path(out)
    rows, definitions = authenticate(config)
    manifest = json.loads((out / 'MANIFEST.json').read_text())
    if (manifest['schema'] != 'slurp-clean-context-test-processor-v1'
            or manifest['recordings'] != 12991 or manifest['decisions'] != 60
            or manifest['context_limit'] != 4096
            or manifest['cache_root'] != str(out / 'cache')
            or manifest['input_manifest_sha256'] != config['input_manifest_sha256']
            or file_hash(out / 'inputs.jsonl') != manifest['receipts_sha256']):
        raise ValueError('complete processor output manifest required')
    cache, prepare, identity = runtime(config, out)
    if identity != manifest['processor_sha256']:
        raise ValueError('processor identity differs')
    maximum = 0
    with (out / 'inputs.jsonl').open() as stream:
        for source, line in zip(rows, stream, strict=True):
            row = json.loads(line)
            for key in ('id', 'audio_sha256', 'request_sha256', 'rendered_sha256'):
                if row[key] != source[key]:
                    raise ValueError('output roster identity/order differs')
            request = reconstruct(source, definitions)
            binding = prepare.identity(request, source['audio_sha256'])
            if (row['cache_key'] != key_for(binding) or
                    file_hash(cache.root / (row['cache_key']+'.pt')) != row['cache_sha256']):
                raise ValueError('cache file identity differs')
            value = cache.read(binding)
            if value is None:
                raise ValueError('cache entry missing')
            inputs, positions, _ = value
            hashes = {k: tensor_identity(v) for k, v in dict(inputs, positions=positions).items()}
            actual = (inputs['input_ids'][0] == 151706).nonzero().flatten()
            if (hashes != row['tensors'] or positions.tolist() != row['positions']
                    or positions.shape != (1, 60) or not torch.equal(positions[0], actual)
                    or inputs['input_features'].dtype != torch.bfloat16
                    or inputs['input_ids'].shape[-1] != row['tokens'] or row['tokens'] > 4096):
                raise ValueError('cached execution tensors differ')
            maximum = max(maximum, row['tokens'])
    if maximum != manifest['max_tokens']:
        raise ValueError('maximum token length differs')
    return dict(recordings=len(rows), max_tokens=maximum, processor_sha256=identity,
        output_manifest_sha256=file_hash(out / 'MANIFEST.json'),
        scope='Fresh-process full receipt/cache verification; no independent processor recomputation or model inference.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    if not args.verify:
        out.mkdir(exist_ok=False)
    print(json.dumps((verify if args.verify else run)(config, out), indent=2))
