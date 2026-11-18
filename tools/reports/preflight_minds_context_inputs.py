"""CPU processor preparation for the frozen, full native-descriptor population."""
import json
import os
import time
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.prepare_minds_context_inputs import reconstruct


def authenticate(config):
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:raise ValueError('Frozen CPU dependency changed: '+path)
    root = Path(config['inputs'])
    if file_hash(root/'MANIFEST.json') != config['input_manifest_sha256']:
        raise ValueError('Input manifest changed')
    manifest = json.loads((root/'MANIFEST.json').read_text())
    proof = json.loads(Path(config['conversion_verification']).read_text())
    if (manifest['schema'] != 'minds14-native-descriptor-16k-inputs-v1'
            or manifest['recordings'] != 8168 or manifest['decisions'] != 14
            or manifest['training_dispatch_allowed'] or manifest['evaluation_dispatch_allowed']
            or proof['manifest_sha256'] != config['input_manifest_sha256']
            or proof['recordings'] != 8168
            or not all(proof[k] for k in ('complete_population_and_order',
                'complete_waveforms_recomputed_exact','full_duration_preserved','task_text_unchanged'))):
        raise ValueError('Independently verified full resampling required')
    for name, digest in manifest['files'].items():
        if file_hash(root/name) != digest:raise ValueError('Input package changed')
    definitions = json.loads((root/'definitions.json').read_text())
    if (definitions['semantics'] != 'exclusive_event'
            or [c['id'] for c in definitions['candidates']] != manifest['candidate_ids']
            or len(set(manifest['candidate_ids'])) != 14):
        raise ValueError('Complete shared menu required')
    rows = [json.loads(line) for line in (root/'inputs.jsonl').read_text().splitlines()]
    if len(rows) != 8168 or len({r['id'] for r in rows}) != 8168:
        raise ValueError('Full unique population required')
    for row in rows:reconstruct(row, definitions)
    return rows, definitions


def run(config, out):
    import torch
    import soundfile as sf
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.cached_context_preparation import CachedContextPreparer, processor_cache_identity
    from multimodal_jev.encoded_audio_cache import EncodedAudioCache, key_for
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.ordered_prefetch import OrderedPrefetch
    from multimodal_jev.span_request_identity import tensor_identity
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('No NPU devices allowed')
    torch.set_num_threads(1)
    out = Path(out);require_training_storage(out)
    rows, definitions = authenticate(config)
    identity = processor_cache_identity(config['source_policy'])
    cache = EncodedAudioCache(out/'cache')
    def factory():
        proc = Qwen3ASRProcessor.from_pretrained(Path(config['source_policy'])/'processor',local_files_only=True)
        vocab = proc.tokenizer.get_vocab();proc.tokenizer.get_vocab = lambda:dict(vocab)
        return proc
    prepare = CachedContextPreparer(factory,cache,processor_sha256=identity,
        decision_token_id=151706,audio_token_id=151676,max_tokens=4096)
    def encode(row):
        request = reconstruct(row,definitions)
        value = prepare((request,row['audio_sha256']))
        receipt = value.audio_receipt
        samples = sf.info(row['audio']).frames
        if (receipt['input_samples'] != samples or receipt['output_samples'] != samples
                or receipt['input_waveform_sha256'] != receipt['output_waveform_sha256']
                or receipt['retained_original_interval'] != [0,samples]):
            raise ValueError('Shared raw preprocessing changed evidence')
        features_mask = value.inputs['input_features_mask']
        frames = int(features_mask.sum())
        # Record complete feature-mask evidence. Processor-level duration
        # coverage must be checked against its pinned feature extractor.
        key = key_for(prepare.identity(request,row['audio_sha256']))
        return dict(id=row['id'],audio_sha256=row['audio_sha256'],request_sha256=row['request_sha256'],
            rendered_sha256=row['rendered_sha256'],cache_key=key,
            cache_sha256=file_hash(cache.root/(key+'.pt')),
            tensors={k:tensor_identity(v) for k,v in dict(value.inputs,positions=value.positions).items()},
            positions=value.positions.tolist(),tokens=value.inputs['input_ids'].shape[-1],
            audio_samples=samples,feature_frames=frames,audio_receipt=receipt)
    start = time.monotonic();max_tokens=0;max_samples=0
    with (out/'inputs.jsonl').open('x') as stream:
        with OrderedPrefetch(encode,rows,workers=2,ahead=4) as prepared:
            for index,row in enumerate(prepared,1):
                stream.write(json.dumps(row,separators=(',',':'))+'\n')
                max_tokens=max(max_tokens,row['tokens']);max_samples=max(max_samples,row['audio_samples'])
                if index%256==0:
                    stream.flush()
                    atomic_json(out/'progress.json',dict(completed=index,total=len(rows),
                        max_tokens=max_tokens,seconds=time.monotonic()-start))
        stream.flush();os.fsync(stream.fileno())
    result=dict(schema='minds14-context-processor-v1',recordings=len(rows),decisions=14,
        max_tokens=max_tokens,max_samples=max_samples,context_limit=4096,processor_sha256=identity,
        input_manifest_sha256=config['input_manifest_sha256'],receipts_sha256=file_hash(out/'inputs.jsonl'),
        seconds=time.monotonic()-start,cache_root=str(cache.root),
        scope='Full CPU input encoding only; independent cache and processor duration audit pending; no inference.')
    commit_json(out/'MANIFEST.json',result)
    return result


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False)
    print(json.dumps(run(json.loads(Path(a.config).read_text()),out),indent=2))
