#!/usr/bin/env python3
"""CPU real-audio check of prospective candidate interventions, without labels."""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash


def run(config, output):
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.span_contract import encode_span_request, span_labels
    from multimodal_jev.span_request_identity import tensor_identity
    from multimodal_jev.span_candidate_intervention import VARIANTS, from_input, intervene
    if list(Path('/dev').glob('davinci[0-9]*')): raise ValueError('CPU only')
    for path, digest in config['pins'].items():
        if file_hash(path) != digest: raise ValueError('pinned input changed: ' + path)
    if config['pins'].get(config['inputs']) != file_hash(config['inputs']):
        raise ValueError('frozen source inputs required')
    source = json.loads(Path(config['inputs']).read_text())
    rows = source['records']
    if not source['passed'] or len(rows) != 178 or len({r['id'] for r in rows}) != 178:
        raise ValueError('complete accepted input population required')
    grouped = {}
    for row in rows:
        arm = row['arm']
        if arm not in ('D', 'Q'): raise ValueError('D/Q input required')
        group = grouped.setdefault(row['recording_id'], {})
        if arm in group: raise ValueError('duplicate source/arm')
        group[arm] = row
    if len(grouped) != 89 or any(set(g) != {'D', 'Q'} for g in grouped.values()):
        raise ValueError('89 paired sources required')
    torch.set_num_threads(2)
    processor = Qwen3ASRProcessor.from_pretrained(config['model'], local_files_only=True)
    records = []
    for recording, group in grouped.items():
        requests = {arm: from_input(row['request']) for arm, row in group.items()}
        if (requests['D'].questions != (requests['Q'].questions[-1],)
                or requests['D'].audio != requests['Q'].audio
                or requests['D'].requirements != requests['Q'].requirements):
            raise ValueError('matched final, audio and requirements required')
        audio_path = requests['D'].audio
        if any(file_hash(audio_path) != row['audio_sha256'] for row in group.values()):
            raise ValueError('audio identity changed')
        audio, audio_receipt = load_audio(audio_path, return_receipt=True)
        for variant in VARIANTS:
            for arm in ('D', 'Q'):
                request, binding = intervene(requests[arm], variant)
                batch, positions = encode_span_request(processor, request, audio)
                features = {k: tensor_identity(batch[k]) for k in ('input_features', 'input_features_mask')}
                if any(features[k] != group[arm][k] for k in features):
                    raise ValueError('audio features differ from frozen input')
                length = batch['input_ids'].shape[-1]
                if length > 4096: raise ValueError('untruncated context exceeds limit')
                token_identity = tensor_identity(batch['input_ids'])
                if variant == 'native' and (token_identity != group[arm]['input_ids']
                        or positions.tolist() != group[arm]['positions']):
                    raise ValueError('native encoding differs from frozen input')
                records.append(dict(id=recording+'::'+arm+'::'+variant, recording_id=recording,
                    arm=arm, variant=variant, request=request.model_input(), binding=binding,
                    audio_sha256=group[arm]['audio_sha256'], audio_processing=audio_receipt,
                    input_tokens=length, positions=positions.tolist(),
                    candidate_mapping=span_labels(processor.tokenizer, request.questions),
                    input_ids=token_identity, **features))
        atomic_json(output/'progress.json', dict(requests=len(records), total=712, updated_unix=time.time()))
    return dict(passed=True, recordings=89, requests=len(records), records=records,
        maximum_input_tokens=max(r['input_tokens'] for r in records),
        scope='Input-only CPU candidate-order and symbol-binding intervention; no model or labels. '
              'Display order and symbol binding change together; not isolated token causality.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json', run(json.loads(a.config.read_text()), a.output))
        atomic_json(a.output/'terminal.json', dict(exit_code=0))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json', dict(exit_code=1, error=repr(exc))); raise
