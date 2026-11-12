#!/usr/bin/env python3
"""Input-only fixed-menu MAC-SLU Span processor probe, without model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def run(config):
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.mac_slu_projection import input_request
    from multimodal_jev.span_contract import encode_span_request, span_labels
    torch.set_num_threads(2)
    for name, digest in config['pins'].items():
        if file_hash(name) != digest: raise ValueError('pinned source changed')
    release = Path(config['release'])
    manifest = json.loads((release/'MANIFEST.json').read_text())
    for name, digest in manifest['files'].items():
        # Do not read targets in this input-only probe.
        if name == 'train.targets.jsonl': continue
        if file_hash(release/name) != digest: raise ValueError('input release changed')
    rows = [json.loads(line) for line in (release/'train.inputs.jsonl').open()]
    usable = [r for r in rows if r['audio'] and r['audio']['native_span_audio_compatible']]
    selected = sorted(usable, key=lambda r: hashlib.sha256(('processor-v1:'+r['id']).encode()).hexdigest())[:3]
    if len(selected) != 3: raise ValueError('three compatible input records required')
    bank = json.loads((release/'bank.json').read_text())
    # Fixed public domains with different question counts; no gold filtering.
    pairs = [('地图', '导航'), ('音乐', '播放音乐'), ('影视', '播放影视')]
    processor = Qwen3ASRProcessor.from_pretrained(config['model'], local_files_only=True)
    records = []
    for row, (domain, intent) in zip(selected, pairs, strict=True):
        audio = row['audio']
        if file_hash(audio['path']) != audio['sha256']: raise ValueError('audio changed')
        waveform, receipt = load_audio(audio['path'], return_receipt=True)
        request = input_request(audio['path'], bank, domain, intent)
        batch, positions = encode_span_request(processor, request, waveform)
        mapping = span_labels(processor.tokenizer, request.questions)
        if any(m['values'] != (False, True) or len(m['token_ids']) != 2 for m in mapping):
            raise ValueError('Boolean candidates differ')
        records.append(dict(id=row['id'], request=request.model_input(), audio_sha256=audio['sha256'],
            audio_processing=receipt, spans=len(request.questions), input_tokens=batch['input_ids'].shape[-1],
            positions=positions.tolist(), candidates=[m['token_ids'] for m in mapping],
            input_shapes={k:list(v.shape) for k,v in batch.items()}))
    return dict(passed=True, records=records, target_files_opened=False, model_weights_loaded=False,
                scope='Three deterministic TRAIN inputs with fixed domains; CPU tokenization/audio expansion '
                      'and ordered native Span positions only, no prediction, parity or task metrics.',
                language_caveat='Existing frozen ASR Span template includes English system prefix; '
                                'Chinese schema is preserved verbatim, no Chinese task-accuracy claim.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(exist_ok=False)
    try:
        result = run(json.loads(a.config.read_text()))
        atomic_json(a.output/'result.json', result)
        atomic_json(a.output/'terminal.json', dict(exit_code=0, finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time()))
        raise
