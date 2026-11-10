#!/usr/bin/env python3
"""Whole-inventory context arithmetic with real tokenizer and encoder lengths.

Reads pinned headers/filenames only. This is not waveform, engine, quality, or
Tier7 business-protocol acceptance. Every independent variant is retained;
Tier7 uses the same provisional ordered role-labeled layout as the prior probe.
"""
import argparse
from collections import Counter
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def groups(row):
    audios = row['audios']
    if not audios:
        raise ValueError('empty query')
    if row['tier'] == 7:
        if (row['audio_semantics'] != 'ordered_conversation_turns'
                or [a['turn'] for a in audios] != sorted({a['turn'] for a in audios})
                or any(a['role'] not in ('user', 'agent') for a in audios)):
            raise ValueError('ordered unique conversation turns required')
        return [audios]
    if row['audio_semantics'] != 'independent_variants':
        raise ValueError('independent variants required')
    return [[a] for a in audios]


def samples_and_frames(header):
    if (type(header['frames']) is not int or header['frames'] <= 0
            or header['sample_rate'] != 24000 or header['channels'] != 1
            or header['sample_width_bytes'] != 2):
        raise ValueError('audited 24k mono PCM16 headers required')
    samples = (header['frames'] * 16000 + 23999) // 24000
    return samples, (samples + 159) // 160


def run(args):
    import torch
    from transformers import Qwen3OmniMoeProcessor
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import _get_feat_extract_output_lengths
    begin = time.monotonic(); torch.set_num_threads(2)
    for name in ('inventory', 'headers', 'prompt', 'prior_identity', 'prior_result'):
        if file_hash(getattr(args, name)) != getattr(args, name + '_hash'):
            raise ValueError('registered input changed: ' + name)
    rows = [json.loads(s) for s in args.inventory.open()]
    headers = [json.loads(s) for s in args.headers.open()]
    index = {h['path']: h for h in headers}
    audio = [a for row in rows for a in row['audios']]
    if (len(rows) != 16843 or len(audio) != 36419 or len(index) != len(headers)
            or len({a['path'] for a in audio}) != len(audio)
            or set(index) != {a['path'] for a in audio}
            or len({(r['tier'], r['query_id']) for r in rows}) != len(rows)
            or any(a['sha256'] != index[a['path']]['manifest_sha256'] for a in audio)):
        raise ValueError('complete pinned query/audio/header coverage required')
    frames = {p: samples_and_frames(h)[1] for p, h in index.items()}
    lengths = _get_feat_extract_output_lengths(torch.tensor(list(frames.values()))).tolist()
    lengths = dict(zip(frames, lengths, strict=True))
    max_length = max(lengths.values())
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model, local_files_only=True)
    if processor.feature_extractor.hop_length != 160:
        raise ValueError('registered hop length required')
    saved = args.prompt.read_text(); start = '<|im_start|>system\n'; end = '<|im_end|>\n'
    if not saved.startswith(start) or saved.count(start) != 1:
        raise ValueError('registered system layout differs')
    system = saved[len(start):].split(end, 1)[0]
    tokenizer = processor.tokenizer; aid = tokenizer.convert_tokens_to_ids('<|audio_pad|>')
    cache = {}

    def count(audios):
        layout = tuple((a.get('turn'), a.get('role')) for a in audios)
        if layout not in cache:
            content = []
            for a in audios:
                if 'turn' in a:
                    content.append(dict(type='text', text=f"Conversation turn {a['turn']}, speaker role {a['role']}:"))
                content.append(dict(type='audio', audio=''))
            messages = [dict(role='system', content=system), dict(role='user', content=content)]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
            if ids.count(aid) != len(audios):
                raise ValueError('ordered placeholders differ')
            # Exercise full expansion at each layout with the maximum clip token
            # length. Special audio tokens must remain atomic at both boundaries.
            expanded = prompt.replace('<|audio_pad|>', '<|audio_pad|>' * max_length)
            expanded_ids = tokenizer(expanded, add_special_tokens=False)['input_ids']
            expected = []
            for token in ids:
                expected.extend([aid] * max_length if token == aid else [token])
            if expanded_ids != expected:
                raise ValueError('audio token expansion is not additive')
            cache[layout] = len(ids) - len(audios)
        return cache[layout] + sum(lengths[a['path']] for a in audios)

    prior_identity = json.loads(args.prior_identity.read_text())
    prior = json.loads(args.prior_result.read_text())
    comparisons = []
    for name, audios in prior_identity['selections']:
        actual = next(r for r in prior['records'] if r['selection'] == name)
        predicted = count(audios)
        if (predicted != actual['input_tokens']
                or [frames[a['path']] for a in audios] != actual['vllm_frames']):
            raise ValueError('header arithmetic differs from prior real vLLM processor')
        comparisons.append(dict(selection=name, input_tokens=predicted, exact_match=True))
    budgets = (4096, 8192, 16384, 32768)
    counts = Counter(); exceeded = {str(n): Counter() for n in budgets}; maxima = {}
    with (args.output / 'requests.jsonl').open('x') as stream:
        for row in rows:
            for audios in groups(row):
                total = count(audios); tier = str(row['tier'])
                record = dict(tier=row['tier'], query_id=row['query_id'],
                              paths=[a['path'] for a in audios], audio_count=len(audios),
                              input_tokens=total, audio_frames=[frames[a['path']] for a in audios],
                              audio_tokens=[lengths[a['path']] for a in audios])
                stream.write(json.dumps(record) + '\n'); counts[tier] += 1
                if tier not in maxima or total > maxima[tier]['input_tokens']:
                    maxima[tier] = record
                for n in budgets:
                    if total + 1024 > n:
                        exceeded[str(n)][tier] += 1
    if sum(counts.values()) != 30731:
        raise ValueError('complete request population differs')
    atomic_json(args.output / 'result.json', dict(status='arithmetic_complete',
        requests=sum(counts.values()), by_tier=dict(counts), maxima=maxima,
        exceeded_with_1024_output={n: dict(c) for n, c in exceeded.items()},
        text_layouts=len(cache), prior_real_processor_comparisons=comparisons,
        input_hashes={n: file_hash(getattr(args,n)) for n in ('inventory','headers','prompt','prior_identity','prior_result')},
        requests_sha256=file_hash(args.output/'requests.jsonl'),
        processor_source_sha256=file_hash(inspect.getfile(type(processor))),
        length_helper_source_sha256=file_hash(inspect.getfile(_get_feat_extract_output_lengths)),
        runner_sha256=file_hash(__file__), elapsed_seconds=time.monotonic()-begin,
        answers_read=False, waveforms_read=False, model_loaded=False,
        scope='Whole pinned inventory header-derived hop-aligned token budget with real tokenizer/helper; provisional Tier7 layout, not full waveform processing or engine/quality acceptance.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('inventory','headers','prompt','prior_identity','prior_result','model','output'):
        p.add_argument('--'+name.replace('_','-'), type=Path, required=True)
    for name in ('inventory','headers','prompt','prior_identity','prior_result'):
        p.add_argument('--'+name.replace('_','-')+'-hash', required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    try:
        run(a)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc))); raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0))
