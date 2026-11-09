"""Explicit chronological Qwen3-ASR adaptation for official SFC conversations.

This is a project-specific prompt contract, not the stock ASR chat template.
The latter collapses all audio turns and drops user text. Features are computed
for all recordings together with the stock processor, in placeholder order.
"""
from __future__ import annotations

AUDIO = '<|audio_start|><|audio_pad|><|audio_end|>'


def load_sfc_audio(path):
    """Explicit mono conversion and polyphase resampling, retaining provenance."""
    import math
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    wave, rate = sf.read(path, dtype='float32', always_2d=True)
    if not len(wave) or rate <= 0 or not np.isfinite(wave).all():
        raise ValueError('finite nonempty audio required')
    channels = wave.shape[1]
    mono = wave.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        mono = resample_poly(mono, 16000 // divisor, rate // divisor)
    return np.ascontiguousarray(mono, dtype=np.float32), {
        'source_rate': rate, 'source_channels': channels, 'source_samples': len(wave),
        'target_rate': 16000, 'target_samples': len(mono),
        'conversion': 'channel mean then scipy.signal.resample_poly when needed'}


def _text(value):
    if not isinstance(value, str) or '<|' in value or '|>' in value:
        raise ValueError('plain text without Qwen control tokens required')
    return value


def render_prompt(inputs):
    """Accept only input fields, never supervision/metadata."""
    if set(inputs) != {'messages', 'audios'}:
        raise ValueError('only input messages and audios are allowed')
    messages = inputs['messages']
    if not messages or len(messages) % 2 != 1:
        raise ValueError('alternating user/assistant ending in user required')
    if len(inputs['audios']) != (len(messages) + 1) // 2:
        raise ValueError('one ordered recording per user turn required')
    segments = []
    for i, message in enumerate(messages):
        role = 'user' if i % 2 == 0 else 'assistant'
        if message['role'] != role:
            raise ValueError('unexpected conversation role order')
        content = _text(message['content'])
        if role == 'user':
            if content.count('<audio>') != 1:
                raise ValueError('one audio placeholder per user turn required')
            if i == 0:
                # Match the official inference route: first text becomes system
                # context. Remove only its literal placeholder; never duplicate it.
                system = content.replace('<audio>', '')
                segments.append('<|im_start|>system\nEnglish\n' + system + '<|im_end|>\n')
                content = AUDIO
            else:
                content = content.replace('<audio>', AUDIO)
        elif '<audio>' in content:
            raise ValueError('assistant history must not contain audio placeholders')
        segments.append('<|im_start|>' + role + '\n' + content + '<|im_end|>\n')
    return ''.join(segments) + '<|im_start|>assistant\n'


def hf_inputs(processor, inputs, audios):
    import torch
    if len(audios) != len(inputs['audios']):
        raise ValueError('waveform/input alignment required')
    raw = render_prompt(inputs)
    # Dummy text/recording pairs satisfy the stock processor's 1:1 check.
    # Extract all recordings together: waveform padding affects the final STFT
    # frame and valid mask of shorter clips. Separate processing is not equivalent
    # to the vLLM multi-audio processor, even if audio token counts happen to agree.
    features = processor(text=[AUDIO] * len(audios), audio=audios,
                         sampling_rate=16000, return_tensors='pt', padding=True)
    token = processor.audio_token
    if token != '<|audio_pad|>':
        raise ValueError('unexpected ASR audio token')
    lengths = [int((ids == processor.audio_token_id).sum()) for ids in features['input_ids']]
    if any(n < 1 for n in lengths):
        raise ValueError('empty audio token expansion')
    parts = raw.split(token)
    if len(parts) != len(audios) + 1:
        raise ValueError('placeholder alignment failure')
    expanded = parts[0] + ''.join(token * n + tail for n, tail in zip(lengths, parts[1:]))
    batch = dict(processor.tokenizer(expanded, return_tensors='pt', add_special_tokens=False))
    if int((batch['input_ids'] == processor.audio_token_id).sum()) != sum(lengths):
        raise ValueError('audio expansion/tokenizer mismatch')
    for key in ('input_features', 'input_features_mask'):
        batch[key] = features[key]
    return batch


def vllm_request(inputs, audios):
    if len(audios) != len(inputs['audios']):
        raise ValueError('waveform/input alignment required')
    return {'prompt': render_prompt(inputs),
            'multi_modal_data': {'audio': [(wave, 16000) for wave in audios]}}
