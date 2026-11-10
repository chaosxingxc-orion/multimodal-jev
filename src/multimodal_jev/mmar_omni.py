"""MMAR native generative contract for local frozen Omni; no answer access."""
import hashlib
from io import BytesIO
import json
from pathlib import Path

from .mmar import model_input
from .omni_waveform_boundary import align_to_hop
from .sfc_audio import load_sfc_audio

INPUT_SHA = 'd4ef0e308b5a27a163b83c06a9a0c89a6b2a64d8c4f7815dbee6d08a3187ad9c'
INSTRUCTION = ('Listen to the audio and answer the question using the provided choices. '
               'Give only the answer text, without an option label or explanation.')


def task_text(inputs):
    if set(inputs) != {'question', 'choices', 'audio'}:
        raise ValueError('only question, choices and audio are allowed')
    model_input(dict(question=inputs['question'], choices=inputs['choices'], audio_path=inputs['audio']))
    if any('<|' in s or '|>' in s for s in [inputs['question'], *inputs['choices']]):
        raise ValueError('model control tokens are not task text')
    # JSON array preserves ordering, duplicates, empty strings and whitespace.
    return INSTRUCTION + '\nQuestion: ' + inputs['question'] + '\nChoices: ' + json.dumps(inputs['choices'], ensure_ascii=False)


def prepare_request(processor, row, audio_root):
    import soundfile as sf
    if set(row) != {'id', 'input', 'audio'}:
        raise ValueError('input-only release row required')
    text = task_text(row['input'])
    path = Path(row['input']['audio']).resolve(strict=True)
    if not path.is_relative_to(Path(audio_root).resolve(strict=True)):
        raise ValueError('audio escapes dataset root')
    payload = path.read_bytes()
    info = sf.info(BytesIO(payload))
    expected = row['audio']
    if (hashlib.sha256(payload).hexdigest() != expected['sha256']
            or (info.frames, info.samplerate, info.channels) !=
            (expected['frames'], expected['sample_rate'], expected['channels'])):
        raise ValueError('release audio bytes or header changed')
    wave, conversion = load_sfc_audio(BytesIO(payload))
    if len(wave) != (info.frames * 16000 + info.samplerate - 1) // info.samplerate:
        raise ValueError('complete resampled audio required')
    if processor.feature_extractor.hop_length != 160:
        raise ValueError('registered processor hop required')
    aligned = align_to_hop([wave], 160)[0]
    messages = [dict(role='system', content='You are a helpful assistant.'),
                dict(role='user', content=[dict(type='audio', audio=aligned), dict(type='text', text=text)])]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prompt.count('<|audio_pad|>') != 1 or text not in prompt:
        raise ValueError('complete audio and task text required')
    receipt = dict(conversion=conversion, source_sha256=expected['sha256'],
                   right_zero_padding_samples=len(aligned)-len(wave), actor_samples=len(aligned),
                   expected_frames=len(aligned)//160,
                   waveform_sha256=hashlib.sha256(aligned.astype('<f4', copy=False).tobytes()).hexdigest())
    return dict(prompt=prompt, multi_modal_data={'audio': [(aligned, 16000)]}), receipt


def score_text(rows, predictions, publisher_match):
    """Score raw generated descriptions, retaining missing and truncated outputs.

    No JSON parsing, label-to-answer conversion, fence removal or candidate match
    repair. The original publisher word-set matcher is supplied by the caller.
    """
    expected = {r['id']: r for r in rows}
    if not rows or len(expected) != len(rows):
        raise ValueError('nonempty unique source IDs required')
    found = {}
    for pred in predictions:
        if pred['id'] not in expected or pred['id'] in found:
            raise ValueError('unknown or duplicate prediction ID')
        if type(pred['raw_text']) is not str or pred['finish_reason'] not in ('stop', 'length'):
            raise ValueError('raw text and terminal generation reason required')
        found[pred['id']] = pred
    correct = exact = truncated = 0
    groups = {}
    for row in rows:
        pred = found.get(row['id'])
        text = pred['raw_text'] if pred else ''
        matched = bool(pred is not None and publisher_match(row['answer'], text, row['choices']))
        correct += int(matched)
        exact += int(pred is not None and text == row['answer'])
        truncated += int(pred is not None and pred['finish_reason'] == 'length')
        for field in ('modality', 'category', 'sub-category'):
            bucket = groups.setdefault(field, {}).setdefault(str(row.get(field, 'unknown')), dict(correct=0, total=0))
            bucket['correct'] += int(matched)
            bucket['total'] += 1
    for values in groups.values():
        for bucket in values.values():
            bucket['accuracy'] = bucket['correct'] / bucket['total']
    return dict(total=len(rows), predictions=len(found), missing=len(rows)-len(found),
                complete=len(found)==len(rows), publisher_correct=correct,
                publisher_accuracy=correct/len(rows), exact_description_correct=exact,
                exact_description_accuracy=exact/len(rows), length_limited=truncated, by_group=groups,
                denominator_policy='all source questions; missing predictions incorrect; raw text including length-limited outputs')
