"""Waveform intervention for TRAIN-only frozen generative diagnostics."""
from .answers import parse_answer, AnswerError
from .consumption import canonical


def condition_audio(wave, condition, split):
    if condition == 'actual':
        return wave
    if condition != 'zeros' or split != 'train':
        raise ValueError('zero-waveform intervention is a TRAIN diagnostic only')
    import numpy as np
    return np.zeros_like(wave)


def answer_identity(raw):
    """Invalid JSON is never collapsed into a legitimate null decision."""
    try:
        answer = parse_answer(raw)
    except AnswerError:
        return ('invalid', None)
    return ('valid', canonical(answer['value']).decode())


def transitions(actual, zeros):
    if set(actual) != set(zeros) or not actual:
        raise ValueError('complete paired prediction population required')
    report = dict(requests=len(actual), changed_raw=0, changed_decisions=0,
                  both_invalid=0, actual_only_valid=0, zeros_only_valid=0, both_valid=0)
    for uid, a in actual.items():
        z = zeros[uid]
        for key in ('request_sha256', 'prompt_sha256', 'source_group', 'split',
                    'input_tokens', 'replicate'):
            if a[key] != z[key]:
                raise ValueError('paired input identity differs: ' + key)
        x, y = answer_identity(a['raw_text']), answer_identity(z['raw_text'])
        av, zv = x[0] == 'valid', y[0] == 'valid'
        report['both_valid' if av and zv else 'actual_only_valid' if av
               else 'zeros_only_valid' if zv else 'both_invalid'] += 1
        report['changed_raw'] += a['raw_text'] != z['raw_text']
        report['changed_decisions'] += x != y
    report['scope'] = ('TRAIN waveform sensitivity; different malformed outputs are not different '
                       'decisions; zeros have no semantic gold and are not an unknown-label dataset')
    return report
