import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from multimodal_jev.mmar_omni import prepare_request, score_text, task_text


def test_task_preserves_choices_and_excludes_supervision():
    inp = dict(question='Which sound?', choices=['B ', '', 'A', 'A'], audio='a.wav')
    assert json.loads(task_text(inp).split('\nChoices: ')[1]) == inp['choices']
    for key in ('answer', 'transcript'):
        with pytest.raises(ValueError, match='only'):
            task_text({**inp, key: 'secret'})
    with pytest.raises(ValueError, match='control tokens'):
        task_text({**inp, 'question': '<|audio_pad|>'})


def test_full_long_audio_and_original_bytes(tmp_path):
    path = tmp_path/'long.wav'
    wave = np.zeros((31*24000+1, 2), dtype=np.float32)
    wave[-100:] = .5
    sf.write(path, wave, 24000, subtype='PCM_16')
    row = dict(id='x', input=dict(question='Which?', choices=['A', 'B'], audio=str(path)),
               audio=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          frames=len(wave), sample_rate=24000, channels=2))
    def render(messages, **kwargs):
        return '<|audio_pad|>' + messages[1]['content'][1]['text']
    processor = SimpleNamespace(feature_extractor=SimpleNamespace(hop_length=160), apply_chat_template=render)
    request, receipt = prepare_request(processor, row, tmp_path)
    actual, rate = request['multi_modal_data']['audio'][0]
    assert rate == 16000 and len(actual) > 31*16000
    assert receipt['conversion']['target_samples'] == (len(wave)*16000+23999)//24000
    assert 0 <= receipt['right_zero_padding_samples'] < 160
    assert actual[31*16000-30] > .4  # Evidence after the usual 30-second cutoff survives.
    (tmp_path/'other').mkdir()
    with pytest.raises(ValueError, match='escapes'):
        prepare_request(processor, row, tmp_path/'other')


def test_changed_audio_rejected(tmp_path):
    path = tmp_path/'a.wav'; sf.write(path, np.zeros(160), 16000)
    row = dict(id='x', input=dict(question='Which?', choices=['A', 'B'], audio=str(path)),
               audio=dict(sha256='0'*64, frames=160, sample_rate=16000, channels=1))
    with pytest.raises(ValueError, match='changed'):
        prepare_request(None, row, tmp_path)


def test_raw_scoring_keeps_denominator_and_does_not_repair():
    rows = [dict(id=str(i), choices=['cat', 'dog'], answer='cat', modality='sound') for i in range(3)]
    preds = [dict(id='0', raw_text='cat', finish_reason='length'),
             dict(id='1', raw_text='```cat```', finish_reason='stop')]
    seen = []
    def match(answer, text, choices):
        seen.append(text); return text == answer
    out = score_text(rows, preds, match)
    assert seen == ['cat', '```cat```']
    assert out['publisher_correct'] == 1 and out['publisher_accuracy'] == 1/3
    assert out['missing'] == 1 and out['length_limited'] == 1 and not out['complete']
    assert out['by_group']['modality']['sound']['total'] == 3
    for bad in ([preds[0], preds[0]], [dict(id='unknown', raw_text='cat', finish_reason='stop')]):
        with pytest.raises(ValueError, match='unknown or duplicate'):
            score_text(rows, bad, match)
