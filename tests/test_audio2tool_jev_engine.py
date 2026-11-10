import copy
import hashlib
from types import SimpleNamespace

import pytest

from multimodal_jev import audio2tool_jev_engine as engine
from multimodal_jev.audio2tool_inference import digest


def fixture():
    ids = [1, 2, 3]
    case = dict(input_tokens=3, input_ids_sha256=digest(ids))
    response = SimpleNamespace(finished=True, prompt_token_ids=ids,
        outputs=[SimpleNamespace(token_ids=[42], finish_reason='length', text='raw')])
    return case, response


def test_capacity_checks_canonical_jev_token_digest():
    case, response = fixture()
    result = engine.accept_response([response], case)
    assert result['token_ids'] == [42] and result['raw_text'] == 'raw'
    response.outputs[0].finish_reason = 'stop'
    assert engine.accept_response([response], case)['finish_reason'] == 'stop'


@pytest.mark.parametrize('fault', ['order', 'multiple', 'unfinished', 'empty', 'too_many', 'bad_stop', 'bool', 'text'])
def test_capacity_rejects_wrong_or_nonterminal_output(fault):
    case, response = fixture()
    if fault == 'order': response.prompt_token_ids = [1, 3, 2]
    if fault == 'multiple': response.outputs *= 2
    if fault == 'unfinished': response.finished = False
    if fault == 'empty': response.outputs[0].token_ids = []
    if fault == 'too_many': response.outputs[0].token_ids = [1, 2]
    if fault == 'bad_stop': response.outputs[0].finish_reason = 'abort'
    if fault == 'bool': response.outputs[0].token_ids = [True]
    if fault == 'text': response.outputs[0].text = None
    with pytest.raises(ValueError): engine.accept_response([response], case)


def test_request_and_complete_frames_match_prior():
    receipts = [dict(expected_frames=5, actor_waveform_sha256='a')]
    request = dict(prompt='test', multi_modal_data={'audio':[(object(), 16000)]})
    prior = dict(prompt_sha256=hashlib.sha256(b'test').hexdigest(), audio_receipts=receipts, frames=[5])
    engine.validate_boundary(request, receipts, prior)
    bad = copy.deepcopy(prior); bad['frames'] = [4]
    with pytest.raises(ValueError): engine.validate_boundary(request, receipts, bad)


def test_bad_decision_rejected_before_audio_load(monkeypatch):
    processor = SimpleNamespace(feature_extractor=SimpleNamespace(hop_length=160))
    def reject(*args): raise ValueError('invalid actor input')
    def forbidden(*args): raise AssertionError('audio read before input validation')
    monkeypatch.setattr(engine, 'messages', reject)
    monkeypatch.setattr(engine, 'load_audio', forbidden)
    with pytest.raises(ValueError, match='invalid actor input'):
        engine.prepare_request(processor, dict(input=dict(context=dict(audios=[{}]))), 'unknown', '.', {}, b'')
