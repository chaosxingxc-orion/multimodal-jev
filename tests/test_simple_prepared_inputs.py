"""Shared renderer authenticates real tensors; audio/processor are substituted."""
import json
from types import SimpleNamespace

import pytest
torch = pytest.importorskip('torch')

from multimodal_jev.simple_prepared_inputs import prepared
from multimodal_jev.span_request_identity import tensor_identity


def fixture(monkeypatch):
    import multimodal_jev.simple_prepared_inputs as module
    import multimodal_jev.audio_policy as audio
    import multimodal_jev.span_contract as contract
    import multimodal_jev.policy_identity as identity
    from multimodal_jev.span_contract import SpanRequest, SpanQuestion
    from multimodal_jev.task_schema import DecisionSchema
    schema = DecisionSchema.from_dict(dict(id='intent', decision_type='choice', question='Choose intent',
        allow_unknown=False, candidates=[dict(id=str(i), description=str(i)) for i in range(60)]))
    request = SpanRequest('/audio.wav', (SpanQuestion('Final', schema),), 'Final', ('Choose from the defined intents.',))
    processor = SimpleNamespace(tokenizer=None)
    inputs = dict(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3, dtype=torch.long),
                  input_features=torch.ones(1, 2, 3), input_features_mask=torch.ones(1, 3, dtype=torch.long))
    receipt = dict(input_waveform_sha256='a'*64, output_waveform_sha256='b'*64, config={}, sample_rate=16000)
    positions = torch.tensor([[2]])
    mapping = [dict(token_ids=list(range(60)), values=[str(i) for i in range(60)])]
    row = dict(id='audio', audio_sha256='audiohash', request=json.loads(json.dumps(request.model_input())),
        audio_processing=receipt, positions=positions.tolist(), candidate_mapping=mapping,
        **{k: tensor_identity(inputs[k]) for k in ('input_ids', 'input_features', 'input_features_mask')})
    monkeypatch.setattr(module, 'file_hash', lambda path: 'audiohash')
    monkeypatch.setattr(audio, 'load_audio', lambda *a, **k: (None, receipt))
    monkeypatch.setattr(contract, 'encode_span_request', lambda *a: (inputs, positions))
    monkeypatch.setattr(contract, 'span_labels', lambda *a: mapping)
    monkeypatch.setattr(identity, 'checkpoint_identity', lambda p: (p, {}))
    return dict(model='base', base_id='base', adapter='adapter', adapter_id='adapter', max_model_len=4096), dict(records=[row]), processor


def test_shared_renderer_reuses_processor_and_binds_policy_and_tensors(monkeypatch):
    config, artifact, processor = fixture(monkeypatch)
    result = list(prepared(config, artifact, processor=processor))
    assert len(result) == 1 and result[0][-1] is processor
    assert result[0][5]['policy_id'] == 'base:adapter'
    assert result[0][5]['inputs']['input_features'] == artifact['records'][0]['input_features']


@pytest.mark.parametrize('fault', ['base', 'adapter', 'audio', 'features', 'positions', 'mapping', 'context'])
def test_shared_renderer_rejects_changed_model_audio_or_inputs(monkeypatch, fault):
    config, artifact, processor = fixture(monkeypatch)
    if fault in ('base', 'adapter'): config[fault + '_id'] = 'changed'
    elif fault == 'audio': artifact['records'][0]['audio_sha256'] = 'changed'
    elif fault == 'features': artifact['records'][0]['input_features'] = {}
    elif fault == 'positions': artifact['records'][0]['positions'] = [[1]]
    elif fault == 'mapping': artifact['records'][0]['candidate_mapping'] = []
    else: config['max_model_len'] = 2
    with pytest.raises(ValueError): list(prepared(config, artifact, processor=processor))
