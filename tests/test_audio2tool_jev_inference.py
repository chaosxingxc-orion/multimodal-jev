import json
from types import SimpleNamespace

import pytest

from multimodal_jev import audio2tool_jev_inference as inference
from multimodal_jev.audio2tool_inference import digest
from multimodal_jev.audio2tool_jev_ledger import records


class Tensor:
    def __init__(self, value): self.value = value
    def tolist(self): return self.value
    def sum(self, axis): return self


def setup(monkeypatch, fault=None):
    bank = dict(decisions=[dict(id=k) for k in ('p', 'a', 'c')])
    context = dict(mode='single_audio', audios=[{}])
    rows = [dict(id='record', input=dict(context=context, task_bank_sha256=digest(bank)))]
    budgets = [dict(id='record', frames=[10], audio_tokens=[1])]
    profiles = {(inference.layout(context), d['id']):dict(text_tokens=2) for d in bank['decisions']}
    loads = []; calls = []; options = []
    wave = object()
    def load(*args):
        loads.append(args); return [wave], [dict(expected_frames=10, source='fixture')]
    monkeypatch.setattr(inference, 'load_audio', load)
    monkeypatch.setattr(inference, 'messages', lambda row, decision, *args:decision)
    class Processor:
        feature_extractor = SimpleNamespace(hop_length=160)
        def apply_chat_template(self, messages, **kw): return messages
        def __call__(self, **kw):
            options.append(kw)
            return dict(input_ids=[Tensor([1, 2, 3])], feature_attention_mask=Tensor([9 if fault=='frames' else 10]),
                        input_features='fixture')
    class Engine:
        def generate(self, batch, sampling, **kw):
            calls.append(batch)
            assert batch[0]['multi_modal_data']['audio'][0][0] is wave
            if fault == 'runtime' and len(calls) == 2: raise RuntimeError('synthetic engine failure')
            ids = [1, 3, 2] if fault=='ids' else [1, 2, 3]
            if fault=='length':
                answer = SimpleNamespace(text='truncated invalid json', token_ids=[9]*1024, finish_reason='length')
            else:
                answer = SimpleNamespace(text='not json' if len(calls)==1 else '', token_ids=[9], finish_reason='stop')
            return [SimpleNamespace(finished=True, prompt_token_ids=ids, outputs=[answer])]
    return Processor(), Engine(), rows, bank, budgets, profiles, loads, calls, options


@pytest.mark.parametrize('fault', [None, 'length'])
def test_complete_independent_calls_and_raw_projection(tmp_path, monkeypatch, fault):
    processor, engine, rows, bank, budgets, profiles, loads, calls, options = setup(monkeypatch, fault)
    root = tmp_path/'ledger'
    result = inference.generate_all(processor, engine, object(), rows, bank, budgets, profiles, '.', b'', root,
                                    lambda *a, **kw:None, lambda _:True)
    assert len(loads)==1 and len(calls)==3 and len(options)==3
    assert [c[0]['prompt'] for c in calls]==['p', 'a', 'c']
    assert all(o['truncation'] is False and o['sampling_rate']==16000 for o in options)
    observed = list(records(root, rows, bank))
    assert len(observed)==3
    assert result['length_limited']==(3 if fault=='length' else 0)
    out = tmp_path/'predictions.jsonl'; inference.export(root, rows, bank, out)
    assert len(out.read_text().splitlines())==3
    if fault is None:
        assert [json.loads(l)['raw_text'] for l in out.read_text().splitlines()]==['not json', '', '']


@pytest.mark.parametrize('fault', ['runtime', 'ids', 'frames', 'nonfinite'])
def test_failure_stops_without_fabricating_remaining_rows(tmp_path, monkeypatch, fault):
    processor, engine, rows, bank, budgets, profiles, loads, calls, options = setup(monkeypatch, fault)
    root=tmp_path/'ledger'
    with pytest.raises((ValueError, RuntimeError)):
        inference.generate_all(processor, engine, object(), rows, bank, budgets, profiles, '.', b'', root,
                               lambda *a, **kw:None, lambda _:fault!='nonfinite')
    assert len(calls)==(2 if fault=='runtime' else 1 if fault=='ids' else 0)
    assert not (root/'complete.json').exists()
    events=[json.loads(l)['event'] for p in root.glob('part-*.jsonl') for l in p.read_text().splitlines()]
    attempts=[e for e in events if e['kind']=='attempt']; terminal=[e for e in events if e['kind']=='terminal']
    assert len(attempts)==len(calls)==len(terminal)
    if terminal: assert terminal[-1]['response']['finish_reason']=='runtime_error'
