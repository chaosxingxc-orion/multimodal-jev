from copy import deepcopy

import pytest

from multimodal_jev import audio2tool_jev_runtime as runtime
from multimodal_jev.audio2tool_inference import digest
from multimodal_jev.audio2tool_input import REGISTRY_SHA256


def setup(monkeypatch):
    b=dict(decisions=[dict(id='t.present',decision_type='noul',question='required'),
                      dict(id='t.calls',decision_type='typed_call_list'),
                      dict(id='tools.sequence',decision_type='typed_tool_sequence')])
    monkeypatch.setattr(runtime,'task_bank',lambda _:b)
    monkeypatch.setattr(runtime,'system_text',lambda _:runtime.TASK+'tool(): definition')
    # Native renderer is real except independently tested registry loading.
    import multimodal_jev.audio2tool_input as native
    monkeypatch.setattr(native,'system_text',lambda _:native.TASK+'tool(): definition')
    inputs=dict(mode='foreground_mixture',registry_sha256=REGISTRY_SHA256,audios=[dict(
        path='public/tier8_intent_blending_audios/query_00001/mixed.wav',sha256='a'*64,bytes=100)])
    r=dict(id='test',tier=8,split='external_test',protocol=runtime.PROTOCOL,native_request_sha256='b'*64,
        input=dict(context=inputs,task_bank_sha256=digest(b)))
    return b,r


@pytest.mark.parametrize('decision,format_text',[('t.present','JSON boolean'),('t.calls','array of argument objects'),('tools.sequence','tool-name strings')])
def test_single_decision_replaces_native_output_contract_and_retains_audio(monkeypatch,decision,format_text):
    b,r=setup(monkeypatch);wave=object()
    result=runtime.messages(r,decision,[wave],b,b'fixture')
    assert result[1]['content'][1]['audio'] is wave
    assert 'foreground' in result[0]['content'] and 'foreground' in result[1]['content'][0]['text']
    assert format_text in result[0]['content']
    assert 'objects with exactly two keys' not in result[0]['content']


@pytest.mark.parametrize('mutation',['gold','missing_hint','bank','decision','split'])
def test_rejects_actor_visible_gold_and_bank_changes(monkeypatch,mutation):
    b,r=setup(monkeypatch);decision='t.present'
    if mutation=='gold':r['native_target']={}
    if mutation=='missing_hint':r['input']['unavailable_decisions']=['t.calls']
    if mutation=='bank':b=deepcopy(b);b['decisions'].pop()
    if mutation=='decision':decision='unregistered'
    if mutation=='split':r['split']='train'
    with pytest.raises(ValueError):runtime.messages(r,decision,[object()],b,b'fixture')
