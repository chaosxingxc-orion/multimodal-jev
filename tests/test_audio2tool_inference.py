import copy
import json
from types import SimpleNamespace

import pytest

from multimodal_jev.audio2tool_inference import Ledger,accept_generation,digest


def response(text='not JSON',ids=None,finish='stop'):
    return [SimpleNamespace(finished=True,prompt_token_ids=[1,2],outputs=[
        SimpleNamespace(text=text,token_ids=[3] if ids is None else ids,finish_reason=finish)])]


def test_raw_malformed_text_and_length_cutoff_are_preserved():
    assert accept_generation(response(),[1,2])['raw_text']=='not JSON'
    assert accept_generation(response('',[3]),[1,2])['raw_text']==''
    assert accept_generation(response('{',[3]*1024,'length'),[1,2])['length_limited']


@pytest.mark.parametrize('mutation',['ids','unfinished','multiple','oversize','empty','short_length','abort'])
def test_actual_generation_contract(mutation):
    r=response()
    if mutation=='ids':r[0].prompt_token_ids=[2,1]
    if mutation=='unfinished':r[0].finished=False
    if mutation=='multiple':r*=2
    if mutation=='oversize':r[0].outputs[0].token_ids=[3]*1025
    if mutation=='empty':r[0].outputs[0].token_ids=[]
    if mutation=='short_length':r[0].outputs[0].finish_reason='length'
    if mutation=='abort':r[0].outputs[0].finish_reason='abort'
    with pytest.raises(ValueError):accept_generation(r,[1,2])


def begin(ledger,key):
    row=dict(id=key,tier=4,input={'same_audio':'same'})
    attempt=ledger.begin(row,{'prompt':'fixed'},[],[1,2])
    return row,attempt


def test_release_duplicates_independent_and_no_replay(tmp_path):
    ledger=Ledger(tmp_path/'run');rows=[]
    for key in ('468','469'):
        row,attempt=begin(ledger,key);rows.append(row)
        ledger.finish(attempt,accept_generation(response(key),[1,2]),.1)
        with pytest.raises(FileExistsError):ledger.finish(attempt,accept_generation(response(),[1,2]),.1)
        with pytest.raises(FileExistsError):begin(ledger,key)
    result=ledger.complete(rows)
    assert result['records']==2
    predictions=[json.loads(l) for l in (tmp_path/'run/predictions.jsonl').open()]
    assert predictions==[dict(id=k,raw_text=k,error=None) for k in ('468','469')]
    with pytest.raises(FileExistsError):Ledger(tmp_path/'run')
    with pytest.raises(FileExistsError):ledger.complete(rows)


def test_interrupted_or_failed_attempt_cannot_be_completed(tmp_path):
    ledger=Ledger(tmp_path/'run');row,attempt=begin(ledger,'1')
    with pytest.raises(ValueError):ledger.complete([row])
    ledger.failure(attempt,RuntimeError('engine failed'),.2)
    with pytest.raises(ValueError):ledger.complete([row])
    assert not (tmp_path/'run/predictions.jsonl').exists()
    record=json.loads((tmp_path/'run/records'/(digest('1')+'.json')).read_text())
    assert record['error']=="RuntimeError('engine failed')"


def test_tampered_inputs_or_record_identity_rejected(tmp_path):
    ledger=Ledger(tmp_path/'run');row,attempt=begin(ledger,'1')
    ledger.finish(attempt,accept_generation(response(),[1,2]),.1)
    changed=copy.deepcopy(row);changed['input']['same_audio']='other'
    with pytest.raises(ValueError):ledger.complete([changed])
    p=tmp_path/'run/records'/(digest('1')+'.json');record=json.loads(p.read_text());record['id']='other';p.write_text(json.dumps(record))
    with pytest.raises(ValueError):ledger.complete([row])
