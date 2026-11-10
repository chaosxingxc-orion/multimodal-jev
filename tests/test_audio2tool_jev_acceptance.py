import copy
import hashlib
import json

import pytest

from multimodal_jev import audio2tool_jev_acceptance as acceptance
from multimodal_jev.audio2tool_inference import digest
from multimodal_jev.audio2tool_jev_ledger import DecisionLedger, export_predictions
from multimodal_jev.baseline import file_hash


def fixture(tmp_path):
    bank=dict(decisions=[dict(id='p'),dict(id='a')])
    audio=dict(path='fixture.wav',sha256='a'*64,bytes=526)
    row=dict(id='one',input=dict(context=dict(audios=[audio]),task_bank_sha256=digest(bank)))
    budget=dict(id='one',frames=[2],audio_tokens=[1])
    receipt=dict(**audio,conversion=dict(source_rate=24000,source_channels=1,source_samples=241,
        target_rate=16000,target_samples=161,conversion='channel mean then scipy.signal.resample_poly when needed'),
        right_zero_padding_samples=159,actor_samples=320,expected_frames=2,actor_waveform_sha256='b'*64)
    header=dict(frames=241,sample_rate=24000,channels=1,sample_width_bytes=2,manifest_sha256=audio['sha256'])
    root=tmp_path/'ledger'
    with DecisionLedger(root,[row],bank,shard_decisions=1) as ledger:
        for d in bank['decisions']:
            a=ledger.begin(row,d['id'],d['id'],[receipt],[1,2])
            ledger.finish(a,dict(raw_text='invalid JSON',error=None,token_ids=[3],finish_reason='stop',length_limited=False,
                input_tokens=2,input_ids_sha256=digest([1,2])),.1)
        ledger.complete()
    predictions=tmp_path/'predictions.jsonl';export_predictions(root,[row],bank,predictions)
    return root,[row],bank,[budget],predictions,{audio['path']:header},lambda r,b,d:(d,[1,2]),lambda _: 'invalid JSON'


def rehash(root,mutation):
    h=json.loads((root/'manifest.json').read_text());complete=json.loads((root/'complete.json').read_text())
    chain=digest(h);context_sha=None;attempt_sha=None
    for name in complete['shards']:
        path=root/name;lines=[json.loads(l) for l in path.read_text().splitlines()]
        for line in lines:
            e=line['event'];mutation(e)
            if e['kind']=='context':context_sha=digest(e)
            elif e['kind']=='attempt':e['context_sha256']=context_sha;attempt_sha=digest(e)
            else:e['attempt_sha256']=attempt_sha
            line['previous']=chain;line['sha256']=digest({k:line[k] for k in ('index','previous','event')});chain=line['sha256']
        path.write_text(''.join(json.dumps(l)+'\n' for l in lines));complete['shards'][name]=file_hash(path)
    complete['final_chain']=chain;(root/'complete.json').write_text(json.dumps(complete))


@pytest.mark.parametrize('fault',[None,'missing_prediction','extra_prediction','repaired_projection','coordinated_text',
                                  'coordinated_audio','prompt','token_order','header','budget_order'])
def test_full_raw_audio_prompt_and_token_audit(tmp_path,fault):
    args=list(fixture(tmp_path));root,rows,bank,budgets,predictions,headers,expected,decode=args
    lines=[json.loads(l) for l in predictions.read_text().splitlines()]
    if fault=='missing_prediction':lines.pop()
    if fault=='extra_prediction':lines.append(lines[0])
    if fault in ('repaired_projection','coordinated_text'):
        for p in lines:p['raw_text']='true'
    if fault=='coordinated_text':rehash(root,lambda e:e['response'].update(raw_text='true') if e['kind']=='terminal' else None)
    if fault=='coordinated_audio':rehash(root,lambda e:e['audio_receipts'][0]['conversion'].update(target_samples=160) if e['kind']=='context' else None)
    if fault=='prompt':args[6]=lambda r,b,d:('other',[1,2])
    if fault=='token_order':args[6]=lambda r,b,d:(d,[2,1])
    if fault=='header':headers['fixture.wav']['frames']=500
    if fault=='budget_order':budgets[0]['id']='other'
    predictions.write_text(''.join(json.dumps(p)+'\n' for p in lines))
    if fault is None:
        result=acceptance.audit_ledger(*args)
        assert result['decisions']==2 and result['records']==1 and result['output_tokens']==2
        assert result['generation_seconds']==.2 and result['targets_read'] is False
    else:
        with pytest.raises(ValueError):acceptance.audit_ledger(*args)


def test_prompt_expansion_preserves_special_tokens_and_sample_audio_lengths(monkeypatch):
    monkeypatch.setattr(acceptance,'validate',lambda _:None)
    monkeypatch.setattr(acceptance,'messages',lambda *args:'P<audio>Q<audio>R')
    class Tokenizer:
        def convert_tokens_to_ids(self,_):return 99
        def __call__(self,prompt,**kw):return dict(input_ids=[1,99,2,99,3])
    class Processor:
        tokenizer=Tokenizer()
        def apply_chat_template(self,*args,**kw):return 'P<|audio_pad|>Q<|audio_pad|>R'
    row=dict(id='one',input=dict(context=dict(mode='dialogue_last_user',audios=[dict(turn=0,role='user'),dict(turn=1,role='user')])))
    key=(acceptance.layout(row['input']['context']),'decision')
    profile=dict(text_tokens=3,prompt_sha256=hashlib.sha256(b'P<|audio_pad|>Q<|audio_pad|>R').hexdigest())
    expected=acceptance.prompt_expander(Processor(),{},b'',{key:profile})
    _,ids=expected(row,dict(id='one',audio_tokens=[2,3]),'decision')
    assert ids==[1,99,99,2,99,99,99,3]
    ids[0]=999
    assert expected(row,dict(id='one',audio_tokens=[1,1]),'decision')[1]==[1,99,2,99,3]
    with pytest.raises(ValueError):expected(row,dict(id='one',audio_tokens=[1,0]),'decision')
