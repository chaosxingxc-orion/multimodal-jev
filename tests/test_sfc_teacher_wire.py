import json
from pathlib import Path
import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_teacher_feedback import TEMPLATE
from multimodal_jev import sfc_teacher_wire as wire
from test_sfc_prompt_candidates import proposal
from test_sfc_jev import load_tool


def fixture(tmp_path, monkeypatch, executor_source=None):
    payload, context = proposal()
    # Synthetic accounting population, never a claim of real accepted inference.
    summary = context['all_decision_summary'][0]
    context['all_decision_summary'] = [dict(summary, evidence_id=f'synthetic:{i}#f00.present') for i in range(78)]
    context['representatives'] = [dict(context['representatives'][0], evidence_id='synthetic:0#f00.present')]
    context['counts']['bundles'] = 6; context['source_groups'] = 4
    payload['diagnosis'][0]['evidence_ids'] = ['synthetic:0#f00.present']
    payload['candidates'][0]['evidence_ids'] = ['synthetic:0#f00.present']
    root = tmp_path/'request'; root.mkdir()
    messages = [dict(role='system',content=TEMPLATE),dict(role='user',content=canonical(context).decode())]
    (root/'messages.private.json').write_bytes(canonical(messages))
    prepared = dict(partition_sha256='a'*64,API_sent=False,selection_or_test_feedback_used=False,
        bundles=6,decisions=78,max_candidates=2,max_instruction_characters=1500,
        messages_sha256=file_hash(root/'messages.private.json'),context_bytes=len(messages[1]['content'].encode()),
        representatives=1)
    (root/'prepared.json').write_bytes(canonical(prepared))
    executor = tmp_path/'executor.py'; executor.write_text(executor_source or 'raise SystemExit(7)\n')
    monkeypatch.setattr(wire,'EXECUTOR_SHA',file_hash(executor))
    reference = tmp_path/'reference.json'
    reference.write_bytes(canonical(dict(model='glm-5.3-flash',endpoint='https://example.invalid',api_key_env='UNSET')))
    wire.configure(root,reference,executor,file_hash(root/'prepared.json'),'a'*64)
    return root,payload


def response(root,payload,events=None):
    cfg=wire.read(root/'executor-config.private.json'); intent_hash=file_hash(root/'teacher-call-intent.json')
    messages=wire.read(root/'messages.private.json'); out=root/'response';out.mkdir()
    (root/'process-launch.json').write_bytes(canonical(dict(pid=123,intent_sha256=intent_hash)))
    (root/'process-terminal.json').write_bytes(canonical(dict(exit_code=0,intent_sha256=intent_hash)))
    (out/'intent.json').write_bytes(canonical(dict(config=cfg,messages_sha256=cfg['messages']['sha256'])))
    body=dict(model=cfg['model'],messages=messages,max_tokens=32768,stream=True,
              stream_options={'include_usage':True},thinking={'type':'enabled'})
    (out/'request.json').write_bytes(canonical(body))
    content=json.dumps(payload);usage=dict(prompt_tokens=100,completion_tokens=200,total_tokens=300)
    if events is None:
        events=[dict(id='response1',model='glm-5.3-flash',choices=[dict(index=0,
            delta=dict(content=content,reasoning_content='PRIVATE_REASONING'),finish_reason='stop')]),
            dict(id='response1',model='glm-5.3-flash',choices=[],usage=usage)]
    (out/'response.sse').write_text(''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n')
    parsed=dict(id='response1',model='glm-5.3-flash',usage=usage,choices=[dict(index=0,finish_reason='stop',
        message=dict(role='assistant',content=content,reasoning_content='PRIVATE_REASONING'))])
    (out/'response.json').write_bytes(canonical(parsed))
    result=dict(state='response_received',http_status=200,finish_reasons=['stop'],model='glm-5.3-flash',
                response_sha256=file_hash(out/'response.sse'),usage=usage)
    (out/'result.json').write_bytes(canonical(result))
    return intent_hash


def update(path, mutate):
    data=wire.read(path);mutate(data);path.write_bytes(canonical(data))


def test_wire_stages_only_final_text_and_keeps_review_pending(tmp_path,monkeypatch):
    root,payload=fixture(tmp_path,monkeypatch); ih=response(root,payload)
    receipt=wire.stage_wire(root,root/'candidates',ih)
    assert receipt['wire_authenticated'] and receipt['usage_known']
    assert receipt['semantic_review']=='pending' and not receipt['candidate_accepted']
    assert not receipt['inference_dispatched'] and receipt['usage']['total_tokens']==300
    assert 'PRIVATE_REASONING' not in (root/'candidates/proposal.private.json').read_text()
    assert (root/'candidates/p1.txt').stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):wire.stage_wire(root,root/'candidates',ih)


@pytest.mark.parametrize('change',['config','prepared','messages','executor','intent','request','dispatch',
    'process','content','wire_hash','model','usage','truncated','no_done','late_data','provider_error','second_choice'])
def test_custody_and_incomplete_stream_rejected_before_staging(tmp_path,monkeypatch,change):
    root,payload=fixture(tmp_path,monkeypatch);ih=response(root,payload);out=root/'response'
    if change=='config':update(root/'executor-config.private.json',lambda d:d.update(max_tokens=999999))
    if change=='prepared':update(root/'prepared.json',lambda d:d.update(decisions=77))
    if change=='messages':update(root/'messages.private.json',lambda d:d.reverse())
    if change=='executor':Path(wire.read(root/'teacher-call-intent.json')['executor']).write_text('changed')
    if change=='intent':ih='0'*64
    if change=='request':update(out/'request.json',lambda d:d['messages'].append(dict(role='user',content='TEST')))
    if change=='dispatch':update(out/'intent.json',lambda d:d.update(messages_sha256='bad'))
    if change=='process':update(root/'process-terminal.json',lambda d:d.update(exit_code=1))
    if change=='content':update(out/'response.json',lambda d:d['choices'][0]['message'].update(content='{}'))
    if change=='wire_hash':(out/'response.sse').write_text('changed')
    if change=='model':update(out/'response.json',lambda d:d.update(model='different'))
    if change=='usage':update(out/'result.json',lambda d:d.update(usage=None))
    if change in ('truncated','no_done','late_data','provider_error','second_choice'):
        p=out/'response.sse';text=p.read_text()
        if change=='truncated':text=text.replace('"stop"','"length"')
        if change=='no_done':text=text.replace('data: [DONE]\n\n','')
        if change=='late_data':text+='data: {}\n\n'
        if change=='provider_error':text='data: {"error":"failed"}\n\n'+text
        if change=='second_choice':text=text.replace('"index": 0','"index": 1')
        p.write_text(text);update(out/'result.json',lambda d:d.update(response_sha256=file_hash(p)))
    with pytest.raises(ValueError):wire.stage_wire(root,root/'candidates',ih)
    assert not (root/'candidates').exists()


def test_missing_usage_is_unknown_not_zero(tmp_path,monkeypatch):
    root,payload=fixture(tmp_path,monkeypatch);ih=response(root,payload);out=root/'response'
    p=out/'response.sse'
    events=[json.loads(s[6:]) for s in p.read_text().splitlines() if s.startswith('data: {')]
    events[-1].pop('usage')
    p.write_text(''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n')
    update(out/'result.json',lambda d:d.update(usage=None,response_sha256=file_hash(p)))
    update(out/'response.json',lambda d:d.update(usage=None))
    receipt=wire.stage_wire(root,root/'candidates',ih)
    assert receipt['usage'] is None and not receipt['usage_known']
    assert receipt['output_budget_exceeded'] is None


def test_provider_over_budget_preserves_accounting_but_rejects_stage(tmp_path,monkeypatch):
    root,payload=fixture(tmp_path,monkeypatch);ih=response(root,payload);out=root/'response'
    usage=dict(prompt_tokens=100,completion_tokens=40000,total_tokens=40100)
    p=out/'response.sse';events=[json.loads(s[6:]) for s in p.read_text().splitlines() if s.startswith('data: {')]
    events[-1]['usage']=usage
    p.write_text(''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n')
    update(out/'result.json',lambda d:d.update(usage=usage,response_sha256=file_hash(p)))
    update(out/'response.json',lambda d:d.update(usage=usage))
    assert wire.load_proposal(root,ih)[2]['usage']['total_tokens']==40100
    with pytest.raises(ValueError,match='budget'):wire.stage_wire(root,root/'candidates',ih)


def test_failed_dispatch_and_ambiguous_intent_never_replayed(tmp_path,monkeypatch):
    root,_=fixture(tmp_path,monkeypatch);ih=file_hash(root/'teacher-call-intent.json')
    runner=load_tool('run_sfc_teacher_once')
    with pytest.raises(RuntimeError,match='failed'):runner.execute_once(root,ih)
    assert wire.read(root/'process-terminal.json')['exit_code']==7
    with pytest.raises(FileExistsError):runner.execute_once(root,ih)
    (root/'process-terminal.json').unlink()
    with pytest.raises(FileExistsError):runner.execute_once(root,ih)


def test_successful_process_stages_exact_payload_without_api(tmp_path,monkeypatch):
    # A real local child process emits a synthetic transcript: no HTTP request.
    executor_source='''
import hashlib,json,sys
from pathlib import Path
cfg=json.loads(Path(sys.argv[2]).read_text())
out=Path(cfg['output_root']);out.mkdir()
messages=json.loads(Path(cfg['messages']['path']).read_text())
payload=json.loads((out.parent/'synthetic-payload.json').read_text())
def write(name,value): (out/name).write_text(json.dumps(value))
write('intent.json',dict(config=cfg,messages_sha256=cfg['messages']['sha256']))
write('request.json',dict(model=cfg['model'],messages=messages,max_tokens=cfg['max_tokens'],
    stream=True,stream_options={'include_usage':True},thinking=cfg['thinking']))
content=json.dumps(payload);usage=dict(prompt_tokens=1,completion_tokens=2,total_tokens=3)
event=dict(id='synthetic',model=cfg['model'],usage=usage,
    choices=[dict(index=0,delta=dict(content=content),finish_reason='stop')])
raw='data: '+json.dumps(event)+'\\n\\ndata: [DONE]\\n\\n'
(out/'response.sse').write_text(raw)
write('response.json',dict(id='synthetic',model=cfg['model'],usage=usage,choices=[
    dict(index=0,finish_reason='stop',message=dict(role='assistant',content=content))]))
write('result.json',dict(state='response_received',http_status=200,finish_reasons=['stop'],
    model=cfg['model'],usage=usage,response_sha256=hashlib.sha256(raw.encode()).hexdigest()))
'''
    root,payload=fixture(tmp_path,monkeypatch,executor_source)
    (root/'synthetic-payload.json').write_bytes(canonical(payload))
    ih=file_hash(root/'teacher-call-intent.json');runner=load_tool('run_sfc_teacher_once')
    receipt=runner.execute_once(root,ih)
    assert wire.read(root/'process-terminal.json')['exit_code']==0
    assert receipt['wire_authenticated'] and not receipt['candidate_accepted']
    assert wire.read(root/'candidates/proposal.private.json')==payload
    with pytest.raises(FileExistsError):runner.execute_once(root,ih)


def test_parse_rejects_identity_drift_duplicate_keys_and_multiple_finishes():
    event=dict(id='a',model='m',choices=[dict(index=0,delta=dict(content='{}'),finish_reason='stop')])
    first='data: '+json.dumps(event)+'\n\n'
    bad=[first+first,first+'data: {"id":"b","choices":[]}\n\n',
         'data: {"id":"a","id":"b"}\n\n']
    for data in bad:
        with pytest.raises(ValueError):wire.parse_wire(data+'data: [DONE]\n\n')


@pytest.mark.parametrize('change',['test_split','selection','audio_claim','template','population','budget','duplicate_evidence'])
def test_rehashed_feedback_still_must_meet_learning_contract(tmp_path,monkeypatch,change):
    root,_=fixture(tmp_path,monkeypatch)
    messages=wire.read(root/'messages.private.json');context=json.loads(messages[1]['content'])
    if change=='test_split':context['split']='test_id'
    if change=='selection':context['selection_or_test_feedback']=True
    if change=='audio_claim':context['teacher_has_audio']=True
    if change=='template':messages[0]['content']='Different teacher instruction'
    if change=='population':context['source_groups']=3
    if change=='budget':context['extra']='x'*150001
    if change=='duplicate_evidence':context['all_decision_summary'][1]=context['all_decision_summary'][0]
    messages[1]['content']=canonical(context).decode()
    (root/'messages.private.json').write_bytes(canonical(messages))
    update(root/'prepared.json',lambda d:d.update(messages_sha256=file_hash(root/'messages.private.json'),
        context_bytes=len(messages[1]['content'].encode())))
    with pytest.raises(ValueError):wire.prepared_feedback(root,file_hash(root/'prepared.json'),'a'*64)
