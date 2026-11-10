import copy
import hashlib
import json

import pytest

from multimodal_jev.consumption import canonical
from multimodal_jev.sfc import call_multiset, parse_calls
from multimodal_jev.sfc_consumption import split_record
from multimodal_jev.sfc_jev import split_bundle
from multimodal_jev.sfc_jev_runtime import messages, vllm_request, FOLLOWUP
from multimodal_jev.sfc_jev_score import decode, native_text, score_bundle, SchemaError
from test_sfc_jev import record
from test_sfc_jev import load_tool


PREFIX = ('Please utilize the following functions and environment details to give the correct function call based on the audio request. '
          'The output must be a JSON list containing one or more function call strings. '
          'All string arguments *within* each function call must be double-quoted. '
          'If a required parameter is missing, use "NAN" as its value. '
          'Example (single call): ["add_reminder(content="Submit weekly report", time="NAN")"]. '
          'Example (multiple calls): ["search_calendar(time="tomorrow")", '
          '"add_reminder(content="call mom", time="2025-11-11 10:00:00")"]\n\n')


def bundle():
    row = record()
    row['input']['messages'][0]['content'] = row['input']['messages'][0]['content'].replace('Official instruction\n', PREFIX)
    row['input']['messages'][2]['content'] = FOLLOWUP
    # Explicit nested schema for synthetic object/boolean/null coverage.
    text = row['input']['messages'][0]['content']
    text = text.replace('"extra": {"type": "object"}',
                        '"extra": {"type": "object", "properties": {"a": {"type": "array", "items": {"type": "integer"}}}}')
    row['input']['messages'][0]['content'] = text
    for c in row['supervision']['calls']:
        if 'extra' in c['arguments']: c['arguments']['extra'] = {'a':[1,2]}
    row['supervision']['official_solution'] = native_text(row['supervision']['calls'])
    return split_bundle(*split_record(row))


def predictions(r,g):
    return [dict(example_id=r['id'], decision_id=key, request_sha256=hashlib.sha256(canonical(r)).hexdigest(),
                 raw_text=json.dumps(value)) for key,value in g['answers'].items()]


def test_runtime_replaces_only_reviewed_instructions_preserves_context_and_wave_order():
    r,g = bundle(); before=copy.deepcopy(r); a,b=object(),object()
    result=messages(r,'f00.present',[a,b])
    assert r == before
    assert [m['role'] for m in result] == ['system','user','assistant','user']
    assert result[2] == r['input']['context']['messages'][1]
    assert result[1]['content'][0]['audio'] is a and result[-1]['content'][-1]['audio'] is b
    assert 'one or more function call strings' not in result[0]['content']
    assert 'complete and final list of all required function calls' not in result[-1]['content'][0]['text']
    sections=r['input']['context']['messages'][0]['content'][len(PREFIX):]
    assert sections in result[0]['content']
    assert 'Output exactly one JSON boolean' in result[0]['content']
    assert 'f01.present' not in result[0]['content']
    assert 'Output exactly one JSON array' in messages(r,'f00.calls',[a,b])[0]['content']
    class Processor:
        def apply_chat_template(self,actual,**kwargs):
            assert actual==result
            assert kwargs==dict(tokenize=False,add_generation_prompt=True)
            return 'prompt'
    assert vllm_request(Processor(),r,'f00.present',[a,b])['multi_modal_data']['audio']==[(a,16000),(b,16000)]


@pytest.mark.parametrize('change',['prefix','followup','target','definition','wave_count'])
def test_runtime_rejects_unreviewed_or_changed_inputs(change):
    r,g=bundle(); waves=[object(),object()]
    if change=='prefix':r['input']['context']['messages'][0]['content']='extra business rule\n'+r['input']['context']['messages'][0]['content']
    if change=='followup':r['input']['context']['messages'][2]['content']='special instruction <audio>'
    if change=='target':r['target']='secret'
    if change=='definition':r['input']['decisions'][0]['question']='different'
    if change=='wave_count':waves.pop()
    with pytest.raises(ValueError):messages(r,'f00.present',waves)


@pytest.mark.parametrize('raw',['```json\ntrue\n```','True','1','null','{"value":true}','true trailing'])
def test_noul_strict_raw_format(raw):
    assert not decode({'decision_type':'noul'},raw)['format_valid']


def decision():
    return dict(decision_type='typed_call_list',value_schema=dict(type='array',items=dict(type='object',
        properties={'x':dict(type='array',items=dict(type='integer')), 'state':dict(type='string',enum=['on','off'])},required=['x'])))


@pytest.mark.parametrize('raw,format_valid,schema_valid',[
    ('[]',True,True), ('[{"x":[1,2],"state":"on"}]',True,True),
    ('[{"x":"NAN","state":"NAN"}]',True,True),
    ('[{"x":[true]}]',True,False), ('[{"x":["NAN"]}]',True,False),
    ('[{"x":null}]',True,False), ('[{}]',True,False),
    ('[{"x":[],"undeclared":1}]',True,False), ('[{"x":[],"state":"bad"}]',True,False),
    ('[{"x":[],"x":[1]}]',False,False), ('[{"x":[NaN]}]',False,False),
    ('[{"x":[1e999]}]',False,False), ('["f(x=[])"]',False,False), ('{}',False,False)])
def test_recursive_schema_and_missing_sentinel(raw,format_valid,schema_valid):
    result=decode(decision(),raw)
    assert result['format_valid'] is format_valid and result['schema_valid'] is schema_valid


def test_unsupported_schema_is_not_counted_as_model_error():
    d=decision();d['value_schema']['maxItems']=3
    with pytest.raises(SchemaError):decode(d,'[]')


def test_serializer_preserves_nested_types_quotes_and_duplicates():
    calls=[dict(name='f',arguments=dict(text='say "hi"\n中文',data={'x':[True,None,2,2.5]},missing='NAN'))]*2
    assert call_multiset(parse_calls(native_text(calls)))==call_multiset(calls)


def test_oracle_scores_and_noul_contradictions_do_not_gate_call_assembly():
    r,g=bundle(); p=predictions(r,g)
    result=score_bundle(r,g,p)
    assert result['all_decisions_correct'] and result['native_calls_exact']
    assert result['strict_native']['exact_correct']==1
    assert result['counts']['consistent_function_triples']==2
    p[0]['raw_text']='false'
    result=score_bundle(r,g,p)
    assert result['native_calls_exact'] and not result['all_decisions_correct']
    assert result['counts']['consistent_function_triples']==1
    # A bad call branch must not silently become an empty/no-call prediction.
    p[2]['raw_text']='[{"x":[1]}]'
    result=score_bundle(r,g,p)
    assert not result['native_calls_valid'] and result['native_prediction']=='INVALID_JEV_CALL_LIST'


@pytest.mark.parametrize('change',['missing','duplicate','unknown','binding'])
def test_score_requires_complete_input_bound_predictions(change):
    r,g=bundle(); p=predictions(r,g)
    if change=='missing':p.pop()
    if change=='duplicate':p.append(p[0])
    if change=='unknown':p[0]['decision_id']='f99.calls'
    if change=='binding':p[0]['request_sha256']='0'*64
    with pytest.raises(ValueError):score_bundle(r,g,p)


def test_evaluator_covers_complete_release_and_passes_native_calls_to_scorer(tmp_path,monkeypatch):
    from multimodal_jev.baseline import file_hash
    r,g=bundle();root=tmp_path/'release';root.mkdir();split='train_id'
    for kind,value in (('requests',r),('targets',g)):
        (root/f'{split}.{kind}.jsonl').write_bytes(canonical(value))
    pin=dict(bundles=1,decisions=6,**{kind+'_sha256':file_hash(root/f'{split}.{kind}.jsonl') for kind in ('requests','targets')})
    (root/'MANIFEST.json').write_bytes(canonical(dict(splits={split:pin})))
    pred=tmp_path/'pred.jsonl';pred.write_bytes(b''.join(map(canonical,predictions(r,g))))
    scorer=tmp_path/'scorer.py';scorer.write_text('# fixture')
    evaluator=load_tool('evaluate_sfc_jev')
    def native(manifest,predictions,official,output,digest):
        row=json.loads(manifest.read_text());p=json.loads(predictions.read_text())
        assert call_multiset(parse_calls(p['raw_text']))==call_multiset(row['supervision']['calls'])
        assert official==scorer and digest==file_hash(scorer)
        return {'fixture':True}
    monkeypatch.setattr(evaluator,'evaluate_native',native)
    report=evaluator.evaluate(root,split,pred,scorer,tmp_path/'output',file_hash(scorer))
    assert report['counts']['all_decisions_correct']==1
    assert json.loads((tmp_path/'output/terminal.json').read_text())['exit_code']==0
    pred.write_bytes(canonical(predictions(r,g)[0]))
    with pytest.raises(ValueError,match='complete split'):
        evaluator.evaluate(root,split,pred,scorer,tmp_path/'incomplete',file_hash(scorer))
