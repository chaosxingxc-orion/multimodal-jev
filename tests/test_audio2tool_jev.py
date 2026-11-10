from copy import deepcopy

import pytest

from multimodal_jev import audio2tool_jev as jev


def bank(monkeypatch):
    tools=[dict(tool_id=str(i),tool_name='t'+str(i),signature='t'+str(i)+'()',
        description='tool '+str(i),argument_defaults='',argument_constraints='',domain='d',category='c') for i in range(152)]
    monkeypatch.setattr(jev,'registry',lambda *args:tools)
    return jev.task_bank(b'fixture')


def gold():
    return dict(native_target=dict(primary=dict(name='t0',arguments={'v':True}),
        additional_annotations=[dict(name='t1',arguments={'v':None},argument_evidence='explicit_literal_call'),
                                dict(name='t0',arguments={'v':1},argument_evidence='explicit_literal_call')]))


def test_full_bank_and_lossless_duplicate_order_and_types(monkeypatch):
    b=bank(monkeypatch);g=gold()
    assert len(b['decisions'])==457
    answers={d['id']:jev.answer(g,d)['value'] for d in b['decisions']}
    calls=jev.reconstruct(b,answers)
    assert [c['name'] for c in calls]==['t0','t1','t0']
    assert type(calls[0]['arguments']['v']) is bool and type(calls[2]['arguments']['v']) is int
    assert calls[1]['arguments']['v'] is None
    assert answers['tool.2.calls']==[] and answers['tool.2.absent'] is True


def test_missing_extra_argument_is_not_empty_or_null_annotation(monkeypatch):
    b=bank(monkeypatch);g=gold();g['native_target']['additional_annotations'][1].update(arguments=None,argument_evidence='unavailable')
    a={d['id']:jev.answer(g,d) for d in b['decisions']}
    assert a['tool.0.calls']==dict(status='unavailable',reason='source_additional_arguments_missing',value=None)
    assert a['tool.0.present']['value'] is True and a['tool.0.absent']['value'] is False
    assert a['tool.1.calls']==dict(status='available',value=[{'v':None}])
    assert a['tool.2.calls']==dict(status='available',value=[])
    assert a['tools.sequence']['value']==['t0','t1','t0']


@pytest.mark.parametrize('mutation',['missing','contradiction','omission','repeat','unknown','bad_type'])
def test_reconstruct_rejects_inconsistent_decisions(monkeypatch,mutation):
    b=bank(monkeypatch);a={d['id']:jev.answer(gold(),d)['value'] for d in b['decisions']}
    if mutation=='missing':del a['tool.1.present']
    if mutation=='contradiction':a['tool.0.absent']=True
    if mutation=='omission':a['tools.sequence'].pop()
    if mutation=='repeat':a['tools.sequence'].append('t1')
    if mutation=='unknown':a['tools.sequence'][0]='unknown'
    if mutation=='bad_type':a['tool.0.calls']=['invalid']
    with pytest.raises(ValueError):jev.reconstruct(b,a)


def test_request_never_contains_gold_or_coverage(monkeypatch):
    b=bank(monkeypatch)
    monkeypatch.setattr(jev,'validate',lambda _:None)
    n=dict(id='id',tier=1,split='external_test',input={'audio':'fixture'},provenance={'hidden':'original'})
    r=jev.request(n,b)
    assert set(r['input'])=={'context','task_bank_sha256'}
    assert r['input']['context']==n['input']
    n['input']['audio']='changed';assert r['input']['context']['audio']=='fixture'
    with pytest.raises(ValueError):jev.request(dict(n,split='train'),b)
