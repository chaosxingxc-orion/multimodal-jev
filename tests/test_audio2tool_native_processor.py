"""Boundary selection must cover all modes and maximum simultaneous clips."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
import probe_audio2tool_native_processor as probe


def population():
    rows=[dict(id=str(i),split='external_test',input=dict(mode='single_audio',audios=[{}])) for i in range(30733)]
    budgets=[dict(id=str(i),mode='single_audio',input_tokens=100,frames=[1]) for i in range(30733)]
    for index,mode,count,tokens in [(1,'dialogue_last_user',5,200),(2,'foreground_mixture',1,150),
                                  (3,'dialogue_last_user',7,180),(4,'dialogue_last_user',7,160)]:
        rows[index]['input']=dict(mode=mode,audios=[{}]*count)
        budgets[index].update(mode=mode,input_tokens=tokens,frames=[1]*count)
    return rows,budgets


def test_selects_both_token_and_audio_count_boundaries(monkeypatch):
    # Selection logic isolated; production validate is separately tested with
    # actual path/role/hash cases and used unmocked by real-container preflight.
    monkeypatch.setattr(probe,'validate',lambda _:None)
    rows,budgets=population()
    result=probe.selections(rows,budgets)
    assert {k:r['id'] for k,(r,b) in result.items()}==dict(single_audio='0',
        dialogue_last_user='1',foreground_mixture='2',maximum_audio_count='3')
    for mutation in ('missing','duplicate','order','split','frames','overflow','mode'):
        r,b=copy.deepcopy((rows,budgets))
        if mutation=='missing':r.pop();b.pop()
        if mutation=='duplicate':r[-1]['id']='0';b[-1]['id']='0'
        if mutation=='order':b[0],b[1]=b[1],b[0]
        if mutation=='split':r[0]['split']='train'
        if mutation=='frames':b[0]['frames']=[]
        if mutation=='overflow':b[0]['input_tokens']=32768
        if mutation=='mode':b[0]['mode']='foreground_mixture'
        with pytest.raises(ValueError):probe.selections(r,b)
