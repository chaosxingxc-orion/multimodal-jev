import json
from pathlib import Path

import pytest

from tools.reports import run_context_service_batch_study as m
from tools.reports.context_capacity_probe_operations import namespace


@pytest.mark.parametrize('capacity',[4,16,True,128])
def test_capacity_probe_refuses_undeclared_batch(capacity):
    with pytest.raises(ValueError):namespace(dict(batch_queries=capacity,batch_tokens=131072))


@pytest.mark.parametrize('failure',[None,'predecessor','parity','training'])
def test_sequence_orders_stages_and_stops_without_retry(tmp_path,monkeypatch,failure):
    baseline={'role':'baseline'};candidate={'role':'candidate'};probe={'role':'probe'}
    paths={}
    for name,value in [('training_operations',candidate),('probe_operations',probe),
                       ('predecessor_driver',{'pid':123}),
                       ('predecessor_terminal',{'exit_code':1 if failure=='predecessor' else 0})]:
        p=tmp_path/(name+'.json');p.write_text(json.dumps(value));paths[name]=str(p)
    c=dict(paths,output=str(tmp_path/'study'),pins={},baseline_config='b',baseline_local='l')
    calls=[]
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    monkeypatch.setattr(m,'load_config',lambda *a:(baseline,{}))
    monkeypatch.setattr(m,'validate',lambda _:None)
    monkeypatch.setattr(m,'controls',lambda o:({},[(4,16384)]*2 if o==baseline else [(32,131072)]*2))
    monkeypatch.setattr(m,'wait_driver',lambda *a:calls.append('wait'))
    def stage(name):
        calls.append(name)
        if failure==name:raise ValueError('fixture '+name)
        return {'stage':name}
    monkeypatch.setattr(m,'namespace',lambda _:dict(validate=lambda _:None,launch=lambda *a:stage('parity')))
    monkeypatch.setattr(m,'audit_collected',lambda _:stage('baseline_audit'))
    monkeypatch.setattr(m,'train',lambda *a:stage('training'))
    monkeypatch.setattr(m,'collect_and_audit',lambda *a,**kw:stage('collect'))
    monkeypatch.setattr(m,'compare',lambda *a:stage('compare'))
    if failure:
        with pytest.raises(ValueError):m.execute(c,{})
    else:m.execute(c,{})
    expected=['wait','baseline_audit','parity','training','collect','compare']
    if failure=='predecessor':expected=['wait']
    elif failure in ('parity','training'):expected=expected[:expected.index(failure)+1]
    assert calls==expected
    terminal=json.loads((tmp_path/'study/terminal.json').read_text())
    assert terminal['exit_code']==int(failure is not None)
    if failure:assert terminal['restart'] is False
