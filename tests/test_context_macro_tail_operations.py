"""Tail audit state machine; model/credit/checkpoint arithmetic use fixtures."""
import ast
import json
import shutil

import pytest

from test_context_macro_training_operations import fixture as command_fixture
from test_context_macro_formal_operations import package as sealed_fixture
from test_context_macro_tail_candidate import config as tail_config
from test_context_macro_formal_audit import fixture as audit_fixture
from tools.reports import context_macro_tail_operations as tail
from tools.reports import context_macro_training_operations as finite
from tools.reports import audit_context_macro_training as audit


@pytest.mark.parametrize('role',['preflight','client','service-0','service-1'])
def test_tail_commands_keep_device_and_storage_boundary(tmp_path,role):
    c,_=command_fixture(tmp_path)
    old=finite.command(c,role);new=tail.command(c,role)
    boundary=old.index(finite.IMAGE)+1
    assert old[:boundary]==new[:boundary]
    if role.startswith('service-'):assert old==new
    elif role=='client':assert new[boundary:boundary+6]==old[boundary:boundary+6]
    else:
        ast.parse(new[-1]);assert 'contract(c);select(batches' in new[-1]


def sealed(tmp_path,monkeypatch):
    c,t,r=sealed_fixture(tmp_path,monkeypatch)
    t.update(tail_config());r['status']='macro_tail_prepared_only'
    (tmp_path/'package/training.json').write_text(json.dumps(t))
    (tmp_path/'PREPARED.json').write_text(json.dumps(r))
    return c,t,r


def test_only_explicit_tail_validator_accepts_package(tmp_path,monkeypatch):
    c,_,_=sealed(tmp_path,monkeypatch)
    assert tail.validate(c)['status']=='macro_tail_prepared_only'
    with pytest.raises(ValueError):finite.validate(c)


@pytest.mark.parametrize('change',[{'stop_update':100},{'microbatch':16},
    {'training_dispatch_allowed':True},{'resume_root':'/resume'},{'algorithm':'rloo'}])
def test_tail_contract_rejects_other_training(tmp_path,monkeypatch,change):
    c,t,_=sealed(tmp_path,monkeypatch);t.update(change)
    (tmp_path/'package/training.json').write_text(json.dumps(t))
    with pytest.raises(ValueError):tail.validate(c)


def tail_audit_fixture(tmp_path,monkeypatch,fault=None):
    package,collection,checks=audit_fixture(tmp_path,monkeypatch)
    old_read=audit.read
    c=dict(old_read(package/'training.json'),**tail_config())
    expected=audit.expected_contract(c);expected['tail_admission']=tail.contract(c)
    journal=collection/'journal/formal'
    for p in journal.iterdir():
        if p.name not in ('batch-00000000','batch-00000001'):shutil.rmtree(p)
    def read(path):
        if path==package/'training.json':return c
        if path==collection/'run/result.json':
            return dict(passed=True,start_update=0,stop_update=2,schedule_updates=2,
                learning_population=40513,engineering_prefix=True,tail_admission=tail.contract(c))
        r=old_read(path)
        if path.name=='result.json':r['stop_update']=2
        if path.name.startswith('checkpoint-') and path.name!='checkpoint-scope.json':r['contract']=expected
        if path.name.startswith('update-'):
            r.update(real_queries=65 if r['update']==1 else 128,checkpoint=True)
            if fault=='denominator':r['real_queries']=128
            if fault=='parity':r['behavior_parity_history'][0][0]={'passed':False,'error':'bad'}
        if fault=='stale_replica' and path.name=='activation-0002.json':
            r['receipt']['replica_receipts'][1]['policy_id']='p1'
        return r
    visited=[]
    def targets(batch,receipt,planned,*args,**kwargs):
        visited.append((planned['epoch'],planned['index'],sum(s['weight'] for s in planned['slots'])))
        return {}
    monkeypatch.setattr(audit,'read',read);monkeypatch.setattr(audit,'update_targets',targets)
    return package,collection,checks,visited


def test_tail_audit_checks_original_schedule_denominator_and_two_updates(tmp_path,monkeypatch):
    package,collection,checks,visited=tail_audit_fixture(tmp_path,monkeypatch)
    result=tail.audit_outputs(package,collection)
    assert result['status']=='finite_dp2_tail_outputs_audited'
    assert result['tail_admission']['formal_history'] is False
    assert checks==[1,2] and len(result['updates'])==2
    assert visited==[(0,316,65),(1,0,128)]


@pytest.mark.parametrize('fault',['denominator','parity','stale_replica'])
def test_tail_audit_rejects_false_success(tmp_path,monkeypatch,fault):
    package,collection,_,_=tail_audit_fixture(tmp_path,monkeypatch,fault)
    with pytest.raises(ValueError):tail.audit_outputs(package,collection)


def test_no_dispatch_without_explicit_diagnostic_flag(monkeypatch):
    monkeypatch.setattr(finite,'run_services',lambda *a,**kw:pytest.fail('unexpected launch'))
    with pytest.raises(ValueError,match='disabled'):tail.run({}, {})
