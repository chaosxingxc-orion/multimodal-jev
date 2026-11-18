"""Service capacity selection preserves failed/slow arms and requires real audits."""
import json
from pathlib import Path

import pytest

from test_compare_context_service_batch import fixture, compare
from test_prepare_context_macro_formal import setup
from tools.reports import prepare_context_macro_formal as m


@pytest.mark.parametrize('ratio,expected', [(2.,'candidate'),(.8,'baseline'),(1.,'baseline')])
def test_complete_timers_select_fastest_and_preserve_regression(tmp_path,ratio,expected):
    ops,audits=fixture(tmp_path,ratio)
    assert m.select_service_arm(compare(*ops,audits))==expected


@pytest.mark.parametrize('fault',['partial','nan','missing_audit','failed_audit','samples','cache','warmup'])
def test_reject_incomplete_service_evidence(tmp_path,fault):
    ops,audits=fixture(tmp_path); c=compare(*ops,audits)
    if fault=='partial':c['candidate']['optimizer_steps']=99
    if fault=='nan':c['candidate']['all_updates']['complete_seconds']=float('nan')
    if fault=='missing_audit':c['source_audits'].pop()
    if fault=='failed_audit':c['source_audits'][1]['status']='failed'
    if fault=='samples':c['matched_actual_queries']=False
    if fault=='cache':c['matched_reference_misses']=False
    if fault=='warmup':c['candidate']['warmup_updates']=0
    with pytest.raises(ValueError):m.select_service_arm(c)


def test_fresh_audits_required_before_selecting(tmp_path,monkeypatch):
    ops,audits=fixture(tmp_path)
    c=dict(arms={n:dict(config=n,local=n) for n in ('baseline','candidate')})
    mapping=dict(comparison=c,baseline=ops[0],candidate=ops[1]); calls=[]
    monkeypatch.setattr(m,'load_config',lambda config,local:(mapping[config],{}))
    def audit(o):
        calls.append(o)
        if o==ops[1]:raise ValueError('actual audit failed')
        return audits[0]
    monkeypatch.setattr(m,'audit_collected',audit)
    with pytest.raises(ValueError,match='actual audit failed'):
        m.efficiency_choice(dict(comparison_kind='service_batch',comparison_config='comparison',comparison_local='local'))
    assert calls==ops


def test_service_choice_uses_shared_formal_packaging(tmp_path,monkeypatch):
    # Runtime/container evidence are fixtures; actual sealing and verification execute.
    c,ops,_=setup(tmp_path,monkeypatch)
    comparison_ops,audits=fixture(tmp_path/'service')
    comparison=compare(*comparison_ops,audits)
    parent=ops['micro8']
    for a in comparison['source_audits']:a['files_sha256']=parent['manifest_sha256']
    arms=dict(arms={n:dict(config='operations',local='micro8') for n in ('baseline','candidate')})
    monkeypatch.setattr(m,'efficiency_choice',lambda c:(arms,{},comparison,'candidate',parent,
        parent['manifest_sha256'],m.SERVICE_SELECTION_RULE))
    result=m.prepare(c,{})
    root=Path(c['output'])/'package'
    evidence=json.loads((root/'FORMAL.json').read_text())
    training=json.loads((root/'training.json').read_text())
    assert evidence['selection_rule']==m.SERVICE_SELECTION_RULE
    assert evidence['comparison']==comparison and evidence['selected_arm']=='candidate'
    assert training['microbatch']==8 and training['stop_update']==634
    assert training['source_policy_id']==json.loads((Path(parent['prepared'])/'package/training.json').read_text())['source_policy_id']
    assert result['formal_launch_admitted'] is False
