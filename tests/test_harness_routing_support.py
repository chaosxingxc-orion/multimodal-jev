"""Caller rule dependence, set labels and input/target separation."""
import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('routing_support',Path(__file__).parents[1]/'tools/jev/audit_harness_routing_support.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ONTOLOGY=[f'event-{i:02d}' for i in range(60)]


def test_complete_ontology_partition_and_overlap():
    for family in m.FAMILIES:
        p=m.policy(ONTOLOGY,family,317)
        assert p==m.policy(list(reversed(ONTOLOGY)),family,317)
        for event in ONTOLOGY:
            out=m.target(event,ONTOLOGY,p)
            assert len(out['valid_final'])==(2 if family=='overlap6' else 1)
            assert out['execution_success'] is None and out['semantic_unknown'] is None
            assert out['no_route']==(not any(out['candidate_applicability'].values()))


def test_candidate_order_and_id_rebinding_preserve_semantic_decision():
    p=m.policy(ONTOLOGY,'overlap6',317)
    permuted=copy.deepcopy(p);permuted['routes'].reverse()
    rebinding={r['id']:f'candidate-{i}' for i,r in enumerate(permuted['routes'])}
    for r in permuted['routes']:r['id']=rebinding[r['id']]
    for event in ONTOLOGY:
        before=m.target(event,ONTOLOGY,p);after=m.target(event,ONTOLOGY,permuted)
        assert set(after['valid_final'])=={rebinding[x] for x in before['valid_final']}
        assert after['candidate_applicability']=={rebinding[k]:v for k,v in before['candidate_applicability'].items()}


def test_same_audio_event_policy_change_can_require_different_final():
    p=m.policy(ONTOLOGY,'partition6',317)
    changed=copy.deepcopy(p)
    changed['routes'][0]['accepts'],changed['routes'][1]['accepts']=changed['routes'][1]['accepts'],changed['routes'][0]['accepts']
    event=p['routes'][0]['accepts'][0]
    assert m.target(event,ONTOLOGY,p)['valid_final']==['route-00']
    assert m.target(event,ONTOLOGY,changed)['valid_final']==['route-01']
    assert p==m.policy(ONTOLOGY,'partition6',317)


def test_unavailable_route_is_no_route_not_unknown_or_failure():
    p=m.policy(ONTOLOGY,'partition6',317);event=p['routes'][0]['accepts'][0]
    p['routes'][0]['enabled']=False
    t=m.target(event,ONTOLOGY,p)
    assert t['valid_final']==['no_route'] and t['semantic_unknown'] is None
    assert t['execution_success'] is None


@pytest.mark.parametrize('fault',['duplicate_id','unknown_event','bad_rule','missing_availability','nonboolean','fallback_collision'])
def test_invalid_policy_is_not_silently_a_negative_label(fault):
    p=m.policy(ONTOLOGY,'partition6',317);event=ONTOLOGY[0]
    if fault=='duplicate_id':p['routes'][1]['id']=p['routes'][0]['id']
    elif fault=='unknown_event':event='not-in-ontology'
    elif fault=='bad_rule':p['routes'][0]['accepts'].append('not-in-ontology')
    elif fault=='missing_availability':p['routes'][0].pop('enabled')
    elif fault=='nonboolean':p['routes'][0]['enabled']=1
    else:p['routes'][0]['id']='no_route'
    with pytest.raises((ValueError,KeyError)):m.target(event,ONTOLOGY,p)


def test_population_accounting_and_same_policies_under_different_gold():
    pairs={k:['domain',k] for k in ONTOLOGY}
    rows=[dict(id=str(i),source_group=str(i),audio_sha256=str(i),role='learning',official_split='train',
        known_pending_source=False,intent=k,scenario='domain',action=k) for i,k in enumerate(ONTOLOGY)]
    report=m.analyze(rows,dict(candidate_pairs=pairs))
    assert report['recordings']==60 and report['new_independent_audio']==0
    assert [r['no_route_recordings'] for r in report['policy_families']]==[0,0,0,30]
    changed=copy.deepcopy(rows)
    for row in changed:row.update(intent=ONTOLOGY[0],action=ONTOLOGY[0])
    second=m.analyze(changed,dict(candidate_pairs=pairs))
    assert [r['policy'] for r in report['policy_families']]==[r['policy'] for r in second['policy_families']]
    assert report['population_sha256']!=second['population_sha256']
    changed[0]['role']='selection'
    with pytest.raises(ValueError):m.analyze(changed,dict(candidate_pairs=pairs))
