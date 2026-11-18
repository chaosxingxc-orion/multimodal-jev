from copy import deepcopy
from pathlib import Path
import runpy

import pytest

api = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/jev/consolidate_intent_presence_references.py'))


def assessment(status, incompatible=()):
    return dict(decision=dict(status=status), incompatible_fields=list(incompatible))


def test_policy_only_change_invalidates_unchanged_card():
    card = dict(semantic_id='a', operator='I_present', definition='same')
    old = api['contract'](card, 'request', {'configs/tasks/intent-request-semantics-v1.json': 'old'})
    new = api['contract'](card, 'request', {'configs/tasks/intent-request-semantics-v2.json': 'new'})
    assert api['incompatibilities'](old, new) == ['act_policy_sha256']
    assert api['coordinate']([assessment('positive', ['act_policy_sha256'])]) == 'needs_rereview'


def test_unrelated_policy_does_not_invalidate_card():
    card = dict(semantic_id='a', operator='I_present', boundary_policy='audio.json')
    policies = {'configs/tasks/intent-request-semantics-v1.json': 'request', 'audio.json': 'audio', 'factual.json': 'old'}
    changed = {**policies, 'factual.json': 'new'}
    assert api['contract'](card, 'request', policies) == api['contract'](card, 'request', changed)
    changed['audio.json'] = 'changed'
    assert api['incompatibilities'](api['contract'](card, 'request', policies), api['contract'](card, 'request', changed)) == ['boundary_policy_sha256']


def test_missing_historical_policy_cannot_inherit_current_policy():
    with pytest.raises(ValueError, match='historical act policy'):
        api['contract'](dict(operator='I_present'), 'request', {})
    with pytest.raises(ValueError, match='historical boundary policy'):
        api['contract'](dict(operator='I_present', boundary_policy='missing'), 'request',
                        {'configs/tasks/intent-request-semantics-v1.json': 'h'})


def test_duplicate_conflicts_not_voted_or_overwritten():
    proposals = [assessment('positive'), assessment('positive'), assessment('negative')]
    original = deepcopy(proposals)
    assert api['coordinate'](proposals) == 'conflict'
    assert api['coordinate'](list(reversed(proposals))) == 'conflict'
    assert proposals == original
    assert api['coordinate']([assessment('positive'), assessment('unavailable')]) == 'conflict'


def test_obsolete_proposal_preserved_but_does_not_vote_on_new_contract():
    proposals = [assessment('negative', ['card_sha256']), assessment('positive')]
    assert api['coordinate'](proposals) == 'positive'
    assert len(proposals) == 2


def test_recording_requires_all_explicit_sources_no_implicit_negative():
    decisions = {('1', 'a'): [assessment('positive')], ('2', 'a'): [assessment('negative')]}
    state = api['recording_state']
    assert state(['1', '3'], 'a', decisions) == 'partial_source_review'
    assert state(['1', '2'], 'a', decisions) == 'source_disagreement'
    assert state(['3'], 'a', decisions) == 'unassessed'
    assert state(['1'], 'other', decisions) == 'unassessed'


def test_reference_proposal_cannot_grant_training_permission():
    meta = dict(scenario='x', action='y', sentence='reference')
    contract = dict(card_sha256='h', act_policy_sha256='policy')
    d = dict(metadata_sha256=api['obj'](meta), reference='reference', native_intent='x_y',
             definition_sha256='h', status='positive', proposed_value=True,
             optimization_allowed=False, audio_listening=False, independent_semantic_review=False)
    api['verify_decision'](d, meta, contract)
    d['optimization_allowed'] = True
    with pytest.raises(ValueError, match='reference-only'):
        api['verify_decision'](d, meta, contract)


def test_review_status_cannot_disagree_with_value():
    meta = dict(scenario='x', action='y', sentence='reference')
    d = dict(metadata_sha256=api['obj'](meta), reference='reference', native_intent='x_y',
             definition_sha256='h', status='negative', proposed_value=None)
    with pytest.raises(ValueError, match='status/value'):
        api['verify_decision'](d, meta, dict(card_sha256='h'))
