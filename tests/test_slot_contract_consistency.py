from copy import deepcopy
from pathlib import Path
import runpy

m = runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/audit_slot_contract_consistency.py'))


def fixture():
    rule = dict(id='role', rule='Typed evidence required.', boundary='Missing annotation is not absence.')
    package = dict(scope='expressed_communicative_content', scope_contract='scope.json',
                   runtime_field_groups=dict(type=['definition', 'role_rules', 'scope'],
                                             policy=['scope_policy', 'role_rules'],
                                             boundary=['shared_rule', 'unresolved_constraint', 'role_rules']))
    definitions = dict(scope=package['scope'], scope_contract='scope.json', types=[
        dict(entity_type=n, scope=package['scope'], definition='A contextual role.', role_rules=[deepcopy(rule)])
        for n in ['a', 'b']])
    policies = dict(scope_contract='scope.json', policies=[
        dict(entity_type=n, scope_policy='Expressed content.', role_rules=[deepcopy(rule)]) for n in ['a', 'b']])
    boundaries = dict(scope_contract='scope.json', rules=[dict(types=['a', 'b'],
        shared_rule='Independent predicates.', unresolved_constraint='Genuinely ambiguous role is unavailable.', role_rules=[])])
    return package, definitions, policies, boundaries


def codes(data):
    return {i['code'] for i in m['audit'](*data)['issues']}


def test_valid_contract_is_not_semantic_certification():
    result = m['audit'](*fixture())
    assert result['status'] == 'pass'
    assert result['semantic_certification'] is False


def test_absence_unknown_conflict_is_detected_not_all_ambiguity_banned():
    data = fixture()
    data[3]['rules'][0]['unresolved_constraint'] = 'If the current input does not establish the role required by either definition, preserve that predicate as unresolved.'
    assert 'absence_conflated_with_unavailable' in codes(data)


def test_shared_role_drift_across_documents_is_detected():
    data = fixture()
    data[2]['policies'][0]['role_rules'][0]['rule'] = 'Opposite rule.'
    assert 'shared_role_rule_drift' in codes(data)


def test_stale_scope_and_unknown_pair_detected():
    data = fixture()
    data[1]['scope'] = 'request_relevant_role'
    data[2]['scope_contract'] = 'historical.json'
    data[1]['types'][0]['definition'] = 'Unrelated background alone does not count.'
    data[3]['rules'][0]['types'] = ['a', 'missing']
    assert {'document_scope_mismatch', 'scope_contract_pointer_mismatch', 'stale_request_scope',
            'unknown_or_invalid_boundary_pair'} <= codes(data)


def test_history_does_not_override_active_semantics():
    data = fixture()
    data[1]['types'][0]['historical_boundary_review'] = 'Unrelated background alone does not count.'
    assert not codes(data)


def test_duplicates_and_title_shortcut_are_detected():
    data = fixture()
    data[3]['rules'][0]['shared_rule'] = 'including descriptive words within a playlist name under an admitted span policy'
    data[3]['rules'].append(deepcopy(data[3]['rules'][0]))
    data[1]['types'].append(deepcopy(data[1]['types'][0]))
    data[2]['policies'][0]['role_rules'] *= 2
    assert {'duplicate_type_or_policy', 'duplicate_boundary_pair', 'duplicate_role_rule_id',
            'title_word_role_shortcut'} <= codes(data)
