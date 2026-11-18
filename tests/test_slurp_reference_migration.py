import copy
from pathlib import Path
import runpy

m = runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/migrate_slurp_annotation_workpack.py'))


def test_incident_boundary_change_invalidates_unchanged_card():
    fields = dict(type=['definition'], policy=['presence_policy'], boundary=['shared_rule'])
    old = dict(slots=[dict(entity_type='a', definition='a')],
               policies=[dict(entity_type='a', presence_policy='explicit', definition_ref='v1')],
               slot_boundary_document=dict(rules=[dict(types=['a', 'b'], shared_rule='old', evidence='historical')]))
    new = copy.deepcopy(old)
    new['policies'][0]['definition_ref'] = 'v2'
    new['slot_boundary_document']['rules'][0]['evidence'] = 'new source'
    contract = m['semantic_contract']
    assert contract(old, 'a', fields) == contract(new, 'a', fields)
    new['slot_boundary_document']['rules'][0]['shared_rule'] = 'changed role'
    assert contract(old, 'a', fields) != contract(new, 'a', fields)


def assessment(value, current=True, identity='a'):
    return dict(proposed_value=value, assessment_id=identity,
                applicability='current_semantics_reference_only' if current else 'stale_semantics_requires_rereview')


def test_stale_disagreement_does_not_override_current_reference_or_admit_it():
    result = m['coordinate_summary']([assessment(False, False), assessment(True, identity='b')])
    assert result['status'] == 'current_reference_proposal'
    assert result['assessment_ids'] == ['a', 'b']
    assert result['admitted_value'] is None and result['optimization_allowed'] is False


def test_current_conflicts_are_preserved_without_latest_wins():
    aa = [assessment(False), assessment(True, identity='b')]
    assert m['coordinate_summary'](aa)['status'] == 'conflicting_current_proposals'
    assert m['coordinate_summary'](aa[::-1])['status'] == 'conflicting_current_proposals'


def test_unknown_is_not_false_or_silently_resolved():
    summarize = m['coordinate_summary']
    assert summarize([assessment(None)])['status'] == 'current_reference_unavailable'
    assert summarize([assessment(None), assessment(False, identity='b')])['status'] == 'current_proposal_with_unresolved_assessment'
    assert summarize([assessment(False, False)])['status'] == 'requires_current_rule_review'
    assert summarize([assessment(True), assessment(True, identity='b')])['independent_review_inferred'] is False


def full_review_fixture():
    text = 'some input'
    source = dict(reference=text, metadata_sha256='metadata')
    record = dict(source_id='s', metadata_sha256='metadata',
                  reference_sha256=m['hashlib'].sha256(text.encode()).hexdigest(),
                  positive={}, negative_types=['a'], unavailable={},
                  negative_coverage=dict(explicitly_assessed_all55=True, source_summary=text, basis='Explicit review'),
                  optimization_allowed=False, actor_input_allowed=False,
                  independent_semantic_review=False, modality='reference_text')
    return dict(schedule=dict(start=0, stop=1, full_population_sources=1,
                             order='existing_annotation_workpack_v1_hash_order'), records=[record]), {'s': source}


def test_sequential_negative_and_authority_requirements():
    import pytest
    review, sources = full_review_fixture()
    validate = m['sequential_values']
    assert validate(review, sources, {'a'})[0][2] is False
    for key, value in [('negative_types', ['a', 'a']), ('independent_semantic_review', True),
                       ('metadata_sha256', 'changed')]:
        broken = copy.deepcopy(review)
        broken['records'][0][key] = value
        with pytest.raises(ValueError):
            validate(broken, sources, {'a'})
    broken = copy.deepcopy(review)
    broken['records'][0]['negative_coverage']['basis'] = ''
    with pytest.raises(ValueError, match='explicit negative'):
        validate(broken, sources, {'a'})


def test_sequential_interval_cannot_silently_skip_or_duplicate_sources():
    import pytest
    review, sources = full_review_fixture()
    review['records'] *= 2
    with pytest.raises(ValueError, match='interval'):
        m['sequential_values'](review, sources, {'a'})
