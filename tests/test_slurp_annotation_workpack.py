from pathlib import Path
import runpy

import pytest


module = runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/build_slurp_annotation_workpack.py'))


def fixture():
    target = dict(source_ids=['a', 'b'], native_target=dict(scenario='email', action='query'))
    row = dict(id='r', role='learning', official_split='train', known_pending_source=False,
               intent='email_query', source_target_sha256=module['content_sha'](target),
               audio_sha256='audio', source_group='g')
    return row, target


def test_multisource_recording_retains_every_binding():
    row, target = fixture()
    bindings = module['source_bindings']([row], {'r': target})
    assert set(bindings) == {'a', 'b'}
    assert all(items[0]['all_source_ids'] == ['a', 'b'] for items in bindings.values())


@pytest.mark.parametrize('change', [{'role': 'selection'}, {'official_split': 'test'},
                                  {'known_pending_source': True}, {'source_target_sha256': 'stale'}])
def test_disallowed_population_or_identity_is_rejected(change):
    row, target = fixture()
    row.update(change)
    with pytest.raises(ValueError):
        module['source_bindings']([row], {'r': target})


def test_duplicate_recording_cannot_double_weight_annotation():
    row, target = fixture()
    with pytest.raises(ValueError, match='duplicate identity'):
        module['source_bindings']([row, row], {'r': target})


def test_reference_fit_cannot_silently_turn_into_exclusive_truth():
    metadata, card = {'sentence': 'example'}, {'semantic_id': 'email_query'}
    d = dict(source_id='a', optimization_allowed=False, semantic_id='email_query',
             metadata_sha256=module['content_sha'](metadata), definition_sha256=module['content_sha'](card),
             exclusive_I_class_truth=None)
    args = ('I_fit', {'a': metadata}, {'email_query': card}, {}, {})
    assert module['check_review'](d, *args) == ('a', 'email_query')
    with pytest.raises(ValueError, match='exclusive'):
        module['check_review'](dict(d, exclusive_I_class_truth=True), *args)
    with pytest.raises(ValueError, match='stale'):
        module['check_review'](dict(d, definition_sha256='old'), *args)


def test_agreeing_reference_merge_preserves_evidence_without_extra_supervision():
    from collections import defaultdict
    reviews=defaultdict(list)
    d=dict(source_id='s',entity_type='radio_name',proposed_semantic_value=None,status='unavailable',
           definition_sha256='definition',policy_sha256='policy',metadata_record_sha256='metadata',
           optimization_allowed=False,evidence='First evidence',limitation='Reference only')
    add=module['merge_reference'];mode='retain_agreeing_reference_assessments_v1'
    assert add(reviews,'s','S','radio_name',d,'report1',mode)
    second=dict(d,evidence='Second wording',limitation='Not independent review')
    assert not add(reviews,'s','S','radio_name',second,'report2',mode)
    assert len(reviews['s'])==1
    e=reviews['s'][0]
    assert e['decision']==d and e['additional_reference_assessments'][0]['decision']==second
    assert not e['independent_review_inferred'] and not e['decision']['optimization_allowed']
    with pytest.raises(ValueError,match='included twice'):
        add(reviews,'s','S','radio_name',second,'report2',mode)


@pytest.mark.parametrize('change',[{'proposed_semantic_value':True},{'status':'positive'},
                                  {'definition_sha256':'new'},{'policy_sha256':'new'},
                                  {'metadata_record_sha256':'new'},{'optimization_allowed':True}])
def test_conflicting_or_stale_duplicate_is_not_resolved_by_order(change):
    from collections import defaultdict
    d=dict(proposed_semantic_value=False,status='negative',definition_sha256='d',policy_sha256='p',
           metadata_record_sha256='m',optimization_allowed=False,evidence='e',limitation='l')
    for first,second in ((d,dict(d,**change)),(dict(d,**change),d)):
        reviews=defaultdict(list);add=module['merge_reference'];mode='retain_agreeing_reference_assessments_v1'
        add(reviews,'s','S','t',first,'r1',mode)
        with pytest.raises(ValueError,match='requires adjudication'):
            add(reviews,'s','S','t',second,'r2',mode)


def test_duplicate_remains_rejected_by_default():
    from collections import defaultdict
    reviews=defaultdict(list);args=(reviews,'s','S','t',{},'r1','reject')
    module['merge_reference'](*args)
    with pytest.raises(ValueError,match='duplicate'):module['merge_reference'](*args)


def test_boundary_identity_and_type_support_are_checked():
    import copy
    d=dict(source_definition_sha256='d',source_policy_sha256='p',rules=[dict(types=['a','b'])])
    validate=module['validate_boundaries'];validate(d,{'a':{},'b':{}},'d','p')
    for change in ('stale','unknown','duplicate'):
        bad=copy.deepcopy(d)
        if change=='stale':bad['source_policy_sha256']='old'
        elif change=='unknown':bad['rules'][0]['types']=['a','z']
        else:bad['rules']*=2
        with pytest.raises(ValueError):validate(bad,{'a':{},'b':{}},'d','p')
