from pathlib import Path
import runpy

import pytest


m = runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/export_semantic_review_layer.py'))


def decision(status='conflict', value=True, native=False):
    return dict(source_id='s', entity_type='time_zone', status=status,
        proposed_semantic_value=value, native_annotated=native, optimization_allowed=False,
        metadata_record_sha256='metadata', definition_sha256='definition', policy_sha256='policy',
        evidence='CST supplies a time standard.', limitation='Reference-only, not audio reviewed.')


def binding(d):
    return dict(source_id='s', entity_type='time_zone', recording_id='r',
        review_sha256=m['content_sha'](d), audio_sha256='audio', source_group='group',
        source_target_sha256='target', complete_source_review=True, optimization_allowed=False)


def test_native_disagreement_retains_proposal_but_never_becomes_admission():
    d = decision()
    p = m['convert'](d)
    assert p['semantic_proposal']['value'] is True
    assert p['semantic_proposal']['status'] == 'proposed'
    assert p['native_projection']['value'] is False
    assert p['native_projection']['agreement'] == 'disagrees'
    assert p['native_projection']['native_annotation_error_certified'] is False
    assert p['admission']['value'] is None and not p['admission']['optimization_allowed']
    assert d == decision()


def test_unresolved_is_not_absence_or_native_agreement():
    p = m['convert'](decision('unavailable', None, False))
    assert p['semantic_proposal']['value'] is None
    assert p['native_projection']['agreement'] == 'not_assessed'


@pytest.mark.parametrize('d', [decision('positive', True, False), decision('conflict', True, True),
    decision('unavailable', False, False), decision('negative', 0, False),
    dict(decision(), optimization_allowed=True), dict(decision(), evidence='')])
def test_invalid_or_already_admitted_review_is_rejected(d):
    with pytest.raises(ValueError):
        m['convert'](d)


def test_rebinding_preserves_identity_and_cannot_promote_truth():
    d = decision(); b = binding(d)
    ps, bs = m['rebind']([d], [b])
    assert bs[0]['review_sha256'] == m['content_sha'](ps[0])
    assert bs[0]['legacy_review_sha256'] == b['review_sha256']
    for key in ('source_id','recording_id','source_group','audio_sha256','source_target_sha256'):
        assert bs[0][key] == b[key]
    assert bs[0]['admitted_value'] is None and not bs[0]['optimization_allowed']
    for bad in ([b,b], [dict(b, source_id='other')], [dict(b, review_sha256='stale')], []):
        with pytest.raises(ValueError):
            m['rebind']([d], bad)
    with pytest.raises(ValueError):
        m['rebind']([d,d], [b])
