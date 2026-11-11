import tarfile

import pytest

from multimodal_jev.mac_slu_audit import audit_audio_members, audit_rows


def row(uid, semantics):
    return dict(id=uid, query='same', split_sens=['same'], semantics=semantics)


def instance(domain, intent, role):
    return {domain: [dict(name='intent', value=intent), dict(name=role, value='value')]}


def test_empty_containers_never_become_negative_gold():
    report = audit_rows([row('1', {}), row('2', {'intent1': {}})])
    assert report['records_without_empty_containers'] == 0
    assert report['explicit_intent_histogram'] == {0: 2}
    assert report['flag_record_counts']['empty_semantics_unresolved'] == 1
    assert report['flag_record_counts']['empty_intent_instance_unresolved'] == 1
    assert report['exact_query_duplicate_groups'][0]['ids'] == ['1', '2']


def test_bound_roles_do_not_collapse_across_domains_or_instances():
    semantics = {'i1': instance('music', 'play', 'song'),
                 'i2': instance('video', 'play', 'title'),
                 'i3': instance('music', 'play', 'singer')}
    report = audit_rows([row('1', semantics)])
    assert report['domain_intent_types'] == 2
    assert report['bound_slot_role_types'] == 3
    assert report['flag_record_counts']['repeated_domain_intent_instances'] == 1


def member(name, size=100):
    item = tarfile.TarInfo(name); item.size = size
    return item


def test_full_audio_denominator_reports_missing_and_extra_ids():
    result = audit_audio_members([member('audio_train/id_1.wav'), member('audio_train/id_3.wav')], ['1', '2'])
    assert result['missing_audio_ids'] == ['2']
    assert result['unlabelled_audio_ids'] == ['3']
    assert not result['exact_id_coverage']


@pytest.mark.parametrize('name', ['/tmp/id_1.wav', '../id_1.wav', 'audio_test/id_1.wav', 'audio_train/id_1.mp3'])
def test_unsafe_or_unexpected_archive_members_rejected(name):
    with pytest.raises(ValueError): audit_audio_members([member(name)], ['1'])


def test_duplicate_and_link_members_rejected():
    with pytest.raises(ValueError):
        audit_audio_members([member('audio_train/id_1.wav')] * 2, ['1'])
    link = member('audio_train/id_1.wav'); link.type = tarfile.SYMTYPE
    with pytest.raises(ValueError): audit_audio_members([link], ['1'])


def test_duplicate_label_ids_rejected():
    with pytest.raises(ValueError): audit_rows([row('1', {}), row('1', {})])
