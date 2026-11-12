from copy import deepcopy

import pytest

from multimodal_jev.mac_slu_projection import (convert_annotation, reconstruct_annotation,
    official_frames, declared_bank, input_request, decision_targets)

SOURCE = '''DOMAIN_INTENT_LIST = """
- music
    - play
- video
    - play
"""
SLOT_LIST = """
- music-name: song name
- video-name: movie name
"""
'''


def frame(domain, intent, values):
    return {domain: [dict(name='intent', value=intent), *[dict(name='name', value=v) for v in values]]}


def test_lossless_repeated_instances_and_values():
    row = dict(id='1', semantics={'i1': frame('music', 'play', ['one', 'two']),
                                'i2': frame('music', 'play', ['three']), 'i3': {}})
    converted = convert_annotation(row)
    assert reconstruct_annotation(converted) == row['semantics']
    assert len(converted['frames']) == 2
    assert converted['unresolved'] == [dict(kind='empty_instance', instance='i3')]
    with pytest.raises(ValueError): official_frames(converted)
    del row['semantics']['i3']
    with pytest.raises(ValueError): official_frames(convert_annotation(row))


def test_official_conversion_preserves_order_multiplicity_and_binding():
    target = convert_annotation(dict(id='1', semantics={
        'i1': frame('music', 'play', ['one']), 'i2': frame('music', 'play', ['two']),
        'i3': frame('video', 'play', ['three'])}))
    values = official_frames(target)
    assert len(values) == 3
    assert [v['slots']['name'] for v in values] == ['one', 'two', 'three']
    assert values[-1]['domain'] == 'video'


def test_inputs_independent_of_gold_and_absence_is_not_false():
    target = convert_annotation(dict(id='1', semantics={'i1': frame('music', 'play', ['one'])}))
    bank = declared_bank(SOURCE, [target])
    request = input_request('/audio.wav', bank, 'video', 'play').model_input()
    assert all(x['status'] == 'unavailable' for x in decision_targets(target, bank, 'video', 'play'))
    changed = convert_annotation(dict(id='1', semantics={'i1': frame('video', 'play', ['secret'])}))
    assert all(x['value'] is True for x in decision_targets(changed, bank, 'video', 'play'))
    assert request == input_request('/audio.wav', bank, 'video', 'play').model_input()
    assert 'secret' not in str(request) and 'semantics' not in request


def test_menu_addition_is_global_not_per_example():
    target = convert_annotation(dict(id='1', semantics={'i1': frame('music', 'pause', [])}))
    bank = declared_bank(SOURCE, [target])
    assert bank['train_only_intent_additions'] == [dict(domain='music', intent='pause')]
    assert input_request('/other.wav', bank, 'music', 'pause')
    with pytest.raises(ValueError): input_request('/other.wav', bank, 'music', 'invented')


def test_empty_mapping_never_silently_becomes_official_empty_list():
    target = convert_annotation(dict(id='1', semantics={}))
    with pytest.raises(ValueError): official_frames(target)
    with pytest.raises(ValueError): convert_annotation(dict(id='1', semantics=[]))


def test_multiple_intent_fields_rejected():
    row = dict(id='1', semantics={'i1': {'music': [dict(name='intent', value='play')] * 2}})
    with pytest.raises(ValueError): convert_annotation(row)


def test_source_metadata_preserved_without_becoming_model_features():
    row = dict(id='1', semantics={'i1': frame('music', 'play', ['one'])})
    row['semantics']['i1']['music'][1].update(hotdegree=0.67, searchlevel=2)
    target = convert_annotation(row)
    assert reconstruct_annotation(target) == row['semantics']
    assert official_frames(target)[0]['slots'] == {'name': 'one'}
    bank = declared_bank(SOURCE, [target])
    assert 'hotdegree' not in str(input_request('/audio.wav', bank, 'music', 'play').model_input())
