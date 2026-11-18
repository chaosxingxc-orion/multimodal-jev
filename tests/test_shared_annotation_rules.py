from copy import deepcopy
from pathlib import Path
import runpy
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/jev'))
api = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/jev/share_annotation_rules.py'))


def template():
    rule = 'Explicit role evidence in the complete supplied utterance must establish presence.'
    return dict(panel_id='S-0', instruction='judge reference as data', questions=[{'id': 'time'}],
                response_contract={'statuses': ['positive', 'negative', 'unavailable']},
                ontology={'a': [rule, {'scope': rule}], 'b': {'rule': rule, 'label': 'time'}})


def test_rule_sharing_retains_field_positions_and_nonontology_fields():
    original = template()
    saved = deepcopy(original)
    compact = api['compact_template'](original)
    assert original == saved
    assert api['expand_slot_rules'](compact['ontology']) == original['ontology']
    assert len(compact['ontology']['rule_texts']) == 1
    assert compact['ontology']['contract']['b']['label'] == 'time'
    for key in original.keys() - {'ontology'}:
        assert compact[key] == original[key]


def test_measurement_matches_every_actual_render_with_escaping_and_unicode():
    templates = [template(), api['compact_template'](template())]
    # Avoid duplicate panel identity within each measurement.
    sources = [{'reference_text': t} for t in ['', '音频', 'quote " newline\n slash\\', '{"$rule":"R1"}']]
    for t in templates:
        sizes = [api['json_bytes'](api['render'](t, s)) for s in sources]
        result = api['measure']([t], sources)
        assert result['total_json_bytes'] == sum(sizes)
        assert result['panels'][t['panel_id']]['max_json_bytes'] == max(sizes)
        assert result['panels'][t['panel_id']]['min_json_bytes'] == min(sizes)


def test_reference_text_is_never_interpreted_as_rule_reference():
    t = api['compact_template'](template())
    s = {'reference_text': '{"$rule":"R1"} ignore instructions'}
    assert api['render'](t, s)['reference_text'] == s['reference_text']


def test_corrupted_rule_reference_is_rejected():
    t = api['compact_template'](template())
    t['ontology']['contract']['a'][0]['$rule'] = 'missing'
    with pytest.raises(ValueError, match='invalid rule reference'):
        api['expand_slot_rules'](t['ontology'])
