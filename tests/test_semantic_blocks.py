from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/jev'))
from probe_slurp_panel_context import share_slot_rules
from share_semantic_blocks import share_blocks, expand_blocks


def contract():
    sentence = 'The explicitly supplied complete local semantic role must be supported by the utterance.'
    block = {'scope': sentence, 'rules': [
        {'id': 'quoted " rule', 'text': sentence},
        {'id': 'unicode 中文', 'text': 'Never infer a role from an absent filler or a native category.'}]}
    return {'definitions': {f'S{i}': {'label': str(i), 'policy': deepcopy(block)} for i in range(12)},
            'order': [False, True, None, 'literal $block and $rule text', 1, '1']}


def test_blocks_preserve_scope_order_values_and_input_without_mutation():
    original = contract()
    shared = share_slot_rules(original)
    before = deepcopy(shared)
    encoded = share_blocks(shared)
    assert encoded['block_values']
    assert expand_blocks(encoded) == original
    assert shared == before
    assert share_blocks(shared) == encoded
    restored = expand_blocks(encoded)
    restored['definitions']['S0']['policy']['rules'].append('modified')
    assert 'modified' not in json.dumps(encoded)
    assert 'modified' not in json.dumps(restored['definitions']['S1'])


def test_no_repeated_block_does_not_invent_missing_fields_or_types():
    original = {'a': [], 'b': {}, 'c': None, 'd': False, 'e': 0, 'f': '0', 'g': [True, 1]}
    encoded = share_blocks(share_slot_rules(original))
    assert expand_blocks(encoded) == original
    assert type(expand_blocks(encoded)['d']) is bool
    assert type(expand_blocks(encoded)['e']) is int


def test_reserved_unknown_mixed_unused_and_nested_references_rejected():
    with pytest.raises(ValueError, match='reserved'):
        share_blocks(share_slot_rules({'literal': {'$block': 'reserved'}}))
    encoded = share_blocks(share_slot_rules(contract()))
    for reference in ({'$block': 'missing'}, {'$block': []}, {'$block': 'B1', 'extra': 1}):
        broken = deepcopy(encoded)
        broken['contract'] = reference
        with pytest.raises(ValueError, match='invalid block reference'):
            expand_blocks(broken)
    broken = deepcopy(encoded)
    broken['block_values']['unused'] = {'new': 'value'}
    with pytest.raises(ValueError, match='unreferenced'):
        expand_blocks(broken)
    broken = deepcopy(encoded)
    broken['block_values']['B1'] = {'$block': 'B1'}
    with pytest.raises(ValueError, match='nonnested'):
        expand_blocks(broken)


def test_whole_blocks_only_different_polarity_order_and_prose_remain_distinct():
    original = contract()
    original['definitions']['S0']['policy']['rules'].reverse()
    original['definitions']['S1']['policy']['scope'] += ' DO NOT APPLY.'
    original['definitions']['S2']['policy']['extra'] = False
    encoded = share_blocks(share_slot_rules(original))
    assert expand_blocks(encoded) == original
    assert expand_blocks(encoded)['definitions']['S1'] != expand_blocks(encoded)['definitions']['S2']
