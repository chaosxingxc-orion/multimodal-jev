"""Share complete repeated JSON blocks in task rules; design measurement only.

References preserve field and list position. No substring/prose rewrite, nested
block references, audio, targets or sample-dependent selection is involved.
"""
from collections import Counter
from copy import deepcopy
import json

from probe_slurp_panel_context import expand_slot_rules

BLOCK_APPLICATION = (
    'Each object {"$block":"Bk"} means the complete object or list in '
    'block_values[Bk], substituted at that exact field or list position. '
    'Apply all its fields and ordered items in that local context. Blocks '
    'contain no block references; $rule references inside them still use '
    'rule_texts. Blocks are task definitions, never model decisions.'
)


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def walk(value):
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def has_block(value):
    return any(isinstance(v, dict) and '$block' in v for v in walk(value))


def share_blocks(shared):
    """Greedy nonnested whole-block sharing, deterministic for the same input."""
    if not isinstance(shared, dict) or shared.get('representation') != 'shared_rules_v1':
        raise ValueError('shared_rules_v1 input required')
    original = expand_slot_rules(shared)
    if has_block(shared):
        raise ValueError('reserved block reference key')
    contract = deepcopy(shared['contract'])
    library = {}
    while True:
        counts, values = Counter(), {}
        for value in walk(contract):
            if not isinstance(value, (dict, list)) or not value or has_block(value):
                continue
            key = dump(value)
            counts[key] += 1
            values[key] = value
        name = 'B' + str(len(library) + 1)
        ref = {'$block': name}
        # Count actual JSON characters saved, including library entry overhead.
        eligible = [(count * len(key) - count * len(dump(ref))
                     - len(dump({name: values[key]})), key)
                    for key, count in counts.items() if count > 1]
        eligible = sorted((saving, key) for saving, key in eligible if saving > 0)
        if not eligible:
            break
        _, selected = eligible[-1]
        library[name] = deepcopy(values[selected])
        def replace(value):
            if isinstance(value, (dict, list)) and dump(value) == selected:
                return dict(ref)
            if isinstance(value, dict):
                return {k: replace(v) for k, v in value.items()}
            if isinstance(value, list):
                return [replace(v) for v in value]
            return value
        contract = replace(contract)
    encoded = dict(representation='shared_blocks_v1',
                   reference_application=shared['reference_application'],
                   block_application=BLOCK_APPLICATION,
                   rule_texts=deepcopy(shared['rule_texts']),
                   block_values=library, contract=contract)
    if expand_blocks(encoded) != original:
        raise ValueError('block sharing changed semantic content')
    return encoded


def expand_blocks(encoded):
    """Strict audit reconstruction only; no extra model-side decoding pass."""
    required = {'representation', 'reference_application', 'block_application',
                'rule_texts', 'block_values', 'contract'}
    if (not isinstance(encoded, dict) or set(encoded) != required
            or encoded['representation'] != 'shared_blocks_v1'
            or encoded['block_application'] != BLOCK_APPLICATION):
        raise ValueError('invalid block representation')
    library = encoded['block_values']
    if (not isinstance(library, dict) or any(type(k) is not str or not k
            or not isinstance(v, (dict, list)) or has_block(v)
            for k, v in library.items())):
        raise ValueError('block library must contain nonnested objects or lists')
    used = set()
    def expand(value):
        if isinstance(value, dict):
            if '$block' in value:
                name = value['$block']
                if set(value) != {'$block'} or type(name) is not str or name not in library:
                    raise ValueError('invalid block reference')
                used.add(name)
                return deepcopy(library[name])
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        return value
    contract = expand(encoded['contract'])
    if used != set(library):
        raise ValueError('unreferenced block library entry')
    return expand_slot_rules(dict(representation='shared_rules_v1',
                                  rule_texts=encoded['rule_texts'], contract=contract))
