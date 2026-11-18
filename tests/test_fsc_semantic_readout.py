import copy
import itertools
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/jev'))
from build_fsc_semantic_readout import compile_design
from audit_semantic_prediction_ledger import decode_spans


def inputs():
    return [json.loads((ROOT / p).read_text()) for p in (
        'configs/tasks/fsc/transfer-protocol-v3.json',
        'configs/tasks/fsc/intent-definitions-transfer-v1.json',
        'configs/tasks/fsc/semantic-presence-definitions-candidate-v1.json')]


def test_full_target_mapping_all_boolean_vectors_and_native_symbols():
    d = compile_design(*inputs())
    assert len(d['native_id_to_triple']) == 31
    assert len(d['semantic_aliases']) == 31
    for panel in d['panels']:
        for layout in d['native_layouts'].values():
            symbols = {e['output_symbol']: e['native_id'] for e in layout['entries']}
            for arm, binding in panel['arms'].items():
                positions = binding['process_span_positions']
                for values in itertools.product((False, True), repeat=len(positions)):
                    process = {str(i): {'A': False, 'B': True} for i in positions}
                    for symbol, native in symbols.items():
                        spans = {str(i): 'B' if v else 'A' for i, v in zip(positions, values)}
                        spans[str(binding['final_span_position'])] = symbol
                        decoded = decode_spans(dict(status='completed', spans=spans),
                            final_position=str(binding['final_span_position']), native_symbols=symbols,
                            process_symbols=process)
                        assert decoded['final'] == native
                        assert decoded['process'] == {str(i): v for i, v in zip(positions, values)}
                if arm == 'Direct':
                    assert positions == [] and binding['final_span_position'] == 0
    assert d['panels'][-1]['arms']['Q_input']['final_span_position'] == 3


def test_order_and_symbol_binding_are_separate_from_semantic_aliases():
    d = compile_design(*inputs())
    layouts = {name: {e['native_id']: e for e in layout['entries']}
               for name, layout in d['native_layouts'].items()}
    for key, base in layouts['canonical'].items():
        order = layouts['candidate_order'][key]
        rebound = layouts['neutral_symbol_rebinding'][key]
        assert base['output_symbol'] == order['output_symbol']
        assert base['candidate_alias'] == order['candidate_alias']
        assert base['display_position'] == rebound['display_position']
    assert any(layouts['canonical'][k]['output_symbol'] != layouts['neutral_symbol_rebinding'][k]['output_symbol'] for k in layouts['canonical'])
    assert all(x['definition_alias'].startswith('I') for x in d['semantic_aliases'])


@pytest.mark.parametrize('fault', ['duplicate_semantic', 'duplicate_native', 'extra_arm', 'missing_panel', 'native_projection'])
def test_invalid_task_bindings_rejected(fault):
    p, n, s = copy.deepcopy(inputs())
    if fault == 'duplicate_semantic': p['structure']['panels'][0]['semantic_ids'][0] = p['structure']['panels'][0]['semantic_ids'][1]
    if fault == 'duplicate_native': n['cards'][0]['native_values'] = n['cards'][1]['native_values']
    if fault == 'extra_arm': p['structure']['arms'].append('Q_credit')
    if fault == 'missing_panel': p['structure']['panels'].pop()
    if fault == 'native_projection': s['cards'][0]['native_equivalence'] = True
    with pytest.raises(ValueError): compile_design(p, n, s)
