from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from build_semantic_actor_menu import compile_panels
from build_semantic_readout_design import compile_design


def design():
    p=compile_panels([f'i{k}' for k in range(65)],[f's{k}' for k in range(55)])
    return compile_design(p,[f'n{k}' for k in range(60)])


def test_native_order_and_rebinding_are_separate_axes():
    d=design();ls=d['native_layouts']
    canonical={e['native_id']:e for e in ls['canonical']['entries']}
    order={e['native_id']:e for e in ls['candidate_order']['entries']}
    rebound={e['native_id']:e for e in ls['neutral_symbol_rebinding']['entries']}
    assert any(canonical[k]['display_position']!=order[k]['display_position'] for k in canonical)
    assert any(canonical[k]['output_symbol']!=rebound[k]['output_symbol'] for k in canonical)
    for k in canonical:
        assert canonical[k]['candidate_alias']==order[k]['candidate_alias']
        assert canonical[k]['output_symbol']==order[k]['output_symbol']
        assert canonical[k]['display_position']==rebound[k]['display_position']
    for layout in ls.values():
        for e in layout['entries']:
            alias=layout['output_symbol_to_candidate_alias'][e['output_symbol']]
            assert layout['scorer_alias_to_native_id'][alias]==e['native_id']


def test_predicates_never_reference_native_classes_and_arms_keep_membership():
    d=design();mapping={x['definition_alias']:x for x in d['semantic_aliases']}
    assert len(mapping)==120
    for p in d['panels']:
        assert p['arms']['Q_input']==p['arms']['Q_credit']
        assert p['arms']['Direct']['final_span_position']==0
        assert p['arms']['Q_input']['final_span_position']==len(p['semantic_questions'])
        for i,q in enumerate(p['semantic_questions']):
            assert q['span_position']==i and not q['definition_alias'].startswith('C')
            assert mapping[q['definition_alias']]['definition_id_offline']==q['definition_id_offline']
            assert q['boolean_output_symbols']=={'A':False,'B':True}
    assert d['panels'][-1]['arms']['Q_input']['final_span_position']==1
