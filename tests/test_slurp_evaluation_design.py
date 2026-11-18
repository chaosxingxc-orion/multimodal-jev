from pathlib import Path
import runpy

m=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/build_slurp_evaluation_design.py'))


def test_order_and_symbol_transforms_change_different_things():
    layouts=m['interventions']([f'intent_{i}' for i in range(60)])
    base=layouts['canonical'];order=layouts['candidate_order'];symbol=layouts['neutral_symbol_rebinding']
    mapping=lambda xs:{x['semantic_id']:x['symbol'] for x in xs}
    assert mapping(base)==mapping(order)
    assert [x['semantic_id'] for x in base]!=[x['semantic_id'] for x in order]
    assert [x['semantic_id'] for x in base]==[x['semantic_id'] for x in symbol]
    assert mapping(base)!=mapping(symbol)
    for layout in layouts.values():
        assert len({x['symbol'] for x in layout})==60
        inverse={x['symbol']:x['semantic_id'] for x in layout}
        assert all(inverse[x['symbol']]==x['semantic_id'] for x in layout)


def test_incomplete_last_slot_panel_never_invents_a_question():
    panel=m['make_panel']({'panel_id':14,'families':{'intent':{'definitions':['i1','i2','i3','i4']},'slot':{'definitions':[]}}})
    assert panel['arms']['direct']==[]
    assert len(panel['arms']['q_small'])==1
    assert len(panel['arms']['q_full'])==4
    assert panel['arms']['q_small'][0] in panel['arms']['q_full']


def test_question_compilation_does_not_depend_on_gold():
    base={'panel_id':0,'families':{'intent':{'definitions':['i1','i2']},'slot':{'definitions':['s1']}}}
    assert m['make_panel'](base)==m['make_panel'](dict(base,gold='different',reference='unrelated'))


def test_declared_readout_order_and_symbol_axes_are_independent():
    layouts=m['interventions']([f'intent_{i}' for i in range(60)])
    designs={k:m['explicit_readout_design'](v) for k,v in layouts.items()}
    mapping=lambda d:{x['semantic_id']:(x['candidate_alias'],x['output_symbol']) for x in d['entries']}
    order=lambda d:[x['semantic_id'] for x in d['entries']]
    assert mapping(designs['canonical'])==mapping(designs['candidate_order'])
    assert order(designs['canonical'])!=order(designs['candidate_order'])
    assert order(designs['canonical'])==order(designs['neutral_symbol_rebinding'])
    assert mapping(designs['canonical'])!=mapping(designs['neutral_symbol_rebinding'])
    for name,d in designs.items():
        assert d['fixed_head_symbol_order']==designs['canonical']['fixed_head_symbol_order']
        # Every possible readout must map bijectively through aliases to meaning.
        for row in d['entries']:
            alias=d['output_symbol_to_candidate_alias'][row['output_symbol']]
            assert d['scorer_alias_to_semantic_id'][alias]==row['semantic_id']
        assert (d['existing_position_bound_symbol_mismatches']==0)==(name=='canonical')


def test_explicit_readout_rejects_missing_duplicate_and_target_fields():
    import copy
    import pytest
    original=m['interventions'](['one','two','three'])['canonical']
    for corruption in ('duplicate_alias','duplicate_id','missing_position','target_field','boolean_position'):
        rows=copy.deepcopy(original)
        if corruption=='duplicate_alias':rows[1]['symbol']=rows[0]['symbol']
        elif corruption=='duplicate_id':rows[1]['semantic_id']=rows[0]['semantic_id']
        elif corruption=='missing_position':rows[0]['position']=9
        elif corruption=='target_field':rows[0]['gold']=True
        else:rows[0]['position']=False
        with pytest.raises(ValueError):m['explicit_readout_design'](rows)


def test_interleaving_preserves_question_membership_and_short_families():
    import pytest
    panel={'panel_id':0,'families':{'intent':{'definitions':['i1','i2','i3']},'slot':{'definitions':['s1','s2']}}}
    legacy=m['make_panel'](panel)
    new=m['make_panel'](panel,'interleaved_I_S')
    assert [q['definition'] for q in new['arms']['q_full']]==['i1','s1','i2','s2','i3']
    assert new['arms']['q_small']==new['arms']['q_full'][:2]
    pairs=lambda qs:{(q['operator'],q['definition']) for q in qs}
    assert pairs(legacy['arms']['q_full'])==pairs(new['arms']['q_full'])
    assert new==m['make_panel'](dict(panel,gold='unseen',reference='unseen'),'interleaved_I_S')
    with pytest.raises(ValueError):m['make_panel'](panel,'unknown')
