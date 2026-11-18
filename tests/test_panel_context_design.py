from pathlib import Path
import runpy

make_request=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/probe_slurp_panel_context.py'))['make_request']


def test_all_arms_share_knowledge_and_definitions_precede_questions():
    intents=[dict(semantic_id='first',definition='First category.'),dict(semantic_id='second',definition='Second category.')]
    slots=[dict(entity_type=t,definition=t+' role.') for t in ('role_a','role_b','unrelated')]
    rules=[dict(types=['role_a','role_b'],shared_rule='Distinguish role_a from role_b.',coexistence_rule='Both may occur.')]
    panel={'families':{'intent':{'definitions':['first','second']},'slot':{'definitions':['role_a']}}}
    reqs=[make_request('/x.wav',panel,intents,slots,rules,n)[0] for n in (0,2,8)]
    assert reqs[0].requirements==reqs[1].requirements==reqs[2].requirements
    assert 'First category.' in str(reqs[0].requirements)
    assert 'Second category.' in str(reqs[0].requirements)
    assert 'role_a role.' in str(reqs[0].requirements) and 'role_b role.' in str(reqs[0].requirements)
    assert 'unrelated' not in str(reqs[0].requirements)
    assert len(reqs[0].questions)==1 and len(reqs[1].questions)==3
    assert len(reqs[2].questions)==4  # no duplicate padding for a short family
    assert all(r.questions[-1].id=='Final' for r in reqs)
    assert len(reqs[2].questions[-1].schema.candidates)==2


def test_changing_audio_does_not_select_different_knowledge():
    intents=[dict(semantic_id='one',definition='One.'),dict(semantic_id='two',definition='Two.')]
    panel={'families':{'intent':{'definitions':['one']},'slot':{'definitions':[]}}}
    a,_=make_request('/a.wav',panel,intents,[],[],8)
    b,_=make_request('/b.wav',panel,intents,[],[],8)
    assert a.questions==b.questions and a.requirements==b.requirements


def test_full_sixty_intents_have_unique_symbols():
    intents=[dict(semantic_id='class_'+str(i),definition='Class '+str(i)+'.') for i in range(60)]
    panel={'families':{'intent':{'definitions':['class_59']},'slot':{'definitions':[]}}}
    request,_=make_request('/x.wav',panel,intents,[],[],8)
    assert len(set(c.id for c in request.questions[-1].schema.candidates))==60
    assert request.questions[-1].schema.candidates[-1].id=='7'


def test_ontology_ids_do_not_corrupt_ordinary_words_or_boundaries():
    import json
    intents=[dict(semantic_id='a',definition='A date and time update.',includes=['A person.'],excludes=['Not every update.']),
             dict(semantic_id='b',definition='Other.',includes=['Other.'],excludes=['First.'])]
    slots=[dict(entity_type='date',definition='A calendar date.'),dict(entity_type='time',definition='A time expression.')]
    panel={'families':{'intent':{'definitions':['a']},'slot':{'definitions':['date','time']}}}
    req,_=make_request('/x.wav',panel,intents,slots,[],8,True)
    supplied=json.loads(req.requirements[2].split(': ',1)[1])
    assert supplied['A']=={k:intents[0][k] for k in ('definition','includes','excludes')}
    assert 'A calendar date.' in req.requirements[3]


def test_current_full_contract_lossless_policy_expansion_and_no_provenance():
    import json
    root=Path(__file__).resolve().parents[1]
    read=lambda p:json.loads((root/p).read_text())
    intents=read('configs/tasks/slurp/intent-definition-candidates-v1.json')['cards']
    slots=read('configs/tasks/slurp/type-definition-candidates-v3.json')['types']
    policies=read('configs/tasks/slurp/slot-presence-policy-candidate-v3.json')['policies']
    rules=read('configs/tasks/slurp/type-boundary-rules-candidate-v4.json')['rules']
    panel={'families':{'intent':{'definitions':[intents[0]['semantic_id']]},'slot':{'definitions':['ingredient','radio_name']}}}
    reqs=[make_request('/x.wav',panel,intents,slots,rules,n,True,policies)[0] for n in (0,2,8)]
    assert reqs[0].requirements==reqs[1].requirements==reqs[2].requirements
    contract=json.loads(reqs[0].requirements[-1].split(': ',1)[1])
    module=runpy.run_path(str(root/'tools/jev/probe_slurp_panel_context.py'))
    pmap={p['entity_type']:p for p in policies}
    for index,s in enumerate(slots,1):
        if 'T'+str(index) not in contract['types']:continue
        entry=contract['types']['T'+str(index)]
        assert {k:entry[k] for k in module['SLOT_FIELDS']}=={k:s[k] for k in module['SLOT_FIELDS']}
        expanded=contract['shared_policy']|entry['policy_overrides']
        assert expanded=={k:pmap[s['entity_type']][k] for k in module['POLICY_FIELDS']}
    for r in contract['boundaries']:
        assert 'unresolved_constraint' in r and 'scope_condition' in r
    text=json.dumps(contract)
    assert 'evidence_source_ids' not in text and 'historical_' not in text and 'annotated_recordings' not in text


def test_full_contract_rejects_missing_policy():
    import pytest
    intents=[dict(semantic_id='a',definition='A',includes=[],excludes=[])]
    slots=[dict(entity_type='s',definition='S')]
    panel={'families':{'intent':{'definitions':['a']},'slot':{'definitions':['s']}}}
    with pytest.raises(ValueError,match='complete unique'):
        make_request('/x',panel,intents,slots,[],2,True,[])


def test_shared_rules_preserve_strings_scopes_and_literal_reference_text():
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/probe_slurp_panel_context.py'))
    text='A calendar date and time are ordinary words; do not replace substrings. '
    contract={'types':{'T1':{'definition':text,'thinking':[text,text+'Different.']},
                       'T2':{'scope':'request','definition':text}},
              'literal':'{"$rule":"R1"}', 'empty':[], 'flag':False}
    encoded=module['share_slot_rules'](contract)
    assert encoded['rule_texts']=={'R1':text}
    assert encoded['contract']['types']['T1']['definition']=={'$rule':'R1'}
    assert encoded['contract']['literal']==contract['literal']
    assert module['expand_slot_rules'](encoded)==contract
    assert contract['types']['T1']['definition']==text


def test_shared_rule_reference_corruption_rejected():
    import copy
    import pytest
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/probe_slurp_panel_context.py'))
    encoded=module['share_slot_rules']({'a':['Repeated rule. '*10]*2})
    for change in ('missing','extra_field','unused','nested_library'):
        bad=copy.deepcopy(encoded)
        if change=='missing':bad['rule_texts'].clear()
        elif change=='extra_field':bad['contract']['a'][0]['scope']='other'
        elif change=='unused':bad['rule_texts']['unused']='Rule'
        else:bad['rule_texts']['R1']={'$rule':'R1'}
        with pytest.raises(ValueError):module['expand_slot_rules'](bad)
    with pytest.raises(ValueError,match='reserved'):
        module['share_slot_rules']({'nested':{'$rule':'A legitimate but reserved input'}})


def test_full_ontology_references_expand_to_each_original_semantic_field():
    import json
    root=Path(__file__).resolve().parents[1]
    read=lambda p:json.loads((root/'configs/tasks/slurp'/p).read_text())
    intents=read('intent-definition-candidates-v1.json')['cards']
    slots=read('type-definition-candidates-v3.json')['types']
    policies=read('slot-presence-policy-candidate-v3.json')['policies']
    rules=read('type-boundary-rules-candidate-v4.json')['rules']
    module=runpy.run_path(str(root/'tools/jev/probe_slurp_panel_context.py'))
    # Cover each type and every incident boundary, including all five role overlays.
    for slot in slots:
        chosen=slot['entity_type']
        panel={'families':{'intent':{'definitions':[intents[0]['semantic_id']]},'slot':{'definitions':[chosen]}}}
        old,info=make_request('/x',panel,intents,slots,rules,8,True,policies)
        shared,newinfo=make_request('/x',panel,intents,slots,rules,8,True,policies,'shared_rules_v1')
        expanded=module['expand_slot_rules'](json.loads(shared.requirements[-1].split(': ',1)[1]))
        assert expanded==json.loads(old.requirements[-1].split(': ',1)[1])
        assert shared.questions==old.questions and shared.requirements[:3]==old.requirements[:3]
        assert info['expanded_slot_contract_sha256']==newinfo['expanded_slot_contract_sha256']
        # Independent extraction: shared policy plus local overrides must recover source policy.
        for index,source in enumerate(slots,1):
            entry=expanded['types'].get('T'+str(index))
            if entry is None:continue
            for field in module['SLOT_FIELDS']:assert entry[field]==source[field]
            policy=next(p for p in policies if p['entity_type']==source['entity_type'])
            assert expanded['shared_policy']|entry['policy_overrides']=={k:policy[k] for k in module['POLICY_FIELDS']}
        expected=[r for r in rules if chosen in r['types']]
        assert len(expected)==len(expanded['boundaries'])
        for source,boundary in zip(expected,expanded['boundaries'],strict=True):
            for field in module['BOUNDARY_FIELDS']:assert boundary[field]==source[field]


def test_shared_representation_is_configured_and_equal_across_arms():
    import pytest
    intents=[dict(semantic_id='a',definition='A',includes=[],excludes=[])]
    panel={'families':{'intent':{'definitions':['a']},'slot':{'definitions':[]}}}
    with pytest.raises(ValueError,match='full policies'):
        make_request('/x',panel,intents,[],[],2,True,slot_representation='shared_rules_v1')
    with pytest.raises(ValueError,match='unknown slot'):
        make_request('/x',panel,intents,[],[],2,True,[],slot_representation='typo')
    # Empty slot panels retain a valid empty contract; audio cannot select rules.
    intents.append(dict(semantic_id='b',definition='B',includes=[],excludes=[]))
    reqs=[make_request('/'+str(n),panel,intents,[],[],n,True,[],'shared_rules_v1')[0] for n in (0,2,8)]
    assert reqs[0].requirements==reqs[1].requirements==reqs[2].requirements
