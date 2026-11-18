import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from build_semantic_actor_menu import compile_panels
from build_semantic_readout_design import compile_design
from compile_semantic_actor_input import make_request, semantic_knowledge, SYMBOLS
from probe_slurp_panel_context import expand_slot_rules
from multimodal_jev.span_contract import span_labels


@pytest.fixture
def design():
    intents=[f'i{k:02d}' for k in range(65)]
    slots=[f's{k:02d}' for k in range(55)]
    readout=compile_design(compile_panels(intents,slots),[f'n{k:02d}' for k in range(60)])
    repeated='This complete rule has enough characters to be shared without changing its local scope or meaning.'
    return dict(readout=readout,native={f'n{k:02d}':dict(descriptor=f'category {k}') for k in range(60)},
        cards={k:dict(definition=k+' intent',includes=['expressed act'],excludes=['topic only'],ambiguity=['mask']) for k in intents},
        acts={k:'request' for k in intents},intent_policies={'request':{'actor_rules':[repeated]}},
        slots={k:dict(definition=k+' role',scope=repeated) for k in slots},
        slot_policies={k:dict(presence_policy='expressed',role_rules=[repeated]) for k in slots},
        boundaries=[dict(types=slots,shared_rule=repeated,scope_condition='local')],
        field_groups={'type':['definition','scope'],'policy':['presence_policy','role_rules'],'boundary':['shared_rule','scope_condition']},
        aliases={(a['operator'],a['definition_id_offline']):a['definition_alias'] for a in readout['semantic_aliases']})


def test_all_arms_keep_knowledge_and_reward_arms_have_identical_requests(design):
    for p in design['readout']['panels']:
        reqs=[make_request(design,'/audio.wav',p['panel_id'],arm)[0] for arm in ('Direct','Q_input','Q_credit')]
        assert reqs[0].requirements==reqs[1].requirements==reqs[2].requirements
        assert reqs[1]==reqs[2]
        assert len(reqs[0].questions)==1
        assert len(reqs[1].questions)==len(p['semantic_questions'])+1
    assert len(reqs[1].questions)==2


def test_contract_expands_exactly_and_neighbor_policies_remain(design):
    p=design['readout']['panels'][0]
    encoded,expanded,info=semantic_knowledge(design,p,'shared_rules_v1')
    assert expand_slot_rules(encoded)==expanded
    assert semantic_knowledge(design,p,'literal_v1')[0]==expanded
    assert info['supplied_slot_definitions']==55
    assert len(expanded['slot']['policies'])==55
    for k,s in design['slots'].items():
        a=design['aliases']['S_present',k]
        assert expanded['slot']['definitions'][a]==s
        assert expanded['slot']['policies'][a]==design['slot_policies'][k]


def test_offline_labels_review_and_audio_path_do_not_route_questions(design):
    before=make_request(design,'/a.wav','semantic-00','Q_input')[0]
    for collection in ('cards','slots','slot_policies','native'):
        for value in design[collection].values():
            value.update(target='SECRET_GOLD',reference_text='SECRET_TRANSCRIPT',evidence_source_ids=['SECRET_ID'])
    design['boundaries'][0]['review_status']='SECRET_REVIEW'
    after=make_request(design,'/b.wav','semantic-00','Q_input')[0]
    assert before.questions==after.questions and before.requirements==after.requirements
    assert 'SECRET' not in json.dumps(after.model_input())


def test_native_order_and_rebinding_match_actual_symbol_readout(design):
    class Tokenizer:
        def encode(self,s,add_special_tokens=False):
            return [ord(s)]
    base=make_request(design,'/a.wav','semantic-00','Q_input')[0]
    for layout_name,layout in design['readout']['native_layouts'].items():
        request=make_request(design,'/a.wav','semantic-00','Q_input',layout_name)[0]
        symbols=span_labels(Tokenizer(),request.questions)[-1]
        by_symbol=dict(zip(symbols['symbols'],symbols['values']))
        assert by_symbol==layout['output_symbol_to_candidate_alias']
        displayed=json.loads(request.requirements[-2].split(': ',1)[1])
        assert [d['candidate_alias'] for d in displayed]==[e['candidate_alias'] for e in sorted(layout['entries'],key=lambda e:e['display_position'])]
        assert request.questions[:-1]==base.questions[:-1]
        assert request.requirements[-1]==base.requirements[-1]


def test_incomplete_semantics_and_bad_bindings_rejected(design):
    changed=copy.deepcopy(design)
    del changed['slot_policies']['s00']['role_rules']
    with pytest.raises(ValueError,match='missing slot field'):
        make_request(changed,'/a.wav','semantic-00','Q_input')
    changed=copy.deepcopy(design)
    changed['readout']['native_layouts']['canonical']['entries'][0]['output_symbol']=SYMBOLS[1]
    with pytest.raises(ValueError,match='native symbols'):
        make_request(changed,'/a.wav','semantic-00','Q_input')



def test_block_representation_preserves_task_and_all_arm_bindings(design):
    from share_semantic_blocks import expand_blocks
    for p in design['readout']['panels']:
        panel = p['panel_id']
        before = make_request(design, '/a.wav', panel, 'Q_input')[0]
        reqs = [make_request(design, '/a.wav', panel, arm, representation='shared_blocks_v1')[0]
                for arm in ('Direct', 'Q_input', 'Q_credit')]
        assert reqs[0].requirements == reqs[1].requirements == reqs[2].requirements
        assert reqs[1] == reqs[2]
        assert reqs[1].questions == before.questions
        assert reqs[1].requirements[:-1] == before.requirements[:-1]
        old = expand_slot_rules(json.loads(before.requirements[-1].split(': ', 1)[1]))
        new = expand_blocks(json.loads(reqs[1].requirements[-1].split(': ', 1)[1]))
        assert old == new

