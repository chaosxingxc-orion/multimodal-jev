from pathlib import Path
from copy import deepcopy
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from build_fsc_semantic_review_packet import semantic_cards,request_wire,reviewed_labels,SEMANTIC_FIELDS


def definitions():
    return {'cards':[dict(semantic_id=f'p{i}',operator='I_present',native_equivalence=False,
                         optimization_allowed=False,related_native_values={'action':'gold'},
                         **{k:'complete rule '+k for k in SEMANTIC_FIELDS}) for i in range(31)]}


def test_semantic_scope_survives_and_native_projection_never_enters_wire():
    d=definitions();original=deepcopy(d);projected=semantic_cards(d)
    assert d==original
    for card in projected.values():
        assert set(card)==set(SEMANTIC_FIELDS)
        assert card['location_constraint']=='complete rule location_constraint'
    t=dict(instruction='Review',ontology=projected,questions=[{'id':'p0'}],response_contract={})
    s=dict(reference_text='Turn the lights on',native_triples=[['activate','lights','none']],
           recordings=['private_path'],optimization_allowed=True)
    w=request_wire(t,s);s['native_triples']=[['deactivate','music','none']]
    assert w==request_wire(t,s) and set(w)=={'instruction','ontology','questions','response_contract','reference_text'}


def test_incomplete_location_constraint_cannot_silently_be_omitted():
    d=definitions();del d['cards'][0]['location_constraint']
    with pytest.raises(ValueError,match='constraint'):semantic_cards(d)
    d=definitions();d['cards'][0]['native_equivalence']=True
    with pytest.raises(ValueError,match='semantic-only'):semantic_cards(d)


def test_unavailable_and_missing_are_not_negative_or_training_labels():
    t={'questions':[{'id':'p0'},{'id':'p1'}]}
    r={'decisions':[{'id':'p0','status':'positive','rationale':'Explicit act'},
                    {'id':'p1','status':'unavailable','rationale':'Domain unresolved'}]}
    out=reviewed_labels(r,t)
    assert out['proposals'][1]['proposed_value'] is None
    assert out['semantic_admission']==0 and all(not p['optimization_allowed'] for p in out['proposals'])
    r['decisions'].pop()
    with pytest.raises(ValueError,match='missing'):reviewed_labels(r,t)
