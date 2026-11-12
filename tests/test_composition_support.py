import pytest
from multimodal_jev.composition_support import audit


def test_tree_leaf_edges_are_not_false_combinations():
    r=audit({'a':('s1','a1'),'b':('s1','a2'),'c':('s2','a2')},{'a':2,'b':1,'c':3})
    assert r['single_component_unique']=={'a':[1],'b':[],'c':[0]}
    assert r['all_single_components_ambiguous_labels']==['b']
    assert r['prospective_split']['heldout_labels']==['b']
    assert r['components'][0]['learning_gold_component_majority_accuracy']==5/6
    assert r['components'][1]['learning_gold_component_majority_accuracy']==5/6


def test_no_supported_component_disappears_with_multiple_removals():
    mapping={f'{i}{j}':(str(i),str(j)) for i in range(3) for j in range(3)}
    a=audit(mapping);b=audit(dict(reversed(list(mapping.items()))))
    assert a['prospective_split']==b['prospective_split']
    split=a['prospective_split'];assert split['heldout_labels']
    for label,witnesses in split['witnesses'].items():
        for j,ids in witnesses.items():
            assert ids and all(mapping[k][int(j)]==mapping[label][int(j)] for k in ids)


def test_unobserved_labels_cannot_supply_fake_training_witnesses():
    mapping={'a':('x','1'),'b':('x','2'),'c':('y','1'),'d':('y','2')}
    r=audit(mapping,{'a':1,'d':1})
    assert r['prospective_split']['heldout_labels']==[]
    assert r['prospective_split']['retained_labels']==['a','d']


def test_none_is_a_legal_component_and_empty_population_has_no_oracle_score():
    r=audit({'a':('on','none'),'b':('off','none')},{})
    assert r['components'][1]['degrees']=={'none':2}
    assert r['components'][0]['learning_gold_component_majority_accuracy'] is None
    assert not r['admission']


def test_pair_shortcuts_and_true_three_component_ambiguity():
    mapping={str((i,j,k)):(str(i),str(j),str(k)) for i in range(2) for j in range(2) for k in range(2)}
    r=audit(mapping)
    assert len(r['requires_all_components_labels'])==8
    assert all(v==3 for v in r['minimum_identifying_components'].values())
    r=audit({'a':('x','u','1'),'b':('x','v','2'),'c':('y','u','2'),'d':('y','v','1')})
    assert len(r['all_single_components_ambiguous_labels'])==4
    assert r['requires_all_components_labels']==[]
    assert all(v==2 for v in r['minimum_identifying_components'].values())


@pytest.mark.parametrize('mapping,counts', [({},None),({'a':('s','a'),'b':('s','a')},None),
    ({'a':('s','a')},{'missing':1}),({'a':('s','a')},{'a':-1}),({'a':('s','a')},{'a':True}),
    ({'a':('s','')},None)])
def test_invalid_ontology_or_population_rejected(mapping,counts):
    with pytest.raises(ValueError):audit(mapping,counts)
