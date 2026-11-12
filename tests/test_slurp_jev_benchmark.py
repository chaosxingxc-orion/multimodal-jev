import pytest
from multimodal_jev.slurp_jev_benchmark import ontology, score

BANK = {'intents':[{'id':'a'},{'id':'b'},{'id':'c'}],'slot_types':['date','time']}


def g(uid,intent,types):
    return dict(id=uid,source_group='source:'+uid,intent=intent,types=dict(zip(BANK['slot_types'],types,strict=True)))


def p(uid,ranking,types):
    return dict(id=uid,intent_ranking=ranking,types=types)


def test_masked_label_does_not_become_false_or_complete_set():
    r=score([g('x','a',[True,None])],[p('x',['a'],['date','time'])],BANK)
    assert r['types']['micro']['f1']==1
    assert r['types']['counts']['fp']==0
    assert r['types']['per_type']['time']['reviewed']==0
    assert r['types']['decision_coverage']==.5
    assert r['types']['set_exact'] is None
    assert r['records'][0]['type_set_exact'] is None


def test_missing_output_never_scores_as_correct_empty():
    r=score([g('x','a',[False,False]),g('y','b',[True,False])],[],BANK)
    assert r['intent']['top1']==0
    assert r['types']['set_exact']==0
    assert r['types']['positive_complete']==0
    assert r['types']['counts']==dict(tp=0,fp=0,fn=1,tn=0,invalid=4)
    assert r['types']['empty_false_positive_rate']==0
    assert r['types']['empty_failure_rate']==1
    assert r['counts']['missing_predictions']==2


def test_known_confusion_macro_and_full_denominator():
    gold=[g('1','a',[True,False]),g('2','b',[True,True]),g('3','a',[False,False])]
    pred=[p('1',['a','b'],['date','time']),p('2',['a','b'],['date']),p('3',['b','a'],[])]
    r=score(gold,pred,BANK)
    assert r['intent']['top1']==pytest.approx(1/3)
    assert r['intent']['top3']==1
    assert r['intent']['macro_f1_supported']==.25
    assert r['intent']['supported_classes']==2
    assert r['types']['micro']==dict(precision=2/3,recall=2/3,f1=2/3)
    assert r['types']['macro_f1_supported']==.5
    assert r['types']['set_exact']==pytest.approx(1/3)
    assert r['types']['positive_complete']==.5
    assert r['types']['empty_failure_rate']==0


@pytest.mark.parametrize('types',[['date','date'],['NONE_OF_FOLD'],['missing'],None,[1]])
def test_protocol_errors_invalidate_type_prediction_without_hiding_intent(types):
    r=score([g('x','a',[True,False])],[p('x',['a'],types)],BANK)
    assert r['intent']['top1']==1
    assert r['types']['counts']['fn']==1
    assert r['counts']['invalid_type_predictions']==1
    assert r['types']['set_exact']==0


@pytest.mark.parametrize('ranking',[[],['a','a'],['wrong'],None,[1]])
def test_invalid_intent_preserves_type_evaluation(ranking):
    r=score([g('x','a',[True,False])],[p('x',ranking,['date'])],BANK)
    assert r['intent']['top1']==0
    assert r['types']['set_exact']==1


def test_unreviewed_intent_not_scored():
    r=score([g('x',None,[True,False])],[p('x',['a'],['date'])],BANK)
    assert r['intent']['top1'] is None
    assert r['intent']['macro_f1_supported'] is None
    assert r['types']['set_exact']==1


@pytest.mark.parametrize('fault',['extra_id','duplicate_id','duplicate_gold','sparse_gold','integer_gold','unknown_intent'])
def test_artifact_and_gold_integrity(fault):
    gold=[g('x','a',[True,False])]; pred=[p('x',['a'],['date'])]
    if fault=='extra_id': pred.append(p('y',['a'],[]))
    if fault=='duplicate_id': pred.append(pred[0])
    if fault=='duplicate_gold': gold.append(gold[0])
    if fault=='sparse_gold': del gold[0]['types']['time']
    if fault=='integer_gold': gold[0]['types']['date']=1
    if fault=='unknown_intent': gold[0]['intent']='unknown'
    with pytest.raises(ValueError):score(gold,pred,BANK)


def test_alias_ambiguity_kept_and_fold_inventory_exact():
    tax={'scenario_action_pairs':[['calendar','query'],['weather','query']],
         'entity_types':['date','time'],'intent_alias_map':[
         {'scenario':'calendar','action':'query','raw_names':['query','calendar_query']},
         {'scenario':'weather','action':'query','raw_names':['query','weather_query']}]}
    folds=[{'fold_id':'a','types':['date']},{'fold_id':'b','types':['time']}]
    bank=ontology(tax,folds,['calendar_query','weather_query'])
    assert bank['ambiguous_raw_aliases']=={'query':['calendar_query','weather_query']}
    assert bank['raw_alias_count']==3
    with pytest.raises(ValueError):ontology(tax,folds+folds,['calendar_query','weather_query'])
    with pytest.raises(ValueError):ontology(tax,folds,['calendar_query'])
    with pytest.raises(ValueError):ontology(tax,[{'fold_id':'a','types':['date']}],['calendar_query','weather_query'])
