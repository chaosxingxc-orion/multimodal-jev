import torch
from multimodal_jev.context_choice import request, distribution, projection, shortlist
from multimodal_jev.context_candidate_task import compile_candidates

SPEC=dict(temperature=1.,beta=1.,reward_gain=1.,shortlist_size=2,seed=317,shortlist_dropout=.25)

def test_open_miss_only_changes_coverage():
    z=torch.tensor([[1.,2.,0.]],requires_grad=True)
    projection(z,z.detach(),torch.tensor([-1]),'open',SPEC).sum().backward()
    assert not z.grad[0,:-1].any() and z.grad[0,-1]>0
    assert torch.allclose(distribution(z,'open',1.).exp().sum(-1),torch.ones(1))

def test_readout_moves_gold_against_competition():
    for mode in ['open','closed']:
        z=torch.zeros((1,3),requires_grad=True)
        projection(z,z.detach(),torch.tensor([1]),mode,SPEC).sum().backward()
        assert z.grad[0,1]<0 and z.grad[0,0]>0
        if mode=='open': assert z.grad[0,-1]<0

def test_compiler_only_displays_proposal():
    ids=['a','b','c']
    chosen=shortlist([.1,.5,.4],ids,'audio-1',SPEC)
    native=compile_candidates('wave',[dict(id=k,definition='intent '+k) for k in ids],['listen'],semantics='exclusive_event')
    q=request(native,'open',chosen)
    assert [t.id for t in q.questions[:-1]]==[ids[i] for i in chosen]
    assert q.questions[-1].id=='__list_covered__'
    assert request(native,'closed',[0,1,2]).model_input()!=native.model_input()

def test_open_scoring_separates_bad_retrieval_and_false_accept():
    from multimodal_jev.choice_metrics import choice_metrics
    assets=[dict(id=str(i),audio_sha256=str(i)) for i in range(3)]
    truth=[dict(a,gold=g) for a,g in zip(assets,['a','b','b'])]
    pred=[dict(id=str(i),slate=['a'],closed=[.75,.25],open=s)
          for i,s in enumerate(([.8,.2],[.1,.9],[.8,.2]))]
    m=choice_metrics(assets,truth,pred,['a','b'])
    assert m['coverage_rate']==1/3 and m['conditional_accuracy']==1
    assert m['false_accept_rate']==.5 and m['open_accuracy']==2/3


def test_variable_slate_support_and_missing_label_at_singleton():
    from multimodal_jev.context_choice import lengths, varied_slate
    ids = ['a', 'b', 'c']
    config = dict(SPEC, candidate_lengths=[1, 2, 8, 'all'], proposal_profiles=['uniform'])
    assert lengths(config, len(ids)) == [1, 2, 3]
    for k in lengths(config,len(ids)):
        one=varied_slate([.8,.5,.2],ids,'u',config,k,0)
        assert one==varied_slate([.2,.5,.8],ids,'u',config,k,0)
        assert len(set(one[0]))==k
    z=torch.zeros((2,2),requires_grad=True)
    projection(z,z.detach(),torch.tensor([0,-1]),'open',SPEC).sum().backward()
    assert not z.grad[:,0].any()
    assert z.grad[0,1] < 0 < z.grad[1,1]


def test_full_panel_reject_counts_as_error_and_brier_includes_none():
    import math,pytest
    from multimodal_jev.choice_metrics import full_ontology_metrics
    a=[dict(id='r',audio_sha256='s')]; gold=[dict(**a[0],gold='a')]
    r=full_ontology_metrics(a,gold,[dict(id='r',logp=[math.log(x) for x in (.3,.1,.6)])],['a','b'])
    assert r['false_none_rate']==1 and r['top1_accuracy']==0 and r['top1_macro_f1']==0
    assert r['categorical_brier']==pytest.approx(.49+.01+.36)
    assert r['categorical_nll']==pytest.approx(-math.log(.3))


def test_semantic_choice_preserves_question_and_per_item_definitions():
    from multimodal_jev.context_decision_contract import ApplicabilityQuestion,ContextDecisionRequest
    req=ContextDecisionRequest('wave',tuple(ApplicabilityQuestion(k,d,'Does this answer the question?')
        for k,d in [('A','a location'),('B','an argument')]),('What does point mean here?',),
        version='jev-semantic-answer-v1')
    out=request(req,'open',[1,0])
    assert out.requirements[0]==req.requirements[0]
    assert [q.definition for q in out.questions[:-1]]==['an argument','a location']
    assert out.questions[-1].id=='__list_covered__'
