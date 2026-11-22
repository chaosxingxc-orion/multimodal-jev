import pytest
from multimodal_jev.intent_triplet_sampling import sample_epoch


def test_triplets_cover_train_without_self_positives_or_false_negatives():
    records = [dict(id=f'{label}{i}', intent=label, role='learning', official_split='train')
               for label in ('up','down','mute') for i in range(4)]
    def sample(rows, epoch=1):
        return list(sample_epoch(rows, seed=318, epoch=epoch))
    triples = sample(records)
    assert triples == sample(reversed(records))
    assert triples != sample(records, 2)
    assert sorted(a for a,p,n in triples) == sorted(r['id'] for r in records)
    for a,p,n in triples:
        assert a != p and a[:-1] == p[:-1] != n[:-1]
    with pytest.raises(ValueError, match='TRAIN'):
        sample(records + [dict(records[0], id='held', role='test')])


def test_relative_negative_is_a_positive_member_of_its_own_card():
    import torch
    from multimodal_jev.intent_triplet_training import components
    markers = torch.tensor([[[1.,0.],[0.,1.],[0.,1.],[0.,0.]]],requires_grad=True)
    scores = torch.zeros((1,4),requires_grad=True)
    terms = components(markers,scores,.1,dict(margin=.1,decision_threshold=.5,gap_weight=1.,anchor_weight=.25))
    terms[0,0].backward()
    assert markers.grad[0,1,0] < 0 < markers.grad[0,2,0]
    assert torch.equal(markers.grad[0,3],torch.zeros(2))
    assert (scores.grad[0,:3] < 0).all() and scores.grad[0,3] > 0


def test_confusion_tail_respects_denominators_ties_and_reaches_all_classes():
    import torch
    from collections import Counter
    from multimodal_jev.intent_triplet_training import confusion_distribution
    labels = ['A','B','C','D']
    means = torch.tensor([[1.,.8,.1,.1],[.8,1.,.2,.4],[.1,.2,1.,.5],[.1,.4,.5,1.]],dtype=torch.float64)
    counts = torch.tensor([20,2,4,12])
    table = confusion_distribution(labels,means*counts[:,None],counts,1.)
    assert set(table['A']) == {'B','C','D'}
    assert table['A']['B'] > table['A']['C'] == table['A']['D'] > 0
    records = [dict(id=f'{label}/{i}',intent=label,role='learning',official_split='train')
               for label in labels for i in range(600)]
    negatives = Counter(n[0] for a,p,n in sample_epoch(records,seed=3,epoch=1,distribution=table) if a[0]=='A')
    assert negatives['B'] > max(negatives['C'],negatives['D']) and set(negatives)=={'B','C','D'}
