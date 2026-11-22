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
