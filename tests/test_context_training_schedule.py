from collections import Counter
import pytest
from multimodal_jev.context_training_schedule import build_epoch


def fixture():
    rows=[dict(id=str(i),intent='common' if i<17 else 'rare',role='learning',
        known_pending_source=False,official_split='train',source_group=str(i//2)) for i in range(23)]
    return rows,{str(i):(20+i%4,40+i) for i in range(23)}


def test_two_epochs_full_population_uniform_tail_and_rank_membership():
    rows,lengths=fixture()
    results=[build_epoch(rows,lengths,epoch=e,global_batch=8,world_size=2,microbatch=2) for e in (0,1)]
    assert results[0]!=results[1]
    assert results[0]==build_epoch(list(reversed(rows)),lengths,epoch=0,global_batch=8,world_size=2,microbatch=2)
    for result in results:
        seen=[]
        for batch in result['batches']:
            for indices in batch['rank_indices']:
                assert len(indices)==4
                for i in indices:
                    slot=batch['slots'][i]
                    assert slot['weight']==(0. if slot['padding'] else 1.)
                    if not slot['padding']:seen.append(slot['id'])
        assert Counter(seen)==Counter(r['id'] for r in rows)
        assert result['report']['padding_slots']==1


@pytest.mark.parametrize('fault',['selection','pending','missing_length','duplicate'])
def test_ineligible_population_rejected(fault):
    rows,lengths=fixture()
    if fault=='selection':rows[0]['role']='selection'
    if fault=='pending':rows[0]['known_pending_source']=True
    if fault=='missing_length':del lengths['0']
    if fault=='duplicate':rows.append(rows[0])
    with pytest.raises(ValueError):build_epoch(rows,lengths,epoch=0)
