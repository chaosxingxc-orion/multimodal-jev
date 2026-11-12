from collections import Counter
import pytest
from multimodal_jev.intent_actor_inputs import schedule


def test_cyclic_schedule_covers_every_source_without_partial_dp_batch():
    items=[{'id':str(i)} for i in range(89)]
    batches=[schedule(items,317,step) for step in range(89)]
    assert all(len(b)==8 and len({x['id'] for x in b})==8 for b in batches)
    assert set(Counter(x['id'] for b in batches for x in b).values())=={8}
    assert schedule(list(reversed(items)),317,10)==batches[10]
    assert schedule(items,318,10)!=batches[10]


@pytest.mark.parametrize('step,size',[(-1,8),(0,0),(0,90),(True,8)])
def test_rejects_invalid_schedule(step,size):
    with pytest.raises(ValueError):schedule([{'id':str(i)} for i in range(89)],317,step,size)
