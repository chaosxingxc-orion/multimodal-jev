import random
import pytest
from multimodal_jev.context_length_buckets import plan
from multimodal_jev.span_length_groups import length_group_indices
from tools.reports.context_bucket_trainer_candidate import generate,repartition


@pytest.mark.parametrize('micro',[8,16,32])
def test_fixed_membership_weights_and_balanced_proxy(micro):
    rng=random.Random(317)
    lengths=[(rng.randrange(100,4096),rng.randrange(100,2000)) for _ in range(128)]
    ranks,r=plan(lengths,microbatch=micro,world_size=2)
    assert all(len(x)==64 for x in ranks)
    assert sorted(i for x in ranks for i in x)==list(range(128))
    assert r['candidate']['slowest_rank_cost_proxy']<=r['baseline']['slowest_rank_cost_proxy']
    assert (ranks,r)==plan(lengths,microbatch=micro,world_size=2)
    weights=[1]*65+[0]*63
    assert sum(weights[i] for x in ranks for i in x)==65


def test_actual_frozen_schedule_admitted_before_new_permutation():
    lengths=[(1000+i%7,100+(i*39)%800) for i in range(128)]
    old=length_group_indices(lengths,microbatch=8,world_size=4,order='length_grouped')
    ranks=repartition(lengths,old,microbatch=8,world_size=2,settings={})
    assert sorted(i for r in ranks for i in r)==list(range(128))
    old[0][0]=old[0][1]
    with pytest.raises(ValueError):repartition(lengths,old,microbatch=8,world_size=2,settings={})


def test_generated_native_verl_loop_keeps_objective_and_pins_strategy():
    from pathlib import Path
    source=generate(Path.cwd())
    assert "settings=c['length_buckets']" in source
    assert "length_buckets=c['length_buckets'],macro_layout=layout" in source
    assert 'actor.update_policy' in source


def test_invalid_bucket_settings_fail():
    with pytest.raises(ValueError):plan([(10,10)]*128,microbatch=8,world_size=2,frame_bucket_width=0)
