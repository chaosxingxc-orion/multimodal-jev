"""Length scheduling preserves update verification joins across DP ranks."""
import copy
import pytest
pytest.importorskip('torch')
from multimodal_jev.span_length_groups import length_group_indices
from multimodal_jev.simple_resident_acceptance import compare_updated_batch
from test_simple_resident_acceptance import fixture


@pytest.mark.parametrize('micro',[8,16])
def test_post_update_acceptance_joins_original_slots_after_length_permutation(monkeypatch,micro):
    ranks,rollout,sampling=fixture(monkeypatch)
    flat=[row for rank in ranks for row in rank]
    permutation=length_group_indices([(1952+i%29,100+i%53) for i in range(128)],
        microbatch=micro,world_size=4,order='length_grouped')
    reordered=[[flat[i] for i in indices] for indices in permutation]
    assert [r['request_id'] for rank in reordered for r in rank] != [r['request_id'] for r in flat]
    before=copy.deepcopy(sampling)
    report=compare_updated_batch(reordered,rollout,sampling,policy_id='base:updated',updates=2)
    assert report['requests']==128 and report['unique_inputs']==65
    assert sampling==before
    reordered[0][0]=dict(reordered[0][0],candidate_logprobs=[[-4.0943445622221]*60])
    with pytest.raises(ValueError):compare_updated_batch(reordered,rollout,sampling,policy_id='base:updated',updates=2)
