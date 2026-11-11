from copy import deepcopy

import pytest
import torch

from multimodal_jev.omni_probability_batch import build_components
from multimodal_jev.omni_probability_shards import shard_plan, recompute_shard, merge_shards
from multimodal_jev.omni_probability_audit import compare
from multimodal_jev.sfc_reward_batch import digest
from test_omni_actor_forward import Batch
from test_omni_probability_batch import fixture
from test_omni_probability_capture import actor_fixture, protocol


class SliceBatch(Batch):
    def __getitem__(self, indices):
        return SliceBatch({k: v[indices] for k, v in self.batch.items()},
                          {k: v[indices] for k, v in self.non_tensor_batch.items()},
                          self.meta_info)


def setup(real=False):
    args = fixture()
    for record in args['records']:
        record['log_probs'] = [-token/8 for token in record['token_ids']]
    parts = build_components(**args)
    if real:
        from verl import DataProto
        batch = DataProto.from_dict(**parts)
    else:
        batch = SliceBatch(parts['tensors'], parts['non_tensors'], parts['meta_info'])
    return args['records'], batch


@pytest.mark.parametrize('real', [False, True])
def test_complete_rank_capture_reordered_input_and_numeric_audit(real):
    if real:
        pytest.importorskip('verl')
    records, batch = setup(real)
    old = batch.batch['old_log_probs'].clone(); meta = deepcopy(batch.meta_info)
    actor, calls = actor_fixture(token_specific=True)
    runs = []
    for size in (1, 2):
        plan = shard_plan(records, world_size=2, micro_batch_size=size)
        shards = [recompute_shard(actor, batch[[2, 0, 3, 1]], records,
                  name=f'micro{size}', rank=rank, plan=plan, plan_sha256=digest(plan),
                  pad_token_id=0) for rank in (1, 0)]
        runs.append(merge_shards(records, shards, name=f'micro{size}',
                                plan=plan, plan_sha256=digest(plan)))
        assert plan['forwards_per_rank'] == 2 // size
    p = protocol(records)
    assert compare(records, runs, protocol=p, protocol_sha256=digest(p))['numerical_gate_passed']
    assert calls == [1, 1, 2, 2]
    assert torch.equal(old, batch.batch['old_log_probs']) and batch.meta_info == meta


@pytest.mark.parametrize('kind', ['missing_rank', 'duplicate_rank', 'wrong_rank', 'missing_response',
    'duplicate_response', 'wrong_assignment', 'tokens', 'record_hash', 'plan_hash',
    'roster_hash', 'name', 'nan', 'positive', 'bool_rank'])
def test_merge_rejects_incomplete_or_misassigned_evidence(kind):
    records, batch = setup(); actor, _ = actor_fixture(token_specific=True)
    plan = shard_plan(records, world_size=2, micro_batch_size=1)
    shards = [recompute_shard(actor, batch, records, name='micro1', rank=rank,
              plan=plan, plan_sha256=digest(plan), pad_token_id=0) for rank in (0, 1)]
    s = shards[0]; response = s['run']['responses'][0]
    if kind == 'missing_rank': shards.pop()
    if kind == 'duplicate_rank': shards[1] = deepcopy(s)
    if kind == 'wrong_rank': s['rank'] = 2
    if kind == 'bool_rank': s['rank'] = False
    if kind == 'missing_response': s['run']['responses'].pop()
    if kind == 'duplicate_response': s['run']['responses'].append(deepcopy(response))
    if kind == 'wrong_assignment':
        s['run']['responses'], shards[1]['run']['responses'] = shards[1]['run']['responses'], s['run']['responses']
    if kind == 'tokens': response['token_ids'][0] += 1
    if kind == 'record_hash': response['record_sha256'] = 'c'*64
    if kind == 'plan_hash': s['plan_sha256'] = 'c'*64
    if kind == 'roster_hash': s['records_sha256'] = 'c'*64
    if kind == 'name': s['run']['name'] = 'other'
    if kind == 'nan': response['log_probs'][0] = float('nan')
    if kind == 'positive': response['log_probs'][0] = 1.
    with pytest.raises(ValueError):
        merge_shards(records, shards, name='micro1', plan=plan, plan_sha256=digest(plan))


def test_unequal_schedule_and_changed_plan_rejected_before_forward():
    records, batch = setup(); actor, calls = actor_fixture()
    with pytest.raises(ValueError, match='divide'):
        shard_plan(records, world_size=3, micro_batch_size=1)
    with pytest.raises(ValueError, match='divide'):
        shard_plan(records, world_size=2, micro_batch_size=3)
    plan = shard_plan(records, world_size=2, micro_batch_size=1)
    plan['rank_indices'].reverse()
    with pytest.raises(ValueError, match='unchanged'):
        recompute_shard(actor, batch, records, name='micro1', rank=0, plan=plan,
                        plan_sha256=digest(plan), pad_token_id=0)
    assert calls == []
