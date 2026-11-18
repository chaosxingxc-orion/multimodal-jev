import copy
from pathlib import Path

import pytest

from multimodal_jev.span_length_groups import length_group_indices
from tools.reports.context_macro_trainer_candidate import ALGORITHM, generate, repartition, topology


@pytest.mark.parametrize('micro', [8, 16, 32])
def test_dp2_repartition_preserves_every_optimizer_slot_and_micro_capacity(micro):
    lengths = [(100+i%9*31, 30+i%5*51) for i in range(128)]
    old = length_group_indices(lengths, microbatch=8, world_size=4, order='length_grouped')
    before = copy.deepcopy(old)
    ranks = repartition(lengths, old, microbatch=micro, world_size=2)
    assert old == before and len(ranks) == 2 and all(len(r) == 64 for r in ranks)
    assert sorted(i for r in ranks for i in r) == list(range(128))
    # Distinct exposure weights (including zero padding) keep the same global
    # numerator/denominator regardless of the new rank assignment.
    weights = [int(i < 121) for i in range(128)]
    assert sum(weights[i] for r in ranks for i in r) == 121
    bad = copy.deepcopy(old); bad[0][0], bad[1][0] = bad[1][0], bad[0][0]
    with pytest.raises(ValueError, match='binding'): repartition(lengths, bad, microbatch=micro, world_size=2)


@pytest.mark.parametrize('micro,accum', [(8, 8), (16, 4), (32, 2)])
def test_layout_and_rejection_of_old_topology_algorithm_or_runtime(micro, accum):
    config = dict(physical_devices=[4,5], global_batch=128, microbatch=micro,
        algorithm=ALGORITHM, dual_arm='joint', dual_runtime_identity={'profile':'actual'},
        macro_client=dict(max_queries=128, rollout_queries=32, runtime_identity={'profile':'actual'}))
    assert topology(config, 2, ['davinci4','davinci5'])['accumulation'] == accum
    for change in (dict(algorithm='dual_credit_rloo_clipped_v1'), dict(physical_devices=[6,7]),
                   dict(microbatch=64), dict(global_batch=64), dict(dual_runtime_identity={})):
        with pytest.raises(ValueError): topology(dict(config, **change), 2, ['davinci4','davinci5'])
    with pytest.raises(ValueError): topology(config, 4, ['davinci4','davinci5','davinci6','davinci7'])


def test_generated_trainer_keeps_finite_guard_and_uses_dp2_for_shards_and_collectives():
    source = generate(Path.cwd())
    for old in ('world_size=4', '[None]*4', 'range(4)', 'complete.split(32)', 'BoundDualClient('):
        assert old not in source
    assert "len(commit['rank_shards'])!=world" in source
    assert 'macro_layout=layout' in source
    assert "client=connect_pool(c['macro_client'],bundle=current)" in source
    assert 'experiment_window(c)' in source
    assert "contract['" not in source  # no contract fields stripped to accept DP4 recovery
    assert "if commit['contract']!=contract" in source
    assert 'validate_trainable_scope(named)' in source


@pytest.mark.parametrize('stop',[3,20,100])
def test_efficiency_window_enforces_explicit_global_update_budget(stop):
    from tools.reports.context_macro_trainer_candidate import experiment_window
    c=dict(start_update=0,stop_update=stop,experiment_kind='efficiency',
        checkpoint_interval=1,engineering_prefix=True,training_dispatch_allowed=False)
    assert experiment_window(c)=='efficiency'
    for interval in (1,2,stop):
        assert experiment_window(dict(c,checkpoint_interval=interval))=='efficiency'
    for change in (dict(stop_update=101),dict(stop_update=2),dict(start_update=1),
                   dict(resume_root='/resume'),dict(checkpoint_interval=0),
                   dict(checkpoint_interval=stop+1),dict(checkpoint_interval=1.0),
                   dict(checkpoint_interval=True),dict(checkpoint_interval=None),
                   dict(training_dispatch_allowed=True),dict(experiment_kind='typo')):
        with pytest.raises(ValueError):experiment_window(dict(c,**change))
    with pytest.raises(ValueError):experiment_window({k:v for k,v in c.items() if k!='experiment_kind'})
    short=dict(start_update=0,stop_update=2,checkpoint_interval=1,engineering_prefix=True)
    assert experiment_window(short)=='engineering_prefix'
    with pytest.raises(ValueError, match='every-update'):
        experiment_window(dict(short,checkpoint_interval=2))


def test_declared_cadence_gates_only_recovery_state_not_policy_export():
    from tools.reports.context_macro_trainer_candidate import CHECKPOINT_BLOCK
    assert "ckpt_completed%ckpt_interval==0 or index+1==stop_update" in CHECKPOINT_BLOCK
    # The optimizer shard and its receipt key are the only cadence-gated writes.
    assert CHECKPOINT_BLOCK.count('if save_now:') == 2
    assert "commit_json(root/f'checkpoint-{index+1:04d}-committed.json',ckpt_published)" in CHECKPOINT_BLOCK
    for unconditional in ("commit_file(root/f'execution-{index+1:04d}.pt'",
                          "export_checkpoint_policy(c['source_policy']",
                          "ckpt_published=dict(updates=index+1,contract=contract"):
        assert unconditional in CHECKPOINT_BLOCK
