import copy
from pathlib import Path

import pytest

from tools.reports.context_replica_trainer_candidate import (topology, generate, ALGORITHM,
    prepare_output, referenced_modules)
from test_context_topology_batch_plan import config


def candidate(world):
    settings=config(world)
    return dict(batch_config=settings,physical_devices=list(range(4,4+world)),
        global_batch=64*world,microbatch=16,algorithm=ALGORITHM,dual_arm='joint',
        dual_runtime_identity={'fixture':True},
        macro_client=dict(schema='jev-replica-macro-client-v1',max_queries=64*world,
            rollout_queries=64*world,replicas=[{}]*(4-world),runtime_identity={'fixture':True}))


@pytest.mark.parametrize('world',[2,3])
def test_layout_derives_local64_from_actual_device_scope(world):
    c=candidate(world);mounted=[f'davinci{d}' for d in c['physical_devices']]
    actual=topology(c,world,mounted)
    assert actual['local_batch']==64 and actual['global_batch']==64*world
    assert actual['total_updates']==(634 if world==2 else 424)
    for key,value in [('global_batch',64),('microbatch',32),('physical_devices',[4,4]),
                      ('algorithm','rloo'),('dual_runtime_identity',{})]:
        bad=copy.deepcopy(c);bad[key]=value
        with pytest.raises(ValueError):topology(bad,world,mounted)
    # A shared container may also mount inference cards; only the declared
    # training cards must be present.
    assert topology(c,world,mounted+['davinci6','davinci7'])['world_size']==world
    with pytest.raises(ValueError):topology(c,world,mounted[:-1])


def test_entrypoint_output_creation_is_idempotent_across_torchrun_ranks(tmp_path):
    # torchrun executes the same entrypoint in every rank; both may create the
    # shared output root. A second call must not raise FileExistsError.
    first = prepare_output(tmp_path / 'run')
    second = prepare_output(tmp_path / 'run')
    assert first == second == tmp_path / 'run' and first.is_dir()


def test_every_transform_introduced_module_ships_with_the_package():
    # v3 failed with ModuleNotFoundError: multimodal_jev.slurp_portable_release:
    # the transform rewrote the official-training import but the packager only
    # copied modules from its SOURCES list. Every module the transform first
    # references must therefore be declared there.
    pytest.importorskip('torch')  # the packager imports the audio-preparation stack
    from tools.reports.prepare_context_replica_training_candidate import SOURCES, required_modules
    referenced = referenced_modules(Path.cwd())
    assert {'context_replica_schedule', 'slurp_portable_release'} <= set(referenced)
    # Packaged tool modules have their own imports; those the parent cannot
    # supply must be shipped explicitly (v5 failed on local_config this way).
    required = required_modules(Path.cwd())
    assert 'local_config' in required
    assert set(required) >= set(referenced) - {'_dual_credit_actor_candidate'}


def test_generated_native_trainer_uses_new_schedule_pool_and_global_formula():
    source=generate(Path.cwd())
    assert 'from multimodal_jev.context_replica_schedule import load_schedule' in source
    assert 'from tools.reports.context_replica_macro_client import connect_pool' in source
    assert "global_batch=c['global_batch']" in source
    assert "planned=rank_indices(batches[index],lengths,c['batch_config'])" in source
    for old in ('world_size=4','schedule_updates=634','global_batch=128)',"batches[index]['rank_indices']"):
        assert old not in source
    assert 'experiment_window(c)' in source
    assert "if commit['contract']!=contract" in source
    assert 'validate_trainable_scope(named)' in source
