import json
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.card_device_config import device_count, tpo_schedule
from multimodal_jev.context_topology_batch_plan import derive
from multimodal_jev.local_config import resolve
from tools.reports.build_tpo_gap_targets import shuffled_batches
from tools.reports.native_seal_operations import idle_npus
from tools.reports.prepare_card_scoring_benchmark import formal_policy
from tools.reports.prepare_tpo_evaluation import queue_jobs
from tools.reports.prepare_tpo_windows import prepare


OPS = Path('configs/portable/operations')


@pytest.fixture(autouse=True)
def temporary_test_storage(monkeypatch):
    # Test runners may put tmp_path on tmpfs. Keep real writes and package hashes.
    monkeypatch.setattr('multimodal_jev.local_checkpoint.check_paths', lambda paths: None)
    monkeypatch.setattr('multimodal_jev.source_snapshot.check_paths', lambda paths: None)


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value))
    return file_hash(path)


@pytest.mark.parametrize('pool', [[3], [2, 6], [1, 3, 5, 7]])
def test_stage1_budget_and_eval_workers_follow_pool(tmp_path, pool):
    window = read(OPS/'card-stage1-device-pool-v1.json')['window']
    batch = read(Path('configs/experiments/context-two-stage-margin-r64-dp8-v1.json'))['stage1']['batch_config']
    plan = derive(dict(batch, training_devices=device_count(pool, window, 'training_npus'),
                       batch_rule=window['batch_rule'], microbatch=window['execution_microbatch'],
                       per_dp_replica_batch=window['effective_anchors_per_rank'], source_queries=129))
    assert plan['global_batch'] == len(pool)*16
    assert plan['updates_per_epoch'] == (128 // (len(pool)*16)) + 1
    settings, _ = resolve(read(OPS/'card-paired-device-pool-v1.json')['sample_blocks'], {'CARD_EVAL_DEVICES': pool})
    config = dict(sample_blocks=settings, cache_container={}, eval_container={})
    prep, workers, queue = queue_jobs(config, tmp_path, {}, {'test': {'logical_partitions': 7}},
                                      {}, {}, set(), 'image')
    assert queue['partitions'] == {'test': 7}
    assert prep['devices'] == []
    assert [w['devices'] for w in workers] == [[d] for d in pool]
    for job in workers:
        assert read(tmp_path/(job['name']+'.json'))['macros']['NATIVE_SEAL_CONTAINER_CONFIG']['devices'] == job['devices']


@pytest.mark.parametrize('pool, expected', [([], 0), ([True], 1), ([-1], 1), ([1, 1], 2),
                                            ([1, 3], 8), ([1], True)])
def test_invalid_or_stale_physical_bindings(pool, expected):
    with pytest.raises(ValueError):
        device_count(pool, {'replicas': expected}, 'replicas')


def test_scoring_changes_cards_but_requires_original_trainer(tmp_path, monkeypatch):
    pool = [0, 2, 4, 6]
    mounts = [dict(PathOnHost=f'/dev/davinci{i}', PathInContainer=f'/dev/davinci{i}') for i in pool]
    capture = dict(Id='container', HostConfig=dict(Devices=mounts),
                   State=dict(Status='exited', ExitCode=0, Running=False, OOMKilled=False, Pid=0))
    proof = dict(completed_window=True, passed=True, container_exited=True, updates=20,
                 config_sha256='config', host='host', devices=pool, container_id='container', exit_code=0,
                 policy='/output/policy-0020', policy_id='policy', manifest_sha256='manifest',
                 checkpoint_sha256='checkpoint', inspect_sha256=write(tmp_path/'inspect.json', capture))
    write(tmp_path/'receipt.json', proof)
    c = dict(formal=dict(epoch_handoff=str(tmp_path)), inputs=dict(training=dict(sha256='config')))
    train = dict(stop_update=20, physical_devices=pool)
    scoring = dict(host='host', devices=[1, 3])
    monkeypatch.setattr('subprocess.check_output', lambda *a, **kw:
                        'manifest  /output/policy-0020/MANIFEST.json\n'
                        'checkpoint  /output/checkpoint-0020-committed.json\n')
    assert formal_policy(c, train, scoring)[2] == 'policy'
    write(tmp_path/'receipt.json', dict(proof, devices=scoring['devices']))
    with pytest.raises(ValueError, match='endpoint'):
        formal_policy(c, train, scoring)
    mounts.pop()
    write(tmp_path/'receipt.json', dict(proof, inspect_sha256=write(tmp_path/'inspect.json', capture)))
    with pytest.raises(ValueError, match='endpoint'):
        formal_policy(c, train, scoring)


def test_tpo_seals_disjoint_physical_devices_and_matching_view(tmp_path):
    bindings = read(Path('configs/experiments/card-device-bindings.example.json'))['macros']
    layout = bindings['CARD_STAGE2_LAYOUT']
    schedule = tpo_schedule(layout, read(OPS/'card-tpo-targets-device-pool-v1.json')['schedule'])
    assert schedule['global_batch'] == 256
    batches = shuffled_batches([str(i) for i in range(257)], schedule)
    assert [sum(s['weight'] for s in b['slots']) for b in batches] == [256, 1]
    with pytest.raises(ValueError, match='schedule differs'):
        tpo_schedule(layout, dict(schedule, global_batch=384))
    with pytest.raises(ValueError, match='four complete'):
        tpo_schedule(layout, dict(schedule, microbatch=64))
    data = tmp_path/'targets'
    data.mkdir()
    write(data/'MANIFEST.json', dict(schema='jev-tpo-reward-view-v1', source_policy='p',
          source_policy_sha256='p-hash', view_sha256='v', reference_sha256='r', card_input={'arm': 'a'},
          bidirectional_contrast={}, schedule=schedule, files={}))
    parent = tmp_path/'parent.json'
    write(parent, dict(source_policy_id='p', physical_devices=layout['training_devices'],
                       batch_config=dict(per_dp_replica_batch=layout['per_dp_replica_batch']),
                       training_dataset=dict(manifest_sha256='v'), reference_dataset=dict(manifest_sha256='r'),
                       view_execution=dict(container=dict(cpus=1, memory_gib=4, shm_gib=1))))
    pkg = tmp_path/'package'
    c = dict(package=str(pkg), repository=str(Path.cwd()), targets=str(data), parent_config=str(parent),
             source_patterns=['src/multimodal_jev/card_device_config.py'], io_workers=1,
             deployment=dict(read_only_mounts=[], idle_devices=list(range(8))),
             card_input=dict(policy_sha256='p-hash', inference_cpus=1, inference_devices=[0]))
    with pytest.raises(ValueError, match='independent inference'):
        prepare(c)
    assert not pkg.exists()
    c['card_input']['inference_devices'] = bindings['CARD_STAGE2_INFERENCE_DEVICES']
    prepare(c)
    ops = read(pkg/'operations-local.json')['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    probes, finite, formal = ops['chains'][0]
    assert [j['devices'] for j in probes] == [[1], [5]]
    assert all(j['visible_ordinal'] == '0' for j in probes)
    assert finite[0]['devices'] == formal[0]['devices'] == [0, 2, 4, 6]
    assert 'idle_devices' not in ops
    leaf = read(pkg/'entry/tpo.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    assert leaf['training']['world_size'] == 4


def test_idle_admission_matches_exact_ids():
    assert idle_npus('| No running processes found in NPU 10 |\nNo running processes found in NPU 3') == {3, 10}
