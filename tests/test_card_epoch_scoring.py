import json

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import content_key
from tools.reports import audit_card_scoring_benchmark as audit
from tools.reports import stop_card_stage1_at_epoch as handoff
from tools.reports.card_scoring_benchmark import scoring_lineage
from tools.reports.native_seal_operations import container_memory, verify_memory


def write(path, value):
    path.write_text(json.dumps(value))
    return file_hash(path)


def test_handoff_checks_all_checkpoint_artifacts(tmp_path, monkeypatch):
    formal = tmp_path/'formal'
    root = formal/'run'
    root.mkdir(parents=True)
    train = dict(checkpoint_interval=2, stop_update=4, physical_devices=[0, 1],
                 engineering_prefix=False, source_policy_id='parent', source_contract={'base_id': 'base'})
    config = dict(updates=2, hash_timeout_seconds=30, training={'sha256': write(formal/'config.json', train)})
    execution = root/'execution-0002.pt'
    execution.write_bytes(b'execution')
    shards = {}
    for rank in range(2):
        directory = root/f'rank-{rank}'
        directory.mkdir()
        shard = directory/'checkpoint-0002.pt'
        shard.write_bytes(bytes([rank]))
        shards[str(shard.relative_to(root))] = file_hash(shard)
        write(directory/'update-0002.json', dict(update=2, checkpoint=True, metrics={'actor/pg_loss': .5}))
    policy = root/'policy-0002'
    policy.mkdir()
    weights = policy/'execution.pt'
    weights.write_bytes(b'endpoint')
    manifest = dict(schema='context-policy-execution-v1', contract=train['source_contract'],
                    files={'execution.pt': file_hash(weights)})
    policy_id = handoff.content_key_policy(manifest)
    manifest['policy_id'] = policy_id
    write(root/'checkpoint-0002-committed.json', dict(updates=2, joint_policy_directory=policy.name,
        joint_policy_manifest_sha256=write(policy/'MANIFEST.json', manifest), rank_shards=shards,
        execution_file=execution.name, execution_sha256=file_hash(execution),
        joint_policy=dict(source_policy_id='parent', policy_id=policy_id, contract=train['source_contract'])))

    def local(_, *args, **kwargs):
        import subprocess
        return subprocess.run(args, check=True, capture_output=True, timeout=kwargs.get('timeout'))
    monkeypatch.setattr(handoff, 'remote', local)
    launch = dict(host='test', remote_root=str(tmp_path))
    assert handoff.endpoint(config, train, launch)['policy_id'] == policy_id
    weights.write_bytes(b'wrong weights')
    with pytest.raises(ValueError, match='bytes differ'):
        handoff.endpoint(config, train, launch)


def test_full_train_audit_checks_coverage_and_content(tmp_path):
    root, entry = tmp_path/'score', tmp_path/'entry'
    root.mkdir()
    entry.mkdir()
    panel = dict(labels=['x', 'y'], rows=[{'id': 'a'}], population=2)
    cfg = dict(devices=[0], widths=[2], streaming={'length_window': 8},
               torch_threads=2, max_model_len=100, io_workers=2, prefetch_audio=4,
               cpu_workers=2, audit_min_parallel_blocks=2,
               formal=dict(policy_id='policy', image='image', recordings_per_block=1, updates=143,
                           checkpoint_sha256='checkpoint'),
               policy_sha256='manifest', cards={'sha256': 'cards'}, source={'sha256': 'source'})
    cfg['training'] = dict(sha256=write(entry/'training.json', {
        'epoch_mining': {'forward_rows': 1, 'zipf_exponent': 1.3}}))
    cfg['full'] = dict(sha256=write(entry/'full.json', dict(panel,
        rows=[{'id': 'a', 'intent': 'x'}, {'id': 'b', 'intent': 'y'}],
        source_training_sha256=cfg['training']['sha256'])))
    cfg['panel'] = dict(sha256=write(entry/'panel.json', panel))
    write(entry/'score.json', {'macros': {'NATIVE_SEAL_CONTAINER_CONFIG': cfg}})
    directory = root/'worker-0'
    directory.mkdir()
    token = scoring_lineage(cfg, 0)
    write(directory/'lineage.json', {'sha256': token})

    def payload(uid, sizes):
        return dict(records=[dict(id=uid, prob=[.25, .75])], seconds=2, peak_bytes=1,
                    batch_sizes=sizes, input_timing=dict(workers=2, ahead=4, first_batch=.1,
                                                        input_wait=.2, forward=1))
    write(directory/'baseline.json', payload('a', {'1': 2}))
    write(directory/'batch-2.json', payload('a', {'2': 1}))
    assert not audit.progress(root, entry)['admitted']
    with pytest.raises(FileNotFoundError):
        audit.audit_admission(root, cfg, panel, 1)
    write(directory/'panel-ready.json', dict(lineage=token,
        baseline_sha256=file_hash(directory/'baseline.json'), candidate_sha256=file_hash(directory/'batch-2.json')))
    gate = write(root/'admission.json', audit.audit_admission(root, cfg, panel, 1))
    blocks = []
    for rank, uid in enumerate(('a', 'b')):
        name = f'block-{rank:06d}.json'
        block = payload(uid, {'2': 1})
        blocks.append(dict(name=name, sha256=write(directory/name, dict(
            payload=block, lineage=token, sha256=content_key(block))), reused=rank == 0))
    worker = dict(blocks=blocks, device=0, rows=2, lineage=token, scoring_seconds=4)
    write(directory/'result.json', worker)
    receipt = dict(policy_id='policy', policy_sha256='manifest', full_sha256=cfg['full']['sha256'],
                   workers=[worker], worker_wall_seconds=5, admission_sha256=gate)
    result = audit.audit_formal(root, entry, cfg, panel, receipt, 'container')
    assert (result['pairs'], result['reused_recordings'], result['panel_pairs']) == (4, 1, 2)
    assert result['confusion']['counts'] == [1, 1]
    assert result['confusion']['distribution'] == {'x': {'y': 1.}, 'y': {'x': 1.}}
    assert audit.progress(root, entry)['recordings'] == 2
    saved = directory/blocks[0]['name']
    data = saved.read_bytes()
    saved.unlink()
    assert audit.progress(root, entry)['recordings'] == 1
    with pytest.raises(ValueError, match='missing, extra or out of order'):
        audit.audit_formal(root, entry, cfg, panel, receipt, 'container')
    saved.write_bytes(data)
    tampered = json.loads(data)
    tampered['payload']['records'][0]['prob'][0] = .3
    write(saved, tampered)
    with pytest.raises(ValueError, match='identity differs'):
        audit.progress(root, entry)
    saved.write_bytes(data)

    changed = payload('a', {'2': 1})
    blocks[1]['sha256'] = write(directory/blocks[1]['name'], dict(
        payload=changed, lineage=token, sha256=content_key(changed)))
    write(directory/'result.json', worker)
    with pytest.raises(ValueError, match='recording IDs'):
        audit.audit_formal(root, entry, cfg, panel, receipt, 'container')


def test_container_uses_configured_floor_and_checks_docker():
    floor = container_memory('128g', 1024)
    assert floor == 2**40
    assert container_memory('1536g', 1024) == 1536*2**30
    with pytest.raises(ValueError, match='actual Docker'):
        verify_memory({'HostConfig': {'Memory': 128*2**30}}, ['--memory', str(floor)])
