import json

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import content_key
from tools.reports import audit_card_scoring_benchmark as audit
from tools.reports import stop_card_stage1_at_epoch as handoff
from tools.reports.card_scoring_benchmark import scoring_lineage
from tools.reports.native_seal_operations import container_memory, verify_memory


def test_instance_checkpoint_requires_separate_scoring_reference():
    from multimodal_jev.card_scoring_settings import scoring_settings

    training = {'instance_negative_manifest': {'sha256': 'manifest'}}
    reference = dict(forward_rows=32, card_arm='rules', zipf_exponent=1.0)
    assert scoring_settings(training, reference) == reference
    for bad in (None, dict(reference, forward_rows=True), dict(reference, zipf_exponent=float('nan'))):
        with pytest.raises(ValueError, match='explicit scoring reference'):
            scoring_settings(training, bad)
    assert scoring_settings({'class_scoring': reference}) == reference
    with pytest.raises(ValueError, match='historical'):
        scoring_settings({'class_scoring': reference}, reference)


def test_parallel_tail_rejects_gaps_and_overlap_for_uneven_shards():
    from tools.reports.card_scoring_shards import check_partition

    source = dict(devices=[0, 1], formal=dict(recordings_per_block=2))

    def lane(lower, upper):
        return dict(formal=dict(recordings_per_block=2, block_range=[lower, upper]))

    check_partition(source, [lane(1, 2), lane(2, None)], 11, 1)
    for wrong in ([lane(1, 2), lane(1, None)], [lane(2, None)], [lane(1, 2)]):
        with pytest.raises(ValueError, match='exactly once'):
            check_partition(source, wrong, 11, 1)


def test_parallel_lanes_can_cover_entire_population():
    from tools.reports.card_scoring_shards import check_partition

    source = dict(devices=[0, 1], formal=dict(recordings_per_block=2))

    def lane(lower, upper):
        return dict(formal=dict(recordings_per_block=2, block_range=[lower, upper]))

    check_partition(source, [lane(0, 1), lane(1, None)], 11, 0)
    for wrong in ([lane(0, 2), lane(1, None)], [lane(1, None)], [lane(0, 1)]):
        with pytest.raises(ValueError, match='exactly once'):
            check_partition(source, wrong, 11, 0)


def write(path, value):
    path.write_text(json.dumps(value))
    return file_hash(path)


def test_concurrent_train_pilot_uses_the_same_blocks_as_the_formal_result(tmp_path, monkeypatch):
    from tools.reports.card_scoring_benchmark import (
        commit_scoring_block, concurrent_offsets, merge_replicas, replica_path, scoring_offsets,
    )

    monkeypatch.setattr('multimodal_jev.local_checkpoint.check_paths', lambda _: None)
    c = dict(devices=[4], widths=[2], streaming={'length_window': 8}, io_workers=1,
             prefetch_audio=2, torch_threads=1, max_model_len=512, cpu_workers=1,
             audit_min_parallel_blocks=4, formal=dict(recordings_per_block=2, policy_id='end',
                 image='image', checkpoint_sha256='checkpoint'),
             concurrent_scoring=dict(replicas_per_device=2, max_batch_audio_chunks=8, pilot_recordings=4),
             policy_sha256='manifest', training={'sha256': 'training'}, cards={'sha256': 'cards'},
             source={'sha256': 'source'}, full={'sha256': 'roster'})
    rows = [dict(id=str(i), synthetic=bool(i % 2)) for i in range(11)]
    full = dict(rows=rows, labels=['x', 'y'])
    lineage = scoring_lineage(c, 0)
    offsets = scoring_offsets(c, len(rows))
    all_offsets = []
    for lane in range(2):
        directory = replica_path(tmp_path, 0, lane)
        directory.mkdir(parents=True)
        pilot, remaining = concurrent_offsets(c, offsets, lane, len(rows))
        all_offsets.extend(pilot + remaining)
        receipts = {}
        for off in pilot + remaining:
            path = directory.parent/f'block-{off//2:06d}.json'
            group = rows[off:off+2]
            commit_scoring_block(path, lineage, dict(records=[dict(id=r['id'], prob=[.7, .1]) for r in group],
                seconds=1, peak_bytes=1, batch_sizes={2: len(group)},
                input_timing=dict(workers=1, ahead=2, first_batch=.1, input_wait=.2, forward=.7)))
            receipts[off] = dict(name=path.name, sha256=file_hash(path), reused=lane == 1 and off in pilot)
        write(directory/'pilot.json', dict(blocks=[receipts[off] for off in pilot],
              started=1 if lane == 0 else None, finished=3 if lane == 0 else None))
        write(directory/'result.json', dict(device=4, lineage=lineage, cache_and_model_seconds=.2,
              scoring_seconds=4, rows=sum(len(rows[off:off+2]) for off in pilot+remaining),
              blocks=[receipts[off] for off in sorted(receipts)]))
    assert sorted(all_offsets) == list(offsets)
    result = audit.audit_pilot(tmp_path, c, full)
    assert (result['recordings'], result['human'], result['synthetic'], result['new_recordings']) == (4, 2, 2, 2)
    assert result['new_pairs_per_second'] is None  # Do not extrapolate a partially resumed window.
    merge_replicas(c, tmp_path)
    worker = audit.read(tmp_path/'worker-0'/'result.json')
    assert worker['rows'] == len(rows)
    audit.audit_replica_results(tmp_path, c, 0, worker, offsets, [r['id'] for r in rows])
    pilot_file = replica_path(tmp_path, 0, 0)/'pilot.json'
    broken = audit.read(pilot_file)
    broken['blocks'][0]['sha256'] = 'different'
    write(pilot_file, broken)
    with pytest.raises(ValueError, match='pilot block identity'):
        audit.audit_pilot(tmp_path, c, full)


def test_concurrent_train_rejects_one_inexact_replica(tmp_path):
    from tools.reports.card_scoring_benchmark import replica_path

    c = dict(devices=[4], widths=[2], streaming={'length_window': 8}, io_workers=1,
             prefetch_audio=2, torch_threads=1, max_model_len=512,
             formal=dict(recordings_per_block=2, policy_id='end', image='image', checkpoint_sha256='checkpoint'),
             concurrent_scoring=dict(replicas_per_device=2, max_batch_audio_chunks=8, pilot_recordings=4),
             policy_sha256='manifest', training={'sha256': 'training'}, cards={'sha256': 'cards'},
             source={'sha256': 'source'}, full={'sha256': 'roster'})
    panel = dict(labels=['x', 'y'], rows=[{'id': 'recording'}])
    lineage = scoring_lineage(c, 0)
    data = dict(records=[dict(id='recording', prob=[.75, .25])], seconds=1, peak_bytes=1,
                input_timing=dict(workers=1, ahead=2, first_batch=.1, input_wait=.2, forward=.7))
    for lane in range(2):
        replica_path(tmp_path, 0, lane).mkdir(parents=True)
    directory = tmp_path/'worker-0'
    baseline = write(directory/'baseline.json', dict(data, batch_sizes={'1': 2}))
    write(directory/'lineage.json', {'sha256': lineage})
    for lane in range(2):
        directory = replica_path(tmp_path, 0, lane)
        proof = {'baseline_sha256': baseline} if lane == 0 else {}
        result = write(directory/'batch-2.json', dict(data, batch_sizes={'2': 1}))
        write(directory/'panel-ready.json', dict(lineage=lineage, candidate_sha256=result, **proof))
    assert audit.audit_admission(tmp_path, c, panel, 1)['passed']
    data['records'][0]['prob'][1] = .24
    result = write(directory/'batch-2.json', dict(data, batch_sizes={'2': 1}))
    write(directory/'panel-ready.json', dict(lineage=lineage, candidate_sha256=result))
    with pytest.raises(ValueError, match='replica 1: endpoint probabilities differ'):
        audit.audit_admission(tmp_path, c, panel, 1)


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


def test_scoring_checkpoint_round_trip_preserves_legacy_blocks(tmp_path, monkeypatch):
    from multimodal_jev import local_checkpoint
    from tools.reports.card_scoring_benchmark import commit_scoring_block, read_block

    monkeypatch.setattr(local_checkpoint, 'check_paths', lambda paths: None)
    # The on-device histogram has integer keys, including partial batches.
    payload = dict(batch_sizes={128: 33, 120: 1, 8: 1}, records=[dict(id='a', prob=[.75])])
    legacy = tmp_path/'legacy.json'
    write(legacy, dict(lineage='before', sha256=content_key(payload), payload=payload))
    stored = json.loads(legacy.read_bytes())['payload']
    assert content_key(stored) != content_key(payload)
    assert read_block(legacy, 'before') == stored
    current = tmp_path/'current.json'
    commit_scoring_block(current, 'after', payload)
    assert read_block(current, 'after') == stored
    with pytest.raises(ValueError, match='identity differs'):
        read_block(current, 'before')
    for path in (legacy, current):
        broken = json.loads(path.read_bytes())
        broken['payload']['records'][0]['prob'][0] = .5
        write(path, broken)
        with pytest.raises(ValueError, match='identity differs'):
            read_block(path, broken['lineage'])


def test_scoring_resume_rejects_changed_inputs_and_runtime():
    from tools.reports.prepare_card_scoring_benchmark import validate_resume_config

    old = dict(source=dict(root='/jev-entry/source', sha256='old'),
               formal=dict(checkpoint_sha256='step'), widths=[128], panel=dict(sha256='panel'))
    new = dict(old, source=dict(old['source'], sha256='new'),
               formal=dict(old['formal'], resume_from='/old', resume_config={'sha256': 'prior'}))
    old_files = {'tools/reports/card_scoring_benchmark.py': 'old', 'model.py': 'same'}
    new_files = dict(old_files, **{'tools/reports/card_scoring_benchmark.py': 'new'})
    validate_resume_config(new, old, new_files, old_files)
    with pytest.raises(ValueError, match='identical scoring inputs'):
        validate_resume_config(dict(new, widths=[224]), old, new_files, old_files)
    with pytest.raises(ValueError, match='identical scoring inputs'):
        validate_resume_config(new, old, dict(new_files, **{'model.py': 'changed'}), old_files)


def test_compact_roster_and_fsdp_gate_fail_closed(tmp_path):
    from tools.reports.card_scoring_benchmark import acoustic_binding, restore_binding

    part = dict(id='a', audio='/recording.wav', audio_sha256='audio', intent='x', synthetic=False)
    spec = dict(acoustic_template={'panel': {'questions': ['task']}},
                processor_sha256='processor', acoustic_encoding={'version': 'x'})
    binding = acoustic_binding(part, spec)
    common = {k: v for k, v in binding.items() if k != 'audio_sha256'}
    common['request'] = {'questions': ['task']}
    assert restore_binding(part, common) == binding
    second = dict(part, id='b', intent='y')
    panel = dict(labels=['x', 'y'], population=2,
                 rows=[dict(id=row['id'], intent=row['intent'], synthetic=False, binding=binding)
                       for row in (part, second)])
    config = dict(devices=[0], widths=[2], streaming={'length_window': 8},
                  torch_threads=2, max_model_len=100, io_workers=2, prefetch_audio=4,
                  probability_atol=2e-6, policy_sha256='manifest', cards={'sha256': 'cards'},
                  training={'sha256': 'train'}, source={'sha256': 'source'},
                  formal=dict(policy_id='policy', image='image', recordings_per_block=2,
                              checkpoint_sha256='checkpoint'))
    rows = [part, second]
    config['full'] = dict(sha256=write(tmp_path/'full.json', dict(panel, rows=rows, acoustic_common=common)))
    config['full_shards'] = [dict(sha256=write(tmp_path/'full-0.json', dict(rows=rows, acoustic_common=common)))]
    assert audit.audit_roster(tmp_path, config, panel)['rows'] == rows
    bad = dict(rows=rows[::-1], acoustic_common=common)
    write(tmp_path/'full-0.json', bad)
    with pytest.raises(ValueError, match='shard 0'):
        audit.audit_roster(tmp_path, config, panel)

    out, fsdp = tmp_path/'worker-0', tmp_path/'fsdp-0'
    out.mkdir()
    fsdp.mkdir()
    lineage = scoring_lineage(config, 0)
    write(out/'lineage.json', dict(sha256=lineage))
    predictions = dict(seconds=2., peak_bytes=1, records=[
        dict(id='a', prob=[.75, .25]), dict(id='b', prob=[.25, .75])])
    timing = dict(workers=2, ahead=4, first_batch=.1, input_wait=.2, forward=1.)
    baseline = write(out/'baseline.json', dict(predictions, batch_sizes={'1': 4}, input_timing=timing))
    candidate = write(out/'batch-2.json', dict(predictions, batch_sizes={'2': 2}, input_timing=timing))
    write(out/'panel-ready.json', dict(lineage=lineage, baseline_sha256=baseline, candidate_sha256=candidate))
    reference = dict(predictions, batch_sizes={'1': 4}, policy_id='policy', rank=0)
    write(fsdp/'reference.json', reference)
    assert audit.audit_admission(tmp_path, config, panel, 1)['fsdp']['max_probability_abs'] == 0
    reference['records'][0]['prob'][0] = .6
    write(fsdp/'reference.json', reference)
    with pytest.raises(ValueError, match='FSDP reference parity failed'):
        audit.audit_admission(tmp_path, config, panel, 1)
