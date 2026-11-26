import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.card_acoustic_pack import SCHEMA
from multimodal_jev.card_tpo_schedule import check_batches
from tools.reports.build_tpo_bidirectional_targets import compile_schedule
from tools.reports.build_tpo_gap_targets import shuffled_batches
from tools.reports.card_tpo_batches import CardTPOBatches
from tools.reports.card_tpo_bucket_package import load, prepare
from tools.reports.prepare_paired_card_evaluation import bind_completed_endpoints
from tools.reports.prepare_tpo_windows import prepare as package


def write(path, data):
    path.write_text(json.dumps(data))


def test_frozen_rewards_survive_actor_rebatching_and_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr('multimodal_jev.local_checkpoint.check_paths', lambda paths: None)
    target = tmp_path/'targets'
    target.mkdir()
    rows = {str(i): dict(index=i, audio_sha256=str(i), gold=i % 2) for i in range(17)}
    design = dict(epochs=2, global_batch=8, microbatch=1, seed=13)
    prob = {k: [0.13+i*.037, 0.88-i*.031] for i, k in enumerate(rows)}
    reward = dict(temperature=1., probability_epsilon=1e-6, max_displacement=1.,
                  audio_to_label=.5, label_to_audio=.5, brier=1.)
    before = shuffled_batches(rows, design)
    projected, batches, *_ = compile_schedule(dict(schedule=design, reward=reward, updates=len(before)),
                                                rows, before, list(rows), prob)
    cards = dict(catalog=[dict(id='a', prefix=[1], suffix=[9]),
                          dict(id='b', prefix=[1, 2], suffix=[9])], arms=dict(rules=[0, 1]))
    for name, content in [('rows', rows), ('batches', batches), ('cards', cards)]:
        write(target/(name+'.json'), content)
    manifest = dict(schema='jev-tpo-reward-view-v1', candidate_ids=['a', 'b'], schedule=design,
                    source_policy='p', updates=len(batches), card_input=dict(arm='rules',
                        checkpoint_sha256='cp', cards_sha256=file_hash(target/'cards.json'),
                        acoustic_common=dict(context_encoding=dict(audio_token_id=8))))
    torch.save(projected, target/'targets.pt')
    manifest.update(source_policy_sha256='policy', view_sha256='view', reference_sha256='reference',
                    bidirectional_contrast={}, files={f.name: file_hash(f) for f in target.iterdir()})
    write(target/'MANIFEST.json', manifest)
    pinned_targets = file_hash(target/'targets.pt')
    acoustic = tmp_path/'index.json'
    entries = {uid: [0, 0, 1, uid, 1+int(uid)//6, 1, 100, 'int64', 'h'] for uid in rows}
    entries['excluded'] = entries['0']
    write(acoustic, dict(schema=SCHEMA, identity='index', token=8, entries=entries))
    cache = dict(path=str(acoustic), sha256=file_hash(acoustic), identity='index')
    buckets = dict(lookahead_batches=2, acoustic_engine='unique_recording_v1',
                   acoustic_index=str(acoustic), acoustic_mount='/remote/acoustics', buckets=[
                       dict(max_tokens=4, batch_size_per_rank=4, microbatch_per_rank=2),
                       dict(max_tokens=None, batch_size_per_rank=2, microbatch_per_rank=1)])
    prepare(buckets, dict(packed_acoustics=cache), target, tmp_path, manifest, 2)
    sha = file_hash(tmp_path/'optimizer-buckets.json')
    target_sha = file_hash(target/'MANIFEST.json')
    plan = load(tmp_path, sha, target_sha, 2, 2)
    assert file_hash(target/'targets.pt') == pinned_targets
    assert sum(b['real'] for b in plan['updates']) == 34
    assert len({b['local_batch'] for b in plan['updates']}) > 1
    assert len(plan['updates']) != len(batches)
    assert any(len([s for s in b['slots'] if s['padding']]) == 1 for b in plan['updates'])
    controls = {(b['epoch'], s['id']): projected[b['epoch'], rows[s['id']]['index']]
                for b in batches for s in b['slots'] if not s['padding']}
    for b in plan['updates']:
        for s in b['slots'][:b['real']]:
            assert torch.equal(controls[b['epoch'], s['id']], projected[b['epoch'], rows[s['id']]['index']])
    write(tmp_path/'training.json', dict(packed_acoustics=cache))
    write(tmp_path/'receipt.json', dict(policy_id='p', checkpoint_sha256='cp', policy='/test/policy',
          passed=True, container_exited=True, config_sha256=file_hash(tmp_path/'training.json'),
          manifest_sha256='policy'))
    write(tmp_path/'parent.json', dict(source_policy='/jev-endpoint', source_policy_id='p',
          physical_devices=[0,2], batch_config=dict(per_dp_replica_batch=4),
          training_dataset=dict(manifest_sha256='view'), reference_dataset=dict(manifest_sha256='reference'),
          view_execution=dict(container=dict(cpus=2, memory_gib=4, shm_gib=1))))
    monkeypatch.setattr('multimodal_jev.source_snapshot.check_paths', lambda paths: None)
    pkg = tmp_path/'package'
    package(dict(package=str(pkg), repository=str(Path.cwd()), targets=str(target),
        parent_config=str(tmp_path/'parent.json'), io_workers=1,
        source_patterns=['tools/reports/card_tpo_bucket_package.py'],
        card_input=dict(inference_devices=[1,3], inference_cpus=1, endpoint_receipt=str(tmp_path/'receipt.json')),
        optimizer_bucketing=buckets, deployment=dict(read_only_mounts=['/obsolete:/jev-endpoint'])))
    ops = json.loads((pkg/'operations-local.json').read_text())['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    assert ops['entry_pins']['optimizer-buckets.json'] == sha
    assert ops['read_only_mounts'] == ['/test/policy:/jev-endpoint', '/remote/acoustics:/jev-stage2-acoustics']
    staged = pkg/'entry'
    for phase in ('infer-0', 'infer-1', 'finite', 'tpo'):
        job = json.loads((staged/(phase+'.json')).read_text())['macros']['NATIVE_SEAL_CONTAINER_CONFIG']['training']
        assert job['optimizer_sha'] == sha
    evaluated = bind_completed_endpoints(dict(stage1_updates=42, completed_endpoints=dict(
        stage1_receipt=tmp_path/'receipt.json', stage2_package=pkg)))
    assert evaluated['stage2_updates'] == len(plan['updates'])
    plan['updates'][0]['max_tokens'] += 1
    with pytest.raises(ValueError, match='geometry'):
        check_batches(plan['updates'], plan['lengths'], plan['design'], 2, 2)
    with pytest.raises(ValueError, match='changed'):
        load(tmp_path, 'incorrect', target_sha, 2, 2)


def test_prefetch_preserves_mask_card_order_and_local_gradient():
    options = dict(audio_token_id=8, decision_token_id=9, pad_token_id=0,
                   audio_chunk_size=100, audio_infer_window=100)
    cards = dict(labels=['a', 'b'], catalog=[dict(id='a', prefix=[2], suffix=[9]),
        dict(id='b', prefix=[3, 4], suffix=[9])], arms=dict(rules=[0, 1]))
    features = {uid: dict(input_ids=torch.full((1, n), 8),
                         input_features=torch.ones(1, 1, 100)*n,
                         input_features_mask=(torch.arange(100) < n*8)[None].long())
                for uid, n in [('short', 1), ('long', 2)]}
    slots = [dict(id=uid, padding=False, weight=1) for uid in features]
    batches = [dict(index=0, slots=slots, microbatch=2)]
    packing = SimpleNamespace(read=features.__getitem__)
    engine = dict(lengths=dict(short=4, long=5))
    weight = torch.nn.Parameter(torch.tensor(0.5))

    def model(inputs, positions, *, return_logits):
        assert return_logits
        assert inputs['attention_mask'].tolist() == [[1,1,1,0,0], [1,1,1,1,0],
                                                     [1,1,1,1,0], [1,1,1,1,1]]
        assert inputs['audio_owner_tokens'].tolist() == [0, 0, 1, 2, 1, 2]
        assert inputs['input_features'].shape[0] == 2
        assert positions.flatten().tolist() == [2, 3, 3, 4]
        return (inputs['input_ids']*inputs['attention_mask']).sum(-1, keepdim=True)*weight

    with CardTPOBatches(batches, {}, cards, dict(arm='rules'), {}, dict(ahead=2, workers=2),
                        options, 1, packed=packing, optimizer=engine) as stream:
        data, _ = stream.consume(0, 'p')
        z = stream.forward(model, data, policy_id='p', device='cpu', dtype=torch.float32, **options)
        torch.testing.assert_close(z, torch.tensor([[9.5, 12.], [13.5, 16.]]))
        z.sum().backward()
    assert weight.grad.item() == 102.
