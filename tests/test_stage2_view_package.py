"""Real CPU release -> package -> actor data seam, without NPU substitutes for admission."""
import json
from pathlib import Path
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev import stage2_dataset_release as dataset,stage2_view_runtime as runtime
from tools.reports import prepare_stage2_view_package as package
from tools.reports.context_stage2_runtime_runner import plan
from test_stage2_dataset_release import setup


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return dict(path=str(p),sha256=file_hash(p))


def fixture(tmp_path,monkeypatch,world=2):
    monkeypatch.setattr(package,'check_paths',lambda paths:None)  # fixture outputs may use image TMPDIR
    a,ck,_,ref=setup(tmp_path,monkeypatch);parent=tmp_path/'parent';view=tmp_path/'view'
    endpoint=write(parent/'initial/endpoint-commit.json',ck)
    a['endpoint']['stage1_commit_sha256']=endpoint['sha256']
    a['batch_config']['training_devices']=world
    r=dataset.publish(view,**a)
    base=dict(stage='task',start_update=0,experiment_kind='engineering_prefix',source_policy_id=a['reference'].policy_id,
        reference_dataset=ref,release_sha256=ref['identity']['release_sha256'],physical_devices=[4,5],
        algorithm='dual_credit_grpo_floor_clipped_v1',dual_arm='joint',seed=317,prompt_layout='prefix-v1',
        objective_extension={'root':'/data/orig/objective-extension','files':{}},objective_plugin={'path':'/data/orig/objective-extension/'+package.OBJECTIVE},pins={})
    write(parent/'config.json',base)
    for n in ('a','b'):write(parent/(n+'.json'),dict(policy={}))
    files={str(f.relative_to(parent)):file_hash(f) for f in parent.rglob('*') if f.is_file()}
    m=dict(remote_root='/data/orig',files=files,training_devices=[4,5],
        inference_services=[dict(name='a',device=6),dict(name='b',device=7)],
        layout=dict(configs=['PREPARED.json','config.json','run.py'],services={'a':'a.json','b':'b.json'}))
    h=write(parent/'PREPARED.json',m)
    bound=dataset.bind_config(base,root=view,manifest_sha256=r['manifest_sha256'],consumer=r['consumer'],reference=a['reference'])
    b=write(tmp_path/'bound',bound);receipt=write(tmp_path/'receipt',dict(r,bound_config_sha256=b['sha256']))
    design=package.ROOT/'configs/experiments/stage2-probability-margin-candidate-v0.0.126.json'
    return a,dict(schema='jev-stage2-view-package-v1',parent=dict(root=str(parent),manifest_sha256=h['sha256']),
        window='prefix',remote_root='/data/new',output=str(tmp_path/'pkg'),objective_design=dict(path=str(design),sha256=file_hash(design)),
        bound_config=b,dataset_receipt=receipt,recovery=None,admission=None)


@pytest.mark.parametrize('window',['prefix','tail','formal'])
def test_view_packaged_once_with_dynamic_budget(tmp_path,monkeypatch,window):
    a,c=fixture(tmp_path,monkeypatch);c['window']=window
    result=package.run(c,{})
    root=Path(c['output']);cfg=json.loads((root/'config.json').read_bytes());manifest=runtime.pinned(root/'PREPARED.json',result['manifest_sha256'])
    pop,batches,consumer=runtime.consume(cfg,a['rows'],a['task'],directory=root/'view-data')
    assert len(pop)==5 and len(cfg['reference_dataset']['population'])==20
    assert [b['real_queries'] for b in batches]==[5,5] and cfg['total_updates']==2 and consumer['real_exposures']==10
    assert all(len(b['slots'])==128 for b in batches)
    assert not result['training_dispatch_allowed'] and cfg['training_dispatch_allowed'] is False
    argv=plan(c['remote_root'],cfg,manifest,package_root=root)
    assert '/data/new/view-entry.py' in argv['actor_command'] and '--preflight' in argv['preflight_command']
    assert file_hash(root/'objective-extension'/package.OBJECTIVE)==cfg['objective_plugin']['sha256']
    assert cfg['pins']['/data/new/view-runtime.py']==file_hash(root/'view-runtime.py')
    assert cfg['objective_plugin']['options']['probability_margin']==.1
    with pytest.raises(ValueError):package.run(c,{})
    (root/'view-data/batches.json').write_text('[]')
    with pytest.raises(ValueError,match='changed'):runtime.consume(cfg,a['rows'],a['task'],directory=root/'view-data')


def test_refuse_foreign_bound_config_before_publication(tmp_path,monkeypatch):
    _,c=fixture(tmp_path,monkeypatch);path=Path(c['bound_config']['path'])
    cfg=json.loads(path.read_bytes());cfg['total_updates']=634
    c['bound_config']=write(path,cfg)
    with pytest.raises(ValueError,match='same audited'):package.run(c,{})
    assert not Path(c['output']).exists()


def test_recovery_contract_keeps_prefix_identity_and_tail_uses_epoch_cursor(tmp_path,monkeypatch):
    _,c=fixture(tmp_path,monkeypatch);package.run(c,{})
    cfg=json.loads((Path(c['output'])/'config.json').read_bytes())
    # A distinct three-batch epoch makes an accidental [0,1] tail mapping visible.
    cfg['batch_config']['source_queries']=257;cfg.update(total_updates=6,evaluation_steps=[0,6])
    assert runtime.indices(dict(cfg,view_window='tail'))==[2,3]
    replay=dict(cfg,view_window='recovery',start_update=1,resume_root='/resume')
    assert runtime.checkpoint_contract(cfg)==runtime.checkpoint_contract(replay)
    assert runtime.checkpoint_contract(replay)!=runtime.checkpoint_contract(dict(cfg,view_window='tail'))
    with pytest.raises(ValueError):runtime.indices(dict(replay,stop_update=6))


def test_foreign_or_missing_formal_receipts_rejected(tmp_path):
    ids={};lineage={'parent':'abc'}
    for kind in ('prefix','tail','recovery'):
        ids[kind]=write(tmp_path/kind,dict(status='stage2_view_'+kind+'_outputs_audited',container_id=kind,
            updates=[0] if kind=='recovery' else [0,1],lineage=lineage,world_size=2,
            storage_audit={'actual_readonly_mounts_verified':True},fresh_process_replay_verified=True,
            prefix_audit_sha256=ids.get('prefix',{}).get('sha256'),replay={'exact_objective_traces':True}))
    assert len(package.admission(ids,lineage))==3
    with pytest.raises(ValueError):package.admission(ids,{'parent':'other'})
    with pytest.raises(ValueError):package.admission({'prefix':ids['prefix']},lineage)


def test_dp6_view_launch_tail_and_topology_binding(tmp_path, monkeypatch):
    a, request = fixture(tmp_path, monkeypatch, world=6)
    request.update(window='tail', topology=dict(
        training_devices=[0, 1, 2, 3, 4, 5],
        inference_services=[dict(name='a', device=6), dict(name='b', device=7)]))
    result = package.run(request, {})
    root = Path(request['output'])
    cfg = json.loads((root/'config.json').read_bytes())
    manifest = runtime.pinned(root/'PREPARED.json', result['manifest_sha256'])
    _, batches, consumer = runtime.consume(cfg, a['rows'], a['task'], directory=root/'view-data')
    assert cfg['global_batch'] == 384
    assert all(len(batch['slots']) == 384 and batch['real_queries'] == 5 for batch in batches)
    assert consumer['real_exposures'] == 10
    launch = plan(request['remote_root'], cfg, manifest, package_root=root)
    assert '--nproc-per-node=6' in launch['actor_command']
    assert launch['train_visible'] == '0,1,2,3,4,5'
    assert launch['services'] == [('a', '6'), ('b', '7')]
    service = json.loads((root/'b.json').read_bytes())['policy']
    assert service['physical_devices'] == [7]
    assert service['process_device_scope']['visible_ordinal'] == '7'
    with pytest.raises(ValueError, match='topology'):
        plan(request['remote_root'], cfg, dict(manifest, training_devices=[4, 5]), package_root=root)
    (root/'view-runtime.py').write_text('raise RuntimeError("changed runtime executed")')
    with pytest.raises(ValueError, match='pinned packaged view runtime'):
        plan(request['remote_root'], cfg, manifest, package_root=root)


def test_invalid_execution_profile_and_duplicate_device_rejected(tmp_path, monkeypatch):
    _, request = fixture(tmp_path, monkeypatch, world=6)
    request['topology'] = dict(training_devices=[0, 1, 2, 3, 4, 5],
                               inference_services=[dict(name='a', device=5), dict(name='b', device=7)])
    with pytest.raises(ValueError, match='distinct'):
        package.run(request, {})
    assert not Path(request['output']).exists()
    design = json.loads(Path(request['objective_design']['path']).read_bytes())
    profile = json.loads((package.ROOT/design['execution']['source']).read_bytes())
    profile['optimizer']['lr'] = float('nan')
    with pytest.raises(ValueError, match='optimizer'):
        runtime.validate_execution(profile)


def test_shared_audio_only_reads_and_checks_content(tmp_path):
    path = tmp_path/'sample'
    path.write_bytes(b'audio')
    digest = file_hash(path)
    mirrored = path.with_name(digest+'.audio')
    path.rename(mirrored)
    mirror = runtime.ReadOnlyAudioMirror(tmp_path)
    assert mirror.resolve(digest) == mirrored
    assert list(tmp_path.iterdir()) == [mirrored]
    mirrored.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        mirror.resolve(digest)


def test_activation_checkpointing_preserves_outputs_gradients_and_names():
    import copy
    import torch
    from torch import nn
    from torch.utils.checkpoint import DefaultDeviceType
    from tools.reports.stage2_view_actor import configure_activation_checkpointing

    class Layer(nn.Linear):
        calls = 0

        def forward(self, x, *, scale):
            self.calls += 1
            return super().forward(x).square() * scale

    original = nn.ModuleDict({'alternate': nn.ModuleList([Layer(4, 4), Layer(4, 4)])})
    candidate = copy.deepcopy(original)
    setting = dict(enabled=True, module_lists=['alternate'])
    with pytest.raises(ValueError, match='missing'):
        configure_activation_checkpointing(candidate, dict(setting, module_lists=['alternate', 'missing']))
    assert candidate['alternate'][0].forward == Layer.forward.__get__(candidate['alternate'][0])
    configure_activation_checkpointing(candidate, setting)
    assert list(dict(candidate.named_parameters())) == list(dict(original.named_parameters()))
    assert list(candidate.state_dict()) == list(original.state_dict())

    def evaluate(model):
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        for layer in model['alternate']:
            x = layer(x, scale=0.5)
        x.sum().backward()
        return x

    prior = DefaultDeviceType.get_device_type()
    try:
        DefaultDeviceType.set_device_type('cpu')
        torch.testing.assert_close(evaluate(candidate), evaluate(original), atol=0, rtol=0)
    finally:
        DefaultDeviceType.set_device_type(prior)
    for p, q in zip(candidate.parameters(), original.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    assert [layer.calls for layer in candidate['alternate']] == [2, 2]
    profile = json.loads((package.ROOT/'configs/experiments/stage2-view-execution-v0.0.183.json').read_text())
    assert runtime.validate_execution(profile)['activation_checkpointing']['enabled']
    for invalid in (dict(enabled=False, module_lists=['alternate']),
                    dict(enabled=True, module_lists=[]),
                    dict(enabled=True, module_lists=['alternate', 'alternate']),
                    dict(enabled=True, module_lists=['alternate', 'alternate.child'])):
        with pytest.raises(ValueError, match='activation_checkpointing'):
            runtime.validate_execution(dict(profile, activation_checkpointing=invalid))
