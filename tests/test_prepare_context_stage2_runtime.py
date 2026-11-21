import copy
import json
from pathlib import Path

import pytest

from tools.reports import prepare_context_stage2_runtime as module


def inputs():
    from tools.reports.context_cached_reference_trainer import BASELINE_OBJECTIVE
    design=json.loads((Path.cwd()/'configs/experiments/context-two-stage-v0.0.11.json').read_text())
    binding=dict(source_step=design['stage1']['reference_endpoint'],base_id='base',source_policy_id='new-policy',source_contract={'base_id':'base'},
        release_sha256='release',stage_lora_transition={'mode':'reuse'},
        stage2_objective=dict(BASELINE_OBJECTIVE),batch_config=design['stage2']['batch_config'])
    original=dict(base_id='base',source_policy_id='old-policy',source_contract={'base_id':'base'},
        release_sha256='release',source_root='/data/old/source',source_policy='/data/old-policy/policy',
        dual_runtime_manifest='/data/old/RUNTIME-SOURCES.json',physical_devices=list(range(6)),
        global_batch=384,stop_update=212,experiment_kind='formal_source',evaluation_steps=[0,212],
        resume_root='/data/resume',reference_cache='/data/old/cache',
        reference_dataset={'stale':True},macro_client={'schema':'jev-replica-macro-client-v1'})
    services={name:dict(source_root='/data/old/source',source_policy='/data/old-policy/policy',
        socket_path='/data/old/service-'+name+'/socket',exchange='/data/old/service-'+name+'/exchange',
        dual_runtime_manifest='/data/old/RUNTIME-SOURCES.json',
        policy=dict(snapshot='/data/old-policy/policy',policy_id='old-policy',max_batch_queries=48,
            max_batch_tokens=16384,device_scope_module={'name':'context_configurable_device_scope','sha256':'hash'}))
        for name in ('a','b')}
    c=dict(training_devices=[4,5],inference_devices=[6,7],schedule_sha256='schedule',
        parent_schedule_sha256='parent',parent_release_sha256='release',attempt_id='new-attempt')
    return original,services,c,binding,{'source_manifest_sha256':'same-code'},'/data/new'


def test_configuration_drops_old_topology_policy_and_resume_without_mutation():
    args=inputs();before=copy.deepcopy(args)
    config,services,plan=module.configured(*args)
    assert args==before
    assert config['physical_devices']==[4,5] and config['global_batch']==128
    assert config['stop_update']==2 and config['checkpoint_interval']==1
    assert config['experiment_kind']=='engineering_prefix' and not config['training_dispatch_allowed']
    assert not {'resume_root','evaluation_steps','reference_cache','reference_dataset'} & config.keys()
    assert plan['total_updates']==634 and plan['tail_real_queries']==65
    assert config['source_policy']=='/data/new/initial/policy'
    assert config['source_policy_id']=='new-policy'
    assert config['initial_adapter']==config['reference_adapter']=='/data/new/initial/adapter'
    assert config['macro_client']['max_queries']==128 and config['macro_client']['rollout_queries']==32
    for name,device,ordinal in [('a',6,'2'),('b',7,'3')]:
        service=services[name];policy=service['policy']
        assert service['source_policy']==config['source_policy']
        assert policy['snapshot']==config['source_policy'] and policy['policy_id']=='new-policy'
        assert policy['physical_devices']==[device] and policy['tensor_parallel_size']==1
        assert policy['process_device_scope']['mounted_physical']==[4,5,6,7]
        assert policy['process_device_scope']['visible_ordinal']==ordinal
        assert policy['max_batch_queries']==32 and policy['max_batch_tokens']==131072
        assert service['dual_runtime_identity']==config['dual_runtime_identity']==args[-2]


@pytest.mark.parametrize('fault',['shared','six-trainers','bool-device','service-count','batch48','base','release','attempt'])
def test_conflicting_topology_and_source_rejected(fault):
    a=list(inputs());old,services,c,binding,_,_=a
    if fault=='shared':c['inference_devices']=[5,7]
    elif fault=='six-trainers':c['training_devices']=list(range(6))
    elif fault=='bool-device':c['training_devices']=[True,5]
    elif fault=='service-count':services.pop('b')
    elif fault=='batch48':binding['batch_config']['inference_batch_queries']=48
    elif fault=='base':old['base_id']='other'
    elif fault=='release':old['release_sha256']='other'
    else:c['attempt_id']='../escape'
    with pytest.raises(ValueError):module.configured(*a)


def test_explicit_model_binding_updates_actor_services_and_pins(tmp_path):
    from test_context_model_profile import LARGE_CONFIG, write_model
    from multimodal_jev.cached_context_preparation import processor_cache_identity
    original,services,c,binding,runtime,remote=inputs()
    c['model_binding']={'io_workers':2}
    model=write_model(tmp_path,dict(LARGE_CONFIG,audio_token_id=151676))
    weights=model/'model.safetensors';weights.write_bytes(b'fixture only')
    base_id,files=module.checkpoint_identity(model)
    profile=module.from_hf(model)
    interface=dict(runtime_profile='profile',request_version='request',output={'field':'prob'},
                   token={'id':151706})
    original.update(model='/shared/old',runtime_profile='profile',context_runtime_profile='profile',
                    audio_token_id=151676,decision_token_id=151706,
                    source_contract=dict(interface,base_id=original['base_id']),
                    pins={'/shared/old/weights':'obsolete','/shared/data':'stay'})
    binding.update(base_id=base_id,source_contract=dict(interface,base_id=base_id,
                   lora=dict.fromkeys(profile.lora_targets()),branch=profile.decision_head()))
    pins={str(model/n):h for n,h in files.items()}
    training=dict(model=str(model),pins=pins,runtime_profile='profile',
                  **{k:binding[k] for k in ('base_id','source_contract','release_sha256')})
    training_path=tmp_path/'config.json';training_path.write_text(json.dumps(training))
    initial=tmp_path/'initial';(initial/'policy/processor').mkdir(parents=True)
    assets={}
    for name in ('tokenizer.json','tokenizer_config.json'):
        processor=initial/'policy/processor'/name;processor.write_text('{}')
        assets['processor/'+name]=module.file_hash(processor)
    (initial/'policy/MANIFEST.json').write_text(json.dumps(
        {'files':assets}))
    (initial/'source-plan.json').write_text(json.dumps(dict(
        source_operations={'package':str(tmp_path)},training_config_sha256=module.file_hash(training_path))))
    (initial/'design.json').write_text((Path.cwd()/'configs/experiments/context-model-profile-1.7b-v1.json').read_text())
    runtime['processor_sha256']=processor_cache_identity(initial/'policy')
    geometry=module.model_from_endpoint(c,initial,binding,runtime)
    config,replicas,_=module.configured(original,services,c,binding,runtime,remote,geometry)
    assert (config['model'],config['base_id'],config['source_contract']['branch']['hidden_size'])==(str(model),base_id,2048)
    assert all((s['policy']['model'],s['policy']['base_id'],s['policy']['contract'])==
               (str(model),base_id,binding['source_contract']) for s in replicas.values())
    assert module.external_pins(original,c,'/data/old',geometry)==dict(pins,**{'/shared/data':'stay'})
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError,match='model bytes differ'):
        module.model_from_endpoint(c,initial,binding,runtime)


def test_package_keeps_source_identity_and_replaces_old_external_policy_pins(tmp_path,monkeypatch):
    from test_prepare_context_cached_runtime import setup as runtime_setup
    from tools.reports.prepare_context_cached_runtime import prepare as runtime_prepare
    from test_context_infonce_stage2_source import setup as source_setup
    from tools.reports.context_infonce_stage2_source import prepare as source_prepare
    from multimodal_jev.cached_context_preparation import processor_cache_identity
    def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
    runtime_root=tmp_path/'runtime';runtime_root.mkdir()
    source_root=tmp_path/'origin';source_root.mkdir()
    rc=runtime_setup(runtime_root,monkeypatch);runtime_prepare(rc,{})
    sc,plan=source_setup(source_root,monkeypatch);sr=source_prepare(sc,{})
    initial=Path(sc['output']);binding=module.read(initial/'binding.json')
    parent=Path(rc['output']);receipt=module.read(parent/'PREPARED.json')
    config=module.read(parent/'config.json')
    config.update(source_contract=binding['source_contract'],base_id=binding['base_id'],
        release_sha256=binding['release_sha256'],release='/shared/release',
        dual_runtime_identity=dict(receipt['dual_runtime_identity'],processor_sha256=processor_cache_identity(initial/'policy')))
    old_initial=str(Path(config['source_policy']).parent)
    config['pins'].update({old_initial+'/execution.pt':'obsolete','/shared/model':'retained'})
    write(parent/'config.json',config)
    receipt['dual_runtime_identity']=config['dual_runtime_identity']
    write(parent/'PREPARED.json',receipt)
    # Source audit/export are real. The old runtime and population/schedule are
    # fixture stand-ins; this test does not claim full TRAIN or NPU validation.
    monkeypatch.setattr(module,'audit_runtime',lambda c,p:{'status':'fixture','runtime_identity':config['dual_runtime_identity']})
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    monkeypatch.setattr(module,'load',lambda *a,**k:([{'id':str(i)} for i in range(40513)],{},{}))
    monkeypatch.setattr(module,'load_schedule',lambda *a,**k:([{}]*634,{}))
    schedule=tmp_path/'schedule';ancestor=tmp_path/'ancestor'
    write(schedule/'MANIFEST.json',{'test':'dp2'});write(ancestor/'MANIFEST.json',{'test':'parent'})
    c=dict(runtime={'package':str(parent),'manifest_sha256':module.file_hash(parent/'PREPARED.json')},
        source=dict(sc,manifest_sha256=sr['manifest_sha256']),output=str(tmp_path/'bound'),
        remote_root='/data/bound',training_devices=[4,5],inference_devices=[6,7],
        schedule=str(schedule),schedule_sha256=module.file_hash(schedule/'MANIFEST.json'),
        parent_schedule=str(ancestor),parent_schedule_sha256=module.file_hash(ancestor/'MANIFEST.json'),
        parent_release_sha256=binding['release_sha256'],attempt_id='bound-attempt')
    before={str(p.relative_to(parent)):module.file_hash(p) for p in parent.rglob('*') if p.is_file()}
    result=module.prepare(c,{})
    assert before=={str(p.relative_to(parent)):module.file_hash(p) for p in parent.rglob('*') if p.is_file()}
    dest=Path(c['output']);actual=module.read(dest/'config.json')
    assert result['training_dispatch_allowed'] is False
    assert (dest/'RUNTIME-SOURCES.json').read_bytes()==(parent/'RUNTIME-SOURCES.json').read_bytes()
    assert (dest/'run.py').read_bytes()==(parent/'run.py').read_bytes()
    assert not any(p.startswith(old_initial+'/') for p in actual['pins'])
    assert actual['pins']['/shared/model']=='retained'
    assert actual['pins']['/data/bound/initial/policy/MANIFEST.json']==binding['source_manifest_sha256']
    assert actual['physical_devices']==[4,5] and actual['global_batch']==128
    for key,digest in actual['pins'].items():
        if key.startswith('/data/bound/'):
            assert module.file_hash(dest/Path(key).relative_to('/data/bound'))==digest
    proof=module.audit(dict(c,manifest_sha256=result['manifest_sha256']),{})
    assert proof['devices']==4 and proof['schedule_updates']==634
    assert not proof['training_dispatch_allowed'] and not proof['npu_executed']
    # Rehashing a modified service must not manufacture a trusted binding.
    service=module.read(dest/'service-a.json');service['policy']['max_batch_queries']=4
    write(dest/'service-a.json',service)
    manifest=module.read(dest/'PREPARED.json')
    manifest['files']['service-a.json']=module.file_hash(dest/'service-a.json')
    write(dest/'PREPARED.json',manifest)
    with pytest.raises(ValueError,match='independent binding: service-a.json'):
        module.audit(dict(c,manifest_sha256=module.file_hash(dest/'PREPARED.json')), {})
