import copy
import json
from pathlib import Path

import pytest

from tools.reports import audit_context_cached_training as module


def reference(tmp_path,monkeypatch):
    from test_context_reference_dataset import fixture
    root,kw,_,_,record=fixture(tmp_path,monkeypatch)
    config=dict(root=str(root),manifest_sha256=kw['manifest_sha256'],
        identity=kw['expected_identity'],population=kw['expected_population'])
    behavior=copy.deepcopy(record);behavior['policy_id']='d'*64
    behavior['request_identity']['policy_id']='d'*64
    behavior['request_id']=module.digest(behavior['request_identity'])
    receipt=dict(update=0,fixed_reference_policy_id=record['policy_id'],behavior_policy_id='d'*64,
        runtime_identity=config['identity']['runtime_identity'],reference_misses=0,
        bulk_reference_calls=0,bulk_behavior_calls=1,switches=[],
        queries=[dict(request_id='sample0::slot0',behavior_record=behavior,reference_record=record)])
    batch=dict(records=[dict(id='sample0',request_id='sample0::slot0')])
    return root,config,receipt,batch


def test_exact_reference_on_first_and_repeated_update_without_online_fill(tmp_path,monkeypatch):
    root,c,receipt,batch=reference(tmp_path,monkeypatch)
    verifier=module.ReferenceConsumption(c,root)
    verifier.check(receipt,batch);receipt['update']=1
    verifier.check(receipt,batch)
    assert verifier.samples==2


@pytest.mark.parametrize('fault',['prob','input','sample','slot','switch','miss','reference-call',
    'behavior-call','runtime','policy','empty'])
def test_consumed_cached_reference_faults_fail(tmp_path,monkeypatch,fault):
    root,c,r,b=reference(tmp_path,monkeypatch);v=module.ReferenceConsumption(c,root)
    if fault=='prob':r['queries'][0]['reference_record']['prob'][0]=.5
    elif fault=='input':r['queries'][0]['behavior_record']['request_identity']['inputs']['input_ids']['sha256']='f'*64
    elif fault=='sample':b['records'][0]['id']='unknown'
    elif fault=='slot':r['queries'][0]['request_id']='other'
    elif fault=='switch':r['switches']=[{'policy_id':v.policy}]
    elif fault=='miss':r['reference_misses']=1
    elif fault=='reference-call':r['bulk_reference_calls']=1
    elif fault=='behavior-call':r['bulk_behavior_calls']=2
    elif fault=='runtime':r['runtime_identity']=dict(r['runtime_identity'],profile='other')
    elif fault=='policy':r['queries'][0]['behavior_record']['policy_id']='e'*64
    else:r['queries']=[];b['records']=[]
    with pytest.raises(ValueError):v.check(r,b)
    assert v.samples==0


def test_checkpoint_contract_keeps_shared_fields_and_binds_stage_reference_objective():
    from test_audit_context_replica_training import candidate,manifest
    from test_context_stage_lora import config as lora_config,contract as lora_contract
    from multimodal_jev.context_stage_lora import transition
    from tools.reports.context_cached_reference_trainer import BASELINE_OBJECTIVE
    c=candidate(2);lora=lora_config();c['source_contract']=lora_contract(lora)
    c['stage_lora_transition']=transition(lora,lora,reuse_stage1_adapter=True)
    c['stage2_objective']=dict(BASELINE_OBJECTIVE)
    c['reference_dataset']=dict(manifest_sha256='a'*64,identity={'test':'reference'})
    mounted=module.shared.mounted_names(manifest(2))
    before=copy.deepcopy(c);actual=module.expected_contract(c,mounted)
    base=module.shared.expected_contract(c,mounted)
    assert all(actual[k]==v for k,v in base.items()) and c==before
    assert actual['reference_dataset']['mode']=='readonly_no_online_fill'
    assert actual['stage2_objective']==BASELINE_OBJECTIVE
    c['stage2_objective']['learning_rate']=2e-4
    with pytest.raises(ValueError):module.expected_contract(c,mounted)


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))


def test_shared_auditor_rejects_missing_training_artifacts_after_cache_check(tmp_path,monkeypatch):
    # Real shared audit executes (no stub) and must not pass merely because
    # published reference records match. A complete training result is required.
    from test_audit_context_replica_training import candidate
    root,c,receipt,batch=reference(tmp_path,monkeypatch)
    package=tmp_path/'package';collection=tmp_path/'collection'
    training=candidate(2)
    training.update(experiment_kind='engineering_prefix',engineering_prefix=True,
        training_dispatch_allowed=False,start_update=0,stop_update=2,checkpoint_interval=1,
        reference_dataset=c)
    write(package/'config.json',training);write(package/'PREPARED.json',{})
    for i in range(2):
        write(collection/f'run/rollout-{i:04d}.json',dict(receipt,update=i))
        write(collection/f'run/batch-{i:04d}.json',batch)
    with pytest.raises((KeyError,FileNotFoundError)):
        module.audit_outputs(package,collection,reference_root=root)


@pytest.mark.parametrize('fault',[None,'archive','extra','readonly','wait','config','package'])
def test_public_audit_authenticates_transport_before_training_replay(tmp_path,monkeypatch,fault):
    from test_prepare_context_cached_training import setup
    from tools.reports import prepare_context_cached_training as preparation
    from tools.reports.context_cached_storage_operations import mounts
    from tools.reports.context_replica_training_operations import IMAGE
    import shutil
    spec,_,_,_=setup(tmp_path,monkeypatch)
    # Real package audit with only its external NPU-producing parents stubbed.
    parent=Path(spec['parent']['output']);p=parent/'config.json'
    value=json.loads(p.read_text())
    value.update(encoded_audio_cache='/data/input/cache',audio_mirror_root='/data/audio/run')
    write(p,value)
    spec['manifest_sha256']=preparation.prepare(spec,{})['manifest_sha256']
    package=Path(spec['output']);m=json.loads((package/'PREPARED.json').read_text())
    deployment=dict(tag='cached-test',shared_root='/shared',driver_root='/usr/local/Ascend/driver')
    argv=module.command(deployment,m,json.loads((package/'config.json').read_text()))
    def values(flag):return [argv[i+1] for i,v in enumerate(argv) if v==flag]
    state=dict(Status='exited',ExitCode=0,OOMKilled=False)
    handle=dict(argv=argv,container_id='a'*64,image=IMAGE,remote_root=spec['remote_root'],layout=m['layout'],
        manifest_sha256=spec['manifest_sha256'],config_sha256=module.file_hash(package/'config.json'),
        services={n:module.file_hash(package/p) for n,p in m['layout']['services'].items()})
    observed=dict(Id=handle['container_id'],Image=IMAGE,State=state,
        HostConfig=dict(Privileged=False,NetworkMode='none',Binds=values('-v'),
            Devices=[dict(PathOnHost=v.split(':')[0],PathInContainer=v.split(':')[1],CgroupPermissions='rwm')
                for v in values('--device')]),
        Config=dict(Entrypoint=values('--entrypoint'),Cmd=argv[argv.index(IMAGE)+1:],Env=values('-e')),
        Mounts=[dict(Source=src,Destination=dst,RW=not ro,Type='bind') for src,dst,ro in mounts(argv)])
    collection=tmp_path/'training-results';collection.mkdir()
    for name in m['layout']['configs']+list(m['layout']['services'].values()):shutil.copyfile(package/name,collection/name)
    archive=tmp_path/'transport.tar';archive.write_bytes(b'fixture transport archive')
    transport=dict(status='transport_complete_independent_audit_required',container_id=handle['container_id'],
        image=IMAGE,remote_root=spec['remote_root'],state=state,bad_pins=[],bad_package=[],launch_binding_mismatches=[],
        archive_sha256=module.file_hash(archive),
        files={str(p.relative_to(collection)):module.file_hash(p) for p in collection.rglob('*') if p.is_file()})
    wait=dict(container_id=handle['container_id'],exit_code=0)
    if fault=='archive':archive.write_bytes(b'corrupted')
    elif fault=='extra':(collection/'extra').write_text('extra')
    elif fault=='readonly':
        next(v for v in observed['Mounts'] if v['Destination']=='/data/input/cache')['RW']=True
    elif fault=='wait':wait['container_id']='b'*64
    elif fault=='config':
        write(collection/'config.json',{})
        transport['files']['config.json']=module.file_hash(collection/'config.json')
    elif fault=='package':handle['manifest_sha256']='f'*64
    c=dict(training_package=spec,deployment=deployment,collection=str(collection),archive=str(archive))
    for name,value in [('launch',handle),('inspect',observed),('transport',transport),('wait',wait)]:
        p=tmp_path/(name+'.json');write(p,value);c[name]=str(p)
    calls=[]
    monkeypatch.setattr(module,'audit_outputs',lambda *a,**kw:(calls.append((a,kw)) or {'fixture':'training-replay'}))
    if fault:
        with pytest.raises(ValueError):module.audit(c,{})
        assert not calls
    else:
        result=module.audit(c,{})
        assert result['storage_audit']['actual_readonly_mounts_verified'] and len(calls)==1
        assert calls[0][1]['reference_root']==Path(spec['reference_job']['collection'])/'reference-build/dataset'
