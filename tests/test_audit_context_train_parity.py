import copy
import json
from pathlib import Path
import shutil

import pytest

from tools.reports import audit_context_train_parity as module


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))


def outputs(tmp_path,monkeypatch,*,actual_rollout_queries=32,actual_replicas=2):
    from test_context_train_parity_inputs import fixture
    from test_context_macro_rollout import Client
    from test_context_replica_macro_rollout import pool
    from tools.reports import context_train_macro_parity as producer
    import multimodal_jev.policy_identity as identities
    import tools.reports.context_macro_evaluation as evaluation
    inputs,_,_,_=fixture(tmp_path,monkeypatch)
    package=tmp_path/'package';collection=tmp_path/'collection'
    shutil.copytree(inputs['output'],package/'parity-inputs')
    policy=package/'initial/policy';write(policy/'MANIFEST.json',{})
    write(package/'initial/endpoint-commit.json',{'joint_policy_manifest_sha256':module.file_hash(policy/'MANIFEST.json')})
    commit=package/'initial/endpoint-commit.json'
    contract={'base_id':'base','token':{'id':99}};bundle=dict(policy_id='p0',contract=contract)
    runtime={'profile':'fixture','processor_sha256':'a'*64}
    c=dict(pins={str(commit):module.file_hash(commit)},source_policy=str(policy),source_policy_id='p0',
        source_contract=contract,initial_adapter='adapter',model='base',base_id='base',stage1_commit=str(commit),
        stage1_endpoint=318,expected_population=40513,train_inputs=inputs,dual_runtime_identity=runtime,release_sha256='c'*64,
        audio_token_id=98,output=str(collection/'evaluation'),macro_client=dict(runtime_identity=runtime,
            max_queries=128,rollout_queries=32,replicas=[{},{}],journal_root=str(collection/'parity-journal'),attempt_id='test'))
    # Model calculation/provenance are stubs; cache bytes, persisted macros and
    # independent input/journal/output reconstruction all execute real code.
    monkeypatch.setattr(producer,'verify_runtime',lambda c:runtime)
    monkeypatch.setattr(identities,'checkpoint_identity',lambda p:('base',{}))
    monkeypatch.setattr(producer,'read_execution_bundle',lambda *a,**kw:bundle)
    monkeypatch.setattr(producer,'verify_native_adapter',lambda *a:None)
    monkeypatch.setattr(producer,'validate_endpoint',lambda *a:None)
    monkeypatch.setattr(producer,'check_paths',lambda _:None)
    monkeypatch.setattr(evaluation,'check_paths',lambda _:None)
    selected,_=producer.panel(c,bundle)
    records=[dict(id=a['id'],request_identity=dict(policy_id='p0',inputs=r['inputs'],positions=r['positions']),
        prob=[.5]*60,logits=[0.]*60) for a,r in zip(selected['assets'],selected['input_receipts'],strict=True)]
    reference=collection/'hf/reference.json'
    write(reference,dict(schema='jev-target-width-hf-reference-v1',policy_id='p0',
        panel_sha256=module.digest(selected),records=records))
    write(collection/'hf/panel.json',selected)
    write(collection/'hf/terminal.json',{'exit_code':0})
    write(collection/'hf/hf-runtime.json',{'profile':module.PROFILE})
    write(collection/'hf-process/terminal.json',{'exit_code':0,'pid':123})
    c['hf_reference']=dict(path=str(reference),sha256=module.file_hash(reference))
    clients=[Client() for _ in range(actual_replicas)]
    engine=pool(collection/'parity-journal/test',monkeypatch,clients,rollout_queries=actual_rollout_queries)
    engine.runtime_identity=runtime
    for client in clients:
        client.runtime_identity=runtime;original=client.score_many
        def expanded(rows,_original=original,**kw):
            values=_original(rows,**kw)
            for value in values:value.update(prob=[.5]*60,logits=[0.]*60)
            return values
        client.score_many=expanded
    monkeypatch.setattr(producer,'connect_pool',lambda *a,**kw:engine)
    producer.run(c,{},'macro')
    write(package/'parity-macro.json',c)
    services=[dict(name='a',device=6),dict(name='b',device=7)]
    write(package/'PREPARED.json',dict(remote_root=str(collection),inference_services=services,
        layout={'services':{s['name']:'service-'+s['name']+'.json' for s in services}}))
    write(collection/'supervision/terminal.json',dict(passed=True,actor_started=True,actor_exit_code=0,
        forced_cleanup=False,child_exit_codes=[0,0,0],services=[dict(name=s['name'],exit_code=0) for s in services]))
    for service in services:
        name=service['name'];source='/sealed/source'
        write(package/f'service-{name}.json',dict(source_root=source,
            pins={source+'/tools/reports/serve_context_dual_candidate.py':'entrypoint'},
            policy=dict(max_batch_queries=32,max_batch_tokens=131072)))
        write(collection/f'service-{name}/ready.json',dict(policy_id='p0',contract=contract,
            physical_devices=[service['device']],transport_protocol='bulk_dual_logits_candidate_v1',
            model_forward_mode='fixture',max_batch_queries=32,max_batch_tokens=131072,entrypoint_sha256='entrypoint'))
        write(collection/f'service-{name}/terminal.json',{'exit_code':0})
    return package,collection


def test_independent_replay_uses_real_cache_and_macro_artifacts(tmp_path,monkeypatch):
    package,collection=outputs(tmp_path,monkeypatch)
    result=module.audit_outputs(package,collection)
    assert result['comparison']['exact'] and result['comparison']['queries']==256
    assert result['actor_updates']==0 and not result['full_population_parity']
    assert not result['training_dispatch_allowed']
    assert result['dispatch_audit']['ownership_reconstructed']
    assert not result['dispatch_audit']['engine_internal_batching_verified']


@pytest.mark.parametrize('width,replicas',[(16,2),(64,2),(32,1)])
def test_self_consistent_exact_outputs_do_not_hide_wrong_dispatch(tmp_path,monkeypatch,width,replicas):
    package,collection=outputs(tmp_path,monkeypatch,actual_rollout_queries=width,actual_replicas=replicas)
    # Producer and generic evaluation auditor accepted complete, exact results.
    # Their rehashed journals still cannot prove the declared rollout32 topology.
    assert module.read(collection/'evaluation/comparison.json')['exact']
    with pytest.raises(ValueError,match='whole-rollout assignment'):
        module.audit_outputs(package,collection)


@pytest.mark.parametrize('fault',['marker','tensor','candidate','input-manifest','hf-profile','hf-exit',
    'service-batch','service-device','service-entrypoint','service-exit','forced-cleanup','child-exit',
    'logits','comparison','journal','proof','reference-hash'])
def test_independent_replay_rejects_success_claims_with_corrupted_evidence(tmp_path,monkeypatch,fault):
    package,collection=outputs(tmp_path,monkeypatch)
    if fault in ('marker','tensor','candidate','input-manifest'):
        p=collection/'hf/panel.json';v=module.read(p)
        if fault=='marker':v['input_receipts'][0]['positions'][0]=0
        elif fault=='tensor':v['input_receipts'][0]['inputs']['input_ids']['sha256']='f'*64
        elif fault=='candidate':v['candidate_ids']=list(reversed(v['candidate_ids']))
        else:v['input_manifest_sha256']='f'*64
    elif fault=='hf-profile':p=collection/'hf/hf-runtime.json';v={'profile':'wrong'}
    elif fault=='hf-exit':p=collection/'hf-process/terminal.json';v={'exit_code':7}
    elif fault.startswith('service-'):
        p=collection/'service-a/ready.json';v=module.read(p)
        if fault=='service-batch':v['max_batch_queries']=4
        elif fault=='service-device':v['physical_devices']=[5]
        elif fault=='service-entrypoint':v['entrypoint_sha256']='wrong'
        else:p=collection/'service-a/terminal.json';v={'exit_code':1}
    elif fault in ('forced-cleanup','child-exit'):
        p=collection/'supervision/terminal.json';v=module.read(p)
        if fault=='forced-cleanup':v['forced_cleanup']=True
        else:v['child_exit_codes']=[0,7,0]
    elif fault=='logits':
        p=collection/'hf/reference.json';v=module.read(p);v['records'][0]['logits'][0]=1e-9
    elif fault in ('comparison','reference-hash'):
        p=collection/'evaluation/comparison.json';v=module.read(p)
        if fault=='comparison':v['queries']=1
        else:v['hf_reference']['sha256']='f'*64
    elif fault=='proof':p=collection/'evaluation/macro-output-audit.json';v={}
    else:
        p=next((collection/'parity-journal/test').glob('evaluation-*/COMPLETE.json'));v={}
    write(p,v)
    with pytest.raises((ValueError,KeyError)):module.audit_outputs(package,collection)


@pytest.mark.parametrize('fault',[None,'archive','extra','readonly','wait','config','package'])
def test_public_audit_checks_exact_transport_before_output_replay(tmp_path,monkeypatch,fault):
    from test_prepare_context_train_parity_job import fixture
    from tools.reports import prepare_context_train_parity_job as preparation
    from tools.reports.context_cached_storage_operations import mounts
    from tools.reports.context_replica_training_operations import IMAGE
    spec=fixture(tmp_path,monkeypatch)
    spec['manifest_sha256']=preparation.prepare(spec,{})['manifest_sha256']
    package=Path(spec['output']);m=module.read(package/'PREPARED.json')
    deployment=dict(tag='train-parity-test',shared_root='/shared',driver_root='/usr/local/Ascend/driver')
    argv=module.command(deployment,m,module.read(package/'config.json'),producer=True)
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
    collection=tmp_path/'collected';collection.mkdir()
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
        next(v for v in observed['Mounts'] if v['Destination']=='/data/cache/input')['RW']=True
    elif fault=='wait':wait['container_id']='b'*64
    elif fault=='config':
        write(collection/'parity-hf.json',{})
        transport['files']['parity-hf.json']=module.file_hash(collection/'parity-hf.json')
    elif fault=='package':handle['manifest_sha256']='f'*64
    c=dict(parity_package=spec,deployment=deployment,collection=str(collection),archive=str(archive))
    for name,value in [('launch',handle),('inspect',observed),('transport',transport),('wait',wait)]:
        p=tmp_path/(name+'.json');write(p,value);c[name]=str(p)
    calls=[]
    monkeypatch.setattr(module,'audit_outputs',lambda *a:(calls.append(a) or {'fixture':'output-replay'}))
    if fault:
        with pytest.raises(ValueError):module.audit(c,{})
        assert not calls
    else:
        result=module.audit(c,{})
        assert result['storage_audit']['actual_readonly_mounts_verified'] and len(calls)==1
        assert result['runtime_provenance_verified']
