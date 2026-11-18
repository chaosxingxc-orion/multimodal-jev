import copy
import json
from pathlib import Path

import pytest

from tools.reports import context_replica_transfer_operations as m
from test_context_replica_transfer_client import fixture
from test_prepare_context_train_parity_job import write


def outputs(tmp_path, monkeypatch, width=32):
    c, engine, clients, selected, reference = fixture(tmp_path, monkeypatch, 'FSC_native31')
    # Real replica pool/journals, fake model forward and portable inventory.
    engine.rollout_queries=width
    remote='/data/transfer-fixture'
    package=tmp_path/'package';collection=tmp_path/'collection'
    c['macro_client'].update(attempt_id='test',journal_root=remote+'/transfer-journal',
        max_queries=128,rollout_queries=32,replicas=[{},{}])
    c.update(contract={},source_policy=remote+'/initial/policy')
    engine.root=collection/'transfer-journal/test';engine.root.mkdir(parents=True)
    c['output']=str(collection/'evaluation')
    m.client.run(c,{},'macro')
    import shutil
    shutil.copytree(tmp_path/'hf',collection/'hf')
    # Bundle source evidence with remote-only paths, as in actual deployment.
    source=m.read(c['source_evaluation']['path'])
    for point in source['points']:
        old=Path(point['result']['path']);name=old.name
        target=package/'source-evidence'/name;write(target,m.read(old))
        point['result']=dict(path=remote+'/source-evidence/'+name,sha256=m.file_hash(target))
    path=package/'source-evidence/source-evaluation.json';write(path,source)
    c['source_evaluation']=dict(path=remote+'/source-evidence/source-evaluation.json',sha256=m.file_hash(path))
    for role in ('evaluation','hf'):
        p=collection/role/'source-binding.json';v=m.read(p);v['source_evaluation']=c['source_evaluation'];write(p,v)
    p=collection/'evaluation/result.json';v=m.read(p);v['source_binding_sha256']=m.file_hash(collection/'evaluation/source-binding.json');write(p,v)
    write(package/'parity-macro.json',c)
    services=[dict(name='a',device=6),dict(name='b',device=7)]
    write(package/'PREPARED.json',dict(remote_root=remote,decisions=31,inference_services=services,
        layout={'services':{s['name']:'service-'+s['name']+'.json' for s in services}}))
    write(collection/'hf-process/terminal.json',dict(exit_code=0))
    write(collection/'hf/hf-runtime.json',dict(profile=m.PROFILE))
    write(collection/'supervision/terminal.json',dict(passed=True,actor_started=True,actor_exit_code=0,
        forced_cleanup=False,child_exit_codes=[0,0,0],services=[dict(name=s['name'],exit_code=0) for s in services]))
    for service in services:
        name=service['name'];source_root='/sealed/source'
        write(package/f'service-{name}.json',dict(source_root=source_root,
            pins={source_root+'/tools/reports/serve_context_dual_candidate.py':'entry'},
            policy=dict(max_batch_queries=32,max_batch_tokens=131072,max_decisions=31)))
        write(collection/f'service-{name}/ready.json',dict(policy_id='p0',contract={},
            physical_devices=[service['device']],transport_protocol='bulk_dual_logits_candidate_v1',
            model_forward_mode='fixture',max_batch_queries=32,max_batch_tokens=131072,entrypoint_sha256='entry'))
        write(collection/f'service-{name}/terminal.json',dict(exit_code=0))
    return package,collection


@pytest.mark.parametrize('fault',[None,'hf','service','width','source','supervisor','logit','journal'])
def test_actual_macro_replay_and_relocated_source_admission(tmp_path,monkeypatch,fault):
    package,collection=outputs(tmp_path,monkeypatch,width=16 if fault=='journal' else 32)
    if fault=='hf':write(collection/'hf/hf-runtime.json',{'profile':'wrong'})
    elif fault=='service':
        p=collection/'service-a/ready.json';v=m.read(p);v['max_batch_queries']=4;write(p,v)
    elif fault=='width':
        p=package/'service-a.json';v=m.read(p);v['policy']['max_decisions']=60;write(p,v)
    elif fault=='source':write(package/'source-evidence/score-128.json',{})
    elif fault=='supervisor':
        p=collection/'supervision/terminal.json';v=m.read(p);v['forced_cleanup']=True;write(p,v)
    elif fault=='logit':
        p=collection/'hf/reference.json';v=m.read(p);v['records'][0]['logits'][0]=1e-9;write(p,v)
    if fault:
        with pytest.raises(ValueError):m.audit_outputs(package,collection)
    else:
        result=m.audit_outputs(package,collection)
        assert result['status']=='complete_replica_transfer_audited'
        assert result['recordings']==129 and result['parity']['exact']
        assert len(result['dispatch_audit'])==2
        assert not result['runtime_provenance_verified'] and not result['target_scored']


@pytest.mark.parametrize('failure',[None,'launch','collection','audit','runtime','score'])
def test_lifecycle_stops_on_failure_and_never_scores_targets(tmp_path,monkeypatch,failure):
    from multimodal_jev import local_checkpoint
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    c=dict(transfer_dispatch_allowed=True,operations_root=str(tmp_path/'ops'),tag='test',transfer_package={})
    calls=[]
    def launch(*a):
        calls.append('launch')
        if failure=='launch':raise ValueError('failed')
        return dict(status='wrong' if failure=='collection' else 'transfer_collected_independent_audit_required')
    def audit(*a):
        calls.append('audit')
        if failure=='audit':raise ValueError('failed')
        return dict(status='complete_replica_transfer_audited',runtime_provenance_verified=failure!='runtime',target_scored=failure=='score')
    monkeypatch.setattr(m,'launch',launch);monkeypatch.setattr(m,'audit',audit)
    if failure:
        with pytest.raises(ValueError):m.run(c,{})
    else:m.run(c,{})
    assert calls==(['launch'] if failure in ('launch','collection') else ['launch','audit'])
    assert m.read(Path(c['operations_root'])/'finalization/terminal.json')['exit_code']==int(failure is not None)
    with pytest.raises(FileExistsError):m.run(c,{})


def test_no_launch_without_explicit_dispatch(monkeypatch):
    with pytest.raises(ValueError,match='authorization'):m.launch({}, {})
    with pytest.raises(ValueError,match='authorization'):m.run({}, {})


@pytest.mark.parametrize('fault',[None,'archive','extra','readonly','wait','config','package'])
def test_public_transport_audit_before_output_replay(tmp_path,monkeypatch,fault):
    from test_prepare_context_replica_transfer import setup
    from tools.reports.context_cached_storage_operations import mounts
    from tools.reports.context_replica_training_operations import IMAGE
    import shutil
    spec,_=setup(tmp_path,monkeypatch)
    spec['manifest_sha256']=m.preparation.prepare(spec,{})['manifest_sha256']
    package=Path(spec['output']);manifest=m.read(package/'PREPARED.json')
    deployment=dict(tag='transfer-test',shared_root='/shared',driver_root='/usr/local/Ascend/driver')
    argv=m.transport.command(deployment,manifest,m.read(package/'config.json'),producer=True)
    def values(flag):return [argv[i+1] for i,v in enumerate(argv) if v==flag]
    state=dict(Status='exited',ExitCode=0,OOMKilled=False)
    handle=dict(argv=argv,container_id='a'*64,image=IMAGE,remote_root=spec['remote_root'],layout=manifest['layout'],
        manifest_sha256=spec['manifest_sha256'],config_sha256=m.file_hash(package/'config.json'),
        services={n:m.file_hash(package/p) for n,p in manifest['layout']['services'].items()})
    observed=dict(Id=handle['container_id'],Image=IMAGE,State=state,
        HostConfig=dict(Privileged=False,NetworkMode='none',Binds=values('-v'),
            Devices=[dict(PathOnHost=v.split(':')[0],PathInContainer=v.split(':')[1],CgroupPermissions='rwm')
                for v in values('--device')]),
        Config=dict(Entrypoint=values('--entrypoint'),Cmd=argv[argv.index(IMAGE)+1:],Env=values('-e')),
        Mounts=[dict(Source=src,Destination=dst,RW=not ro,Type='bind') for src,dst,ro in mounts(argv)])
    collection=tmp_path/'collected';collection.mkdir()
    for name in manifest['layout']['configs']+list(manifest['layout']['services'].values()):
        shutil.copyfile(package/name,collection/name)
    archive=tmp_path/'transport.tar';archive.write_bytes(b'fixture transport archive')
    transport=dict(status='transport_complete_independent_audit_required',container_id=handle['container_id'],
        image=IMAGE,remote_root=spec['remote_root'],state=state,bad_pins=[],bad_package=[],launch_binding_mismatches=[],
        archive_sha256=m.file_hash(archive),
        files={str(p.relative_to(collection)):m.file_hash(p) for p in collection.rglob('*') if p.is_file()})
    wait=dict(container_id=handle['container_id'],exit_code=0)
    if fault=='archive':archive.write_bytes(b'corrupt')
    elif fault=='extra':(collection/'extra').write_text('extra')
    elif fault=='readonly':next(v for v in observed['Mounts'] if v['Destination']=='/data/target-cache/input')['RW']=True
    elif fault=='wait':wait['container_id']='b'*64
    elif fault=='config':
        write(collection/'parity-hf.json',{})
        transport['files']['parity-hf.json']=m.file_hash(collection/'parity-hf.json')
    elif fault=='package':handle['manifest_sha256']='f'*64
    c=dict(transfer_package=spec,deployment=deployment,collection=str(collection),archive=str(archive))
    for name,value in [('launch',handle),('inspect',observed),('transport',transport),('wait',wait)]:
        p=tmp_path/(name+'.json');write(p,value);c[name]=str(p)
    calls=[]
    monkeypatch.setattr(m,'audit_outputs',lambda *a:(calls.append(a) or {'fixture':'output-replay'}))
    if fault:
        with pytest.raises(ValueError):m.audit(c,{})
        assert not calls
    else:
        result=m.audit(c,{})
        assert result['storage_audit']['actual_readonly_mounts_verified'] and len(calls)==1
        assert result['runtime_provenance_verified']
