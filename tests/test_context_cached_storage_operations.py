import hashlib
from pathlib import Path

import pytest

from tools.reports.context_cached_storage_operations import command,audit_mounts,storage_preflight,audit_container,mounts
from test_context_replica_training_operations import topology,REMOTE


def fixture():
    c,m,config=topology(2,2)
    config['source_policy']=REMOTE+'/initial/policy'
    config['source_root']=REMOTE+'/source'
    config['reference_dataset']=dict(root='/data/host/reference/dataset',manifest_sha256='a'*64)
    config['pins'].update({REMOTE+'/SOURCE-BINDING-AUDIT.json':'b'*64,
        '/data/host/reference/dataset/MANIFEST.json':'a'*64})
    return c,m,config


def test_readonly_dependencies_and_same_shared_execution_environment():
    c,m,config=fixture();argv=command(c,m,config)
    binds=[argv[i+1] for i,v in enumerate(argv) if v=='-v']
    for name in (config['encoded_audio_cache'],config['audio_mirror_root'],
                 config['reference_dataset']['root'],REMOTE+'/initial',REMOTE+'/SOURCE-BINDING-AUDIT.json'):
        assert name+':'+name+':ro' in binds
    assert REMOTE+':'+REMOTE in binds and REMOTE+'/cache:/jev-tmp' in binds
    assert argv[-1]==REMOTE+'/run.py'
    assert audit_mounts(argv,config)['actual_container_inspect_pending']


@pytest.mark.parametrize('fault',['cache-rw','shadow','reference-rw','missing-reference','duplicate'])
def test_mount_audit_rejects_ineffective_readonly(fault):
    c,m,config=fixture();argv=command(c,m,config)
    cache=config['encoded_audio_cache'];ref=config['reference_dataset']['root']
    if fault in ('cache-rw','reference-rw'):
        name=cache if fault=='cache-rw' else ref
        i=argv.index(name+':'+name+':ro');argv[i]=name+':'+name
    elif fault=='missing-reference':
        i=argv.index(ref+':'+ref+':ro');del argv[i-1:i+1]
    else:
        name=cache if fault=='duplicate' else cache+'/entries'
        argv[0:0]=['-v',name+':'+name]
    with pytest.raises(ValueError):audit_mounts(argv,config)


@pytest.mark.parametrize('fault',['reference-under-run','reference-parent','shared-card','six-training','old-policy'])
def test_unbounded_or_colliding_layout_refused(fault):
    c,m,config=fixture()
    if fault=='reference-under-run':config['reference_dataset']['root']=REMOTE+'/reference'
    elif fault=='reference-parent':config['reference_dataset']['root']=str(Path(REMOTE).parent)
    elif fault=='shared-card':m['inference_services'][0]['device']=0
    elif fault=='six-training':m['training_devices']=list(range(6))
    else:config['source_policy']='/data/old/policy'
    with pytest.raises(ValueError):command(c,m,config)


def test_preflight_only_checks_run_storage_and_reads_inputs(tmp_path):
    cache=tmp_path/'inputs/cache';mirror=tmp_path/'mirror';ref=tmp_path/'reference'
    for p in (cache,mirror,ref):p.mkdir(parents=True)
    manifest=ref/'MANIFEST.json';manifest.write_bytes(b'reference')
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    config=dict(encoded_audio_cache=str(cache),audio_mirror_root=str(mirror),
        reference_dataset=dict(root=str(ref),manifest_sha256=digest),pins={str(manifest):digest})
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    checks=[];runtime=str(tmp_path/'new-runtime')
    script=storage_preflight(config,runtime)
    exec(compile(script,'<storage-preflight>','exec'),{'check_paths':lambda p:checks.append(p)})
    assert checks==[{'run':runtime}] and not Path(runtime).exists()
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    manifest.write_bytes(b'changed')
    with pytest.raises(AssertionError):exec(compile(script,'<test>','exec'),{'check_paths':lambda _:None})


def test_reference_manifest_pin_cannot_be_omitted(tmp_path):
    _,_,config=fixture();config['pins'].pop(config['reference_dataset']['root']+'/MANIFEST.json')
    with pytest.raises(ValueError,match='pinned externally'):storage_preflight(config,REMOTE)


def test_reference_producer_has_no_reference_mount_and_readonly_inputs():
    from tools.reports.context_cached_storage_operations import reference_command
    c,m,config=fixture();reference=config.pop('reference_dataset')
    del config['pins'][reference['root']+'/MANIFEST.json']
    m.update(status='full_train_reference_job_prepared_only',actor_updates=0,expected_population=40513)
    argv=reference_command(c,m,config)
    binds=mounts(argv)
    assert all(dst!=reference['root'] for _,dst,_ in binds)
    assert (config['encoded_audio_cache'],config['encoded_audio_cache'],True) in binds
    assert (REMOTE,REMOTE,False) in binds
    assert audit_mounts(argv,config,producer=True)['actual_container_inspect_pending']
    m['actor_updates']=2
    with pytest.raises(ValueError,match='reference-only'):reference_command(c,m,config)


def test_producer_preflight_does_not_require_or_write_future_outputs(tmp_path):
    cache=tmp_path/'input';mirror=tmp_path/'audio'
    cache.mkdir();mirror.mkdir()
    config=dict(encoded_audio_cache=str(cache),audio_mirror_root=str(mirror),pins={})
    root=tmp_path/'new-job';checks=[]
    script=storage_preflight(config,str(root),producer=True)
    exec(compile(script,'<producer-preflight>','exec'),{'check_paths':lambda v:checks.append(v)})
    assert checks==[{'run':str(root)}] and not root.exists()
    assert not list(cache.iterdir()) and not list(mirror.iterdir())
    config['reference_cache']='/data/online'
    with pytest.raises(ValueError,match='cannot consume'):
        storage_preflight(config,str(root),producer=True)


@pytest.mark.parametrize('fault',[None,'rw','source','missing','extra','network'])
def test_actual_docker_inspect_required_not_just_requested_binds(fault):
    from tools.reports.context_replica_training_operations import IMAGE
    c,m,config=fixture();argv=command(c,m,config)
    def values(flag):return [argv[i+1] for i,v in enumerate(argv) if v==flag]
    handle=dict(argv=argv,container_id='a'*64)
    observed=dict(Id=handle['container_id'],Image=IMAGE,
        State=dict(Status='exited',ExitCode=0,OOMKilled=False),
        HostConfig=dict(Privileged=False,NetworkMode='none',Binds=values('-v'),
            Devices=[dict(PathOnHost=v.split(':')[0],PathInContainer=v.split(':')[1],CgroupPermissions='rwm')
                     for v in values('--device')]),
        Config=dict(Entrypoint=values('--entrypoint'),Cmd=argv[argv.index(IMAGE)+1:],Env=values('-e')),
        Mounts=[dict(Source=src,Destination=dst,RW=not ro,Type='bind') for src,dst,ro in mounts(argv)])
    ref=next(v for v in observed['Mounts'] if v['Destination']==config['reference_dataset']['root'])
    if fault=='rw':ref['RW']=True
    elif fault=='source':ref['Source']='/data/different-reference'
    elif fault=='missing':observed['Mounts'].remove(ref)
    elif fault=='extra':observed['Mounts'].append(dict(Source='/other',Destination='/extra',RW=True,Type='bind'))
    elif fault=='network':observed['HostConfig']['NetworkMode']='host'
    if fault is None:
        result=audit_container(argv,config,handle,observed)
        assert result['actual_readonly_mounts_verified'] and not result['actual_container_inspect_pending']
    else:
        with pytest.raises(ValueError):audit_container(argv,config,handle,observed)
