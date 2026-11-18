import json
from pathlib import Path

import pytest

from tools.reports import context_arithmetic_resume_operations as ops
from tools.reports.context_arithmetic_resume_runtime import expected_config


@pytest.fixture
def package(tmp_path):
    original=tmp_path/'original';original.mkdir();root=tmp_path/'package';root.mkdir()
    def put(base,name,value):
        p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return ops.sha(p)
    old=dict(algorithm='rloo_v1',physical_devices=[4,5,6,7],global_batch=128,microbatch=8,
        encoded_audio_cache='/cache/run/cache',audio_mirror_root='/mirror/run',pins={})
    digest=put(original,'config.json',old);put(original,'PREPARED.json',{})
    service=dict(policy=dict(physical_devices=[7]),pins={});put(original,'service-config.json',service)
    c=expected_config(old,origin_config='/old/config.json',remote_root='/new',step=64)
    c['pins']={};put(root,'config.json',c)
    service['policy'].update(snapshot='/old/run/policy-0064',adapter='/old/run/adapter-0064',policy_id='p')
    put(root,'service-config.json',service)
    put(root,'RESUME.json',dict(schema='context-arithmetic-recovery-prefix-v1',runtime_root='/new',step=64,
        original_manifest_sha256=ops.sha(original/'PREPARED.json'),original_config_sha256=digest,
        behavior_policy_id='p',training_dispatch_allowed=False))
    m=dict(status='arithmetic_recovery_prefix_prepared_only',image=ops.IMAGE,remote_root='/new',
        training_dispatch_allowed=False,finite_dispatch_allowed=False)
    settings=dict(package=str(root),original_package=str(original),host='npu',tag='resume-test',
        shared_root='/shared',driver_root='/driver',finite_recovery_dispatch_allowed=False)
    def seal():
        m['files']={str(p.relative_to(root)):ops.sha(p) for p in root.rglob('*') if p.is_file() and p.name!='PREPARED.json'}
        put(root,'PREPARED.json',m);settings['manifest_sha256']=ops.sha(root/'PREPARED.json')
    seal();return settings,root,seal


def test_explicit_mounts_preserve_original_readonly(package):
    c,_,_=package;m,config=ops.validate(c);argv=ops.command(c,m,config)
    assert '--volumes-from' not in argv
    assert '/old:/old:ro' in argv and '/shared:/shared:ro' in argv
    devices=[argv[i+1] for i,v in enumerate(argv) if v=='--device']
    assert '/dev/davinci0:/dev/davinci0' not in devices
    assert all('/dev/davinci'+str(i)+':/dev/davinci'+str(i) in devices for i in (4,5,6,7))


@pytest.mark.parametrize('fault',['algorithm','end','reference','batch','service','root','scope','modified'])
def test_resealed_conflicting_config_rejected(package,fault):
    c,root,seal=package
    if fault=='modified':(root/'config.json').write_text('{}')
    else:
        name='service-config.json' if fault=='service' else 'RESUME.json' if fault in ('root','scope') else 'config.json'
        p=root/name;value=json.loads(p.read_text())
        if fault=='algorithm':value['algorithm']='grpo_v1'
        if fault=='end':value['stop_update']=634
        if fault=='reference':value['reference_cache']='/old/reference-cache'
        if fault=='batch':value['microbatch']=32
        if fault=='service':value['policy']['policy_id']='wrong'
        if fault=='root':value['runtime_root']='/other'
        if fault=='scope':value['training_dispatch_allowed']=True
        p.write_text(json.dumps(value));seal()
    with pytest.raises(ValueError):ops.validate(c)


def test_launch_disabled_before_remote_calls(package,monkeypatch):
    c,_,_=package
    monkeypatch.setattr(ops.subprocess,'run',lambda *a,**kw:pytest.fail('unexpected remote call'))
    with pytest.raises(ValueError,match='dispatch not enabled'):ops.launch(c,{})


def test_overlapping_output_mount_rejected(package):
    c,_,_=package;m,config=ops.validate(c);c['shared_root']='/new/inputs'
    with pytest.raises(ValueError,match='disjoint'):ops.command(c,m,config)


def test_unmounted_dependency_rejected(package):
    c,_,_=package;m,config=ops.validate(c);config['pins']['/unmounted/data']='hash'
    with pytest.raises(ValueError,match='outside explicit mounts'):ops.command(c,m,config)
