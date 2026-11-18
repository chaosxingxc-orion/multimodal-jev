import ast
import json

import pytest

from tools.reports.context_macro_training_operations import command
from tools.reports.context_macro_probe_operations import inspect_runtime, IMAGE


def fixture(tmp_path):
    package=tmp_path/'package';package.mkdir()
    training=dict(encoded_audio_cache='/local/audio/cache',audio_mirror_root='/local/mirror',
        pins={'/shared/dataset/MANIFEST.json':'hash','/local/audio/cache/a.pt':'hash',
              '/local/mirror/MANIFEST.json':'hash','/local/run/package/code.py':'hash'})
    (package/'training.json').write_text(json.dumps(training))
    c=dict(prepared=str(tmp_path),remote_root='/local/run',tag='dp2-prefix',model='/shared/model',
           shared_root='/shared',driver_root='/driver')
    return c,training


@pytest.mark.parametrize('role,devices',[('client',['davinci4','davinci5']),('preflight',[]),
    ('service-0',['davinci6']),('service-1',['davinci7'])])
def test_physical_isolation_native_torchrun_and_pinned_input_mounts(tmp_path,role,devices):
    c,_=fixture(tmp_path);argv=command(c,role)
    def values(flag):return [argv[i+1] for i,v in enumerate(argv) if v==flag]
    assert [d.split(':')[0].split('/')[-1] for d in values('--device') if d.split(':')[0][-1].isdigit()]==devices
    if role=='client':
        assert argv[argv.index(IMAGE)+1]=='torchrun' and '--nproc-per-node=2' in argv
        assert '--standalone' not in argv and '--master-addr=127.0.0.1' in argv
        assert '--master-port=29500' in argv and '--node-rank=0' in argv
        assert '/shared:/shared:ro' in values('-v')
        assert '/local/audio/cache:/local/audio/cache' in values('-v')
    if role=='preflight':
        assert 'initialize_verl_preserving_qwen3' in argv[-1]
        assert 'verify_native_adapter' in argv[-1]
        tree=ast.parse(argv[-1])
        probe=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='write_text')
        ast.parse(ast.literal_eval(probe.args[0]))
        assert 'timeout=90' in argv[-1] and '127.0.0.1' in argv[-1]
    obs=dict(Id='a'*64,Image=IMAGE,State=dict(Status='exited',ExitCode=0,OOMKilled=False),
        Config=dict(Entrypoint=values('--entrypoint'),Cmd=argv[argv.index(IMAGE)+1:],Env=values('-e')),
        HostConfig=dict(Privileged=False,NetworkMode='none',Binds=values('-v'),Devices=[
            dict(PathOnHost=d.split(':')[0],PathInContainer=d.split(':')[1],CgroupPermissions='rwm')
            for d in values('--device')]))
    inspect_runtime(obs,dict(container_id='a'*64,argv=argv),terminal=True)


def test_unmounted_external_dependency_refused(tmp_path):
    c,t=fixture(tmp_path);t['pins']['/unmounted/private/file']='hash'
    (tmp_path/'package/training.json').write_text(json.dumps(t))
    with pytest.raises(ValueError,match='outside explicit mounts'):command(c,'client')


def test_shared_mount_cannot_cover_runtime(tmp_path):
    c,_=fixture(tmp_path);c['shared_root']='/local'
    with pytest.raises(ValueError,match='disjoint'):command(c,'client')
