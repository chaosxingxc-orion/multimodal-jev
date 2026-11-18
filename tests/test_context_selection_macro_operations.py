import copy
import pytest

from tools.reports.context_selection_macro_operations import command
from tools.reports.context_macro_probe_operations import inspect_runtime,IMAGE


def config():
    return dict(remote_root='/data/user/selection',tag='selection',model='/models/asr',driver_root='/driver',
        release='/datasets/release',cache_root='/data/user/shared-cache')


@pytest.mark.parametrize('role',['client','preflight','service-0','service-1'])
def test_shared_lifecycle_commands_keep_readonly_inputs_and_device_isolation(role):
    argv=command(config(),role)
    def values(flag):return [argv[i+1] for i,v in enumerate(argv) if v==flag]
    devices=values('--device');binds=values('-v')
    if role in ('client','preflight'):
        assert not devices
        assert '/datasets/release:/datasets/release:ro' in binds
        assert '/data/user/shared-cache:/data/user/shared-cache:ro' in binds
        assert 'context_macro_measurement' not in argv[-1]
    else:
        assert len(devices)==4 and devices[0].startswith('/dev/davinci'+str(6+int(role[-1])))
        assert '/datasets/release:/datasets/release:ro' not in binds
    handle=dict(container_id='a'*64,argv=argv)
    obs=dict(Id='a'*64,Image=IMAGE,State=dict(Status='exited',ExitCode=0,OOMKilled=False),
        Config=dict(Entrypoint=values('--entrypoint'),Cmd=argv[argv.index(IMAGE)+1:],Env=values('-e')),
        HostConfig=dict(Privileged=False,NetworkMode='none',Binds=binds,Devices=[
            dict(PathOnHost=v.split(':')[0],PathInContainer=v.split(':')[1],CgroupPermissions='rwm') for v in devices]))
    inspect_runtime(obs,handle,terminal=True)
    bad=copy.deepcopy(obs);bad['HostConfig']['Binds'].append('/:/host')
    with pytest.raises(ValueError):inspect_runtime(bad,handle)


@pytest.mark.parametrize('path',['relative','/data/user/selection/cache','/data','/foo:bar'])
def test_no_input_output_alias_or_mount_injection(path):
    c=config();c['cache_root']=path
    with pytest.raises(ValueError):command(c,'client')
