import copy

import pytest

from tools.reports.context_macro_probe_operations import command, inspect_runtime, IMAGE


def config():
    return dict(remote_root='/local/macro',tag='macro-v1',model='/models/asr',driver_root='/driver')


def observation(role):
    argv=command(config(),role)
    def values(flag):return [argv[i+1] for i,v in enumerate(argv) if v==flag]
    handle=dict(container_id='a'*64,argv=argv)
    obs=dict(Id='a'*64,Image=IMAGE,State=dict(Status='exited',ExitCode=0,OOMKilled=False),
        Config=dict(Entrypoint=values('--entrypoint'),Cmd=argv[argv.index(IMAGE)+1:],Env=values('-e')),
        HostConfig=dict(Privileged=False,NetworkMode='none',Binds=values('-v'),Devices=[
            dict(PathOnHost=v.split(':')[0],PathInContainer=v.split(':')[1],CgroupPermissions='rwm')
            for v in values('--device')]))
    return obs,handle


@pytest.mark.parametrize('role,devices',[('service-0',['/dev/davinci6']),('service-1',['/dev/davinci7']),('client',[]),('preflight',[])])
def test_isolated_roles_and_actual_inspect(role,devices):
    obs,handle=observation(role);inspect_runtime(obs,handle,terminal=True)
    assert [d['PathOnHost'] for d in obs['HostConfig']['Devices'] if d['PathOnHost'][-1].isdigit()]==devices
    assert 'VLLM_ALLOW_INSECURE_SERIALIZATION=1' not in obs['Config']['Env']
    assert '/local/macro/package:/local/macro/package:ro' in obs['HostConfig']['Binds']


@pytest.mark.parametrize('fault',['id','image','device','network','mount','command','env','exit','oom'])
def test_runtime_scope_or_failed_terminal_cannot_pass(fault):
    obs,handle=observation('service-0');bad=copy.deepcopy(obs)
    if fault=='id':bad['Id']='other'
    if fault=='image':bad['Image']='other'
    if fault=='device':bad['HostConfig']['Devices'][0]['PathOnHost']='/dev/davinci4'
    if fault=='network':bad['HostConfig']['NetworkMode']='host'
    if fault=='mount':bad['HostConfig']['Binds'].pop()
    if fault=='command':bad['Config']['Cmd']=['other']
    if fault=='env':bad['Config']['Env']=[]
    if fault=='exit':bad['State']['ExitCode']=1
    if fault=='oom':bad['State']['OOMKilled']=True
    with pytest.raises(ValueError):inspect_runtime(bad,handle,terminal=True)
