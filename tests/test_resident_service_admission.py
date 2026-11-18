import copy
import pytest
from multimodal_jev.resident_service_admission import IMAGE,validate


def fixture():
    launch=dict(container_id='service',image=IMAGE,physical_devices=[4],config_sha256='hash',
        socket_path='/jev-ipc/rollout.sock',exchange='/jev-ipc/exchange',ipc_host='/data/task/ipc',
        base_id='base',runtime_profile='fixed')
    inspect=dict(Id='service',Image=IMAGE,State=dict(Status='running',Running=True,Restarting=False,OOMKilled=False),
        HostConfig=dict(Devices=[dict(PathOnHost='/dev/'+n) for n in ['davinci4','davinci_manager','devmm_svm','hisi_hdc']]),
        Mounts=[dict(Destination='/jev-ipc',Source='/data/task/ipc',RW=True),
                dict(Destination='/mnt/cephfs/xingchao',Source='/mnt/cephfs/xingchao',RW=False)])
    ready={k:copy.deepcopy(launch[k]) for k in ('physical_devices','socket_path','base_id','runtime_profile')}
    return launch,inspect,ready


def test_exact_live_isolated_service():
    assert validate(*fixture(),actor_devices=[0,1,2,3],config_sha256='hash')['passed']


@pytest.mark.parametrize('fault',['overlap','duplicate','bool_device','container','image','exited','restarting',
    'oom','config','device_mount','extra_device','ready','socket','exchange','ipc','shared_rw'])
def test_reject_stale_or_misbound_service(fault):
    launch,inspect,ready=fixture();actor=[0,1,2,3]
    if fault=='overlap':actor[-1]=4
    elif fault=='duplicate':actor[-1]=0
    elif fault=='bool_device':actor[0]=False
    elif fault=='container':inspect['Id']='other'
    elif fault=='image':inspect['Image']='other'
    elif fault=='exited':inspect['State']['Status']='exited'
    elif fault=='restarting':inspect['State']['Restarting']=True
    elif fault=='oom':inspect['State']['OOMKilled']=True
    elif fault=='config':launch['config_sha256']='other'
    elif fault=='device_mount':inspect['HostConfig']['Devices'][0]['PathOnHost']='/dev/davinci5'
    elif fault=='extra_device':inspect['HostConfig']['Devices'].append(dict(PathOnHost='/dev/davinci0'))
    elif fault=='ready':ready['base_id']='other'
    elif fault=='socket':launch['socket_path']=ready['socket_path']='/data/'+'x'*108
    elif fault=='exchange':launch['exchange']='/other'
    elif fault=='ipc':inspect['Mounts'][0]['Source']='/data/other'
    else:inspect['Mounts'][1]['RW']=True
    with pytest.raises(ValueError):validate(launch,inspect,ready,actor_devices=actor,config_sha256='hash')
