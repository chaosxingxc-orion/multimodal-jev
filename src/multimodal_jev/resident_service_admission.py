"""Authenticate a separate TP1 service before connecting a DP4 training actor.

The launcher supplies a fresh docker inspect; a ready file alone is insufficient.
This does not certify the worker's probability parity or grant training admission.
"""
IMAGE='sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'


def validate(launch, inspected, ready, *, actor_devices, config_sha256):
    devices=launch.get('physical_devices')
    if (not isinstance(actor_devices,list) or len(actor_devices)!=4
            or len(set(actor_devices))!=4
            or any(type(i) is not int or not 0<=i<=7 for i in actor_devices)
            or not isinstance(devices,list) or len(devices)!=1
            or type(devices[0]) is not int or not 0<=devices[0]<=7
            or devices[0] in actor_devices):
        raise ValueError('isolated DP4 actor and separate TP1 service required')
    if (not launch.get('container_id') or inspected.get('Id')!=launch['container_id']
            or launch.get('image')!=IMAGE or inspected.get('Image')!=IMAGE
            or inspected['State'].get('Status')!='running'
            or inspected['State'].get('Running') is not True
            or inspected['State'].get('Restarting') is not False
            or inspected['State'].get('OOMKilled') is not False
            or launch.get('config_sha256')!=config_sha256):
        raise ValueError('exact live pinned resident service required')
    physical=[d['PathOnHost'] for d in inspected['HostConfig']['Devices']
              if d['PathOnHost'].rsplit('/',1)[-1].removeprefix('davinci').isdigit()]
    if physical!=['/dev/davinci'+str(devices[0])]:
        raise ValueError('actual service device mounts differ')
    for key in ('physical_devices','socket_path','base_id','runtime_profile'):
        if ready.get(key)!=launch.get(key):raise ValueError('ready service identity differs: '+key)
    address=launch['socket_path']
    if not address.startswith('/jev-ipc/') or len(address.encode())>107:
        raise ValueError('short shared local socket required')
    if launch['exchange']!='/jev-ipc/exchange':raise ValueError('fixed local exchange required')
    mounts={m['Destination']:m for m in inspected['Mounts']}
    ipc=mounts.get('/jev-ipc',{})
    if ipc.get('Source')!=launch['ipc_host'] or ipc.get('RW') is not True:
        raise ValueError('actual shared socket mount differs')
    shared=mounts.get('/mnt/cephfs/xingchao',{})
    if shared.get('Source')!='/mnt/cephfs/xingchao' or shared.get('RW') is not False:
        raise ValueError('shared datasets/models must remain read-only')
    return dict(container_id=launch['container_id'],image=IMAGE,physical_devices=devices,
                socket_path=address,exchange=launch['exchange'],passed=True,
                scope='Live process/config/ready/device/mount authentication only; probability acceptance separate.')
