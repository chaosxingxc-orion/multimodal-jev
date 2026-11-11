from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

from multimodal_jev.omni_fsdp_initialization_audit import validate_ranks

TOOLS = Path(__file__).parents[1]/'tools/jev'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('initialization_audit', TOOLS/'audit_omni_fsdp_initialization.py')
cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)


def fixture():
    structure = {'head.weight': dict(shape=[3, 2], dtype='torch.bfloat16', numel=6),
                 'layers.0.weight': dict(shape=[2, 2], dtype='torch.bfloat16', numel=4)}
    original = {n: dict(shape=v['shape'], dtype=v['dtype'], sha256='a'*64)
                for n, v in structure.items()}
    buffers = {'rope': dict(shape=[2], dtype='torch.float32', sha256='b'*64, persistent=False)}
    units = [dict(name='<root>', parameters=6), dict(name='layers.0', parameters=4)]
    ranks = []
    for rank in (0, 1):
        r = dict(rank=rank, local_rank=rank, device=f'npu:{rank}', visible_devices=2,
                 status='initialization_verified', parameters=10,
                 original_parameters_sha256='c'*64, preflight_sha256='d'*64,
                 buffers={n: {k: v for k, v in b.items() if k != 'persistent'} for n, b in buffers.items()},
                 forward_calls=0, weight_updates=0, units=deepcopy(units),
                 initial_memory=dict(allocated=10, reserved=20), peak_allocated_bytes=15,
                 peak_reserved_bytes=25, elapsed_seconds=1.)
        ranks.append((r, deepcopy(original)))
    return dict(original=original, structure=structure, buffers=buffers, units=units,
                ranks=ranks, original_sha256='c'*64, preflight_sha256='d'*64)


def test_complete_two_rank_state_and_order_independent_units():
    args = fixture(); args['ranks'].reverse(); args['ranks'][0][0]['units'].reverse()
    report = validate_ranks(**args)
    assert report['parameters'] == 10 and report['ranks'] == 2
    assert [r['rank'] for r in report['memory']] == [0, 1]


@pytest.mark.parametrize('kind', ['missing_rank', 'duplicate_rank', 'bool_rank', 'device',
    'value', 'shape', 'missing_param', 'buffer', 'missing_unit', 'duplicate_unit',
    'ownership', 'source', 'preflight', 'forward', 'update', 'peak', 'nan'])
def test_reject_incomplete_changed_state_or_invalid_accounting(kind):
    a = fixture(); r, params = a['ranks'][1]
    if kind == 'missing_rank': a['ranks'].pop()
    if kind == 'duplicate_rank': r['rank'] = 0
    if kind == 'bool_rank': r['rank'] = True
    if kind == 'device': r['device'] = 'npu:0'
    if kind == 'value': params['head.weight']['sha256'] = 'e'*64
    if kind == 'shape': a['original']['head.weight']['shape'] = [6]
    if kind == 'missing_param': params.pop('head.weight')
    if kind == 'buffer': r['buffers']['rope']['dtype'] = 'torch.bfloat16'
    if kind == 'missing_unit': r['units'].pop()
    if kind == 'duplicate_unit': r['units'].append(deepcopy(r['units'][0]))
    if kind == 'ownership':
        a['units'][0]['parameters'] = 4; a['units'][1]['parameters'] = 6
    if kind == 'source': r['original_parameters_sha256'] = 'e'*64
    if kind == 'preflight': r['preflight_sha256'] = 'e'*64
    if kind == 'forward': r['forward_calls'] = 1
    if kind == 'update': r['weight_updates'] = 1
    if kind == 'peak': r['peak_allocated_bytes'] = 26
    if kind == 'nan': r['elapsed_seconds'] = float('nan')
    with pytest.raises(ValueError): validate_ranks(**a)


def runtime_fixture():
    image = cli.successful.__globals__['IMAGE']
    argv = ['docker', 'create', '-e', 'HF_HUB_OFFLINE=1', '-v', '/model:/model:ro', image, 'timeout', '3600s']
    devices = ['/dev/davinci4', '/dev/davinci5', '/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc']
    actual = dict(Id='a'*64, Image=image, RestartCount=0,
        State=dict(Status='exited', ExitCode=0, OOMKilled=False, Dead=False, Error=''),
        Config=dict(Cmd=['timeout','3600s'], Entrypoint=['/entrypoint.sh'], Env=['HF_HUB_OFFLINE=1']),
        HostConfig=dict(NanoCpus=16*10**9, Memory=256*1024**3, MemorySwap=256*1024**3,
            ShmSize=8*1024**3, NetworkMode='host', IpcMode='private', Privileged=False,
            DeviceRequests=None, DeviceCgroupRules=None, Binds=['/model:/model:ro'],
            Devices=[dict(PathOnHost=d,PathInContainer=d,CgroupPermissions='rwm') for d in devices]))
    return actual, dict(argv=argv,image=image)


def test_exact_prepared_runtime():
    actual, prep = runtime_fixture()
    cli.validate_runtime(actual, prep, 'a'*64, ['/entrypoint.sh'])


@pytest.mark.parametrize('kind', ['live', 'exit', 'oom', 'restart', 'device', 'extra_device',
    'device_rule', 'device_request', 'memory', 'cpu', 'ipc', 'entrypoint', 'command', 'env', 'bind'])
def test_runtime_rejects_failure_or_unregistered_configuration(kind):
    actual, prep = runtime_fixture(); host=actual['HostConfig']; config=actual['Config']
    if kind == 'live': actual['State']['Status']='running'
    if kind == 'exit': actual['State']['ExitCode']=1
    if kind == 'oom': actual['State']['OOMKilled']=True
    if kind == 'restart': actual['RestartCount']=1
    if kind == 'device': host['Devices'][0]['PathOnHost']='/dev/davinci0'
    if kind == 'extra_device': host['Devices'].append(deepcopy(host['Devices'][0]))
    if kind == 'device_rule': host['DeviceCgroupRules']=['c *:* rwm']
    if kind == 'device_request': host['DeviceRequests']=[{'Count':-1}]
    if kind == 'memory': host['Memory']=128*1024**3
    if kind == 'cpu': host['NanoCpus']=2*10**9
    if kind == 'ipc': host['IpcMode']='host'
    if kind == 'entrypoint': config['Entrypoint']=[]
    if kind == 'command': config['Cmd']=['true']
    if kind == 'env': config['Env']=[]
    if kind == 'bind': host['Binds'].append('/:/host:rw')
    with pytest.raises(ValueError): cli.validate_runtime(actual,prep,'a'*64,['/entrypoint.sh'])
