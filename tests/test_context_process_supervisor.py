"""Real subprocess/socket failure paths; no simulated NPU success."""
import json
import os
from pathlib import Path
import sys

import pytest

from tools.jev.context_process_supervisor import supervise


SERVICE = '''
import json,socket,struct,sys
from pathlib import Path
address,ready=sys.argv[1:]
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
s.bind(address);s.listen(1)
Path(ready).write_text('ready')
c,_=s.accept()
def exact(n):
 b=b''
 while len(b)<n:
  p=c.recv(n-len(b))
  if not p: raise EOFError()
  b+=p
 return b
command=json.loads(exact(struct.unpack('!I',exact(4))[0]))
assert command==dict(operation='close',sequence=0)
b=json.dumps(dict(sequence=0,ok=True,result=None)).encode()
c.sendall(struct.pack('!I',len(b))+b)
c.close();s.close();Path(address).unlink()
'''


def settings(tmp_path, actor='raise SystemExit(7)', service=SERVICE):
    address=tmp_path/'s';ready=tmp_path/'ready'
    return dict(service_command=[sys.executable,'-c',service,str(address),str(ready)],
                actor_command=[sys.executable,'-c',actor], service_env=dict(os.environ),
                actor_env=dict(os.environ),output=tmp_path/'supervision',
                socket_path=address,ready_path=ready,startup_timeout=2.,shutdown_timeout=.3)


GREETING_SERVICE = '''
import json,socket,struct,sys
from pathlib import Path
address,ready,repeat=sys.argv[1:]
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
s.bind(address);s.listen(1)
Path(ready).write_text('ready')
c,_=s.accept()
def exact(n):
 b=b''
 while len(b)<n:
  p=c.recv(n-len(b))
  if not p: raise EOFError()
  b+=p
 return b
def send(o):
 b=json.dumps(o).encode()
 c.sendall(struct.pack('!I',len(b))+b)
greeting=dict(protocol='replica-macro-runtime-hello-v1',max_request_queries=48,
              runtime_identity=dict(filler='x'*9000),policy_id='p',contract=dict(k='v'*9000))
for _ in range(int(repeat)): send(greeting)
command=json.loads(exact(struct.unpack('!I',exact(4))[0]))
assert command==dict(operation='close',sequence=0)
send(dict(sequence=0,ok=True,result=None))
c.close();s.close();Path(address).unlink()
'''


def greeting_settings(tmp_path, repeat=1, actor='raise SystemExit(7)'):
    args = settings(tmp_path, actor=actor, service=GREETING_SERVICE)
    args['service_command'] = [sys.executable, '-c', GREETING_SERVICE,
                               str(tmp_path / 's'), str(tmp_path / 'ready'), str(repeat)]
    return args


def test_greeting_service_close_acknowledged_after_protocol_frame(tmp_path):
    """A live replica service greets on accept; close must skip that frame.

    The greeting is a large HELLO frame, so treating it as the acknowledgement
    trips both the old size guard and the reply comparison.
    """
    report = supervise(**greeting_settings(tmp_path))
    assert not report['passed'] and report['actor_exit_code'] == 7
    assert report['close_ack'] == dict(sequence=0, ok=True, result=None)
    assert 'close_error' not in report
    assert report['service_exit_code'] == 0


def test_repeated_greeting_is_not_acknowledgement(tmp_path):
    report = supervise(**greeting_settings(tmp_path, repeat=2))
    assert 'close_ack' not in report
    assert 'did not acknowledge close' in report['close_error']


def test_actor_dies_before_connect_service_closes_normally(tmp_path):
    report=supervise(**settings(tmp_path))
    assert not report['passed']
    assert report['actor_exit_code']==7
    assert report['service_exit_code']==0
    assert report['close_ack']['ok']
    assert not report['forced_cleanup']
    assert json.loads((tmp_path/'supervision/terminal.json').read_text())==report


def test_actor_success_and_service_shutdown(tmp_path):
    report=supervise(**settings(tmp_path,actor='pass'))
    assert report['passed'] and report['child_exit_codes']==[0,0]


def test_service_startup_failure_never_starts_actor(tmp_path):
    with pytest.raises(RuntimeError,match='before readiness'):
        supervise(**settings(tmp_path,service='raise SystemExit(9)'))
    report=json.loads((tmp_path/'supervision/terminal.json').read_text())
    assert not report['actor_started'] and report['child_exit_codes']==[9]


def test_service_deadline_cleans_owned_process(tmp_path):
    args=settings(tmp_path,service='import time;time.sleep(60)')
    args['startup_timeout']=.15
    with pytest.raises(TimeoutError,match='startup deadline'):
        supervise(**args)
    report=json.loads((tmp_path/'supervision/terminal.json').read_text())
    assert not report['actor_started'] and report['forced_cleanup']
    assert all(code is not None for code in report['child_exit_codes'])


def test_stale_readiness_rejected_before_launch(tmp_path):
    args=settings(tmp_path)
    Path(args['ready_path']).touch()
    with pytest.raises(ValueError,match='existing handles'):
        supervise(**args)
    assert not Path(args['output']).exists()


def test_hung_service_after_actor_failure_is_bounded(tmp_path):
    service=SERVICE[:SERVICE.index('c,_=s.accept()')]+'import time;time.sleep(60)'
    report=supervise(**settings(tmp_path,service=service))
    assert not report['passed'] and report['forced_cleanup']
    assert report['actor_exit_code']==7 and report['service_exit_code']!=0
    assert 'close_error' in report


def test_service_dies_while_actor_is_live(tmp_path):
    service=SERVICE[:SERVICE.index('c,_=s.accept()')]+'import time;time.sleep(.15);raise SystemExit(9)'
    with pytest.raises(RuntimeError,match='actor remained live'):
        supervise(**settings(tmp_path,service=service,actor='import time;time.sleep(60)'))
    report=json.loads((tmp_path/'supervision/terminal.json').read_text())
    assert report['actor_started'] and not report['passed']
    assert report['child_exit_codes'][0]==9
    assert report['child_exit_codes'][1] is not None


def test_actor_can_close_service_before_its_own_exit(tmp_path):
    actor=('from tools.jev.context_process_supervisor import close_unconnected_service;'
           f'close_unconnected_service({str(tmp_path / "s")!r},1);'
           'import time;time.sleep(.1)')
    report=supervise(**settings(tmp_path,actor=actor))
    assert report['passed'] and not report['forced_cleanup']


def declared(tmp_path, count, actor='raise SystemExit(7)', root=None):
    root = root or tmp_path
    root.mkdir(parents=True, exist_ok=True)
    services = []
    for i in range(count):
        address = root / f's{i}'; ready = root / f'ready{i}'
        services.append(dict(name=f's{i}',
                             command=[sys.executable, '-c', SERVICE, str(address), str(ready)],
                             env=dict(os.environ), socket_path=address, ready_path=ready))
    return dict(services=services, actor_command=[sys.executable, '-c', actor],
                actor_env=dict(os.environ), output=root / 'supervision',
                startup_timeout=2., shutdown_timeout=.3)


@pytest.mark.parametrize('count', [1, 2, 3])
def test_declared_service_count_runs_one_actor_over_every_service(tmp_path, count):
    from tools.jev.context_process_supervisor import supervise_services
    report = supervise_services(**declared(tmp_path, count, actor='pass'))
    assert report['passed'] and report['actor_started']
    assert [s['name'] for s in report['services']] == [f's{i}' for i in range(count)]
    assert all(s['exit_code'] == 0 and s['close_ack']['ok'] for s in report['services'])
    assert len(report['child_exit_codes']) == count + 1
    assert json.loads((tmp_path / 'supervision/terminal.json').read_text()) == report


def test_single_service_legacy_fields_only_for_one_service(tmp_path):
    from tools.jev.context_process_supervisor import supervise_services
    one = supervise_services(**declared(tmp_path / 'one', 1, actor='pass'))
    assert one['service_exit_code'] == 0 and one['close_ack']['ok']
    two = supervise_services(**declared(tmp_path / 'two', 2, actor='pass'))
    assert 'service_exit_code' not in two and 'close_ack' not in two


def test_declared_service_startup_failure_stops_before_actor(tmp_path):
    from tools.jev.context_process_supervisor import supervise_services
    args = declared(tmp_path, 2)
    args['services'][1]['command'] = [sys.executable, '-c', 'raise SystemExit(9)']
    with pytest.raises(RuntimeError, match='s1 exited before readiness'):
        supervise_services(**args)
    report = json.loads((tmp_path / 'supervision/terminal.json').read_text())
    assert not report['actor_started'] and report['child_exit_codes'][1] == 9


@pytest.mark.parametrize('change', ['count0', 'count9', 'duplicate-name', 'shared-socket',
                                    'shared-ready', 'stale-ready', 'extra-key'])
def test_declared_service_specification_is_validated(tmp_path, change):
    from tools.jev.context_process_supervisor import supervise_services
    args = declared(tmp_path, 2 if change not in ('count0', 'count9') else 1)
    services = args['services']
    if change == 'count0':
        args['services'] = []
    elif change == 'count9':
        args['services'] = [dict(services[0], name=f's{i}') for i in range(9)]
    elif change == 'duplicate-name':
        services[1]['name'] = services[0]['name']
    elif change == 'shared-socket':
        services[1]['socket_path'] = services[0]['socket_path']
    elif change == 'shared-ready':
        services[1]['ready_path'] = services[0]['ready_path']
    elif change == 'stale-ready':
        Path(services[1]['ready_path']).touch()
    elif change == 'extra-key':
        services[1]['device'] = 7
    with pytest.raises(ValueError):
        supervise_services(**args)
    assert not (tmp_path / 'supervision').exists()
