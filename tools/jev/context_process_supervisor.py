"""Candidate same-container lifecycle; no model or algorithm changes.

Commands must be fresh process trees owned by this invocation. Actor runtime is
unbounded; only startup and shutdown have deadlines. Never reconnect to an
existing training job or retry an actor after uncertain execution.
"""
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import time


def stop_owned(process, grace):
    """Stop this invocation's process group, including worker descendants."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # A process leader can exit before its workers; clean the owned group too.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


MAX_FRAME = 1 << 20


def close_unconnected_service(address, timeout):
    """Use the existing sequence-zero close contract after actor tree exits.

    Connected services should already observe EOF. A rejected/uncertain close
    is recorded and handled by bounded shutdown, never retried.

    A replica service greets every accepted connection with its protocol
    frame before serving it, so the first frame here may be that greeting and
    not the close acknowledgement. Skip exactly one such frame; anything after
    it must be the acknowledgement, and a second greeting is an error.
    """
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(address))
        raw = json.dumps(dict(operation='close', sequence=0)).encode()
        connection.sendall(struct.pack('!I', len(raw)) + raw)

        def exact(n):
            value = bytearray()
            while len(value) < n:
                part = connection.recv(n - len(value))
                if not part:
                    raise ConnectionError('service closed before acknowledgement')
                value.extend(part)
            return value

        def frame():
            size = struct.unpack('!I', exact(4))[0]
            if size > MAX_FRAME:
                raise ValueError('oversized service frame')
            return json.loads(exact(size))

        reply = frame()
        if isinstance(reply, dict) and isinstance(reply.get('protocol'), str) and reply['protocol']:
            reply = frame()
            if isinstance(reply, dict) and 'protocol' in reply:
                raise ValueError('service did not acknowledge close')
        if reply != dict(sequence=0, ok=True, result=None):
            raise ValueError('service did not acknowledge close')
        return reply


def supervise(*, service_command, actor_command, service_env, actor_env, output,
              ready_path, socket_path, startup_timeout=900., shutdown_timeout=30.):
    if startup_timeout <= 0 or shutdown_timeout <= 0:
        raise ValueError('positive lifecycle deadlines required')
    output = Path(output)
    if Path(ready_path).exists() or Path(socket_path).exists():
        raise ValueError('fresh service paths required; existing handles not adopted')
    output.mkdir(exist_ok=False)
    children = []
    report = dict(passed=False, actor_started=False, forced_cleanup=False)
    with (output / 'service.log').open('w') as service_log, (output / 'actor.log').open('w') as actor_log:
        try:
            service = subprocess.Popen(service_command, env=service_env, stdout=service_log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            children.append(service)
            deadline = time.monotonic() + startup_timeout
            while not Path(ready_path).exists():
                if service.poll() is not None:
                    raise RuntimeError('service exited before readiness')
                if time.monotonic() >= deadline:
                    raise TimeoutError('service startup deadline')
                time.sleep(.05)
            if service.poll() is not None or not Path(socket_path).is_socket():
                raise RuntimeError('ready service requires live process and socket')
            actor = subprocess.Popen(actor_command, env=actor_env, stdout=actor_log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            children.append(actor)
            report['actor_started'] = True
            while actor.poll() is None:
                if service.poll() is not None:
                    # Successful actor shutdown can follow its acknowledged close.
                    try:
                        actor.wait(timeout=shutdown_timeout)
                    except subprocess.TimeoutExpired:
                        raise RuntimeError('service exited while actor remained live')
                    break
                time.sleep(.05)
            report['actor_exit_code'] = actor.returncode
            stop_owned(actor, shutdown_timeout)
            if service.poll() is None:
                try:
                    report['close_ack'] = close_unconnected_service(socket_path, shutdown_timeout)
                except (OSError, ValueError, ConnectionError) as exc:
                    report['close_error'] = repr(exc)
            try:
                service.wait(timeout=shutdown_timeout)
            except subprocess.TimeoutExpired:
                report['forced_cleanup'] = True
                stop_owned(service, shutdown_timeout)
            report['service_exit_code'] = service.returncode
            report['passed'] = (actor.returncode == 0 and service.returncode == 0
                                and not report['forced_cleanup'])
        except BaseException as exc:
            report['error'] = repr(exc)
            raise
        finally:
            for child in reversed(children):
                if child.poll() is None:
                    report['forced_cleanup'] = True
                stop_owned(child, shutdown_timeout)
            report['child_exit_codes'] = [child.returncode for child in children]
            report['finished_unix'] = time.time()
            # Preserve fsync: caller must place this candidate on local storage.
            with (output / 'terminal.json').open('w') as terminal:
                json.dump(report, terminal, indent=2)
                terminal.write('\n')
                terminal.flush()
                os.fsync(terminal.fileno())
    return report


def _validate_service(spec):
    required = {'name', 'command', 'env', 'ready_path', 'socket_path'}
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError('explicit service name/command/env/ready/socket required')
    name = spec['name']
    if (not isinstance(name, str) or not name or not name.isidentifier()
            or not isinstance(spec['command'], list) or not spec['command']
            or not isinstance(spec['env'], dict)):
        raise ValueError('well-formed service command and environment required')
    return name


def supervise_services(*, services, actor_command, actor_env, output,
                       startup_timeout=900., shutdown_timeout=30.):
    """Start any number of declared inference services, then one actor tree.

    The service list is data supplied by the package; device visibility and
    capacities stay in each service environment/configuration.
    """
    if not isinstance(services, list) or not 1 <= len(services) <= 8:
        raise ValueError('one to eight independent inference services required')
    names = [_validate_service(spec) for spec in services]
    if len(set(names)) != len(names):
        raise ValueError('distinct service names required')
    if startup_timeout <= 0 or shutdown_timeout <= 0:
        raise ValueError('positive lifecycle deadlines required')
    if (not isinstance(actor_command, list) or not actor_command
            or not isinstance(actor_env, dict)):
        raise ValueError('explicit actor command and environment required')
    output = Path(output)
    handles = [(spec['name'], Path(spec['ready_path']), Path(spec['socket_path']))
               for spec in services]
    if len({h[1] for h in handles}) != len(handles) or len({h[2] for h in handles}) != len(handles):
        raise ValueError('distinct service ready and socket paths required')
    for _, ready, sock in handles:
        if ready.exists() or sock.exists():
            raise ValueError('fresh service paths required; existing handles not adopted')
    output.mkdir(exist_ok=False)
    children = []; processes = []; logs = []
    report = dict(passed=False, actor_started=False, forced_cleanup=False,
                  services=[dict(name=n) for n in names])
    try:
        for spec, entry in zip(services, report['services'], strict=True):
            log = (output / ('service-' + spec['name'] + '.log')).open('w'); logs.append(log)
            child = subprocess.Popen(spec['command'], env=spec['env'], stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child); processes.append((entry, child, Path(spec['ready_path']),
                                                      Path(spec['socket_path'])))
        deadline = time.monotonic() + startup_timeout
        for entry, child, ready, sock in processes:
            while not ready.exists():
                if child.poll() is not None:
                    raise RuntimeError('service ' + entry['name'] + ' exited before readiness')
                if time.monotonic() >= deadline:
                    raise TimeoutError('service ' + entry['name'] + ' startup deadline')
                time.sleep(.05)
            if child.poll() is not None or not sock.is_socket():
                raise RuntimeError('ready service requires live process and socket')
        actor_log = (output / 'actor.log').open('w'); logs.append(actor_log)
        actor = subprocess.Popen(actor_command, env=actor_env, stdout=actor_log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        children.append(actor)
        report['actor_started'] = True
        while actor.poll() is None:
            if any(child.poll() is not None for _, child, _, _ in processes):
                try:
                    actor.wait(timeout=shutdown_timeout)
                except subprocess.TimeoutExpired:
                    raise RuntimeError('service exited while actor remained live')
                break
            time.sleep(.05)
        report['actor_exit_code'] = actor.returncode
        stop_owned(actor, shutdown_timeout)
        for entry, child, _, sock in processes:
            if child.poll() is None:
                try:
                    entry['close_ack'] = close_unconnected_service(sock, shutdown_timeout)
                except (OSError, ValueError, ConnectionError) as exc:
                    entry['close_error'] = repr(exc)
            try:
                child.wait(timeout=shutdown_timeout)
            except subprocess.TimeoutExpired:
                report['forced_cleanup'] = True
                stop_owned(child, shutdown_timeout)
            entry['exit_code'] = child.returncode
        report['passed'] = (actor.returncode == 0 and not report['forced_cleanup']
                            and all(e.get('exit_code') == 0 for e in report['services']))
        if len(report['services']) == 1:
            report['service_exit_code'] = report['services'][0]['exit_code']
            for key in ('close_ack', 'close_error'):
                if key in report['services'][0]:
                    report[key] = report['services'][0][key]
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        for log in logs:
            log.close()
        for child in reversed(children):
            if child.poll() is None:
                report['forced_cleanup'] = True
            stop_owned(child, shutdown_timeout)
        report['child_exit_codes'] = [child.returncode for child in children]
        report['finished_unix'] = time.time()
        # Preserve fsync: caller must place this candidate on local storage.
        with (output / 'terminal.json').open('w') as terminal:
            json.dump(report, terminal, indent=2)
            terminal.write('\n')
            terminal.flush()
            os.fsync(terminal.fileno())
    return report
