#!/usr/bin/env python3
"""One frozen MMAR capacity attempt after the existing decode resource queue."""
import argparse
import fcntl
from pathlib import Path
import time

import watch_audio2tool_decode as decode
from multimodal_jev.audio2tool_acceptance import check_container
from multimodal_jev.audio2tool_inference import write_once
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.mmar_inference import admit_capacity, population

native = decode.native
jev = decode.jev
PREPARATION = '336425517e6742f1c4acb757c2ea512238b06be7360985d16a8c6544ee782e48'
DECODE_PLAN = 'a075ca5f0d7fc915a9683479d5370a79399981eee789ac46e4533a7401bf87f9'
DECODE_LAUNCH = 'f512b0997c3202a3e7af787f7a660a8d71e105b0014eba94aae5af0a7a821121'
PREFLIGHT_ID = '3c36124d99bccb50936514bc9001730a6d1dce2067103c844a791602d4a6faff'
PREFLIGHT_INTENT = '7407e2b97f5814550a74e0ae72f6842de8b42c06fd46889e1dbc92232e5c6141'
PREFLIGHT_RESULT = 'c2b174882edad7601c870c27c71e4b0452cc8747cffe5b667b05b6a7b42c304c'


def dependency_ready(base):
    """Preserve queue order; decode execution is not a task-quality acceptance."""
    root = base/'audio2tool-decode-profile-v1'
    if file_hash(root/'queue-plan.json') != DECODE_PLAN or file_hash(root/'evaluation-launch.json') != DECODE_LAUNCH:
        raise ValueError('original decode resource predecessor required')
    launch = native.read(root/'evaluation-launch.json')
    terminal = root/'evaluation-terminal.json'
    if not terminal.exists():
        proc = Path(f"/proc/{launch['pid']}")
        try:
            stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
            argv = [s.decode() for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
        except FileNotFoundError:
            raise RuntimeError('original decode supervisor missing without terminal') from None
        if len(stat) < 20 or stat[19] != launch['process_start'] or stat[0] == 'Z' or argv != launch['argv']:
            raise RuntimeError('decode supervisor replaced/dead; inspect original attempt')
        return False
    if native.read(terminal)['exit_code'] != 0:
        raise RuntimeError('decode predecessor failed; no MMAR dispatch')
    decode.check_pins(root, DECODE_PLAN)
    if not decode.dependency_ready(base):
        raise ValueError('original decode capacity dependencies not complete')
    plan = native.read(root/'intent.json')
    launch = native.read(root/'launch.json')
    item = native.inspect(launch['container_id'])
    check_container(item, plan, launch['container_id'])
    complete = native.read(root/'execution-complete.json')
    result = native.read(root/'run/result.json')
    if (native.read(root/'run/terminal.json')['exit_code'] != 0
            or complete['container_id'] != item['Id'] or complete['result_sha256'] != file_hash(root/'run/result.json')
            or result['status'] != 'profile_complete' or result['model_loaded'] is not True
            or len(result['records']) != 16 or result['weight_updates'] != 0 or result['targets_read'] is not False):
        raise ValueError('successful original decode execution required')
    return True


def check_pins(root, pin):
    if file_hash(root/'queue-plan.json') != pin or file_hash(root/'preparation.json') != PREPARATION:
        raise ValueError('fixed MMAR capacity preparation required')
    queue = native.read(root/'queue-plan.json')
    prep = native.read(root/'preparation.json')
    if (queue['schema'] != 'mmar-capacity-queue-v1' or queue['automatic_retry'] is not False
            or queue['dependency_seconds'] != 350000 or queue['observation_seconds'] != 3900
            or queue['preparation_sha256'] != PREPARATION
            or queue['resource_lock'] != str(root.parent/'npu67-sampling-resource.lock')
            or prep['resource_lock'] != queue['resource_lock']
            or prep['required_predecessor'] != str(root.parent/'audio2tool-decode-profile-v1')
            or prep['physical_devices'] != [6, 7] or prep['wall_seconds'] != 3600
            or prep['automatic_retry'] is not False or prep['weight_updates'] != 0 or prep['targets_mounted']):
        raise ValueError('bounded original MMAR queue required')
    for n, h in queue['source_files'].items():
        if file_hash(root/'queue-source'/n) != h:
            raise ValueError('queue source changed')
    for n, h in prep['source_files'].items():
        if file_hash(root/'source'/n) != h:
            raise ValueError('original inference source changed')
    for name, key in [('inputs.jsonl', 'inputs_sha256'), ('asset-audit.json', 'asset_audit_sha256')]:
        if file_hash(root/name) != prep[key]:
            raise ValueError('prepared MMAR inputs/model identity changed')
    prior = root.parent/'mmar-omni-inference-preflight-v1'
    if (file_hash(prior/'intent.json') != PREFLIGHT_INTENT
            or file_hash(prior/'run/result.json') != PREFLIGHT_RESULT
            or prep['preflight_container_id'] != PREFLIGHT_ID):
        raise ValueError('accepted original runner preflight required')
    before = native.read(prior/'intent.json')
    if prep['source_files'] != before['source_files'] or prep['source_commit'] != before['source_commit']:
        raise ValueError('capacity runner must match original preflight source')
    item = native.successful(PREFLIGHT_ID, prep['image'])
    a = before['argv']; host = item['HostConfig']
    if (item['Config']['Cmd'] != a[a.index(prep['image'])+1:]
            or host['NanoCpus'] != 2*10**9 or host['Memory'] != 20*1024**3
            or host['NetworkMode'] != 'none' or host['Devices'] or host['Privileged']
            or set(host['Binds']) != {a[i+1] for i, v in enumerate(a[:-1]) if v == '-v'}):
        raise ValueError('actual CPU runner preflight differs')
    population(root/'inputs.jsonl', root.parent/'mmar-omni-native-preflight-v1/run')
    return queue, prep


def execute(root, pin):
    if any((root/n).exists() for n in ('intent.json', 'launch.json', 'run', 'evaluation-terminal.json')):
        raise FileExistsError('existing MMAR capacity attempt; no replay')
    queue, prep = check_pins(root, pin)
    deadline = time.monotonic()+queue['dependency_seconds']
    while True:
        try:
            ready = dependency_ready(root.parent)
            atomic_json(root/'queue-progress.json', dict(phase='waiting_decode_execution', ready=ready, heartbeat_unix=time.time()))
            if ready:
                break
        except jev.OBSERVATION_ERRORS as exc:
            atomic_json(root/'queue-progress.json', dict(phase='dependency_observation_retry', error_type=type(exc).__name__, heartbeat_unix=time.time()))
        if time.monotonic() > deadline:
            raise TimeoutError('dependency observation expired; no dispatch')
        time.sleep(20)
    with Path(queue['resource_lock']).open('a') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                locked = False
            else:
                locked = True
            try:
                free = locked and jev.free_devices(root)
            except jev.OBSERVATION_ERRORS:
                free = False
            if free:
                break
            if locked:
                fcntl.flock(lock, fcntl.LOCK_UN)
            atomic_json(root/'queue-progress.json', dict(phase='waiting_npu67', heartbeat_unix=time.time()))
            if time.monotonic() > deadline:
                raise TimeoutError('resource observation expired; no dispatch')
            time.sleep(20)
        queue, prep = check_pins(root, pin)
        if not dependency_ready(root.parent):
            raise ValueError('predecessor execution changed')
        # Reuse the tested create-ID-persist-start transaction; no replay on ambiguity.
        cid, intent_hash = jev.dispatch(root, prep)
        deadline = time.monotonic()+queue['observation_seconds']
        while True:
            try:
                item = native.inspect(cid)
                if item['Id'] != cid or item['Image'] != prep['image']:
                    raise ValueError('capacity container identity changed')
                atomic_json(root/'queue-progress.json', dict(phase='capacity_running', container_id=cid, state=item['State'], heartbeat_unix=time.time()))
                if item['State']['Status'] in ('exited', 'dead'):
                    break
            except jev.OBSERVATION_ERRORS as exc:
                atomic_json(root/'queue-progress.json', dict(phase='observation_retry', container_id=cid, error_type=type(exc).__name__, heartbeat_unix=time.time()))
            if time.monotonic() > deadline:
                raise TimeoutError('observation expired; inspect same ID, never replay')
            time.sleep(20)
        inspection = root/'capacity-inspection.json'
        write_once(inspection, item)
        rows, prior, _ = population(root/'inputs.jsonl', root.parent/'mmar-omni-native-preflight-v1/run')
        accepted = admit_capacity(root, intent_hash, inspection, file_hash(inspection), rows, prior)
        write_once(root/'acceptance.json', accepted)
        atomic_json(root/'queue-progress.json', dict(phase='capacity_accepted', container_id=cid, heartbeat_unix=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan-hash', required=True)
    args = parser.parse_args()
    with (args.root/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.root/'evaluation-terminal.json').exists():
            raise FileExistsError('preserve previous terminal')
        try:
            execute(args.root, args.plan_hash)
        except BaseException as exc:
            write_once(args.root/'evaluation-terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time(), automatic_retry=False))
            raise
        write_once(args.root/'evaluation-terminal.json', dict(exit_code=0, finished_unix=time.time(), automatic_retry=False))
