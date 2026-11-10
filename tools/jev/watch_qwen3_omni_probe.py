#!/usr/bin/env python3
"""Wait for registered ASR jobs; dispatch and score one isolated Omni probe."""
import argparse,json,shlex,subprocess,time,os,sys
from pathlib import Path
from multimodal_jev.baseline import atomic_json


def inspect(cid):
    return json.loads(subprocess.check_output(['ssh','-o','ConnectTimeout=8','npu','docker inspect '+cid],text=True,timeout=20))[0]


def wait(root,cid,image,deadline):
    while True:
        try:item=inspect(cid)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,json.JSONDecodeError) as exc:
            atomic_json(root/'observation.json',dict(container_id=cid,error_type=type(exc).__name__,phase='observation_retry'))
            if time.monotonic()>deadline:raise TimeoutError('observation expired; no restart')
            time.sleep(20);continue
        if item['Id']!=cid or item['Image']!=image:raise ValueError('container identity mismatch')
        atomic_json(root/'observation.json',dict(container_id=cid,state=item['State'],heartbeat_unix=time.time()))
        if item['State']['Status'] in ('exited','dead'):return item['State']
        if time.monotonic()>deadline:raise TimeoutError('wait expired; no restart')
        time.sleep(20)


def run(root):
    plan=json.loads((root/'intent.json').read_text());deadline=time.monotonic()+7200
    observation_seconds=plan.get('observation_seconds',3900)
    cfg=json.loads((root/'config.json').read_text())
    maximum = 86700 if cfg.get('evaluation_phase') in ('test','slurp-native-test','slurp-jev-transfer') else 7500
    if type(observation_seconds) is not int or not 3900 <= observation_seconds <= maximum:
        raise ValueError('bounded observation deadline required')
    for dep in plan['wait_for']:
        state=wait(root,dep['container_id'],dep['image'],deadline)
        if state['ExitCode']!=0:raise RuntimeError('dependency failed; inspect before dispatch')
    ids=subprocess.check_output(['ssh','npu','docker ps -q'],text=True).split()
    devices=plan['physical_devices']
    if devices not in ([4,5],[6,7]):raise ValueError('authorized NPU pair required')
    declared={plan['argv'][i+1].split(':')[0] for i,x in enumerate(plan['argv'][:-1]) if x=='--device'}
    expected={'/dev/davinci'+str(d) for d in devices}
    if {d for d in declared if d.removeprefix('/dev/davinci').isdigit()}!=expected:
        raise ValueError('declared NPU devices differ from plan')
    for cid in ids:
        item=inspect(cid)
        if any(d['PathOnHost'] in expected for d in item['HostConfig'].get('Devices') or []):
            raise RuntimeError('target device mapped by another running container')
    result=subprocess.run(['ssh','npu',shlex.join(plan['argv'])],capture_output=True,text=True,timeout=90)
    (root/'dispatch.log').write_text(result.stdout+result.stderr);cid=result.stdout.strip()
    if result.returncode or len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid):
        raise RuntimeError('dispatch ambiguous; no automatic retry')
    atomic_json(root/'launch.json',{**plan,'container_id':cid})
    state=wait(root,cid,plan['image'],time.monotonic()+observation_seconds)
    atomic_json(root/'docker-terminal.json',dict(container_id=cid,state=state))
    logs=subprocess.run(['ssh','npu','docker logs '+cid],capture_output=True,text=True,timeout=30)
    (root/'container.log').write_text(logs.stdout+logs.stderr)
    if state['ExitCode']!=0:raise RuntimeError('Omni probe failed; inspect retained logs')
    cfg=json.loads((root/'config.json').read_text())
    if json.loads((root/'run/terminal.json').read_text())['exit_code']!=0:
        raise RuntimeError('application failed')
    if cfg.get('audio_condition','actual')=='zeros':
        return  # Silence has no native gold; paired diagnostic runs separately.
    if cfg.get('evaluation_phase') in ('slurp-jev-probe','slurp-jev-transfer'):
        subprocess.run([plan['scorer_python'],str(root/'source/tools/jev/score_slurp_jev_full.py'),
            '--run',str(root/'run'),'--release',plan['jev_release'],
            '--repository',plan['scorer_repository'],'--output',str(root/'score')],
            env={**os.environ,'PYTHONPATH':str(root/'source/src')},check=True)
        return
    if cfg.get('task_format')=='slurp-native':
        subprocess.run([plan['scorer_python'],str(root/'source/tools/jev/score_slurp_native.py'),
            '--run',str(root/'run'),'--targets',str(Path(cfg['requests']).with_name(cfg['split']+'.targets.jsonl')),
            '--repository',plan['scorer_repository'],'--output',str(root/'score')],
            env={**os.environ,'PYTHONPATH':str(root/'source/src')},check=True)
        return
    subprocess.run([sys.executable,str(root/'source/tools/jev/score_frozen_requests.py'),'--run',str(root/'run'),
        '--targets',str(Path(cfg['requests']).with_name(cfg['split']+'.targets.jsonl')),'--six-views','--output',str(root/'score.json')],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:run(a.root)
    except BaseException as exc:
        atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),automatic_retry=False));raise
    atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time()))
