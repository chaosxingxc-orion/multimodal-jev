#!/usr/bin/env python3
"""Wait for a pinned asset download, then run and score one frozen audio probe."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.six_view_score import score_six_views


def run(root,assets):
    base=root.parent
    intent=json.loads((root/'intent.json').read_text())
    deadline=time.monotonic()+7200
    while not (assets/'terminal.json').exists():
        dispatch=json.loads((assets/'dispatch.json').read_text());pid=dispatch['pid']
        cmd=Path('/proc')/str(pid)/'cmdline'
        if not cmd.exists() or str(assets/'download.py').encode() not in cmd.read_bytes():
            raise RuntimeError('asset downloader missing without terminal; no relaunch')
        atomic_json(root/'observation.json',dict(phase='waiting_assets',pid=pid,heartbeat_unix=time.time()))
        if time.monotonic()>deadline:raise TimeoutError('asset wait expired; no restart')
        time.sleep(20)
    if json.loads((assets/'terminal.json').read_text())['exit_code']!=0:
        raise RuntimeError('asset download failed; preserve original evidence')
    cfg=json.loads((root/'config.json').read_text())
    if file_hash(root/'config.json')!=intent['config_sha256']:raise ValueError('config changed')
    ids=subprocess.check_output(['ssh','npu','docker ps -q'],text=True).split()
    if ids:
        items=json.loads(subprocess.check_output(['ssh','npu','docker inspect '+' '.join(ids)],text=True))
        if any(d['PathOnHost']=='/dev/davinci4' for item in items for d in item['HostConfig'].get('Devices') or []):
            raise RuntimeError('physical card4 occupied; no dispatch')
    result=subprocess.run(['ssh','-o','ConnectTimeout=8','npu',shlex.join(intent['argv'])],capture_output=True,text=True,timeout=90)
    (root/'dispatch.log').write_text(result.stdout+result.stderr);cid=result.stdout.strip()
    if result.returncode or len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid):
        raise RuntimeError('ambiguous dispatch; no retry')
    launch={**intent,'container_id':cid};atomic_json(root/'launch.json',launch)
    deadline=time.monotonic()+4200
    while True:
        try:
            item=json.loads(subprocess.check_output(['ssh','-o','ConnectTimeout=8','npu','docker inspect '+cid],text=True,timeout=20))[0]
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,json.JSONDecodeError) as exc:
            atomic_json(root/'observation.json',dict(phase='observation_error',error_type=type(exc).__name__,container_id=cid))
            if time.monotonic()>deadline:raise TimeoutError('observation deadline; no restart')
            time.sleep(20);continue
        if item['Id']!=cid or item['Image']!=intent['image']:raise ValueError('container identity changed')
        state=item['State'];atomic_json(root/'observation.json',dict(phase='inference',container_id=cid,state=state,heartbeat_unix=time.time()))
        if state['Status'] in ('exited','dead'):
            atomic_json(root/'docker-terminal.json',dict(container_id=cid,state=state))
            logs=subprocess.run(['ssh','npu','docker logs '+cid],capture_output=True,text=True,timeout=30)
            (root/'container.log').write_text(logs.stdout+logs.stderr)
            if state['ExitCode']!=0:raise RuntimeError('probe container failed')
            break
        if time.monotonic()>deadline:raise TimeoutError('probe observation expired; no restart')
        time.sleep(20)
    out=root/'run';terminal=json.loads((out/'terminal.json').read_text())
    result=json.loads((out/'result.json').read_text());identity=json.loads((out/'identity.json').read_text())
    if (terminal['exit_code']!=0 or result['status']!='passed' or result['requests']!=6
            or result['weight_updates']!=0 or result['model_files_unchanged'] is not True
            or identity['config']!=cfg or file_hash(cfg['requests'])!=identity['requests_sha256']
            or file_hash(cfg['instruction'])!=identity['instruction_sha256']
            or file_hash(out/'predictions.jsonl')!=result['predictions_sha256']):
        raise ValueError('complete unchanged frozen probe required')
    rows=[json.loads(line) for line in Path(cfg['requests']).open()]
    rows=[r for r in rows if r['source_group'] in cfg['source_groups']]
    targets_path=Path(cfg['requests']).with_name('train.targets.jsonl')
    targets=[json.loads(line) for line in targets_path.open()]
    pred=[json.loads(line) for line in (out/'predictions.jsonl').open()]
    metrics=score_six_views(rows,targets,pred)
    atomic_json(root/'score.json',dict(metrics=metrics,targets_sha256=file_hash(targets_path),
        result_sha256=file_hash(out/'result.json'),scope='single TRAIN group runtime/interface probe; not benchmark efficacy'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    a=p.parse_args()
    try:run(a.root,a.assets)
    except BaseException as exc:
        atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),automatic_retry=False));raise
    atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time()))
