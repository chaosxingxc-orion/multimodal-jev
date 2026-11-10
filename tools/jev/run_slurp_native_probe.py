#!/usr/bin/env python3
"""One detached native probe after successful full release construction."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from multimodal_jev.baseline import atomic_json, file_hash
from run_selected_frozen_test import observe, wait_container
from multimodal_jev.end_to_end import require_audio_task_route


def execute(root):
    require_audio_task_route(json.loads((root/'config-template.json').read_text()))
    plan=json.loads((root/'plan.json').read_text())
    state=Path(plan['release_state']); deadline=plan['deadline_unix']
    while not (state/'terminal.json').exists():
        if time.time()>deadline:raise TimeoutError('release wait deadline; no retry')
        atomic_json(root/'progress.json',{'phase':'waiting_complete_native_release',
                                         'heartbeat_unix':time.time(),'builder_pid':plan['release_builder_pid']})
        time.sleep(15)
    if json.loads((state/'terminal.json').read_text())['exit_code']!=0:
        raise ValueError('native release builder failed')
    release_result=json.loads((state/'result.json').read_text()); release=Path(release_result['path'])
    manifest=json.loads((release/'MANIFEST.json').read_text())
    if manifest['release_id']!=release_result['release_id']:raise ValueError('native release identity changed')
    if file_hash(root/'config-template.json')!=plan['config_template_sha256']:
        raise ValueError('probe config changed')
    if file_hash(root/'groups.json')!=plan['groups_sha256']:raise ValueError('probe groups changed')
    cfg=json.loads((root/'config-template.json').read_text())
    cfg['requests']=str(release/'train.requests.jsonl')
    cfg['taxonomy']=str(release/'taxonomy.json')
    cfg['taxonomy_sha256']=manifest['identity']['taxonomy_sha256']
    if file_hash(cfg['requests'])!=manifest['splits']['train']['requests_sha256']:
        raise ValueError('release requests changed')
    if file_hash(cfg['taxonomy'])!=cfg['taxonomy_sha256']:raise ValueError('release taxonomy changed')
    rows=[json.loads(line) for line in Path(cfg['requests']).open()]
    selected=[r for r in rows if r['source_group'] in set(cfg['source_groups'])]
    if len(selected)!=plan['expected_recordings'] or {r['source_group'] for r in selected}!=set(cfg['source_groups']):
        raise ValueError('probe recording coverage changed')
    atomic_json(root/'config.json',cfg)
    prior=observe(plan['prior_device_container_id'],plan['image'])
    if prior['State']['Status']!='exited' or prior['State']['ExitCode']!=0:
        raise ValueError('prior device run not successfully complete')
    if (root/'dispatch-intent.json').exists():raise ValueError('previous dispatch must be reconciled; no retry')
    launch={k:plan[k] for k in ['argv','image','source_commit','physical_device']}
    launch.update(root=str(root),config_sha256=file_hash(root/'config.json'),
                  release_id=manifest['release_id'],groups_sha256=plan['groups_sha256'])
    atomic_json(root/'dispatch-intent.json',launch)
    result=subprocess.run(['ssh','-o','ConnectTimeout=8','npu',shlex.join(plan['argv'])],
                          capture_output=True,text=True,timeout=90)
    (root/'dispatch.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('dispatch failed; no retry')
    cid=result.stdout.strip()
    if len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid):raise ValueError('ambiguous container ID')
    launch['container_id']=cid;atomic_json(root/'launch.json',launch)
    item=wait_container(cid,plan['image'],root,'native_slurp_probe',deadline)
    atomic_json(root/'docker-terminal.json',{'container_id':cid,'state':item['State']})
    with (root/'score.log').open('x') as log:
        subprocess.run([plan['scorer_python'],str(root/'source/tools/jev/score_slurp_native.py'),
                        '--run',str(root/'run'),'--targets',str(release/'train.targets.jsonl'),
                        '--repository',plan['scorer_repository'],'--output',str(root/'score')],
                       env={**os.environ,'PYTHONPATH':str(root/'source/src')},
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    atomic_json(root/'terminal.json',{'exit_code':0,'finished_unix':time.time(),
                                     'scope':'native TRAIN probe complete; not official test performance'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    try:execute(args.root)
    except BaseException as exc:
        atomic_json(args.root/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time(),
                                             'automatic_retry':False})
        raise
