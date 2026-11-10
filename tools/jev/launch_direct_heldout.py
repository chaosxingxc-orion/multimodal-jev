#!/usr/bin/env python3
"""Freeze a completed TRAIN selection and dispatch full paired TEST runs once."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.direct_test_lock import validate_population
from select_direct_prompt import select


def launch(selection_path, root):
    selection = json.loads(selection_path.read_text())
    teacher = selection_path.parent
    if json.loads((teacher/'evaluation-terminal.json').read_text())['exit_code'] != 0:
        raise ValueError('selection supervisor must complete')
    reports = {r['id']: r for r in selection['comparisons']}
    parent_run = Path(reports['parent']['run']); parent = parent_run.parent
    identity = json.loads((parent_run/'identity.json').read_text())
    partition = parent.parent/'frozen-fsc-asr-text-learning64x4-v1/learning-selection-split.json'
    targets = Path(identity['config']['requests']).with_name('train.targets.jsonl')
    recomputed = select(parent_run, {k: Path(v['run']) for k,v in reports.items() if k != 'parent'},
                        teacher/'review.json', targets, partition)
    def structural(value):
        if isinstance(value,dict):
            return {k:structural(v) for k,v in value.items() if k!='mcnemar_exact_two_sided_p'}
        if isinstance(value,list):return [structural(v) for v in value]
        return value
    if structural(recomputed) != structural(selection) or selection['test_feedback_used'] is not False:
        raise ValueError('selection audit changed')
    for report in reports.values():
        prior = json.loads((Path(report['run']).parent/'launch.json').read_text())
        item = json.loads(subprocess.check_output(['ssh','npu','docker inspect '+prior['container_id']],text=True))[0]
        if (item['Id'] != prior['container_id'] or item['Image'] != prior['image']
                or item['State']['Status'] != 'exited' or item['State']['ExitCode'] != 0):
            raise ValueError('selection run not completed on expected image')
    cfg = identity['config']
    request_path = Path(cfg['requests']).with_name('test.requests.jsonl')
    rows = [json.loads(line) for line in request_path.open()]
    groups = sorted({r['source_group'] for r in rows})
    if len(groups) != 3793 or len(rows) != 22758:
        raise ValueError('full FSC TEST release required')
    # Inspect allocation, never claim another experiment's device.
    ids = subprocess.check_output(['ssh','npu','docker ps -q'],text=True).split()
    if ids:
        active = json.loads(subprocess.check_output(['ssh','npu','docker inspect '+' '.join(ids)],text=True))
        for item in active:
            if any(d['PathOnHost'] in ('/dev/davinci5','/dev/davinci6')
                   for d in item['HostConfig'].get('Devices') or []):
                raise ValueError('requested device already mapped by running container')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('commit reviewed source before dispatch')
    root.mkdir(parents=True,exist_ok=False)
    source=root/'source';source.mkdir()
    archive=subprocess.Popen(['git','archive',commit],stdout=subprocess.PIPE)
    subprocess.run(['tar','-x','-C',str(source)],stdin=archive.stdout,check=True)
    archive.stdout.close()
    if archive.wait():raise RuntimeError('source export failed')
    (root/'selection.json').write_bytes(selection_path.read_bytes())
    ignored={'split','requests','source_groups','instruction','purpose','wall_seconds','heldout_lock'}
    lock=dict(schema='jev-direct-heldout-lock-v1',test_feedback_used=False,source_selection_split='train',
              selection_sha256=file_hash(selection_path),requests_sha256=file_hash(request_path),
              targets_sha256=file_hash(request_path.with_name('test.targets.jsonl')),
              requests=len(rows),source_groups=len(groups),models=identity['models'],
              parent_prompt_sha256=identity['instruction_sha256'],
              selected_prompt_sha256=selection['selected_instruction_sha256'],
              actor_config={k:v for k,v in cfg.items() if k not in ignored})
    atomic_json(root/'lock.json',lock)
    prior=json.loads((parent/'launch.json').read_text())
    names=['parent']+(['selected'] if selection['selected_id']!='parent' else [])
    launches=[]
    for name,device in zip(names,[5,6],strict=False):
        run=root/name;run.mkdir()
        prompt=Path(reports['parent']['instruction'] if name=='parent' else selection['selected_instruction'])
        (run/'instruction.txt').write_bytes(prompt.read_bytes())
        config={**cfg,'split':'test','requests':str(request_path),'source_groups':groups,
                'instruction':str(run/'instruction.txt'),'purpose':'direct-audio-locked-full-test',
                'wall_seconds':10800,'heldout_lock':str(root/'lock.json')}
        validate_population(config,rows);atomic_json(run/'config.json',config)
        argv=[a.replace(str(parent),str(root)).replace(parent.name,root.name+'-'+name)
              .replace('/dev/davinci4','/dev/davinci'+str(device)) for a in prior['argv']]
        argv=[str(run/'config.json') if a==str(root/'config.json') else
              str(run/'run') if a==str(root/'run') else '11100s' if a=='3900s' else a for a in argv]
        plan=dict(argv=argv,image=prior['image'],source_commit=commit,root=str(run),physical_device=device,
                  config_sha256=file_hash(run/'config.json'),lock_sha256=file_hash(root/'lock.json'))
        atomic_json(run/'launch-intent.json',plan)
        result=subprocess.run(['ssh','-o','ConnectTimeout=8','npu',shlex.join(argv)],capture_output=True,text=True,timeout=90)
        (run/'dispatch.log').write_text(result.stdout+result.stderr)
        cid=result.stdout.strip()
        if result.returncode or len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid):
            raise RuntimeError('ambiguous dispatch; inspect exact intent before any recovery')
        plan['container_id']=cid;atomic_json(run/'launch.json',plan)
        (run/'supervisor.py').write_bytes((parent/'supervisor.py').read_bytes())
        with (run/'supervisor.log').open('ab') as log:
            proc=subprocess.Popen([sys.executable,str(run/'supervisor.py')],stdout=log,stderr=log,start_new_session=True)
        plan['supervisor_pid']=proc.pid;atomic_json(run/'launch.json',plan);launches.append(plan)
    atomic_json(root/'launches.json',launches)
    env={**os.environ,'PYTHONPATH':str(source/'src')}
    with (root/'evaluation.log').open('ab') as log:
        proc=subprocess.Popen([sys.executable,str(source/'tools/jev/watch_direct_heldout.py'),
            '--root',str(root)],env=env,stdout=log,stderr=log,start_new_session=True)
    atomic_json(root/'evaluation-launch.json',dict(pid=proc.pid,source_commit=commit,automatic_retry=False))
    return launches


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();print(json.dumps(launch(a.selection,a.root),indent=2))
