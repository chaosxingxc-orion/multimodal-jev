#!/usr/bin/env python3
"""Prepare and run the full frozen TEST roster, sequentially on NPU4/5."""
import argparse
import fcntl
import json
import shutil
import subprocess
import time
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.container_inputs import require_visible_inputs
from multimodal_jev.omni_test_contract import select_test
from watch_qwen3_omni_probe import run as dispatch_and_score
from compare_omni_test import compare


def test_argv(prior, parent, child, selection_root, test_root, key):
    argv = [x.replace(str(parent),str(child))
            .replace(selection_root.name+'-parent',test_root.name+'-'+key)
            .replace('infer_qwen3_omni_selection.py','infer_qwen3_omni_test.py') for x in prior]
    index = argv.index('timeout')
    if argv[index+3] != '7200s':
        raise ValueError('registered selection timeout required')
    argv[index+3] = '86400s'
    # The immutable selection result and protocol are nested TEST inputs.
    argv[2:2] = ['-v',str(selection_root)+':'+str(selection_root)+':ro']
    return argv


def prepare(root, protocol_path):
    protocol = json.loads(protocol_path.read_text())
    selection_root = Path(protocol['selection_result']).parent
    if json.loads((selection_root/'evaluation-terminal.json').read_text())['exit_code'] != 0:
        raise ValueError('completed selection required')
    queue = json.loads((selection_root/'queue-intent.json').read_text())
    commit = queue['source_commit']
    prior = json.loads((selection_root/'parent/launch.json').read_text())
    if prior['image'] != protocol['image'] or prior['physical_devices'] != [4,5]:
        raise ValueError('same image and authorized NPU4/5 required')
    instructions = {key:selection_root/key/'instruction.txt' for key in protocol['prompts']}
    for key,path in instructions.items():
        if file_hash(path) != protocol['prompts'][key]:
            raise ValueError('frozen TEST prompt changed')
    root.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(protocol_path,root/'test-protocol.json')
    source=root/'source';source.mkdir()
    archive=subprocess.Popen(['git','archive',commit],stdout=subprocess.PIPE)
    subprocess.run(['tar','-x','-C',str(source)],stdin=archive.stdout,check=True)
    archive.stdout.close()
    if archive.wait(): raise RuntimeError('source snapshot export failed')
    if not (source/'tools/jev/infer_qwen3_omni_test.py').exists():
        raise ValueError('TEST runner must be frozen before selection inference')
    rows=[json.loads(line) for line in Path(protocol['actor_config']['requests']).open()]
    parent_cfg=json.loads((selection_root/'parent/config.json').read_text())
    order=['parent',*sorted(set(instructions)-{'parent'})]
    for key in order:
        child=root/key;child.mkdir()
        shutil.copytree(source,child/'source')
        shutil.copyfile(instructions[key],child/'instruction.txt')
        shutil.copyfile(parent_cfg['asset_audit'],child/'asset-audit.json')
        shutil.copyfile(protocol_path,child/'test-protocol.json')
        cfg=dict(protocol['actor_config'],instruction=str(child/'instruction.txt'),
            asset_audit=str(child/'asset-audit.json'),test_protocol=str(child/'test-protocol.json'),
            test_protocol_sha256=file_hash(protocol_path))
        select_test(cfg,rows)
        atomic_json(child/'config.json',cfg)
        argv=test_argv(prior['argv'],selection_root/'parent',child,selection_root,root,key)
        require_visible_inputs(argv,cfg)
        atomic_json(child/'intent.json',dict(argv=argv,image=prior['image'],source_commit=commit,
            physical_devices=[4,5],root=str(child),wait_for=[],config_sha256=file_hash(child/'config.json'),
            automatic_retry=False,observation_seconds=86700,test_protocol_sha256=file_hash(protocol_path),prompt_id=key))
    receipt=dict(source_commit=commit,order=order,protocol_sha256=file_hash(protocol_path),
        dispatched=False,automatic_retry=False,physical_devices=[4,5],requests_per_prompt=22758,
        outer_seconds_per_prompt=86400)
    atomic_json(root/'queue-intent.json',receipt)
    return receipt


def run(root):
    queue=json.loads((root/'queue-intent.json').read_text())
    if file_hash(root/'test-protocol.json') != queue['protocol_sha256']:
        raise ValueError('TEST protocol changed')
    for key in queue['order']:
        if (root/key/'launch.json').exists() or (root/key/'dispatch.log').exists():
            raise FileExistsError('prior TEST dispatch; no automatic replay')
    for key in queue['order']:
        child=root/key
        try:
            intent=json.loads((child/'intent.json').read_text())
            if file_hash(child/'config.json') != intent['config_sha256']:
                raise ValueError('prepared TEST config changed')
            cfg=json.loads((child/'config.json').read_text())
            select_test(cfg,[json.loads(line) for line in Path(cfg['requests']).open()])
            require_visible_inputs(intent['argv'],cfg)
            dispatch_and_score(child)
        except BaseException as exc:
            atomic_json(child/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()))
            raise
        atomic_json(child/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time()))
    atomic_json(root/'test-comparison.json',compare(root))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--protocol',type=Path)
    a=p.parse_args()
    if a.prepare:
        if a.protocol is None: p.error('--protocol required')
        print(json.dumps(prepare(a.root,a.protocol),indent=2))
    else:
        with (a.root/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (a.root/'evaluation-terminal.json').exists():
                raise FileExistsError('preserve prior TEST queue attempt')
            try: run(a.root)
            except BaseException as exc:
                atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),automatic_retry=False))
                raise
            atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time(),comparative_acceptance='passed'))
