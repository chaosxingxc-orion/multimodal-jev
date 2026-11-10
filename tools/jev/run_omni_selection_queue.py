#!/usr/bin/env python3
"""Prepare or run an immutable sequential parent/candidate Omni selection queue."""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.container_inputs import require_visible_inputs
from multimodal_jev.omni_selection_contract import select_registered
from select_omni_prompt import select
from watch_qwen3_omni_probe import run as dispatch_and_score


def selection_argv(prior_argv, parent, child, controls, root, key):
    argv = [x.replace(str(parent),str(child)).replace(controls.name+'-actual',root.name+'-'+key)
            .replace('infer_qwen3_omni_probe.py','infer_qwen3_omni_selection.py')
            for x in prior_argv]
    index = argv.index('timeout')
    if argv[index+3] != '3600s':
        raise ValueError('unexpected parent timeout command')
    argv[index+3] = '7200s'
    return argv


def prepare(root, controls, protocol_path, instructions, source_commit):
    protocol = json.loads(protocol_path.read_text())
    if set(instructions) != set(protocol['prompts']):
        raise ValueError('exact complete registered prompt roster required')
    for key, path in instructions.items():
        if file_hash(path) != protocol['prompts'][key]:
            raise ValueError('registered prompt changed')
    parent = controls/'actual'
    prior = json.loads((parent/'launch.json').read_text())
    for name in ('actual','zeros'):
        result = controls/name/'run/result.json'
        if file_hash(result) != protocol['controls_'+name+'_result_sha256']:
            raise ValueError('control result binding changed')
    if json.loads((controls/'evaluation-terminal.json').read_text())['exit_code'] != 0:
        raise ValueError('completed controls required')
    commit = subprocess.check_output(['git','rev-parse',source_commit+'^{commit}'], text=True).strip()
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(protocol_path, root/'selection-protocol.json')
    source = root/'source'; source.mkdir()
    archive = subprocess.Popen(['git','archive',commit], stdout=subprocess.PIPE)
    subprocess.run(['tar','-x','-C',str(source)], stdin=archive.stdout, check=True)
    archive.stdout.close()
    if archive.wait():
        raise RuntimeError('source snapshot export failed')
    rows = [json.loads(line) for line in Path(protocol['actor_config']['requests']).open()]
    base_cfg = json.loads((parent/'config.json').read_text())
    order = ['parent', *sorted(set(instructions)-{'parent'})]
    for key in order:
        child = root/key; child.mkdir()
        shutil.copytree(source, child/'source')
        shutil.copyfile(instructions[key], child/'instruction.txt')
        shutil.copyfile(base_cfg['asset_audit'], child/'asset-audit.json')
        shutil.copyfile(base_cfg['partition'], child/'partition.json')
        shutil.copyfile(protocol_path, child/'selection-protocol.json')
        cfg = dict(protocol['actor_config'], instruction=str(child/'instruction.txt'),
            partition=str(child/'partition.json'), asset_audit=str(child/'asset-audit.json'),
            selection_protocol=str(child/'selection-protocol.json'),
            selection_protocol_sha256=file_hash(protocol_path))
        select_registered(cfg, rows)
        atomic_json(child/'config.json', cfg)
        argv = selection_argv(prior['argv'],parent,child,controls,root,key)
        require_visible_inputs(argv,cfg)
        intent = dict(argv=argv, image=prior['image'], source_commit=commit,
            physical_devices=prior['physical_devices'], root=str(child), wait_for=[],
            config_sha256=file_hash(child/'config.json'), automatic_retry=False,
            observation_seconds=7500, selection_protocol_sha256=file_hash(protocol_path), prompt_id=key)
        atomic_json(child/'intent.json',intent)
    receipt = dict(source_commit=commit, order=order, protocol_sha256=file_hash(protocol_path),
        dispatched=False, automatic_retry=False, physical_devices=prior['physical_devices'],
        requests_per_prompt=1536, outer_seconds_per_prompt=7200)
    atomic_json(root/'queue-intent.json',receipt)
    return receipt


def run(root):
    queue = json.loads((root/'queue-intent.json').read_text())
    protocol = root/'selection-protocol.json'
    if file_hash(protocol) != queue['protocol_sha256']:
        raise ValueError('queue protocol changed')
    # This queue is a one-shot dispatch; a failed/ambiguous attempt needs audit.
    for key in queue['order']:
        if (root/key/'launch.json').exists() or (root/key/'dispatch.log').exists():
            raise FileExistsError('prior dispatch evidence exists; no automatic replay')
    for key in queue['order']:
        child = root/key
        try:
            intent = json.loads((child/'intent.json').read_text())
            if file_hash(child/'config.json') != intent['config_sha256']:
                raise ValueError('prepared config changed')
            cfg = json.loads((child/'config.json').read_text())
            select_registered(cfg,[json.loads(line) for line in Path(cfg['requests']).open()])
            require_visible_inputs(intent['argv'],cfg)
            dispatch_and_score(child)
        except BaseException as exc:
            atomic_json(child/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()))
            raise
        atomic_json(child/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time()))
    report = select(protocol,{k:root/k/'run' for k in queue['order']})
    atomic_json(root/'selection-result.json',report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--controls',type=Path)
    p.add_argument('--protocol',type=Path)
    p.add_argument('--source-commit')
    p.add_argument('--instruction',action='append',default=[])
    a = p.parse_args()
    if a.prepare:
        if not all((a.controls,a.protocol,a.source_commit)):
            p.error('prepare requires controls, protocol and source-commit')
        instructions = dict(x.split('=',1) for x in a.instruction)
        if len(instructions)!=len(a.instruction):
            raise ValueError('duplicate prompt ID')
        print(json.dumps(prepare(a.root,a.controls,a.protocol,instructions,a.source_commit),indent=2))
    else:
        try:
            run(a.root)
        except BaseException as exc:
            atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),automatic_retry=False))
            raise
        atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time()))
