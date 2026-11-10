#!/usr/bin/env python3
"""Prepare a complete transfer shard with explicit mounts, without dispatch."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.container_inputs import require_visible_inputs
from multimodal_jev.omni_slurp_transfer_contract import select_transfer_shard
from multimodal_jev.slurp_jev_shards import materialize_shard


def prepare(root,protocol_path,shard_id,prompt_id,native_probe,source_commit,devices):
    if devices not in ([4,5],[6,7]):raise ValueError('authorized NPU pair required')
    protocol=json.loads(protocol_path.read_text())
    if prompt_id not in protocol['prompts']:raise ValueError('registered transfer prompt required')
    selection=Path(protocol['selection_result']).parent
    instruction=selection/prompt_id/'instruction.txt'
    if file_hash(instruction)!=protocol['prompts'][prompt_id]:raise ValueError('selected prompt changed')
    parent_cfg=json.loads((selection/'parent/config.json').read_text())
    old=json.loads((native_probe/'launch.json').read_text())
    if old['image']!=protocol['image']:raise ValueError('same runtime image required')
    if json.loads((native_probe/'evaluation-terminal.json').read_text())['exit_code']!=0:
        raise ValueError('successful complete native probe launch reference required')
    commit=subprocess.check_output(['git','rev-parse',source_commit+'^{commit}'],text=True).strip()
    root.mkdir(parents=True,exist_ok=False);source=root/'source';source.mkdir()
    archive=subprocess.Popen(['git','archive',commit],stdout=subprocess.PIPE)
    subprocess.run(['tar','-x','-C',str(source)],stdin=archive.stdout,check=True)
    archive.stdout.close()
    if archive.wait():raise RuntimeError('source export failed')
    for relative in ('tools/jev/infer_qwen3_omni_slurp_transfer.py',
                     'src/multimodal_jev/omni_slurp_transfer_contract.py'):
        if not (source/relative).exists():raise ValueError('transfer snapshot missing')
    shutil.copyfile(protocol_path,root/'transfer-protocol.json')
    shutil.copyfile(instruction,root/'instruction.txt')
    shutil.copyfile(parent_cfg['asset_audit'],root/'asset-audit.json')
    receipt=materialize_shard(Path(protocol['shard_plan']),shard_id,root/'inputs')
    cfg=dict(protocol['runtime'],split='test',evaluation_phase='slurp-jev-transfer',
        source_groups=receipt['source_groups'],shard_id=shard_id,prompt_id=prompt_id,
        purpose='frozen-FSC-selected-prompt-SLURP-Jev-full-TEST-shard',device='npu:0',wall_seconds=86400,
        requests=str(root/'inputs/test.requests.jsonl'),instruction=str(root/'instruction.txt'),
        asset_audit=str(root/'asset-audit.json'),transfer_protocol=str(root/'transfer-protocol.json'),
        transfer_protocol_sha256=file_hash(protocol_path))
    select_transfer_shard(cfg,[json.loads(line) for line in Path(cfg['requests']).open()])
    atomic_json(root/'config.json',cfg)
    argv=[]
    for arg in old['argv']:
        arg=arg.replace(str(native_probe),str(root)).replace(native_probe.name,root.name)
        arg=arg.replace('infer_qwen3_omni_slurp_probe.py','infer_qwen3_omni_slurp_transfer.py')
        if arg=='3600s':arg='86400s'
        if arg.startswith('HCCL_IF_BASE_PORT='):arg='HCCL_IF_BASE_PORT='+('62500' if devices==[4,5] else '63500')
        if devices==[6,7]:arg=arg.replace('/dev/davinci4','/dev/davinci6').replace('/dev/davinci5','/dev/davinci7')
        argv.append(arg)
    # Protocol references remain immutable at their original paths.
    mounts={selection,Path(protocol['shard_plan']).parent}
    for mount in sorted(mounts):argv[2:2]=['-v',str(mount)+':'+str(mount)+':ro']
    require_visible_inputs(argv,cfg)
    plan=json.loads(Path(protocol['shard_plan']).read_text())
    intent=dict(argv=argv,image=protocol['image'],source_commit=commit,physical_devices=devices,
        root=str(root),config_sha256=file_hash(root/'config.json'),wait_for=[],automatic_retry=False,
        observation_seconds=86700,scorer_python=old['scorer_python'],scorer_repository=old['scorer_repository'],
        jev_release=plan['release'],expected_recordings=receipt['recordings'],
        transfer_protocol_sha256=file_hash(protocol_path),shard_id=shard_id,prompt_id=prompt_id,
        scope='Prepared one full-task shard; complete transfer requires all registered shards')
    atomic_json(root/'intent.json',intent)
    return dict(root=str(root),source_commit=commit,physical_devices=devices,shard_id=shard_id,
        prompt_id=prompt_id,requests=receipt['requests'],recordings=receipt['recordings'],dispatched=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','protocol','native-probe'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('shard','prompt','source-commit'):p.add_argument('--'+name,required=True)
    p.add_argument('--devices',choices=['4,5','6,7'],required=True);a=p.parse_args()
    print(json.dumps(prepare(a.root,a.protocol,a.shard,a.prompt,a.native_probe,a.source_commit,
        list(map(int,a.devices.split(',')))),indent=2))
