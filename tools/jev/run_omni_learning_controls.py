#!/usr/bin/env python3
"""Gate on completed v3 lifecycle, then sequential fixed64 actual/zero runs."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from watch_qwen3_omni_probe import inspect,wait,run as dispatch_and_score
from compare_omni_waveform import compare


def run(root,predecessor):
    old=json.loads((predecessor/'launch.json').read_text())
    state=wait(root,old['container_id'],old['image'],time.monotonic()+3900)
    if state['Status']!='exited' or state['ExitCode']!=0:raise RuntimeError('v3 lifecycle gate failed; no controls dispatched')
    terminal=json.loads((predecessor/'run/terminal.json').read_text())
    result=json.loads((predecessor/'run/result.json').read_text())
    if terminal['exit_code']!=0 or result['requests']!=6 or not result['model_files_unchanged']:
        raise ValueError('complete unchanged v3 probe required')
    deadline=time.monotonic()+120
    while not (predecessor/'evaluation-terminal.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('v3 scoring terminal not available; no dispatch')
        time.sleep(5)
    if json.loads((predecessor/'evaluation-terminal.json').read_text())['exit_code']!=0:
        raise RuntimeError('v3 supervisor/scorer gate failed')
    score=json.loads((predecessor/'score.json').read_text())
    if score['requests']!=6 or score['coverage']!=1.:raise ValueError('v3 scoring coverage incomplete')
    intent=json.loads((root/'queue-intent.json').read_text());partition=Path(intent['partition'])
    if file_hash(partition)!=intent['partition_sha256']:raise ValueError('partition changed')
    groups=json.loads(partition.read_text())['learning'];commit=intent['source_commit']
    for condition in ['actual','zeros']:
        child=root/condition;child.mkdir(exist_ok=False)
        shutil.copytree(root/'source',child/'source')
        cfg=json.loads((predecessor/'config.json').read_text())
        cfg.update(source_groups=groups,audio_condition=condition,partition=str(partition),partition_sha256=file_hash(partition),
                   instruction=str(child/'source/configs/tasks/frozen-decision-prompt-v1.txt'),asset_audit=str(child/'asset-audit.json'),
                   purpose='frozen-qwen3-omni-learning64-waveform-control')
        atomic_json(child/'config.json',cfg);shutil.copyfile(predecessor/'asset-audit.json',child/'asset-audit.json')
        argv=[x.replace(str(predecessor),str(child)).replace(predecessor.name,root.name+'-'+condition) for x in old['argv']]
        plan={**old,'root':str(child),'argv':argv,'source_commit':commit,'wait_for':[],
              'config_sha256':file_hash(child/'config.json'),'revision_reason':'registered64TRAIN waveform controls'}
        plan.pop('container_id');atomic_json(child/'intent.json',plan)
        try:dispatch_and_score(child)
        except BaseException as exc:
            atomic_json(child/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
        atomic_json(child/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time()))
    report=compare(root/'actual/run',root/'zeros/run')
    atomic_json(root/'result.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--predecessor',type=Path,required=True);a=p.parse_args()
    try:run(a.root,a.predecessor)
    except BaseException as exc:
        atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),automatic_retry=False));raise
    atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time()))
