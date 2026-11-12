#!/usr/bin/env python3
"""Train beta0/beta0.5 independently and require full post-export acceptance."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_pair_pilot import pilot_rows
from multimodal_jev.policy_identity import checkpoint_identity
from verify_span_pair_pilot import verify


def run(root,output):
    read=lambda p:json.loads(p.read_text())
    launch=read(root/'launch.json');manifest=read(root/'source-manifest.json')
    if launch['launch_exit_code']!=0:raise ValueError('authenticated launch required')
    if launch['source_manifest_sha256']!=file_hash(root/'source-manifest.json'):
        raise ValueError('source manifest changed')
    for name,digest in manifest['files'].items():
        if file_hash(root/'source'/name)!=digest:raise ValueError('pilot snapshot changed')
    tools=Path(__file__).parent;accepted=[]
    def child(script,c,stage):
        stage.mkdir(exist_ok=False);atomic_json(stage/'config.json',c)
        with (stage/'process.log').open('x') as log:
            subprocess.run([sys.executable,str(tools/script),'--config',str(stage/'config.json'),'--output',str(stage/'run')],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    def receipt(stage,c):
        atomic_json(stage/'source-manifest.json',manifest)
        atomic_json(stage/'launch.json',dict(root=str(stage),image=launch['image'],physical_devices=c['physical_devices'],
            config_sha256=file_hash(stage/'config.json'),source_manifest_sha256=file_hash(stage/'source-manifest.json'),
            controller_sha256=file_hash(Path(__file__)),execution='paired engineering controller stage'))
    for name in ('pair0','pair05'):
        config=read(root/(name+'.json'));protocol,batch,_,_=pilot_rows(config)
        if config['physical_devices']!=launch['physical_devices']:raise ValueError('device differs')
        atomic_json(output/'progress.json',dict(stage='training',branch=name,updated_unix=time.time()))
        training=output/name;child('train_span_pair_pilot.py',config,training)
        atomic_json(training/'pending-audit.json',verify(training))
        adapter=training/'run/policy-1/adapter'
        # Reuse only the original gate population descriptor, not its adapter
        # or historical source snapshot. The paired protocol identifies this update.
        old=read(Path(protocol['runtime_admission'])/'config.json')
        c={k:v for k,v in old.items() if k not in ('hf_run','pins','adapter','adapter_id')}
        c.update(adapter=str(adapter),adapter_id=checkpoint_identity(adapter)[0],
                 physical_devices=config['physical_devices'],pins=dict(config['pins']),
                 engineering_protocol=config['protocol'],engineering_pair_weight=config['pair_weight'])
        transfer=Path(c['transfer_protocol']);c['pins'][str(transfer)]=file_hash(transfer)
        for p in adapter.iterdir():
            if p.is_file():c['pins'][str(p)]=file_hash(p)
        hf=output/(name+'-hf');atomic_json(output/'progress.json',dict(stage='hf_reload',branch=name,updated_unix=time.time()))
        child('infer_span_rule_baseline.py',c,hf);receipt(hf,c)
        runtime=output/(name+'-runtime');rc={**c,'hf_run':str(hf),'pins':dict(c['pins'])}
        for n in ('config.json','source-manifest.json','launch.json','run/result.json','run/records.jsonl','run/terminal.json'):
            rc['pins'][str(hf/n)]=file_hash(hf/n)
        atomic_json(output/'progress.json',dict(stage='vllm_full_gate',branch=name,updated_unix=time.time()))
        child('check_span_rule_runtime.py',rc,runtime);receipt(runtime,rc)
        report=verify(training,hf,runtime)
        if not report['accepted_for_reuse']:raise ValueError('full acceptance required')
        atomic_json(training/'verified.json',report);accepted.append(dict(branch=name,adapter_id=report['adapter_id'],audit_sha256=file_hash(training/'verified.json')))
    atomic_json(output/'READY.json',dict(status='passed',branches=accepted,
        scope='Two independent one-step engineering probes with full reload/parity. No continuation, model selection or task gain.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(a.root,a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
