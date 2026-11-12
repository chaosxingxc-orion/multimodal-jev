#!/usr/bin/env python3
"""Prepare a fixed-image resident suite from accepted policy exports; no launch."""
import argparse
from pathlib import Path
import shutil
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.container_pins import validate_bind_pins
from queue_span_native_actor_npu import read,completed
from queue_span_dp4_resume import check_pins


def prepare(template,output,repo):
    template,output,repo=map(Path,(template,output,repo))
    original=read(template/'config.json');plan=read(template/'prepared.json');check_pins(original['pins'])
    if original['sequence'] != ['A','B','A'] or plan['physical_devices'] != [4]:
        raise ValueError('fixed accepted switch plan required')
    check_pins({str(template/'config.json'):plan['config_sha256'],str(template/'source-manifest.json'):plan['source_manifest_sha256']})
    references=[]
    for key in ('A','B'):
        root=Path(original['policies'][key]['gate']);launch=read(root/'launch.json')
        if launch['image'] != plan['image'] or not completed(launch['container_id'],plan['image']):
            raise ValueError('actual successful policy reference required')
        if read(root/'config.json')['adapter_id'] != original['policies'][key]['adapter_id']:
            raise ValueError('original accepted adapter differs')
        references.append(root)
    output.mkdir(exist_ok=False)
    for directory in ('src','tools/jev','tests'):
        shutil.copytree(repo/directory,output/'source'/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    manifest={str(p.relative_to(output/'source')):file_hash(p) for p in (output/'source').rglob('*') if p.is_file()}
    atomic_json(output/'source-manifest.json',manifest)
    policies=[dict(key=key,config=str(root/'config.json')) for key,root in zip(('initial','d12'),references,strict=True)]
    pins={str(output/'source'/p):h for p,h in manifest.items()}
    for root in references:
        pins.update(read(root/'config.json')['pins'])
        for name in ('config.json','launch.json','run/READY.json','run/terminal.json','run/hf/result.json','run/hf/terminal.json','run/vllm/result.json','run/vllm/terminal.json'):
            pins[str(root/name)]=file_hash(root/name)
    cfg=dict(schema='jev-resident-policy-suite-v1',policies=policies,pins=pins)
    atomic_json(output/'config.json',cfg)
    argv=[s.replace(str(template),str(output)) for s in plan['create_argv']]
    argv[argv.index('--name')+1]='multimodal-jev-'+output.name
    index=next(i for i,s in enumerate(argv) if s.endswith('/probe_span_lora_switch.py'))
    argv[index]=str(output/'source/tools/jev/run_span_resident_policy_suite.py')
    atomic_json(output/'bind-pins.json',validate_bind_pins(argv,pins))
    result=dict(root=str(output),image=plan['image'],physical_devices=[4],create_argv=argv,
        config_sha256=file_hash(output/'config.json'),source_manifest_sha256=file_hash(output/'source-manifest.json'),
        status='prepared_not_launched',scope='Two fresh HF references followed by one resident vLLM; no new training or device coexistence.')
    atomic_json(output/'prepared.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2]);a=p.parse_args()
    prepare(a.template,a.output,a.repo)
