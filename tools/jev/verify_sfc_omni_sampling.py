#!/usr/bin/env python3
"""Verify exact frozen sampling container, source and independent raw records.

Run in an environment with the pinned Transformers tokenizer and SSH access to
server175. No target files are read and no missing inference is replayed.
"""
import argparse
import hashlib
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.omni_sampling_acceptance import audit_records, read_json
from multimodal_jev.omni_rollout_records import sampling_kwargs
from multimodal_jev.sfc_jev_runtime import INSTRUCTION
from multimodal_jev.sfc_reward_batch import digest
from sample_sfc_omni_learning import load_inputs
from watch_qwen3_omni_probe import inspect


def verify_container(root, plan, launch, inspection=None):
    if not (isinstance(launch['container_id'],str) and len(launch['container_id'])==64
            and all(c in '0123456789abcdef' for c in launch['container_id'])):
        raise ValueError('full hexadecimal container identity required')
    state=inspect(launch['container_id']) if inspection is None else inspection
    if (state['Id']!=launch['container_id'] or state['Image']!=plan['image']
            or state['State']['Status']!='exited' or state['State']['ExitCode']!=0
            or state['State'].get('OOMKilled',False)
            or launch['source_commit']!=plan['source_commit'] or launch['physical_devices']!=[6,7]
            or file_hash(root/'config.json')!=plan['config_sha256']):
        raise ValueError('exact successful pinned sampling container required')
    devices={d['PathOnHost'] for d in state['HostConfig'].get('Devices') or []
             if d['PathOnHost'].removeprefix('/dev/davinci').isdigit()}
    if devices!={'/dev/davinci6','/dev/davinci7'} or state['HostConfig'].get('Privileged'):
        raise ValueError('actual container must map only authorized NPU6/7')
    return state


def accept(root, tokenizer, inspection=None):
    plan=read_json(root/'preparation.json');launch=read_json(root/'launch.json')
    config=read_json(root/'config.json')
    verify_container(root,plan,launch,inspection)
    source=root/'source'
    expected_sources={str(p.relative_to(source)) for directory in ('src/multimodal_jev','tools/jev')
                      for p in (source/directory).glob('*.py')}
    if (set(plan['source_files'])!=expected_sources or not expected_sources
            or 'tools/jev/sample_sfc_omni_learning.py' not in expected_sources
            or 'tools/jev/verify_sfc_omni_sampling.py' not in expected_sources):
        raise ValueError('complete source snapshot required')
    for name,pin in plan['source_files'].items():
        if file_hash(source/name)!=pin or file_hash(Path(__file__).parents[2]/name)!=pin:
            raise ValueError('verifier/source snapshot differs from registered source')
    base,rows,preflight,checks=load_inputs(config)
    identity=read_json(root/'run/identity.json')
    asset=read_json(base['asset_audit'])
    if (asset['status']!='verified_inference_assets'
            or asset['repository']!='Qwen/Qwen3-Omni-30B-A3B-Instruct'
            or asset['revision']!='26291f793822fb6be9555850f06dfe95f2d7e695'
            or asset['model']!=base['audio_model'] or not asset['all_publisher_files_present']
            or not asset['files']):
        raise ValueError('pinned official model audit required')
    policy=digest(dict(repository=asset['repository'],revision=asset['revision'],files=asset['files']))
    if (identity['config']!=config or identity['preflight_only'] is not False
            or identity['versions']!=base['versions'] or identity['inputs_sha256']!=digest(rows)
            or identity['policy_sha256']!=policy or identity['model_files']!=asset['files']
            or identity['instruction_sha256']!=hashlib.sha256(INSTRUCTION.encode()).hexdigest()
            or identity['runner_sha256']!=plan['source_files']['tools/jev/sample_sfc_omni_learning.py']):
        raise ValueError('sampling run identity differs')
    required={name for name in plan['source_files'] if name.startswith('src/multimodal_jev/') and name.endswith('.py')}
    if set(identity['source_files'])!=required or any(plan['source_files'][k]!=v for k,v in identity['source_files'].items()):
        raise ValueError('complete source identity required')
    expected_sampling=sampling_kwargs(2,17)
    expected_sampling.update(stop=[],stop_token_ids=[])
    expected_engine=dict(model=base['audio_model'],dtype='bfloat16',tensor_parallel_size=2,seed=17,
        max_model_len=4096,max_num_seqs=1,max_num_batched_tokens=4096,gpu_memory_utilization=.85,
        enforce_eager=True,trust_remote_code=False,limit_mm_per_prompt={'audio':2,'image':0,'video':0},
        enable_prefix_caching=False,logprobs_mode='raw_logprobs',generation_config='vllm')
    if (identity['effective_sampling']!=expected_sampling or identity['engine_arguments']!=expected_engine
            or read_json(root/'run/effective-engine.json')!={'logprobs_mode':'raw_logprobs','generation_config':'vllm'}):
        raise ValueError('effective probability configuration differs')
    expected=[dict(example_id=r['id'],decision_id=d['id'],**{k:checks[r['id'],d['id']][k]
        for k in ('prompt_sha256','input_ids_sha256')}) for r in rows for d in r['input']['decisions']]
    if identity['prompt_evidence']!=expected:raise ValueError('processor input evidence differs')
    for name,pin in preflight['processor_assets'].items():
        if file_hash(Path(base['audio_model'])/name)!=pin:raise ValueError('tokenizer assets changed')
    result=audit_records(root/'run',rows,checks,tokenizer,policy_sha256=policy,
        instruction_sha256=identity['instruction_sha256'],sampling=expected_sampling)
    return dict(**result,container_id=launch['container_id'],image=plan['image'],
        source_commit=plan['source_commit'],result_sha256=file_hash(root/'run/result.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inspection',type=Path);p.add_argument('--inspection-hash');a=p.parse_args()
    inspection=None
    if a.inspection or a.inspection_hash:
        if not a.inspection or not a.inspection_hash or file_hash(a.inspection)!=a.inspection_hash:
            raise ValueError('exact coordinator-captured Docker inspection required')
        inspection=read_json(a.inspection)
    if a.output.exists():raise FileExistsError('preserve earlier acceptance')
    cfg=read_json(a.root/'config.json');base=read_json(cfg['baseline_config'])
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(base['audio_model'],local_files_only=True)
    print(json.dumps(result:=accept(a.root,tokenizer,inspection),indent=2))
    atomic_json(a.output,result)
