#!/usr/bin/env python3
"""Independent full-action parity audit of a fixed rule coverage probe."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_parity import compare_span_policies
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.span_runtime_profile import PROFILE
from multimodal_jev.span_rule_population import configured_rule_rows


def verify(root):
    read=lambda p:json.loads(p.read_text())
    config=read(root/'config.json');launch=read(root/'launch.json');run=root/'run'
    if (launch['physical_devices']!=[5] or launch['config_sha256']!=file_hash(root/'config.json')
            or launch['source_manifest_sha256']!=file_hash(root/'source-manifest.json')
            or read(run/'terminal.json')['exit_code']!=0):raise ValueError('launch or terminal differs')
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('pinned input changed')
    suite=read(run/'suite-config.json')
    for path,digest in suite['pins'].items():
        if file_hash(path)!=digest:raise ValueError('reconstructed suite asset changed')
    result=read(run/'result.json');base_id=checkpoint_identity(config['model'])[0];adapter_id=checkpoint_identity(config['adapter'])[0]
    if (result['base_id']!=base_id or result['adapter_id']!=adapter_id or result['runtime_profile']!=PROFILE
            or result['tensor_parallel_size']!=1 or result['optimizer_updates']!=0 or result['generated_tokens']!=0):
        raise ValueError('policy or runtime differs')
    rows=configured_rule_rows(config)
    hf=[json.loads(s) for s in (Path(config['hf_run'])/'run/records.jsonl').read_text().splitlines()]
    if [r['id'] for r in rows]!=[r['id'] for r in hf] or [r['id'] for r in rows]!=[r['id'] for r in result['records']]:
        raise ValueError('full fixed coverage differs')
    audits=[]
    for row,reference,actual in zip(rows,hf,result['records'],strict=True):
        if (reference['request_id']!=actual['request_id'] or reference['request_identity']!=actual['request_identity']
                or reference['request_identity']['request']!=row['request']
                or actual['hf_candidate_logprobs']!=reference['candidate_logprobs']):
            raise ValueError('independent HF input or probabilities differ')
        audit=compare_span_policies(reference['candidate_logprobs'],actual['candidate_logprobs'])
        if audit!=actual['parity']:raise ValueError('stored parity differs')
        audits.append(audit)
    if result['passed']!=all(a['passed'] for a in audits):raise ValueError('aggregate parity differs')
    workers=read(run/'worker.json')
    if len(workers)!=1:raise ValueError('one worker required')
    worker=workers[0]
    if (worker['model_class']!='Qwen3ASRForSpanDecisions' or worker['lora_dtype']!='torch.bfloat16'
            or worker['head_projection_dtype']!='torch.bfloat16' or worker['registered_adapter_ids']!=[1]
            or len(worker['wrapper_modules'])!=28
            or any(v!=['torch.bfloat16'] for v in worker['actual_lora_buffer_dtypes'].values())):
        raise ValueError('actual BF16 native adapter execution differs')
    return dict(root=str(root),passed=result['passed'],base_id=base_id,adapter_id=adapter_id,requests=len(rows),
        passed_requests=sum(a['passed'] for a in audits),maximum_logprob_error=max(a['maximum_logprob_error'] for a in audits),
        joint_ratio_range=[min(a['joint_ratio_min'] for a in audits),max(a['joint_ratio_max'] for a in audits)],
        argmax_changed_spans=sum(len(a['argmax_changed_spans']) for a in audits),
        physical_devices=[5],training_admitted=False,
        files={str(p):file_hash(p) for p in [root/'config.json',run/'result.json',run/'identity.json',run/'worker.json',run/'terminal.json']},
        scope='Declared learning runtime coverage on one immutable policy; no automatic training admission')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=verify(args.root);atomic_json(args.output,result);print(json.dumps(result,indent=2))
