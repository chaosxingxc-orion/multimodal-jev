#!/usr/bin/env python3
"""Independently authenticate completed bounded NPU updates and policy gates."""
import argparse
import json
from pathlib import Path
import torch
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.span_training_chain import read, check_pins, checkpoint_receipt, require_ready
from multimodal_jev.span_runtime_profile import verify_suite_admission
from multimodal_jev.span_parity import compare_span_policies


def verify(root):
    config=read(root/'config.json');run=root/'run'
    launch=read(root/'launch.json')
    if (launch['physical_devices']!=[4] or launch['config_sha256']!=file_hash(root/'config.json')
            or launch['source_manifest_sha256']!=file_hash(root/'source-manifest.json')):
        raise ValueError('launch identity differs')
    require_ready(run)
    step=dict(config['step']);step['pins']=dict(step['pins']);check_pins(step['pins'])
    verify_suite_admission(step)
    previous,_=checkpoint_receipt(step['initial_optimizer_checkpoint'],step['pins'])
    initial_updates=previous['completed_updates']
    versions=read(run/'accepted-versions.json')
    if len(versions)!=config['num_updates']:
        raise ValueError('incomplete update chain')
    summaries=[]
    for offset,version in enumerate(versions):
        number=initial_updates+offset+1
        training=run/f'update-{number:04d}';suite=run/f'acceptance-{number:04d}'
        require_ready(training);require_ready(suite)
        child=read(training.with_suffix('.config.json'));check_pins(child['pins'])
        if (child['initial_optimizer_checkpoint']!=step['initial_optimizer_checkpoint']
                or child['runtime_acceptance']!=step['runtime_acceptance']):
            raise ValueError('child consumed a different accepted parent')
        checkpoint=training/'policy-1';receipt,pins=checkpoint_receipt(checkpoint)
        selection=read(training/'selection.json');cursor=previous['training_state']
        if (receipt['completed_updates']!=number or selection!=receipt['training_state']
                or selection['batch_index']!=cursor['next_batch_index']
                or selection['next_batch_index']!=selection['batch_index']+1):
            raise ValueError('optimizer count or sample cursor differs')
        for key in ('schema','seed','schedule_sha256','epoch_batches','manifest_sha256','question_order'):
            if selection[key]!=cursor[key]:raise ValueError('learning contract changed')
        resume=read(training/'optimizer-resume.json');export=read(training/'export.json')
        if (resume['previous_updates']!=number-1 or resume['optimizer_sha256']!=previous['optimizer_sha256']
                or resume['optimizer']['optimizer_steps']!=[number-1]
                or export['optimizer']['optimizer_steps']!=[number]):
            raise ValueError('Adam state did not resume and advance once')
        state=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=True)
        states=state['optimizer']['optimizer']['state']
        if (state['completed_updates']!=number or state['training_state']!=selection
                or len(states)!=112 or any(int(s['step'])!=number for s in states.values())
                or any(v.dtype!=torch.float32 for s in states.values() for v in s.values() if torch.is_tensor(v))):
            raise ValueError('stored Adam state differs')
        if read(training/'actor-reload.json')['maximum_logprob_error']!=0:
            raise ValueError('independent actor reload differs')
        acceptance=suite/'vllm/result.json'
        step.update(initial_adapter=str(checkpoint/'adapter'),initial_optimizer_checkpoint=str(checkpoint),
                    runtime_acceptance=str(acceptance))
        step['pins'].update(pins);step['pins'][str(acceptance)]=file_hash(acceptance)
        verify_suite_admission(step)
        hf=read(suite/'hf/result.json');vllm=read(acceptance)
        if hf['base_id']!=receipt['base_id'] or hf['adapter_id']!=receipt['adapter_id']:
            raise ValueError('independent HF reference policy differs')
        reference={r['request_id']:r for r in hf['records']}
        errors=[]
        for row in vllm['records']:
            other=reference[row['request_id']]
            if other['request_identity']!=row['request_identity']:
                raise ValueError('HF/vLLM input differs')
            probs=other['candidate_logprobs']
            if isinstance(probs[0][0],list):probs=[p[0] for p in probs]
            if probs!=row['hf_candidate_logprobs']:raise ValueError('HF reference values differ')
            gate=compare_span_policies(probs,row['candidate_logprobs'])
            if not gate['passed']:raise ValueError('new version probability gate failed')
            errors.append(gate['maximum_logprob_error'])
        expected=dict(completed_updates=number,checkpoint=str(checkpoint),adapter_id=receipt['adapter_id'],
            optimizer_sha256=receipt['optimizer_sha256'],checkpoint_receipt_sha256=file_hash(checkpoint/'CHECKPOINT.json'),
            runtime_acceptance=str(acceptance),acceptance_sha256=file_hash(acceptance),next_batch_index=selection['next_batch_index'])
        if version!=expected:raise ValueError('published accepted version differs')
        summaries.append(dict(completed_updates=number,selection=selection,adapter_id=receipt['adapter_id'],
            recordings=len(errors),maximum_logprob_error=max(errors),gradient_norm=read(training/'update.json')['gradient_norm']))
        previous=receipt
    if read(run/'last-accepted.json')!=versions[-1] or read(run/'READY.json')['completed_updates']!=previous['completed_updates']:
        raise ValueError('final chain receipt differs')
    return dict(root=str(root),passed=True,initial_completed_updates=initial_updates,
        completed_updates=previous['completed_updates'],versions=summaries,
        scope='Bounded single-rank real verl-loss/vLLM updates with restored Adam and full policy gates; not distributed Ray training or held-out gains')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();result=verify(args.root);atomic_json(args.output,result);print(json.dumps(result,indent=2))
