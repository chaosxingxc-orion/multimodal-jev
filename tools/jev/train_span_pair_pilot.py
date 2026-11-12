#!/usr/bin/env python3
"""One direct-audio BF16 LoRA paired update; output never claims READY."""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash


def run(config,output):
    import torch
    import torch_npu
    from multimodal_jev.span_pair_pilot import pilot_rows
    from multimodal_jev.span_runtime_topology import runtime_topology
    from multimodal_jev.span_runtime_profile import initialize_verl_preserving_qwen3,configure_actor_profile
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.sft import initialize_adapter
    from multimodal_jev.span_model import NativeAudioSpanModel
    from multimodal_jev.span_contract import SpanRequest,SpanQuestion,encode_span_request,span_labels
    from multimodal_jev.task_schema import DecisionSchema
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.span_request_identity import bind_span_request
    from multimodal_jev.span_actor_update import record_rollout
    from multimodal_jev.span_pair_actor_update import update_pair_actor
    from multimodal_jev.span_pair_credit import expected_pair_reward,exact_pair_credit
    from multimodal_jev.bf16_lora import BF16MasterAdamW
    from multimodal_jev.span_checkpoint import save_span_checkpoint
    from verify_span_rule_runtime import verify

    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('pair pilot pinned asset changed')
    protocol,batch,rows,pairs=pilot_rows(config)
    topology=runtime_topology(dict(runtime_profile=batch['runtime_profile'],physical_devices=config['physical_devices']))
    if topology['size']!=1 or sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))!=topology['devices']:
        raise ValueError('isolated authorized NPU required')
    gate=Path(protocol['runtime_admission']);audit=verify(gate)
    base_id=checkpoint_identity(batch['model'])[0];adapter_id=checkpoint_identity(batch['adapter'])[0]
    if (not audit['passed'] or audit['base_id']!=base_id or audit['adapter_id']!=adapter_id
            or base_id!=protocol['base_id'] or adapter_id!=protocol['initial_adapter_id']):
        raise ValueError('initial full runtime gate differs')
    atomic_json(output/'admission.json',audit)
    refs={r['id']:r for r in json.loads((gate/'run/result.json').read_text())['records']}
    torch.set_num_threads(4);torch.npu.set_device(0);torch.manual_seed(17);torch.npu.manual_seed_all(17)
    atomic_json(output/'verl-initialization.json',initialize_verl_preserving_qwen3())
    atomic_json(output/'runtime-profile.json',configure_actor_profile(batch['runtime_profile']))
    processor,base=load_runtime(batch['model'],'npu:0','bfloat16')
    adapted=initialize_adapter(base,dict(initial_adapter=batch['adapter'],lora_execution_dtype='bfloat16')).eval()
    model=NativeAudioSpanModel(adapted.get_base_model()).eval()
    prepared=[];ids=[];tokens=0;seconds=0.;policy_id=base_id+':'+adapter_id
    for row in rows:
        raw=row['request'];request=SpanRequest(raw['audio'],tuple(SpanQuestion(q['id'],DecisionSchema.from_dict(q['schema']))
            for q in raw['questions']),raw['final_id'],tuple(raw['requirements']),raw['version'])
        if file_hash(request.audio)!=row['provenance']['audio_sha256']:raise ValueError('audio changed')
        audio,receipt=load_audio(request.audio,return_receipt=True)
        inputs,positions=encode_span_request(processor,request,audio)
        mapping=span_labels(processor.tokenizer,request.questions)
        if any(m['values']!=(False,True) for m in mapping):raise ValueError('Boolean candidate support required')
        candidates=[m['token_ids'] for m in mapping]
        uid,identity=bind_span_request(request,receipt,inputs,positions,candidates,policy_id)
        if refs[row['id']]['request_id']!=uid or refs[row['id']]['request_identity']!=identity:
            raise ValueError('input differs from initial runtime gate')
        prepared.append((inputs,positions,candidates));ids.append(uid)
        tokens+=inputs['input_ids'].numel();seconds+=len(audio)/16000
    def forward():
        outputs=[]
        for inputs,positions,candidates in prepared:
            device_inputs={k:v.to('npu:0',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
            outputs.append(model(device_inputs,positions.to('npu:0'),candidates)[1])
        return tuple(torch.cat([r[j] for r in outputs]) for j in range(3))
    behavior=tuple(torch.tensor([refs[r['id']]['candidate_logprobs'][j] for r in rows]) for j in range(3))
    rollout=record_rollout(policy_id,ids,behavior,32,torch.Generator().manual_seed(17))
    targets=torch.tensor([r['targets'][-1] for r in rows],dtype=torch.long);beta=config['pair_weight']
    with torch.no_grad():before=forward()
    atomic_json(output/'before.json',dict(candidate_logprobs=[p.cpu().tolist() for p in before],request_ids=ids))
    torch.save(dict(candidate_logprobs=behavior,targets=targets,pairs=pairs,request_ids=ids,policy_id=policy_id,
                    advantages=exact_pair_credit(behavior[-1],targets,pairs,beta)),output/'pair-credit.pt')
    params=[p for p in model.parameters() if p.requires_grad];optimizer=BF16MasterAdamW(params,lr=protocol['lr'])
    atomic_json(output/'optimizer-before.json',optimizer.receipt())
    with (output/'update-claimed.json').open('x') as handle:
        json.dump(dict(policy_id=policy_id,pair_weight=beta,protocol_sha256=file_hash(config['protocol'])),handle)
    update=update_pair_actor(rollout,forward,optimizer,params,targets,pairs,beta,policy_id,ids)
    state=dict(schema='jev-pair-engineering-pilot-v1',protocol_sha256=file_hash(config['protocol']),pair_weight=beta,completed_updates=1)
    next_id=save_span_checkpoint(output/'policy-1',adapted,model,optimizer,base_id,1,state)
    with torch.no_grad():after=forward()
    atomic_json(output/'after.json',dict(candidate_logprobs=[p.cpu().tolist() for p in after],policy_id=next_id,
        expected_pair_reward=expected_pair_reward(after[-1].cpu(),targets,pairs,beta).tolist()))
    atomic_json(output/'update.json',dict(**update,optimizer=optimizer.receipt(),initial_policy_id=policy_id,
        exported_policy_id=next_id,input_tokens=tokens,audio_seconds_with_repetition=seconds,
        unique_recordings=4,training_admitted=False,checkpoint_ready=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    try:
        run(json.loads(a.config.read_text()),a.output)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
