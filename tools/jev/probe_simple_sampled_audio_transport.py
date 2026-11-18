#!/usr/bin/env python3
"""CPU replay of accepted real-audio vLLM records into native verl DataProto.

No model update or new rollout. The two-step prefix serves only as an engineering
transport fixture; it is not the final Stage1 reference or an RL learning run.
"""
import argparse,json,time
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash


def run(config):
    import torch
    from check_slurp_native_span_policy import prepared
    from verify_slurp_simple_parity import verify
    from multimodal_jev.slurp_official_training import load,example
    from multimodal_jev.simple_sampled_rollout import actor_batch
    from multimodal_jev.span_hybrid_actor import make_batch
    from multimodal_jev.span_policy import joint_logprobs
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU-only replay required')
    torch.set_num_threads(4)
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('replay dependency changed')
    gate=Path(config['gate']);accepted=verify(gate)
    if accepted['requests']!=256 or accepted['updates']!=2:raise ValueError('accepted original256 engineering fixture required')
    inference=json.loads((gate/'config.json').read_text());train=json.loads((Path(accepted['training_root'])/'config.json').read_text())
    artifact=json.loads(Path(inference['inputs']).read_text())
    result=json.loads((gate/'run/vllm/result.json').read_text())
    population,task,_=load(train['release'],train['release_sha256'],'learning',purpose='optimize');by_id={r['id']:r for r in population}
    policy=inference['base_id']+':'+inference['adapter_id'];entries=[]
    for row,inputs,positions,candidates,rid,identity,processor in prepared(inference,artifact):
        source=by_id[row['recording_id']]
        request,target,_=example(source,task,'direct','native')
        if json.loads(json.dumps(request.model_input()))!=identity['request']:raise ValueError('learning input differs')
        entries.append(dict(identity=identity,item=dict(policy_id=policy,inputs=inputs,positions=positions,candidates=candidates,
            targets=torch.tensor([target['targets']]),supervised=torch.tensor([target['supervised']]),
            span_weights=torch.tensor([target['span_weights']]),request_weight=1.,real_request=True)))
    if len(entries)!=256:raise ValueError('complete256 transport fixture required')
    schedule_receipts=[]
    if config.get('verify_scheduled_compiler',False):
        from multimodal_jev.simple_audio_batch import compile_batch,renderer_for_policy
        from multimodal_jev.slurp_training import content_hash
        schedule=json.loads(Path(train['epoch_file']).read_text())
        if (schedule['report']['global_batch']!=128
                or content_hash(schedule['batches'])!=schedule['report']['batches_sha256']):
            raise ValueError('original full global128 epoch required')
        old={r['identity']['request']['audio']:r for r in entries}
        new=[]
        for slots in schedule['batches'][:2]:
            compiled,renderer,receipt=compile_batch(by_id,task,slots,processor,policy_id=policy,global_batch=128)
            assert receipt['global_real_requests']==128
            rebound=renderer_for_policy(renderer,policy)
            assert [r[4:6] for r in rebound]==[r[4:6] for r in renderer]
            for entry in compiled:
                previous=old[entry['identity']['request']['audio']]
                assert entry['identity']==previous['identity']
                for name in ('targets','supervised','span_weights'):
                    assert torch.equal(entry['item'][name],previous['item'][name])
            new.extend(compiled);schedule_receipts.append(receipt)
        entries=new
    reports=[]
    for algorithm in ['grpo_v1','rloo_v1']:
        items,receipt=actor_batch(entries,result,result,policy_id=policy,reference_policy_id=policy,seed=317,update=0)
        batch=make_batch(items,stage='task',coefficient=1.,reduction='real_request_mean',algorithm=algorithm,
                         kl_coefficient=.01,reference_policy_id=policy)
        if len(batch)!=256 or any(len(c)!=64 for c in batch.split(64)):raise ValueError('DP4 transport split differs')
        mixed=0;allwrong=0;correct=0
        for item,payload in zip(items,batch.non_tensor_batch['multi_modal_inputs'],strict=True):
            rewards=(item['actions'][:,:,0]==item['targets'][:,0,None]).float()
            centered=rewards-rewards.mean(-1,keepdim=True)
            expected=centered/(rewards.std(-1,keepdim=True)+1e-6) if algorithm=='grpo_v1' else centered*16/15
            torch.testing.assert_close(expected,payload['sampled_advantages'],atol=1e-6,rtol=1e-6)
            assert torch.equal(payload['old_joint'],joint_logprobs(item['behavior'],item['actions']))
            correct+=int(rewards.sum());mixed+=0<float(rewards.sum())<16;allwrong+=float(rewards.sum())==0
        reports.append(dict(algorithm=algorithm,requests=256,samples=4096,mixed_reward_groups=mixed,
            all_wrong_groups=allwrong,correct_samples=correct,global_real_requests=receipt['global_real_requests']))
    return dict(passed=True,policy_id=policy,requests=256,reports=reports,
        scheduled_compiler_verified=bool(schedule_receipts),schedule_receipts=schedule_receipts,
        scope='Actual accepted audio/vLLM artifact replay through regenerated CPU tensors, G16 sampler and native verl DataProto/advantages. No NPU update, fresh rollout lifecycle, final Stage1 reference or efficacy claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json',run(json.loads(a.config.read_text())))
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
