"""Collective admission for equally weighted complete typed request groups.

Call on every data-parallel rank before the native actor's first collective.
Only metadata is gathered; audio and sampled actions stay on their owner rank.
This gate does not establish FSDP initialization or rollout synchronization.
"""
import torch
import torch.distributed as dist


def admit_rank_batch(batch, config, process_group=None):
    """Fail on all ranks when partitioning would bias or split request groups."""
    summary = dict(error=None, requests=[], policy=config.jev_policy_id,
                   group_size=config.jev_group_size,
                   rows=len(batch), micro=config.ppo_micro_batch_size_per_gpu)
    summary['algorithm'] = batch.meta_info.get('jev_advantage_algorithm','grpo_v1')
    try:
        if (summary['algorithm'] not in ('grpo_v1','rloo_v1')
                or summary['algorithm'] != config.get('jev_advantage_algorithm','grpo_v1')):
            raise ValueError('configured group advantage differs from behavior batch')
        size=config.jev_group_size
        micro=config.ppo_micro_batch_size_per_gpu
        if (type(size) is not int or size<2 or type(micro) is not int or micro<1
                or micro%size or len(batch)!=config.ppo_mini_batch_size
                or not len(batch) or len(batch)%micro):
            raise ValueError('local full minibatch of complete groups required')
        if batch.meta_info['use_dynamic_bsz'] or batch.meta_info['temperature']!=1.:
            raise ValueError('static untransformed policy required')
        for name in ('responses','response_mask','old_log_probs','advantages'):
            value=batch.batch[name]
            if value.shape!=(len(batch),1) or not torch.isfinite(value).all():
                raise ValueError('finite whole-action transport required')
        if not torch.all(batch.batch['response_mask']==1):
            raise ValueError('one valid whole action per sample required')
        payload=batch.non_tensor_batch['multi_modal_inputs']
        for start in range(0,len(batch),size):
            group=payload[start:start+size];item=group[0]['item']
            rollout=item['rollout']
            if (rollout.policy_id!=config.jev_policy_id or len(rollout.request_ids)!=1
                    or any(row['item'] is not item or row['group_size']!=size for row in group)
                    or [row['sample_index'] for row in group]!=list(range(size))
                    or batch.batch['responses'][start:start+size,0].tolist()!=list(range(size))):
                raise ValueError('stale policy or incomplete ordered group')
            if not torch.equal(batch.batch['old_log_probs'][start:start+size].detach().cpu(),
                               rollout.old_joint_logprobs.reshape(-1,1)):
                raise ValueError('old action logprob transport changed')
            summary['requests'].append(rollout.request_ids[0])
    except Exception as exc:
        summary['error']=str(exc)
    gathered=[None]*dist.get_world_size(process_group)
    dist.all_gather_object(gathered,summary,group=process_group)
    errors=[(rank,row['error']) for rank,row in enumerate(gathered) if row['error']]
    if errors:raise ValueError(f'collective typed batch admission failed: {errors}')
    shapes={(row['rows'],row['group_size'],row['micro'],row['policy'],row['algorithm']) for row in gathered}
    if len(shapes)!=1:
        raise ValueError('equal complete groups, microbatches and policy required on all ranks')
    ids=[uid for row in gathered for uid in row['requests']]
    if len(ids)!=len(set(ids)):
        raise ValueError('request duplicated across data-parallel ranks')
    return dict(world_size=len(gathered),global_requests=len(ids),
                global_actions=sum(row['rows'] for row in gathered),
                local_requests=len(summary['requests']),passed=True)
