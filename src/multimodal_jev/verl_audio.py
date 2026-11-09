"""Audio batch adapter for the pinned verl DataProto/actor conventions.

Use remove_padding=False, sequence_parallel_size=1 and no fused actor kernels
until packed audio execution is verified. This module does not start Ray workers.
"""
from __future__ import annotations
import json
import math
from collections import Counter


def make_batch(processor,examples,audios,response_ids,old_log_probs,rewards,group_size):
    import numpy as np
    import torch
    from verl import DataProto
    from .audio_policy import hf_prompt
    n=len(examples)
    if n==0 or any(len(v)!=n for v in (audios,response_ids,old_log_probs,rewards)):
        raise ValueError('aligned nonempty batch required')
    if type(group_size) is not int or group_size<2:
        raise ValueError('GRPO requires at least two responses per prompt')
    counts=Counter(ex.example_id for ex in examples)
    if any(count!=group_size for count in counts.values()):
        raise ValueError('incomplete GRPO prompt group')
    # source_group joins counterfactual views for pair rewards. GRPO must center
    # only repeated responses to the SAME prompt, never across different schemas.
    identities={}
    for ex,audio in zip(examples,audios):
        signature=json.dumps(ex.model_input(),sort_keys=True)
        waveform=np.asarray(audio)
        if waveform.ndim!=1 or not waveform.size or not np.isfinite(waveform).all():
            raise ValueError('finite mono audio required')
        import hashlib
        signature+=(hashlib.sha256(waveform.tobytes()).hexdigest())
        if ex.example_id in identities and identities[ex.example_id]!=signature:
            raise ValueError('different prompt/audio assigned same GRPO uid')
        identities[ex.example_id]=signature
    prompts=[dict(hf_prompt(processor,ex,audio)) for ex,audio in zip(examples,audios)]
    max_prompt=max(p['input_ids'].shape[-1] for p in prompts)
    if any(not ids for ids in response_ids):raise ValueError('empty response is not trainable')
    max_response=max(map(len,response_ids))
    max_frames=max(p['input_features'].shape[-1] for p in prompts)
    pad=processor.tokenizer.pad_token_id
    if pad is None:raise ValueError('padding token required')
    batch_prompt=torch.full((n,max_prompt),pad,dtype=torch.long)
    batch_response=torch.full((n,max_response),pad,dtype=torch.long)
    attention=torch.zeros((n,max_prompt+max_response),dtype=torch.long)
    response_mask=torch.zeros((n,max_response),dtype=torch.long)
    old=torch.zeros((n,max_response),dtype=torch.float32)
    token_rewards=torch.zeros_like(old)
    mm=np.empty(n,dtype=object)
    uid=np.empty(n,dtype=object)
    source_group=np.empty(n,dtype=object)
    for i,(p,ex,ids,logps,reward) in enumerate(zip(prompts,examples,response_ids,old_log_probs,rewards)):
        length=len(ids);plen=p['input_ids'].shape[-1]
        if len(logps)!=length or not all(math.isfinite(v) for v in logps) or not math.isfinite(reward):
            raise ValueError('finite aligned rollout log-probs and rewards required')
        if any(type(token) is not int or not 0<=token<len(processor.tokenizer) for token in ids):
            raise ValueError('response contains invalid token id')
        batch_prompt[i,-plen:]=p['input_ids'][0]
        batch_response[i,:length]=torch.tensor(ids)
        attention[i,max_prompt-plen:max_prompt+length]=1
        response_mask[i,:length]=1
        old[i,:length]=torch.tensor(logps)
        token_rewards[i,length-1]=reward
        frames=p['input_features'].shape[-1]
        mm[i]={'input_features':torch.nn.functional.pad(p['input_features'],(0,max_frames-frames)),
               'input_features_mask':torch.nn.functional.pad(p['input_features_mask'],(0,max_frames-frames))}
        uid[i]=ex.example_id;source_group[i]=ex.metadata.get('source_group',ex.example_id)
    positions=(attention.cumsum(-1)-1).clamp_min(0)*attention
    result=DataProto.from_dict(tensors={
        'prompts':batch_prompt,'responses':batch_response,
        'input_ids':torch.cat((batch_prompt,batch_response),dim=-1),
        'attention_mask':attention,'position_ids':positions,'response_mask':response_mask,
        'old_log_probs':old,'token_level_rewards':token_rewards},
        non_tensors={'multi_modal_inputs':mm,'uid':uid,'source_group':source_group},
        meta_info={'audio_contract':'jev-qwen3-asr-v1','group_size':group_size,
                   'requires_remove_padding':False,'requires_sequence_parallel_size':1})
    return result


def compute_advantages(batch,norm_adv_by_std=True):
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    from .advantage_config import normalization
    normalization({'norm_adv_by_std':norm_adv_by_std})
    advantages,returns=compute_grpo_outcome_advantage(
        batch.batch['token_level_rewards'],batch.batch['response_mask'],
        batch.non_tensor_batch['uid'],norm_adv_by_std_in_grpo=norm_adv_by_std)
    batch.batch['advantages']=advantages;batch.batch['returns']=returns
    return batch


def policy_loss(batch,new_log_probs,clip_ratio=0.2):
    from omegaconf import OmegaConf
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    if new_log_probs.shape!=batch.batch['old_log_probs'].shape:
        raise ValueError('log-prob shape differs from rollout')
    config=OmegaConf.create({'clip_ratio':clip_ratio,'clip_ratio_low':None,'clip_ratio_high':None,
                            'clip_ratio_c':3.0,'global_batch_info':{}})
    return compute_policy_loss_vanilla(batch.batch['old_log_probs'],new_log_probs,
        batch.batch['advantages'],batch.batch['response_mask'],config=config)
