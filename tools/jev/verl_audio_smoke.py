#!/usr/bin/env python3
"""CPU integration test using real processor, DataProto, GRPO and PPO loss.

Response tokens are controlled fixtures, NOT sampled policy trajectories.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import torch
from transformers import Qwen3ASRProcessor
from verl.utils.model import extract_multi_modal_inputs
from multimodal_jev.audio_policy import load_audio,answer_ids
from multimodal_jev.task_schema import JevExample,DecisionSchema
from multimodal_jev.verl_audio import make_batch,compute_advantages,policy_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--audio',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4)
    processor=Qwen3ASRProcessor.from_pretrained(args.model,local_files_only=True)
    audio=load_audio(args.audio)
    ex=JevExample('prompt-A','train',args.audio,DecisionSchema('a','noul','Language change requested?'),True,
                  {'source_group':'recording'})
    other=replace(ex,example_id='prompt-B',schema=DecisionSchema('b','noul','Lights off requested?'),target=False)
    examples=[ex,ex,other,other]
    ids=[answer_ids(processor.tokenizer,v) for v in (True,False,None,False)]
    old=[[-2.0]*len(row) for row in ids]
    rewards=[1.0,0.0,-0.1,1.0]
    batch=make_batch(processor,examples,[audio,audio,audio[:len(audio)//2],audio[:len(audio)//2]],ids,old,rewards,2)
    batch=compute_advantages(batch)
    advantage=batch.batch['advantages'];mask=batch.batch['response_mask']
    assert advantage[0,0]>0 and advantage[1,0]<0 and advantage[2,0]<0 and advantage[3,0]>0
    assert torch.equal(advantage[mask==0],torch.zeros_like(advantage[mask==0]))
    for i,row in enumerate(ids):
        assert int(mask[i].sum())==len(row)
        assert float(batch.batch['token_level_rewards'][i].sum())==torch.tensor(rewards[i]).item()
    mm=extract_multi_modal_inputs(batch.non_tensor_batch['multi_modal_inputs'])
    assert mm['input_features'].shape[0]==4 and mm['input_features_mask'].shape[0]==4
    assert mm['input_features_mask'][2].sum()<mm['input_features_mask'][0].sum()
    new=batch.batch['old_log_probs'].detach().clone().requires_grad_(True)
    loss,metrics=policy_loss(batch,new);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(new.grad).all()
    assert new.grad[0,0]<0 and new.grad[1,0]>0
    assert torch.equal(new.grad[mask==0],torch.zeros_like(new.grad[mask==0]))
    # Padding log-probs must not change the objective.
    perturbed=new.detach().clone();perturbed[mask==0]=100
    torch.testing.assert_close(policy_loss(batch,perturbed)[0],loss.detach())
    # Invalid group size and colliding prompt identities must be rejected.
    checks=0
    for candidate_examples,candidate_ids,candidate_logps,candidate_rewards in [
        (examples[:-1],ids[:-1],old[:-1],rewards[:-1]),
        ([ex,replace(other,example_id=ex.example_id),other,other],ids,old,rewards),
    ]:
        try:
            make_batch(processor,candidate_examples,[audio]*len(candidate_examples),
                       candidate_ids,candidate_logps,candidate_rewards,2)
        except ValueError:
            checks+=1
        else:
            raise AssertionError('invalid prompt group accepted')
    assert checks==2
    report={'status':'passed' ,'batch_size':len(examples),'prompt_groups':2,'replicates_per_prompt':2,
            'actor_input_shape':list(batch.batch['input_ids'].shape),
            'audio_features_shape':list(mm['input_features'].shape),
            'reward_centering_per_prompt':True,'padding_gradient_zero':True,
            'positive_advantage_increases_logprob':True,'audio_variable_length_collation':True,
            'metrics':metrics,'scope':'Real verl components and real audio preprocessing; controlled response fixtures, no rollout or RL efficacy claim'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(report,handle,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
