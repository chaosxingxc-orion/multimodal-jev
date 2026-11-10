#!/usr/bin/env python3
"""Compare native actor microbatch autograd with the validated audio actor.

Uses a completed initializer rollout for a fresh replay from that initializer.
It does not continue an unrelated policy or implement native update_policy.
"""
import argparse
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash


def run(args):
    import torch
    import torch.distributed as dist
    from peft import PeftModel
    from omegaconf import OmegaConf
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.rl_run import load_rollout_run
    from multimodal_jev.rl_update import update_actor
    from multimodal_jev.policy_identity import validate_adapter_export
    from multimodal_jev.paired_rollout import compose_rewards
    from multimodal_jev.verl_audio import make_batch,compute_advantages,policy_loss
    from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_logprobs

    cfg=json.loads(args.config.read_text());torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    if cfg.get('parent_run') or not cfg.get('initial_adapter'):
        raise ValueError('fresh initializer replay required')
    identity,examples,records=load_rollout_run(cfg['rollout_run'])
    validate_adapter_export(cfg['model_path'],identity['config']['checkpoint'],cfg['initial_adapter'])
    if file_hash(inspect.getfile(DataParallelPPOActor))!=args.actor_sha256:
        raise ValueError('native actor source changed')
    processor,base=load_runtime(cfg['model_path'],'cpu','float32')
    model=PeftModel.from_pretrained(base,cfg['initial_adapter'],is_trainable=True).eval()
    parameters=[p for p in model.parameters() if p.requires_grad]
    initial=[p.detach().clone() for p in parameters]
    waves=[load_audio(ex.audio) for ex in examples]
    def progress(values):
        atomic_json(args.output/'progress.json',{**values,'heartbeat_unix':time.time()})
    def optimizer():
        return torch.optim.AdamW(parameters,lr=cfg['lr'],weight_decay=0.,foreach=False,fused=False)
    reference_optimizer=optimizer();reference_grad=[]
    # Capture pre-clipping gradients without changing the backward calculation.
    handles=[p.register_hook(lambda g,index=i: capture(index,g)) for i,p in enumerate(parameters)]
    gradients={}
    def capture(index,gradient):
        gradients[index]=gradient.detach().clone()
        return gradient
    reference=update_actor(model,processor,reference_optimizer,examples,waves,records,
        identity['policy_id'],identity['config']['sampling']['n'],max_logprob_error=cfg['max_logprob_error'],
        reward_mode=cfg['reward_mode'],pair_bonus_weight=cfg['pair_bonus_weight'],
        progress=lambda v:progress({'implementation':'reference',**v}))
    for handle in handles:handle.remove()
    if not reference['updated'] or len(gradients)!=len(parameters):
        raise ValueError('registered reference must exercise a complete optimizer update')
    reference_grad=[gradients[i] for i in range(len(parameters))]
    reference_state={'parameters':[p.detach().clone() for p in parameters],
                     'gradients':reference_grad,
                     'exp_avg':[reference_optimizer.state[p]['exp_avg'].clone() for p in parameters],
                     'exp_avg_sq':[reference_optimizer.state[p]['exp_avg_sq'].clone() for p in parameters]}
    del gradients,reference_optimizer
    with torch.no_grad():
        for p,value in zip(parameters,initial):p.copy_(value)
    opt=optimizer();opt.zero_grad(set_to_none=True)
    lookup={ex.example_id:ex for ex in examples};audio=dict(zip(lookup,waves))
    ordered=[lookup[r.example_id] for r in records]
    rewards,_=compose_rewards(ordered,records,cfg['reward_mode'],cfg['pair_bonus_weight'])
    batch=compute_advantages(make_batch(processor,ordered,[audio[r.example_id] for r in records],
        [r.token_ids for r in records],[r.log_probs for r in records],rewards,identity['config']['sampling']['n']))
    native_config=OmegaConf.create({'use_remove_padding':False,'use_fused_kernels':False,
        'ulysses_sequence_parallel_size':1,'use_dynamic_bsz':False,'use_prefix_grouper':False,
        'entropy_from_logits_with_chunking':False,'use_torch_compile':False,
        'fsdp_config':{'dtype':'float32'},'calculate_sum_pi_squared':False,'entropy_checkpointing':False})
    dist.init_process_group('gloo',init_method=(args.output/'gloo-init').resolve().as_uri(),rank=0,world_size=1)
    try:
        actor=DataParallelPPOActor(native_config,model,opt)
        if actor.device_name!='cpu':raise ValueError('CPU-only probe')
        inputs={**batch.batch,**batch.non_tensor_batch}
        with native_actor_cpu_logprobs():
            progress({'implementation':'native','phase':'parity_forward'})
            with torch.no_grad():
                logp=actor._forward_micro_batch(inputs,temperature=1.,calculate_entropy=False)['log_probs']
                error=float((logp-batch.batch['old_log_probs'])[batch.batch['response_mask'].bool()].abs().max())
                if not torch.isfinite(logp).all() or error>cfg['max_logprob_error']:
                    raise ValueError('native actor rollout parity failed before update')
            del logp
            progress({'implementation':'native','phase':'gradient_forward','max_logprob_error':error})
            logp=actor._forward_micro_batch(inputs,temperature=1.,calculate_entropy=False)['log_probs']
            loss,_=policy_loss(batch,logp)
            if not torch.isfinite(loss):raise ValueError('nonfinite native loss')
            loss.backward();native_grad=[p.grad.detach().clone() for p in parameters]
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            opt.step()
        current={'parameters':[p.detach().clone() for p in parameters],'gradients':native_grad,
                 'exp_avg':[opt.state[p]['exp_avg'].clone() for p in parameters],
                 'exp_avg_sq':[opt.state[p]['exp_avg_sq'].clone() for p in parameters]}
        errors={};failures=[]
        for field in current:
            errors[field]=max(float((a-b).abs().max()) for a,b in zip(reference_state[field],current[field]))
            for index,(a,b) in enumerate(zip(reference_state[field],current[field])):
                try:torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-4)
                except AssertionError as exc:failures.append({'field':field,'parameter':index,'error':str(exc)})
        comparison={'reference_metrics':reference,'native_loss':float(loss.detach()),
            'native_grad_norm':float(norm),'native_parity_error':error,'max_errors':errors,
            'failures':failures,'atol':2e-6,'rtol':2e-4,'response_tokens':int(batch.batch['response_mask'].sum())}
        atomic_json(args.output/'comparison.json',comparison)
        # Save both snapshots so independent audits can repeat the tensor comparisons.
        torch.save({'reference':reference_state,'native':current},args.output/'comparison.pt')
        if failures:raise AssertionError('native update equivalence failed; preserve comparison')
        atomic_json(args.output/'result.json',{'status':'passed',**comparison,
            'config_sha256':file_hash(args.config),'comparison_sha256':file_hash(args.output/'comparison.pt'),
            'actor_source_sha256':args.actor_sha256,'cpu_logprob_adapter':True,
            'scope':'real Qwen LoRA gradient/AdamW parity through native microbatch forward with explicit CPU dispatcher; '
                    'not native update_policy, Ray/FSDP, multi-rank model training or efficacy'})
    finally:
        dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--actor-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(a)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()});raise
    atomic_json(a.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})


if __name__=='__main__':main()
