"""Single-rank FSDP capacity/timing diagnostic on synthetic or pinned TRAIN inputs."""
import json
import sys
import time
from pathlib import Path


def run(config, prior, case):
    import torch
    import torch_npu
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from torch.distributed.fsdp.wrap import CustomPolicy
    from multimodal_jev.span_runtime_profile import initialize_verl_preserving_qwen3, configure_actor_profile, SHORT_AUDIO_PROFILE
    initialize_verl_preserving_qwen3()
    configure_actor_profile(SHORT_AUDIO_PROFILE)
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.sft import initialize_adapter
    from multimodal_jev.context_model_profile import from_hf, build_head
    from multimodal_jev.context_stage_lora import from_policy
    from multimodal_jev.context_runtime_profile import install_actor
    from multimodal_jev.context_fsdp import context_fsdp_units
    from multimodal_jev.context_decision_model import ContextDecisionModel
    from multimodal_jev.context_policy_snapshot import load_policy
    from multimodal_jev.card_contrast import Directions, components
    from multimodal_jev.card_training_acoustics import collate
    from tools.reports.stage2_view_actor import configure_activation_checkpointing

    real = config.get('real_lane')
    length,width=(None,real['width']) if real else config['cases'][case]
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['seed'])
    torch.npu.set_device(0)
    torch.npu.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True)
    device=torch.device('npu',0)
    dist.init_process_group('hccl',init_method='tcp://127.0.0.1:29867',rank=0,world_size=1)
    try:
        c=json.loads(Path(prior).read_text())
        print('loading',length,width,flush=True)
        processor,base=load_runtime(c['model'],device,'bfloat16')
        lora=from_policy(c['source_contract'])
        adapted=initialize_adapter(base,dict(lora_rank=lora['rank'],lora_alpha=lora['alpha'],
                                  lora_execution_dtype=lora['execution_dtype'])).eval()
        head=build_head(from_hf(c['model'])).to(device=device,dtype=torch.bfloat16).eval()
        token=c['source_contract']['token']['id']
        model=ContextDecisionModel(adapted.get_base_model().model,head,token).eval()
        load_policy(c['source_policy'],model,expected_policy_id=c['source_policy_id'],
                    expected_contract=c['source_contract'],expected_base_id=c['base_id'])
        check = config.get('paired_check')
        graph = check['graphs'][case] if check else config.get('training_graph')
        hooks,profile=install_actor(adapted,model,c['runtime_profile'],training_graph=graph)
        units=context_fsdp_units(model)
        for p in model.parameters():
            if p.requires_grad:p.data=p.data.float()
        wrapped=FSDP(model,auto_wrap_policy=CustomPolicy(lambda m:m in units),device_id=device,use_orig_params=True,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32,cast_forward_inputs=True))
        configure_activation_checkpointing(model,c['activation_checkpointing'])
        if real:
            from tools.jev.card_real_bucket_panel import measure
            return measure(config,c,wrapped,base,processor,model,device)
        opt=torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],lr=c['lr_schedule']['peak_lr'])
        d=Directions.from_objective(c['instruction_objective'])
        rows=[]
        chunk=base.config.audio_config.n_window*2
        # Inverse length for the registered three stride-two convolutions.
        counts=config['audio_tokens'][str(length)]
        complete,tail=divmod(counts,(chunk+7)//8)
        frames=complete*chunk+(8*tail-7 if tail else 0)
        padframes=((frames+chunk-1)//chunk)*chunk
        for a in range(width):
            sounds=[]
            for k in range(1+d.recordings+d.positives):
                f=torch.randn(1,base.config.audio_config.num_mel_bins,padframes,dtype=torch.bfloat16)
                m=(torch.arange(padframes)[None]<frames).long()
                sounds.append((f,m))
            for j in range(d.width):
                f,m=sounds[max(0,j-d.cards)]
                n=max(counts+2,round(length*config['token_fill'][(a+j)%len(config['token_fill'])]))
                ids=torch.tensor([[13]*(n-counts-1)+[base.config.audio_token_id]*counts+[token]])
                rows.append(dict(request_id=f'{a}/{max(0,j-d.cards)}:{j}',policy_id='probe',positions=torch.tensor([[n-1]]),
                    inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),input_features=f,input_features_mask=m)))
        inp,pos=collate(rows,'probe',decision_token_id=token,pad_token_id=processor.tokenizer.pad_token_id,
                audio_token_id=base.config.audio_token_id,audio_chunk_size=chunk,audio_infer_window=base.config.audio_config.n_window_infer)
        if inp['input_ids'].shape!=(width*d.width,length) or inp['attention_mask'].all():raise ValueError('incorrect padding geometry')
        inp={k:v.to(device) for k,v in inp.items()}; pos=pos.to(device)
        times=[]
        if check:
            from tools.jev.card_split_parity import Witness
            witness = Witness(check, rows, c, base, processor, model, case)
        for i in range(config['warmup']+config['repeats']):
            if check and i in check['check_updates']:
                witness.snapshot(i, wrapped, model)
            opt.zero_grad(set_to_none=True);torch.npu.synchronize();torch.npu.reset_peak_memory_stats()
            t=time.monotonic()
            logits=wrapped(inp,pos,return_logits=True).reshape(width,d.width)
            loss=components(logits,d)[:,0].mean()
            loss.backward()
            norm=wrapped.clip_grad_norm_(1.)
            opt.step();torch.npu.synchronize()
            seconds=time.monotonic()-t
            if check:
                witness.observe(i,logits,loss,norm,wrapped)
            if not torch.isfinite(loss).item() or not torch.isfinite(norm).item():raise ValueError('nonfinite update')
            print('update',i,round(seconds,3),flush=True)
            if i>=config['warmup']:times.append(dict(seconds=seconds,allocated=torch.npu.max_memory_allocated(),
                                               reserved=torch.npu.max_memory_reserved()))
        print('PROBE_RESULT '+json.dumps(dict(case=case,length=length,width=width,pairs=d.width,frames=frames,times=times,
            base_id=c['base_id'],policy_id=c['source_policy_id'])),flush=True)
        if check:
            witness.finish()
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    run(json.loads(Path(sys.argv[1]).read_bytes()),sys.argv[2],int(sys.argv[3]))
