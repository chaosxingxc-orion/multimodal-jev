"""torchrun entry. Uses the shared BF16 actor, immutable targets, no rollout service."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import require_training_storage


def read(path):
    return json.loads(Path(path).read_bytes())


def main(args):
    root, data = args.output, args.targets
    require_training_storage(root)
    if file_hash(args.parent) != args.parent_sha or file_hash(data/'MANIFEST.json') != args.target_sha:
        raise ValueError('parent config or target publication changed')
    c = read(args.parent)
    m = read(data/'MANIFEST.json')
    if any(file_hash(data/k) != v for k,v in m['files'].items()):
        raise ValueError('training target files changed')
    if (m['source_policy'] != c['source_policy_id'] or m['release_sha256'] != c['release_sha256'] or
            m['reference_sha256'] != c['reference_dataset']['manifest_sha256']):
        raise ValueError('target view belongs to another stage1')
    from multimodal_jev.span_runtime_profile import initialize_verl_preserving_qwen3,configure_actor_profile,SHORT_AUDIO_PROFILE
    initialize_verl_preserving_qwen3()
    configure_actor_profile(SHORT_AUDIO_PROFILE)
    from omegaconf import OmegaConf
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from multimodal_jev.context_tpo_actor import MODE, EXTENSION, pack
    from multimodal_jev.context_native_actor import build_actor
    from multimodal_jev.context_audio_collate import forward_context_audio_batch
    from multimodal_jev.context_tpo_pair_margin import validate_spec, verify_update
    from multimodal_jev.context_stage_lora import admit_stage2
    from multimodal_jev.context_fsdp_export import canonical_execution
    from multimodal_jev.context_checkpoint_policy import export_checkpoint_policy
    from multimodal_jev.cached_context_batches import CachedContextBatches
    from multimodal_jev.cached_context_preparation import CachedContextPreparer,processor_cache_identity
    from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache
    from multimodal_jev.context_reference_dataset import digest
    from multimodal_jev.span_request_identity import tensor_identity
    from multimodal_jev.context_lora_checkpoint_scope import validate_trainable_scope
    from tools.reports.stage2_view_actor import load_actor_stack
    from tools.reports.context_cached_reference_trainer import no_online_processor
    rank, world, local = (int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    e, schedule = c['view_execution'], m['schedule']
    local_size = c['batch_config']['per_dp_replica_batch']
    if (schedule['global_batch'] != world*local_size or schedule['microbatch'] != c['microbatch'] or
            local_size % schedule['microbatch']):
        raise ValueError('schedule geometry incompatible with DP')
    torch.set_num_threads(e['actor_threads'])
    torch.npu.set_device(local)
    torch.manual_seed(e['actor_seed'])
    torch.npu.manual_seed_all(e['actor_seed'])
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('hccl',timeout=timedelta(seconds=e['collective_timeout_seconds']))
    try:
        out = root/f'rank-{rank}'
        from multimodal_jev import window_recovery as recovery
        options = c.get('recovery') if args.window == 'formal' else None
        restart = recovery.restart_enabled(options)
        out.mkdir(parents=True,exist_ok=restart)
        transition = admit_stage2(c['stage_lora_transition'],c['source_contract'])
        base,processor,model,wrapped,opt,executions,handle = load_actor_stack(
            c,torch.device('npu',local),out,e,transition)
        validate_trainable_scope([(n,p) for n,p in wrapped.named_parameters() if p.requires_grad])
        choice = m['schema']=='jev-tpo-choices-v1'
        reward_only = choice or m['schema']=='jev-tpo-reward-view-v1'
        pair_spec = m.get('actor_pair_margin')
        if pair_spec is not None:
            if m['schema'] != 'jev-tpo-reward-view-v1' or not m.get('bidirectional_contrast'):
                raise ValueError('actor pair margin requires a bidirectional reward view')
            validate_spec(pair_spec['loss'])
        if reward_only and args.arm!='tpo':
            raise ValueError('reward-only view requires a TPO projection')
        opts = {} if reward_only else m['objective'].copy()
        if args.arm == 'direct':
            opts['projection'] = 0.
        cfg = OmegaConf.create(dict(use_remove_padding=False,use_fused_kernels=False,
            ulysses_sequence_parallel_size=1,use_dynamic_bsz=False,use_prefix_grouper=False,
            entropy_from_logits_with_chunking=False,use_torch_compile=False,fsdp_config={'dtype':'bfloat16'},
            calculate_sum_pi_squared=False,entropy_checkpointing=False,ppo_mini_batch_size=local_size,
            ppo_micro_batch_size_per_gpu=c['microbatch'],ppo_epochs=1,use_kl_loss=False,calculate_entropy=False,
            entropy_coeff=0.,loss_agg_mode='token-mean',use_rollout_log_probs=True,policy_loss={'loss_mode':MODE},
            global_batch_info={},grad_clip=e['grad_clip'],jev_policy_id=c['source_policy_id'],
            jev_context_parity=c['behavior_bounds'],jev_context_objective=opts,jev_reward_projection=reward_only,
            jev_choice=m.get('choice'),jev_probability_gap=m.get('probability_gap'),jev_actor_pair_margin=pair_spec['loss'] if pair_spec is not None else None))
        rows = read(data/'rows.json')
        targets = torch.load(data/'targets.pt',map_location='cpu',weights_only=True)
        card_binding = m.get('card_input')
        if card_binding:
            if (args.window == 'reference' or not reward_only or choice or
                    file_hash(Path(c['source_policy'])/'MANIFEST.json') != m['source_policy_sha256'] or
                    targets.shape != (schedule['epochs'], len(rows), len(m['candidate_ids']))):
                raise ValueError('card TPO policy, window or projected target differs')
            cards = read(data/'cards.json')
            if cards['labels'] != m['candidate_ids'] or m['files']['cards.json'] != card_binding['cards_sha256']:
                raise ValueError('card TPO candidate identity differs')
        else:
            release = Path(c['release'])
            if file_hash(release/'MANIFEST.json') != m['release_sha256']:
                raise ValueError('clean release changed')
            release_files = read(release/'MANIFEST.json')['files']
            for name in ('learning.jsonl','tasks.json'):
                if file_hash(release/name) != release_files[name]:
                    raise ValueError('source training inputs changed')
            task=read(release/'tasks.json')['intent']
            if [i['id'] for i in task['schema']['candidates']] != m['candidate_ids']:
                raise ValueError('candidate order changed')
            with (release/'learning.jsonl').open() as f:
                population={v['id']:v for line in f if (v:=json.loads(line))['id'] in rows}
        batches=read(data/'batches.json')
        bucket_plan, packed = None, None
        bucket_steps = None
        if args.optimizer_sha:
            from multimodal_jev.card_acoustic_pack import PackedAcoustics
            from multimodal_jev.card_training_acoustics import ENGINE
            from tools.reports.card_tpo_bucket_package import load
            if not card_binding or pair_spec or args.window not in ('finite', 'formal'):
                raise ValueError('optimizer bucketing requires immutable recording-wise card targets')
            bucket_plan = load(data.parent, args.optimizer_sha, args.target_sha, world, schedule['epochs'])
            if (bucket_plan['lengths'].keys() != rows.keys() or bucket_plan['acoustic_engine'] != ENGINE):
                raise ValueError('bucket plan inputs changed')
            acoustics = (model.backbone.audio_tower, model.backbone.multi_modal_projector)
            if any(p.requires_grad for part in acoustics for p in part.parameters()):
                raise ValueError('Stage2 acoustic reuse requires frozen acoustic weights')
            packed = PackedAcoustics(bucket_plan['cache'], rows, base.config.audio_token_id, allow_subset=True)
            bucket_steps = bucket_plan['updates']
            if args.window == 'finite':
                bucket_steps = [bucket_steps[i] for i in bucket_plan['finite']]
            batches = [dict(b, index=i) for i, b in enumerate(bucket_steps)]
        end = 2 if args.window == 'finite' else len(batches)
        denominators = [sum(not s['padding'] for s in b['slots']) for b in batches]
        epochs = [b['epoch'] for b in batches]
        if pair_spec is not None:
            pair_inputs = torch.load(data/'actor-pairs.pt', map_location='cpu', weights_only=True)
            rivals, enabled = pair_inputs['rival'], pair_inputs['enabled']
            if (rivals.shape != (len(rows),) or rivals.dtype != torch.long
                    or enabled.shape != (schedule['epochs'], len(rows)) or enabled.dtype != torch.bool
                    or not ((rivals >= 0) & (rivals < len(m['candidate_ids']))).all()
                    or any(int(rivals[row['index']]) == row['gold'] for row in rows.values())):
                raise ValueError('invalid frozen actor pair coordinates')
            if denominators != m['batch_real'] or epochs != pair_spec['epochs']:
                raise ValueError('frozen actor pair schedule changed')
        if bucket_plan:
            batches = [dict(b, slots=b['slots'][rank::world]) for b in batches]
        else:
            batches = [dict(b,slots=b['slots'][rank*local_size:(rank+1)*local_size]) for b in batches]
        ref_records, card_probs = None, None
        if card_binding:
            if ((not bucket_plan and (denominators != m['batch_real'] or len(batches) != m['updates']))
                    or (bucket_plan and denominators != [b['real'] for b in bucket_steps])):
                raise ValueError('card TPO schedule budget differs')
            if args.window == 'finite':
                card_probs = torch.load(data/'reference.pt', map_location='cpu', weights_only=True)
                if card_probs.shape != targets.shape[1:]:
                    raise ValueError('card TPO fixed reference geometry differs')
        else:
            prep=CachedContextPreparer(no_online_processor,ReadOnlyEncodedAudioCache(c['encoded_audio_cache']),
                processor_sha256=processor_cache_identity(c['source_policy']),decision_token_id=model.decision_token_id,
                audio_token_id=base.config.audio_token_id,max_tokens=e['max_input_tokens'],prompt_layout=c['prompt_layout'])
            if args.window == 'finite' and not choice:
                ref_root = Path(c['reference_dataset']['root'])
                if file_hash(ref_root/'MANIFEST.json') != m['reference_sha256']:
                    raise ValueError('parity reference differs')
                if file_hash(ref_root/'records.json') != read(ref_root/'MANIFEST.json')['files']['records.json']:
                    raise ValueError('parity records changed')
                ref_records=read(ref_root/'records.json')
        collate=dict(decision_token_id=model.decision_token_id,pad_token_id=processor.tokenizer.pad_token_id,
            audio_token_id=base.config.audio_token_id,audio_chunk_size=base.config.audio_config.n_window*2,
            audio_infer_window=base.config.audio_config.n_window_infer)
        dynamic=args.next_macro_sha is not None
        if dynamic:
            entry=data.parent
            if file_hash(entry/'next-macro.json')!=args.next_macro_sha:
                raise ValueError('reference policy config changed')
            spec=read(entry/'next-macro.json')
            if (not choice or spec['mode']!='next_macro' or spec['target_sha']!=args.target_sha
                    or spec['parent_sha']!=args.parent_sha or file_hash(entry/'reward-shift.pt')!=spec['reward_shift']):
                raise ValueError('dynamic reference incorrectly bound')
            shifts=torch.load(entry/'reward-shift.pt',map_location='cpu',weights_only=True)
        if choice:
            from multimodal_jev.context_choice_inputs import ChoiceInputs,identity as choice_identity
            from multimodal_jev.local_checkpoint import commit_file
            compiler=ChoiceInputs(processor,collate,e['max_input_tokens'])
            reference=Path('/jev-control/reference') if dynamic else root.parent/'reference'
            ref_file=reference/f'rank-{rank}'/'choice.pt'
            if args.window=='reference':
                reference_rows={}
            else:
                seal=read(reference/'receipt.json')
                if dynamic and seal!=spec['control_reference']:
                    raise ValueError('reference parity source changed')
                if (seal['target_sha']!=args.target_sha or seal['parent_sha']!=args.parent_sha
                        or file_hash(ref_file)!=seal['rank_files'][rank]):
                    raise ValueError('Choice reference binding differs')
                reference_rows=torch.load(ref_file,map_location='cpu',weights_only=True)
        start_step = 0
        contract = dict(stage='card-tpo', parent=args.parent_sha, targets=args.target_sha, arm=args.arm, end=end)
        if bucket_plan:
            contract['optimizer_sha'] = args.optimizer_sha
        named = [(n,p) for n,p in wrapped.named_parameters() if p.requires_grad]
        if options and not card_binding:
            raise ValueError('window recovery requires fixed card targets')
        if restart:
            recovered = recovery.latest(root, contract, 0, end, world)
            if recovered:
                start_step = recovered['step']
                recovery.recover(root,recovered,named,opt,dict(rank=rank,world_size=world,
                                 completed_updates=start_step,contract=contract))
            if rank == 0:
                paths = [root/f'update-{i:04d}.json' for i in range(start_step+1,end+1)]
                paths.extend(root/f'policy-step-{i}' for i in m.get('intermediate_steps',[]) if i > start_step)
                recovery.preserve_tail(root,[*paths,root/'policy',root/'result.json'])
            if start_step == 0:
                recovery.preserve_tail(out, [out/'initial-parity.json'])
            dist.barrier()
        if card_binding:
            from tools.reports.card_tpo_batches import CardTPOBatches
            batch_source = CardTPOBatches(batches, rows, cards, card_binding,
                                         c['card_caches'], e['prefetch'], collate, end, start_step, packed=packed, optimizer=bucket_plan)
            forward = batch_source.forward
        else:
            batch_source = CachedContextBatches(batches,population,task,prep,start_update=0,
                                                stop_update=end,global_batch=local_size)
            forward = forward_context_audio_batch
        with batch_source as stream:
            actor=build_actor(cfg,wrapped,opt,objective=EXTENSION,collation=collate,forward=forward)
            for step in range(start_step,end):
                now=time.monotonic()
                policy=f"{c['source_policy_id']}:{args.arm}:{step}"
                cfg.jev_policy_id=policy
                items,metadata=stream.consume(step,policy)
                if bucket_plan:
                    width = batches[step]['microbatch']
                    cfg.ppo_mini_batch_size = batches[step]['local_batch']
                    cfg.ppo_micro_batch_size_per_gpu = width
                    if len(items) != cfg.ppo_mini_batch_size or len(items) % width:
                        raise ValueError('rank does not match frozen bucket geometry')
                for x,meta,slot in zip(items,metadata,batches[step]['slots'],strict=True):
                    row=rows[meta['id']]
                    if not card_binding:
                        identity=dict(policy_id=c['source_policy_id'],positions=x['positions'][0].tolist(),
                                  inputs={k:tensor_identity(v) for k,v in x['inputs'].items()})
                        if (digest(identity)!=row['request_key'] or int(x['labels'].long().argmax())!=row['gold'] or
                                row['audio_sha256']!=population[meta['id']]['audio_sha256']):
                            raise ValueError('audio, reference binding or gold differs')
                    x.update(request_id=f"{rank}:{x['request_id']}",stage='tpo', gold=row['gold'],
                             paired=slot['paired'],confusion=slot['confusion'],group=row['group'],audio_sha256=row['audio_sha256'],
                             global_real=denominators[step],log_target=(targets[batches[step]['epoch'],row['index']] if reward_only else targets[row['index']]))
                    if 'probability_gap' in m:
                        x['reference_rival'] = row['competitor']
                    if pair_spec is not None:
                        x.update(pair_rival=int(rivals[row['index']]),
                                 pair_enabled=bool(enabled[epochs[step], row['index']]))
                prepared = time.monotonic()-now
                if choice:
                    mode=batches[step]['mode']
                    for x,meta,slot in zip(items,metadata,batches[step]['slots'],strict=True):
                        uid=meta['id']; row=rows[uid]
                        native=stream._request(population[uid]['audio'])
                        x['choice_gold']=compiler.apply(x,native,row,mode,c['source_policy_id'],indices=slot.get('slate'))
                        x['choice_mode']=mode
                    if dynamic or mode!='noul':
                        keys=[choice_identity(x,c['source_policy_id']) for x in items]
                        if dynamic or args.window=='reference' or (args.window=='finite' and step==0):
                            ref_start=time.monotonic()
                            reports=[]
                            with torch.no_grad():
                                for start in range(0,local_size,c['microbatch']):
                                    group=items[start:start+c['microbatch']]
                                    prepared=[{k:x[k] for k in ('policy_id','request_id','inputs','positions')} for x in group]
                                    z=forward_context_audio_batch(wrapped,prepared,policy_id=policy,
                                        device=torch.device('npu',local),dtype=torch.bfloat16,**collate).float().cpu()
                                    if not torch.isfinite(z).all(): raise ValueError('nonfinite choice reference')
                                    kk=keys[start:start+len(group)]
                                    if dynamic:
                                        for i,(x,v) in enumerate(zip(group,z.unbind(),strict=True)):
                                            if v.numel()!=x['positions'].numel(): raise ValueError('missing reference marker')
                                            x['macro_prior']=v.detach()
                                            x['log_target']=((v.double()+shifts[batches[step]['epoch'],rows[metadata[start+i]['id']]['index']]).float()
                                                             if mode=='noul' else v)
                                        if step==0 and not torch.equal(z,torch.stack([reference_rows[k] for k in kk])):
                                            raise ValueError('first macro differs from paired control logits')
                                    if args.window=='reference':
                                        reference_rows.update(zip(kk,z.unbind(),strict=True))
                                    elif step==0 and start<4*c['microbatch']:
                                        prior=torch.stack([reference_rows[k] for k in kk]).sigmoid()
                                        error=float((prior-z.sigmoid()).abs().max())
                                        flips=int(((z>=0)!=(prior>=.5)).sum())
                                        reports.append(dict(rows=len(group),max_abs=error,threshold_flips=flips,
                                            passed=error<=c['behavior_bounds']['probability_max_abs'] and flips==0))
                            if dynamic:
                                dist.barrier()
                                ref_seconds=time.monotonic()-ref_start
                                if step==0: del reference_rows
                            if args.window=='finite' and step==0:
                                commit_json(out/'initial-parity.json',reports)
                                if not all(r['passed'] for r in reports):raise ValueError('Choice reference parity failed')
                        if not dynamic and args.window!='reference':
                            for x,k in zip(items,keys,strict=True):x['log_target']=reference_rows[k]
                    if args.window=='reference':continue
                bucket_probe = bucket_plan and args.window == 'finite'
                if ((ref_records is not None or card_probs is not None) and step == 0) or bucket_probe:
                    if bucket_plan:
                        for part in acoustics:
                            part.eval()
                    reports=[]
                    with torch.no_grad():
                        for group in range(2 if bucket_probe else 4):
                            probe_micro = width if bucket_plan else c['microbatch']
                            start = (group % (len(items)//probe_micro))*probe_micro
                            sample=items[start:start+probe_micro]
                            inputs=[{k:x[k] for k in ('request_id','policy_id','inputs','positions')} for x in sample]
                            z=forward(wrapped,inputs,device=torch.device('npu',local),
                                      dtype=torch.bfloat16,policy_id=policy,**collate)
                            ids = [rows[v['id']] for v in metadata[start:start+len(sample)]]
                            if bucket_probe and step != 0:
                                # After the first update, compare to the native
                                # actor at the current weights, not Stage1 scores.
                                prior = stream.forward_native(wrapped, inputs,
                                    device=torch.device('npu',local), dtype=torch.bfloat16,
                                    policy_id=policy, **collate)
                                prob, actual = prior.sigmoid().float().cpu(), z.sigmoid().float().cpu()
                            elif card_binding:
                                prob=card_probs[[v['index'] for v in ids]]
                                actual=z.sigmoid().float().cpu()
                            else:
                                prob=torch.tensor([ref_records[v['request_key']]['prob'] for v in ids])
                                actual=z.float().sigmoid().cpu()
                            error=float((actual-prob).abs().max())
                            flips=int(((actual>=.5)!=(prob>=.5)).sum())
                            reports.append(dict(rows=len(prob),max_abs=error,threshold_flips=flips,
                                passed=error<=c['behavior_bounds']['probability_max_abs'] and flips==0))
                    if bucket_probe:
                        reports = [dict(r, step=step+1, comparison='frozen_stage1' if step == 0
                                        else 'current_native_actor') for r in reports]
                    receipt_name = f'parity-step-{step+1:04d}.json' if bucket_probe else 'initial-parity.json'
                    commit_json(out/receipt_name, reports)
                    parity_passed = all(r['passed'] for r in reports)
                    if card_binding:
                        reports_by_rank = [None]*world
                        dist.all_gather_object(reports_by_rank, parity_passed)
                        parity_passed = all(reports_by_rank)
                    if not parity_passed:
                        raise ValueError(f'producer vs actor discrepancy; see {receipt_name}')
                    ref_records = card_probs = None
                if bucket_plan:
                    # veRL puts the whole model in train mode at update entry.
                    def freeze_acoustics(*_):
                        for part in acoustics:
                            if part.training:
                                part.eval()
                    acoustic_hook = model.register_forward_pre_hook(freeze_acoustics)
                    torch.npu.reset_peak_memory_stats()
                begin_actor = time.monotonic()
                try:
                    metrics=actor.update_policy(pack(items))
                    torch.npu.synchronize()
                finally:
                    if bucket_plan:
                        acoustic_hook.remove()
                runtime = None
                if bucket_plan:
                    runtime = dict(input_wait_seconds=prepared,
                                   train_seconds=time.monotonic()-begin_actor,
                                   peak_bytes=torch.npu.max_memory_allocated())
                norms=metrics['actor/grad_norm']
                if not all(math.isfinite(v) and v>0 for v in norms):
                    raise ValueError('invalid optimizer gradient norm')
                if dynamic: metrics['jev/reference_seconds']=[ref_seconds]
                gathered=[None]*world
                dist.all_gather_object(gathered, (metrics, runtime))
                if rank==0:
                    gathered, runtimes = zip(*gathered, strict=True)
                    count_prefix = 'jev/pair_count/'
                    names = ['actor/pg_loss', 'actor/grad_norm'] + [
                        k for k in metrics if k.startswith('jev/') and not k.startswith((count_prefix, 'jev/gap_count/'))]
                    reduced={k:sum(float(torch.as_tensor(r[k]).mean()) for r in gathered)/world for k in names}
                    if bucket_plan:
                        # Each TPO microbatch metric includes its accumulation scale.
                        accumulation = batches[step]['local_batch']//width
                        for key in names:
                            if key.startswith('jev/') and not key.endswith('_seconds'):
                                if any(len(r[key]) != accumulation for r in gathered):
                                    raise ValueError('unexpected Stage2 metric accumulation')
                    if not all(math.isfinite(v) for v in reduced.values()):
                        raise ValueError('nonfinite reduced metrics')
                    extra={}
                    if dynamic:
                        if step==0:
                            first=read(Path('/jev-control/tpo/update-0001.json'))
                            for k in ('jev/tpo_loss','actor/grad_norm'):
                                if not math.isclose(reduced[k],first['metrics'][k],rel_tol=1e-5,abs_tol=1e-7):
                                    raise ValueError('first update differs from fixed control: '+k)
                        extra['reference']=dict(mode='next_macro',policy_update=step,macro_slots=local_size*world,
                            seconds_max=max(r['jev/reference_seconds'][0] for r in gathered),config_sha256=args.next_macro_sha)
                    update = dict(step=step+1,seconds=time.monotonic()-now,
                        real=denominators[step],task=batches[step].get('mode','noul'),metrics=reduced,**extra)
                    if bucket_plan:
                        update['bucket'] = {k: batches[step][k] for k in ('bucket','local_batch','microbatch','max_tokens','epoch_end')}
                        update['optimizer_sha'] = args.optimizer_sha
                        update['epoch'] = epochs[step]
                        update['peak_bytes'] = max(r['peak_bytes'] for r in runtimes)
                        update['timing'] = {k: max(r[k] for r in runtimes)
                                            for k in ('input_wait_seconds', 'train_seconds')}
                    if pair_spec is not None:
                        update['epoch'] = epochs[step]
                        update['pair_margin'] = {
                            key.removeprefix(count_prefix): int(sum(sum(r[key]) for r in gathered))
                            for key in metrics if key.startswith(count_prefix)
                        }
                        verify_update(update, m['batch_real'], epochs, pair_spec['batch_eligible'])
                    if 'probability_gap' in m:
                        from multimodal_jev.context_tpo_probability_gap import verify_gap_update
                        update['epoch'] = epochs[step]
                        update['gap_counts'] = {k.split('/')[-1]: int(sum(sum(r[k]) for r in gathered))
                                                for k in metrics if k.startswith('jev/gap_count/')}
                        verify_gap_update(update, step+1, denominators)
                    commit_json(root/f'update-{step+1:04d}.json', update)
                if args.window == 'formal' and step+1 in m.get('intermediate_steps', []):
                    with FSDP.summon_full_params(wrapped,writeback=False,rank0_only=False):
                        if rank == 0:
                            state = canonical_execution({n:p.detach().cpu().to(torch.bfloat16).clone()
                                for n,p in model.named_parameters() if p.requires_grad})
                            export_checkpoint_policy(c['source_policy'], root/f'policy-step-{step+1}', state,
                                policy_id=c['source_policy_id'], contract=c['source_contract'], base_id=c['base_id'])
                            del state
                    dist.barrier()
                if options and step+1 < end and (step+1) % options['interval'] == 0:
                    checkpoint_begin = time.monotonic()
                    recovery.save(root,named,opt,dict(rank=rank,world_size=world,completed_updates=step+1,
                                  contract=contract),{},dist)
                    from multimodal_jev.baseline import atomic_json
                    atomic_json(out/f'recovery-timing-{step+1:04d}.json',dict(seconds=time.monotonic()-checkpoint_begin))
        if args.window=='reference':
            commit_file(ref_file,lambda f:torch.save(reference_rows,f))
            dist.barrier()
            return
        with FSDP.summon_full_params(wrapped,writeback=False,rank0_only=False):
            if rank==0:
                full=canonical_execution({n:p.detach().cpu().to(torch.bfloat16).clone()
                    for n,p in model.named_parameters() if p.requires_grad})
                joint=export_checkpoint_policy(c['source_policy'],root/'policy',full,policy_id=c['source_policy_id'],
                    contract=c['source_contract'],base_id=c['base_id'])
                commit_json(root/'result.json',dict(arm=args.arm,window=args.window,updates=end,policy=joint['policy_id'],
                    target_manifest_sha256=args.target_sha,parent_config_sha256=args.parent_sha))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('parent','targets','output'):
        p.add_argument('--'+key,type=Path,required=True)
    for key in ('parent-sha','target-sha'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--next-macro-sha')
    p.add_argument('--optimizer-sha')
    p.add_argument('--arm',choices=['direct','tpo'],required=True)
    p.add_argument('--window',choices=['reference','finite','formal'],required=True)
    main(p.parse_args())
