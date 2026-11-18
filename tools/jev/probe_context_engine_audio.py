#!/usr/bin/env python3
"""Two-process real-audio HF/vLLM context diagnostic, no optimization.

Run hf then vllm in separate processes so the HF allocation is gone before
engine startup. Retain raw probability differences; no predeclared old-profile
parity tolerance is reused for this new execution path.
"""
import argparse
import copy
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def run(config,output,phase):
    import torch
    import torch_npu
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.context_candidate_task import compile_candidates
    from multimodal_jev.context_decision_contract import encode_context_request
    from multimodal_jev.context_policy_artifacts import read_execution_bundle, export_native_adapter
    from multimodal_jev.context_policy_snapshot import load_policy,export_policy
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.span_request_identity import tensor_identity
    from multimodal_jev.local_runtime_storage import require_training_storage
    require_training_storage(output)
    if (len(config['physical_devices']) != 1 or sorted(p.name for p in Path('/dev').glob('davinci[0-9]*')) !=
            ['davinci'+str(config['physical_devices'][0])]):
        raise ValueError('one isolated NPU required')
    for path,digest in config['pins'].items():
        if file_hash(path) != digest: raise ValueError('dependency changed: '+path)
    if checkpoint_identity(config['model'])[0] != config['base_id']:
        raise ValueError('base identity differs')
    torch.set_num_threads(2); torch.npu.set_device(0); torch.manual_seed(317); torch.npu.manual_seed_all(317)
    device = torch.device('npu',0)
    source = read_execution_bundle(config['source_policy'],expected_policy_id=config['source_policy_id'],
        expected_contract=config['source_contract'],expected_base_id=config['base_id'])
    processor = Qwen3ASRProcessor.from_pretrained(source['processor'],local_files_only=True)
    task = json.loads(Path(config['tasks']).read_text())['intent']
    candidates = [dict(id=x['id'],definition=x['description']) for x in task['schema']['candidates']]
    if len(candidates) != 60: raise ValueError('complete candidate menu required')
    rows = []
    for asset in config['audio']:
        if file_hash(asset['audio']) != asset['sha256']: raise ValueError('audio changed')
        request = compile_candidates(asset['audio'],candidates,task['thinking']['requirements'],semantics='exclusive_event')
        wave = load_audio(asset['audio'])
        inputs,positions = encode_context_request(processor,request,wave,
            decision_token_id=source['contract']['token']['id'],audio_token_id=config['audio_token_id'],max_tokens=4096)
        # Bind the exact BF16 feature representation sent to both implementations.
        inputs = {k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in inputs.items()}
        rows.append((asset,inputs,positions))
    records = []
    if phase in ('hf', 'hf_capture', 'hf_batch'):
        from multimodal_jev.asr import load_runtime
        from multimodal_jev.sft import initialize_adapter
        from multimodal_jev.context_decision_head import ContextDecisionHead
        from multimodal_jev.context_decision_model import ContextDecisionModel
        _,base = load_runtime(config['model'],device,'bfloat16')
        adapted = initialize_adapter(base,dict(lora_rank=4,lora_alpha=8,lora_execution_dtype='bfloat16')).eval()
        head = ContextDecisionHead(1024,8,1024,.1).to(device=device,dtype=torch.bfloat16).eval()
        model = ContextDecisionModel(adapted.get_base_model().model,head,source['contract']['token']['id']).eval()
        named_runtime = config.get('context_runtime_profile')
        if named_runtime is not None:
            from multimodal_jev.context_runtime_profile import install_actor
            candidate_handle, candidate_profile = install_actor(adapted,model,named_runtime)
            atomic_json(output/'context-actor-runtime.json',candidate_profile)
        if phase in ('hf_capture', 'hf_batch'):
            joint = json.loads((output/'joint.json').read_text())
            load_policy(output/'policy',model,expected_policy_id=joint['policy_id'],
                expected_contract=joint['contract'],expected_base_id=config['base_id'])
            fresh_references = config.get('reference_from_current_inputs') is True
            references = ([dict(id=asset['id'],positions=positions.tolist(),
                inputs={k:tensor_identity(v) for k,v in inputs.items()}) for asset,inputs,positions in rows]
                if fresh_references else json.loads((output/'hf.json').read_text()))
            if phase == 'hf_capture' and config.get('diagnostic_packed_alignment') is True:
                from multimodal_jev.span_batch_candidate_runtime import install_candidate
                from multimodal_jev.context_row_head import install_row_head
                candidate_handle,candidate_profile=install_candidate(adapted,model)
                candidate_profile['head']=install_row_head(model.decision_head)
                atomic_json(output/'hf-packed-runtime.json',candidate_profile)
                packed_references=[]
            if phase == 'hf_batch':
                if config.get('batch_runtime') == 'fixed_projection_packed_attention':
                    from multimodal_jev.span_batch_candidate_runtime import install_candidate
                    candidate_handle,candidate_profile=install_candidate(adapted,model)
                    atomic_json(output/'batch-runtime.json',candidate_profile)
                elif config.get('batch_runtime', 'eager') != 'eager':
                    raise ValueError('explicit supported batch runtime required')
                from multimodal_jev.context_audio_collate import forward_context_audio_batch
                from multimodal_jev.local_checkpoint import commit_file
                items = [dict(request_id=asset['id'],policy_id=joint['policy_id'],inputs=inputs,
                              positions=positions) for asset,inputs,positions in rows]
                options = dict(device=device,dtype=torch.bfloat16,decision_token_id=model.decision_token_id,
                    policy_id=joint['policy_id'],pad_token_id=processor.tokenizer.pad_token_id,
                    audio_token_id=config['audio_token_id'],audio_chunk_size=base.config.audio_config.n_window*2,
                    audio_infer_window=base.config.audio_config.n_window_infer)
                named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
                coefficients = torch.linspace(-1,1,60,device=device)[None]/120
                modes = ('serial','batch','batch_row_head') if config.get('isolate_head_batch') else ('serial','batch')
                original_score = model.decision_head.score_logits
                def row_score(hidden,positions,valid):
                    return torch.cat([original_score(hidden[i:i+1,:int(valid[i].sum())],
                        positions[i:i+1],valid[i:i+1,:int(valid[i].sum())]) for i in range(len(hidden))])
                for mode in modes:
                    model.decision_head.score_logits = row_score if mode == 'batch_row_head' else original_score
                    model.zero_grad(set_to_none=True)
                    torch.npu.reset_peak_memory_stats();torch.npu.synchronize();start=time.monotonic()
                    values = []
                    groups = [[item] for item in items] if mode == 'serial' else [items]
                    for group_index,group in enumerate(groups):
                        atomic_json(output/'progress.json',dict(phase='hf_batch',mode=mode,group=group_index))
                        logits = forward_context_audio_batch(model,group,**options)
                        values.append(logits.detach().cpu())
                        (logits*coefficients).sum().backward()
                        del logits
                    torch.npu.synchronize()
                    gradients = {n:p.grad.detach().cpu() for n,p in named if p.grad is not None}
                    if len(gradients)!=len(named) or not all(torch.isfinite(v).all() for v in gradients.values()):
                        raise ValueError('complete finite real-batch gradients required')
                    snapshot = dict(logits=torch.cat(values),gradients=gradients,policy_id=joint['policy_id'],
                        inputs=[{k:tensor_identity(v) for k,v in inputs.items()} for _,inputs,_ in rows],
                        positions=[positions.tolist() for _,_,positions in rows])
                    path=output/f'hf-{mode}-batch.pt'
                    commit_file(path,lambda stream:torch.save(snapshot,stream))
                    records.append(dict(mode=mode,seconds=time.monotonic()-start,
                        allocated=torch.npu.max_memory_allocated(),reserved=torch.npu.max_memory_reserved(),
                        snapshot_sha256=file_hash(path)))
                model.zero_grad(set_to_none=True)
                model.decision_head.score_logits = original_score
                atomic_json(output/'hf-batch.json',dict(records=records,optimizer_updates=0,
                    scope='Two real audio requests, serial/batch gradients and probabilities on identical weights. No throughput or parity admission.'))
                return
            captured = {}
            def capture_hidden(module, args, result):
                captured['hidden'] = result.last_hidden_state.detach().cpu()
            handle = (model.backbone.register_forward_hook(capture_hidden)
                      if config.get('capture_hidden') is True else None)
            from multimodal_jev.local_checkpoint import commit_file
            for index, ((asset, inputs, positions), ref) in enumerate(zip(rows,references,strict=True)):
                if (ref['id'] != asset['id'] or ref['positions'] != positions.tolist() or
                        ref['inputs'] != {k:tensor_identity(v) for k,v in inputs.items()}):
                    raise ValueError('capture reference input differs')
                with torch.no_grad():
                    prob = model({k:v.to(device) for k,v in inputs.items()},positions.to(device)).cpu()
                if handle is not None:
                    value = dict(hidden=captured['hidden'],positions=positions,inputs=inputs,
                        probability=prob,policy_id=joint['policy_id'],
                        head={k:v.detach().cpu() for k,v in model.decision_head.state_dict().items()})
                    commit_file(output/f'hf-hidden-{index}.pt',lambda stream:torch.save(value,stream))
                records.append(dict(id=asset['id'],reference_error=(None if fresh_references else
                    float((prob[0]-torch.tensor(ref['prob'])).abs().max()))))
                if config.get('diagnostic_packed_alignment') is True:
                    packed_references.append(dict(ref,prob=prob[0].tolist()))
                if (index+1)%16==0:
                    atomic_json(output/'progress.json',dict(phase=phase,completed=index+1,total=len(rows)))
            if handle is not None:
                handle.remove()
            if config.get('diagnostic_packed_alignment') is True:
                atomic_json(output/'hf-packed.json',packed_references)
                if config.get('parity_microbatch'):
                    from multimodal_jev.context_audio_collate import forward_context_audio_batch
                    micro=config['parity_microbatch']
                    if type(micro) is not int or micro<1:raise ValueError('positive parity microbatch required')
                    options=dict(device=device,dtype=torch.bfloat16,decision_token_id=model.decision_token_id,
                        policy_id=joint['policy_id'],pad_token_id=processor.tokenizer.pad_token_id,
                        audio_token_id=config['audio_token_id'],audio_chunk_size=base.config.audio_config.n_window*2,
                        audio_infer_window=base.config.audio_config.n_window_infer)
                    batched=[]
                    for start in range(0,len(rows),micro):
                        items=[dict(request_id=a['id'],policy_id=joint['policy_id'],inputs=x,positions=p)
                               for a,x,p in rows[start:start+micro]]
                        with torch.no_grad():
                            probabilities=forward_context_audio_batch(model,items,**options).sigmoid().cpu()
                        for item,prob,ref in zip(items,probabilities,packed_references[start:start+micro],strict=True):
                            batched.append(dict(id=item['request_id'],prob=prob.tolist(),
                                serial_probability_max_error=float((prob-torch.tensor(ref['prob'])).abs().max())))
                    atomic_json(output/'hf-packed-batch.json',dict(microbatch=micro,records=batched))
            atomic_json(output/'hf-capture.json',records)
            return
        load_policy(config['source_policy'],model,expected_policy_id=source['policy_id'],
            expected_contract=source['contract'],expected_base_id=config['base_id'])
        if config.get('assembled_execution') is not None:
            from multimodal_jev.context_fsdp_export import load_assembled_execution
            path=config['assembled_execution']
            if config['pins'].get(path)!=file_hash(path):
                raise ValueError('pinned assembled execution required')
            if config.get('diagnostic_update') is True:
                raise ValueError('assembled policy cannot receive another diagnostic update')
            receipt=load_assembled_execution(model,torch.load(path,map_location='cpu',weights_only=True))
            atomic_json(output/'assembled-execution-load.json',dict(receipt,sha256=file_hash(path)))
        if config.get('diagnostic_update') is True:
            from multimodal_jev.bf16_lora import BF16MasterAdamW
            named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
            before = {n:p.detach().clone() for n,p in named}
            optimizer = BF16MasterAdamW([p for _,p in named],lr=1e-4)
            _,batch,positions = rows[0]
            batch = {k:v.to(device) for k,v in batch.items()}
            positions = positions.to(device)
            logits = model.score_logits(batch,positions)
            prior_prob = logits.detach().sigmoid().cpu()
            # Diagnostic functional gives all decision spans signal without
            # labels or invented task reward. This is not a learning result.
            coefficients = torch.linspace(-1,1,logits.shape[1],device=device)[None]/logits.shape[1]
            diagnostic_loss = (logits*coefficients).sum()
            diagnostic_loss.backward()
            gradients = {n:float(p.grad.float().norm()) for n,p in named if p.grad is not None}
            if (len(gradients) != len(named) or not all(torch.isfinite(p.grad).all() for _,p in named) or
                    not any(v > 0 for n,v in gradients.items() if '.lora_B.' in n)):
                raise ValueError('finite head/backbone diagnostic gradients required')
            torch.nn.utils.clip_grad_norm_([p for _,p in named],1.)
            optimizer.step(); optimizer.zero_grad()
            changed = {n:float((p.detach().float()-before[n].float()).abs().max()) for n,p in named}
            if (not any(v > 0 for n,v in changed.items() if '.lora_B.' in n) or
                    not any(v > 0 for n,v in changed.items() if n.startswith('decision_head.'))):
                raise ValueError('actual BF16 LoRA B and branch update required')
            state = optimizer.state_dict()
            checkpoint = dict(names=[n for n,_ in named],source_policy_id=source['policy_id'],
                execution={n:p.detach().cpu().clone() for n,p in named},master_adam=state)
            from multimodal_jev.local_checkpoint import commit_file
            commit_file(output/'diagnostic-update.pt',lambda stream:torch.save(checkpoint,stream))
            with torch.no_grad():
                updated = model(batch,positions).cpu()
                for _,p in named: p.add_(.125)
            restored = torch.load(output/'diagnostic-update.pt',map_location='cpu',weights_only=True)
            if (restored['names'] != [n for n,_ in named] or
                    restored['source_policy_id'] != source['policy_id']):
                raise ValueError('diagnostic checkpoint parameter identity differs')
            with torch.no_grad():
                for n,p in named: p.copy_(restored['execution'][n])
            recovered_optimizer = BF16MasterAdamW([p for _,p in named],lr=1e-4)
            recovered_optimizer.load_state_dict(restored['master_adam'])
            recovered_state = recovered_optimizer.state_dict()
            if any(not torch.equal(a,b) for a,b in zip(state['masters'],recovered_state['masters'],strict=True)):
                raise ValueError('FP32 masters not restored exactly')
            for key,values in state['optimizer']['state'].items():
                for field,value in values.items():
                    got = recovered_state['optimizer']['state'][key][field]
                    if isinstance(value,torch.Tensor) and not torch.equal(value.cpu(),got.cpu()):
                        raise ValueError('Adam tensor not restored exactly')
            with torch.no_grad(): recovered_prob = model(batch,positions).cpu()
            atomic_json(output/'diagnostic-update.json',dict(gradients=gradients,changed_parameters=changed,
                optimizer=recovered_optimizer.receipt(),checkpoint_sha256=file_hash(output/'diagnostic-update.pt'),
                probability_change=float((updated-prior_prob).abs().max()),
                reload_probability_max_error=float((updated-recovered_prob).abs().max()),
                master_adam_exact=True,training_dispatch_allowed=False,
                scope='One real-audio diagnostic functional update; no label/task reward, RL or task gain. '
                      'Actual BF16 LoRA/head change plus FP32 master/Adam restore; no DP recovery proof.'))
            del before,state,restored,checkpoint,recovered_state,optimizer,recovered_optimizer,logits,diagnostic_loss
        contract = copy.deepcopy(source['contract'])
        contract['runtime_profile'] = named_runtime or 'context-vllm-native-bf16-diagnostic-v1'
        if config.get('bind_exclusive_event_contract') is True:
            from multimodal_jev.context_token_contract import validate_processor_extension
            atomic_json(output/'token-extension-audit.json',validate_processor_extension(config['model'],source['processor']))
            if named_runtime!='qwen3-asr-context-bf16-packed-row-v1':
                raise ValueError('explicit named runtime for exclusive-event export required')
            contract['request_version']='jev-context-exclusive_event-v1'
            # This run proves the corrected input contract. Production admission
            # is a separate evidence record after both backends finish.
            contract['token']['registration_status']='test_only'
        identity = export_policy(output/'policy',model,source['processor'],contract)
        bundle = read_execution_bundle(output/'policy',expected_policy_id=identity,
            expected_contract=contract,expected_base_id=config['base_id'])
        export_native_adapter(output/'adapter',bundle)
        atomic_json(output/'joint.json',dict(policy_id=identity,contract=contract))
        for index,(asset,inputs,positions) in enumerate(rows):
            atomic_json(output/'progress.json',dict(phase=phase,index=index))
            torch.npu.synchronize(); start = time.monotonic()
            with torch.no_grad():
                prob = model({k:v.to(device) for k,v in inputs.items()},positions.to(device)).cpu()
            torch.npu.synchronize()
            records.append(dict(id=asset['id'],prob=prob[0].tolist(),positions=positions.tolist(),
                inputs={k:tensor_identity(v) for k,v in inputs.items()},seconds=time.monotonic()-start))
        atomic_json(output/'hf.json',records)
    else:
        from multimodal_jev.context_resident_rollout import FixedContextRollout
        joint = json.loads((output/'joint.json').read_text())
        refs = json.loads((output/('hf-packed.json' if config.get('diagnostic_packed_alignment') else 'hf.json')).read_text())
        cfg = dict(model=config['model'],base_id=config['base_id'],snapshot=str(output/'policy'),
            policy_id=joint['policy_id'],contract=joint['contract'],adapter=str(output/'adapter'),
            physical_devices=config['physical_devices'],tensor_parallel_size=1,max_model_len=4096,max_decisions=60)
        engine = FixedContextRollout(cfg)
        try:
            atomic_json(output/'context-rollout-runtime.json',engine.runtime)
            if config.get('diagnostic_backbone_alignment') is not None:
                from multimodal_jev.span_runtime_profile import install_rollout_profile
                atomic_json(output/'diagnostic-backbone-alignment.json',
                    install_rollout_profile(engine.engine,config['diagnostic_backbone_alignment']))
            if config.get('diagnostic_packed_alignment') is True:
                atomic_json(output/'vllm-packed-runtime.json',dict(
                    attention=engine.engine.llm_engine.collective_rpc('align_jev_candidate_packed_attention'),
                    projection=engine.engine.llm_engine.collective_rpc('align_jev_candidate_fixed_base')))
            for index,((asset,inputs,positions),ref) in enumerate(zip(rows,refs,strict=True)):
                if (ref['id'] != asset['id'] or ref['inputs'] != {k:tensor_identity(v) for k,v in inputs.items()} or
                        ref['positions'] != positions.tolist()):
                    raise ValueError('HF input differs')
                atomic_json(output/'progress.json',dict(phase=phase,index=index))
                if config.get('capture_hidden') is True:
                    engine.engine.llm_engine.collective_rpc('capture_next_jev_context_readout',
                        args=(joint['policy_id'],str(output/f'vllm-hidden-{index}.pt')))
                start = time.monotonic()
                row = engine.score(inputs,positions,audio_token_id=config['audio_token_id'])
                row.update(id=asset['id'],seconds=time.monotonic()-start,
                    probability_max_error=float((torch.tensor(row['prob'])-torch.tensor(ref['prob'])).abs().max()))
                if config.get('capture_hidden') is True:
                    row['hf_hidden_replay'] = engine.engine.llm_engine.collective_rpc(
                        'replay_jev_context_hidden',args=(joint['policy_id'],str(output/f'hf-hidden-{index}.pt')))
                records.append(row)
            atomic_json(output/'vllm.json',dict(records=records,startup=engine.startup,
                parity_admitted=False,training_dispatch_allowed=False,
                scope='Actual fixed-policy native HF/vLLM outputs; no tolerance or old-profile acceptance reused.'))
        finally:
            engine.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['hf','hf_capture','hf_batch','vllm'],required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=a.phase in ('hf_capture','hf_batch','vllm'))
    try:
        run(json.loads(a.config.read_text()),a.output,a.phase)
        atomic_json(a.output/(a.phase+'-terminal.json'),dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/(a.phase+'-terminal.json'),dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
