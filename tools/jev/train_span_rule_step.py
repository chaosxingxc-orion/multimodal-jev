#!/usr/bin/env python3
"""One matched rule SFT or verl update after full probe/runtime admission.

This exports a pending policy, never READY. Independently reload in HF and
vLLM and gate the complete probe before consuming this policy again.
"""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def run(config, output):
    import torch
    import torch_npu
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.sft import initialize_adapter
    from multimodal_jev.span_model import NativeAudioSpanModel
    from multimodal_jev.span_contract import SpanRequest, SpanQuestion, encode_span_request, span_labels
    from multimodal_jev.task_schema import DecisionSchema
    from multimodal_jev.span_request_identity import bind_span_request
    from multimodal_jev.span_runtime_profile import configure_actor_profile, initialize_verl_preserving_qwen3
    from multimodal_jev.span_actor_update import record_rollout, admit_actor, update_span_actor
    from multimodal_jev.span_rule_objectives import rule_sft_loss, rule_sample_rewards, expected_rule_reward
    from multimodal_jev.bf16_lora import BF16MasterAdamW
    from multimodal_jev.span_checkpoint import save_span_checkpoint, restore_span_optimizer
    from multimodal_jev.policy_identity import checkpoint_identity
    from infer_span_rule_baseline import select_rows
    from verify_span_rule_runtime import verify

    from multimodal_jev.span_runtime_topology import runtime_topology
    topology=runtime_topology({**config,'physical_devices':config.get('physical_devices',[5])})
    if topology['size']!=1 or sorted(p.name for p in Path('/dev').glob('davinci[0-9]*')) != topology['devices']:
        raise ValueError('one isolated authorized physical NPU required')
    transfer = config.get('population') == 'transfer-version-gate-v1'
    if not transfer:
        if (config['method'] not in ('sft', 'rl', 'conditional') or config['optimizer_start'] != 'fresh'
                or config['num_updates'] != 1 or config['seed'] != 17):
            raise ValueError('fixed one-update matched pilot required')
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('pinned pilot asset changed: '+path)
    admission_root = Path(config['runtime_admission'])
    verified = verify(admission_root)
    if not verified['passed'] or (not transfer and verified['requests'] != 216):
        raise ValueError('complete rule probe runtime gate required before either method')
    base_id = checkpoint_identity(config['model'])[0]
    adapter_id = checkpoint_identity(config['adapter'])[0]
    if verified['base_id'] != base_id or verified['adapter_id'] != adapter_id:
        raise ValueError('runtime acceptance is for another policy')
    admission_config = json.loads((admission_root/'config.json').read_text())
    if admission_config['release'] != config['release']:
        raise ValueError('runtime acceptance is for another release')
    atomic_json(output/'admission.json', verified)
    release = Path(config['release'])
    # Only a scoped, pinned protocol may use these development controls. The
    # immutable pool manifest remains unadmitted for arbitrary training/eval.
    if transfer:
        from multimodal_jev.span_transfer_runtime import branch_protocol, step_rows, cursor, validate_parent
        protocol = branch_protocol(config)
        if (config['num_updates'] != 1 or config['adapter_id'] != adapter_id
                or admission_config.get('version_index') != config['batch_index']):
            raise ValueError('transfer update or runtime version differs')
        for key in ('transfer_protocol','seed','arm','method','auxiliary_weight'):
            if admission_config.get(key) != config[key]:
                raise ValueError('runtime branch differs')
        if config['batch_index'] == 0 and base_id != protocol['initializer']['base_id']:
            raise ValueError('initializer base differs')
        selected = step_rows(config)
    else:
        protocol = json.loads(Path(config['protocol']).read_text())
        if config['pins'].get(config['protocol']) != file_hash(config['protocol']):
            raise ValueError('pinned scoped pilot protocol required')
        protocol_schema = ('jev-span-rule-conditional-pilot-v1' if config['method'] == 'conditional'
                           else 'jev-span-rule-matched-pilot-v1')
        if (protocol['schema'] != protocol_schema
                or protocol['release_manifest_sha256'] != file_hash(release/'MANIFEST.json')
                or protocol['initial_adapter_id'] != adapter_id
                or protocol['request_ids'] != config['request_ids']
                or protocol['runtime_admission'] != config['runtime_admission']
                or protocol['num_updates'] != 1 or protocol['optimizer_start'] != 'fresh'
                or protocol['arm'] != config['arm'] or config['lr'] != protocol['lr']
                or protocol['seed'] != config['seed']
                or config['samples_per_request'] != protocol['samples_per_request']
                or [config['method'], config['auxiliary_weight']] not in protocol['objectives']):
            raise ValueError('run differs from matched protocol')
        if protocol['data_audit']['passed'] is not True:
            raise ValueError('reviewed rule provenance audit required')
        if config['method'] == 'conditional':
            if (protocol.get('algorithm') != 'exact-conditional-marginal-ppo-v1'
                    or protocol.get('clip_ratio') != 0.2
                    or protocol.get('advantage_standardization') != 'none'):
                raise ValueError('explicit conditional-credit algorithm contract required')
        rows = {r['id']:r for r in select_rows(release)}
        batch = json.loads((release/'schedule.json').read_text())['batches'][0]
        if config['request_ids'] != batch['arms'][config['arm']]:
            raise ValueError('pilot must use the fixed first balanced batch')
        selected = [rows[k] for k in config['request_ids']]
    width = len(selected[0]['targets'])
    if (len(selected) != 8 or len({r['provenance']['source_group'] for r in selected}) != 4
            or any(r['split'] != 'learning' or not all(r['supervised']) for r in selected)
            or any(sum(r['targets'][j] is True for r in selected) != 4 for j in range(width))):
        raise ValueError('reviewed per-span balanced learning batch required')
    runtime = json.loads((admission_root/'run/result.json').read_text())
    references = {r['id']:r for r in runtime['records']}
    torch.set_num_threads(4); torch.npu.set_device(0)
    torch.manual_seed(config['seed']); torch.npu.manual_seed_all(config['seed'])
    # Both arms have the same import/environment and restore the accepted HF
    # equations before model construction, even when only SFT uses autograd.
    atomic_json(output/'verl-initialization.json', initialize_verl_preserving_qwen3())
    atomic_json(output/'runtime-profile.json', configure_actor_profile(config['runtime_profile']))
    processor, base = load_runtime(config['model'], 'npu:0', 'bfloat16')
    adapted = initialize_adapter(base, dict(initial_adapter=config['adapter'], lora_execution_dtype='bfloat16')).eval()
    model = NativeAudioSpanModel(adapted.get_base_model()).eval()
    prepared = []; ids = []; token_count = 0; audio_seconds = 0.
    policy_id = base_id+':'+adapter_id
    for row in selected:
        raw = row['request']
        request = SpanRequest(raw['audio'], tuple(SpanQuestion(q['id'], DecisionSchema.from_dict(q['schema']))
            for q in raw['questions']), raw['final_id'], tuple(raw['requirements']), raw['version'])
        if file_hash(request.audio) != row['provenance']['audio_sha256']:
            raise ValueError('audio changed')
        audio, receipt = load_audio(request.audio, return_receipt=True)
        inputs, positions = encode_span_request(processor, request, audio)
        mapping = span_labels(processor.tokenizer, request.questions)
        if any(m['values'] != (False, True) for m in mapping):
            raise ValueError('fixed Boolean support required')
        candidates = [m['token_ids'] for m in mapping]
        uid, identity = bind_span_request(request, receipt, inputs, positions, candidates, policy_id)
        ref = references[row['id']]
        if ref['request_id'] != uid or ref['request_identity'] != identity:
            raise ValueError('current input differs from admitted rollout')
        prepared.append((inputs, positions, candidates)); ids.append(uid)
        token_count += inputs['input_ids'].numel(); audio_seconds += len(audio)/16000

    def forward_one(i):
        inputs, positions, candidates = prepared[i]
        device_inputs = {k:v.to('npu:0', dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                         for k,v in inputs.items()}
        return model(device_inputs, positions.to('npu:0'), candidates)[1]

    def forward():
        probabilities = [forward_one(i) for i in range(len(prepared))]
        return tuple(torch.cat([p[j] for p in probabilities]) for j in range(width))

    behavior = tuple(torch.tensor([references[r['id']]['candidate_logprobs'][j] for r in selected])
                     for j in range(width))
    rollout = record_rollout(policy_id, ids, behavior, config['samples_per_request'],
                             torch.Generator().manual_seed(config['seed'] + (config['batch_index'] if transfer else 0)))
    targets = torch.tensor([r['targets'] for r in selected], dtype=torch.long, device='npu:0')
    masks = torch.ones_like(targets, dtype=torch.bool)
    weight = config['auxiliary_weight']
    with torch.no_grad():
        before = forward()
        audits = admit_actor(rollout, before, policy_id, ids)
        expected_before = expected_rule_reward(before, targets, masks, weight).cpu().tolist()
    atomic_json(output/'before.json', dict(admission=audits, expected_reward=expected_before,
        candidate_logprobs=[p.cpu().tolist() for p in before], request_ids=ids))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = BF16MasterAdamW(params, lr=config['lr'])
    if transfer:
        parent = None
        if config['batch_index'] > 0:
            parent = restore_span_optimizer(config['initial_optimizer_checkpoint'], model, optimizer, base_id)
        validate_parent(config, parent)
        atomic_json(output/'optimizer-before.json', optimizer.receipt())
    with (output/'update-claimed.json').open('x') as handle:
        json.dump(dict(policy_id=policy_id, method=config['method'], request_ids=ids), handle)
    if config['method'] == 'rl':
        rewards = rule_sample_rewards(rollout.actions, targets.cpu(), masks.cpu(), weight)
        torch.save(dict(actions=rollout.actions, candidate_logprobs=behavior,
            old_joint_logprobs=rollout.old_joint_logprobs, rewards=rewards,
            request_ids=ids, policy_id=policy_id), output/'behavior.pt')
        update = update_span_actor(rollout, forward, optimizer, params, rewards['total'], policy_id, ids)
        update['nonconstant_reward_groups'] = int((rewards['total'].std(-1) > 0).sum())
    elif config['method'] == 'conditional':
        from multimodal_jev.span_conditional_credit import conditional_credit, verl_conditional_loss
        advantages, probabilities = conditional_credit(behavior, targets.cpu(), masks.cpu(), weight)
        torch.save(dict(candidate_logprobs=behavior, advantages=advantages, probabilities=probabilities,
            request_ids=ids, policy_id=policy_id, targets=targets.cpu(), supervised=masks.cpu()),
            output/'conditional-credit.pt')
        optimizer.zero_grad(set_to_none=True); losses = []; metrics = []
        for i in range(len(prepared)):
            old = tuple(p[i:i+1].to('npu:0') for p in behavior)
            loss, info = verl_conditional_loss(old, forward_one(i), targets[i:i+1], masks[i:i+1], weight)
            loss = loss / len(prepared)
            loss.backward(); losses.append(float(loss.detach()))
            metrics.append({k:float(v) for k,v in info.items()})
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        update = dict(loss=sum(losses), gradient_norm=float(norm), optimizer_updates=1,
            algorithm='exact-conditional-marginal-ppo-v1', advantage_standardization='none',
            sampled_actions_used_for_loss=0, exact_candidates_per_request=2*width,
            per_request_verl_metrics=metrics)
    else:
        optimizer.zero_grad(set_to_none=True); losses = []
        for i in range(len(prepared)):
            loss = rule_sft_loss(forward_one(i), targets[i:i+1], masks[i:i+1], weight)/len(prepared)
            loss.backward(); losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        update = dict(loss=sum(losses), gradient_norm=float(norm), optimizer_updates=1)
    if transfer:
        completed_updates = config['batch_index'] + 1
        state = cursor(config, completed_updates)
    else:
        completed_updates = 1
        state = dict(schema='jev-rule-pilot-cursor-v1', request_ids=config['request_ids'], next_batch_index=1,
            method=config['method'], auxiliary_weight=weight, protocol_sha256=file_hash(config['protocol']))
    next_id = save_span_checkpoint(output/'policy-1', adapted, model, optimizer, base_id, completed_updates, state)
    with torch.no_grad():
        after = forward()
        expected_after = expected_rule_reward(after, targets, masks, weight).cpu().tolist()
    atomic_json(output/'after.json', dict(candidate_logprobs=[p.cpu().tolist() for p in after],
        expected_reward=expected_after, policy_id=next_id))
    update['scope'] = 'One matched learning update; export pending independent HF reload and full HF/vLLM gate; no held-out gain'
    atomic_json(output/'update.json', dict(**update, optimizer=optimizer.receipt(),
        method=config['method'], auxiliary_weight=weight, optimizer_start='resume' if transfer and config['batch_index'] > 0 else 'fresh',
        completed_updates=completed_updates,
        training_requests=8, unique_recordings=4, input_tokens=token_count,
        audio_seconds_with_repetition=audio_seconds, samples_per_request=config['samples_per_request'],
        initial_policy_id=policy_id, exported_policy_id=next_id, training_admitted=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    try:
        run(json.loads(args.config.read_text()),args.output)
        atomic_json(args.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time())); raise
