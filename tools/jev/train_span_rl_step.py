#!/usr/bin/env python3
"""Bounded real-audio verl/vLLM BF16 step, with sequential NPU ownership.

Every subprocess exits before the next backend is loaded. The output root is
exclusive: failed/pending versions cannot be consumed again through this CLI.
Use the pinned NPU image; this is a TP1 integration pilot, not a Ray trainer.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from multimodal_jev.baseline import atomic_json, file_hash


def prepare_actor(config, output, adapter):
    import torch
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.sft import initialize_adapter
    from multimodal_jev.span_training import load_reviewed_bundles
    from multimodal_jev.span_balance import balanced_binary_batches
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.span_contract import encode_span_request, span_labels
    from multimodal_jev.span_request_identity import bind_span_request
    from multimodal_jev.span_model import NativeAudioSpanModel
    from multimodal_jev.policy_identity import checkpoint_identity
    bundles, manifest = load_reviewed_bundles(config['release'], config['manifest_sha256'], config['question_order'])
    batches, balance = balanced_binary_batches(bundles, config['seed'])
    # One complementary pair; every task has one positive and one negative.
    selected = [bundles[i] for i in batches[0]]
    atomic_json(output / 'balance.json', balance)
    base_id = checkpoint_identity(config['model_path'])[0]
    policy_id = base_id + ':' + checkpoint_identity(adapter)[0]
    processor, base = load_runtime(config['model_path'], 'npu:0', 'bfloat16')
    adapted = initialize_adapter(base, {**config, 'initial_adapter':str(adapter), 'lora_execution_dtype':'bfloat16'}).eval()
    model = NativeAudioSpanModel(adapted.get_base_model()).eval()
    prepared = []
    for b in selected:
        audio, receipt = load_audio(b['request'].audio, config.get('audio_preprocessing'), return_receipt=True)
        inputs, positions = encode_span_request(processor, b['request'], audio)
        mapping = span_labels(processor.tokenizer, b['request'].questions)
        candidates = [m['token_ids'] for m in mapping]
        request_id, identity = bind_span_request(b['request'], receipt, inputs, positions, candidates, policy_id)
        targets = []
        for gold, m in zip(b['targets'], mapping, strict=True):
            matches = [j for j, value in enumerate(m['values']) if type(value) is type(gold) and value == gold]
            if len(matches) != 1:
                raise ValueError('complete balanced candidate-index supervision required')
            targets.append(matches[0])
        prepared.append(dict(bundle=b, inputs=inputs, positions=positions, candidates=candidates,
                             request_id=request_id, identity=identity, targets=targets))

    def forward():
        rows = []
        for p in prepared:
            inputs = {k:v.to(device='npu:0', dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                      for k,v in p['inputs'].items()}
            rows.append(model(inputs, p['positions'].to('npu:0'), p['candidates'])[1])
        return tuple(torch.cat([row[j] for row in rows], 0) for j in range(len(rows[0])))
    return adapted, model, base_id, policy_id, prepared, forward


def actor_phase(config, output, phase):
    import torch
    import torch_npu
    from multimodal_jev.bf16_lora import BF16MasterAdamW
    from multimodal_jev.span_actor_update import record_rollout, update_span_actor
    from multimodal_jev.span_checkpoint import save_span_checkpoint, restore_span_optimizer
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.manual_seed(config['seed'])
    torch.npu.manual_seed_all(config['seed'])
    if config.get('runtime_profile'):
        from multimodal_jev.span_runtime_profile import configure_actor_profile
        atomic_json(output/f'actor-runtime-{phase}.json', configure_actor_profile(config['runtime_profile']))
    adapter = output/'policy-1/adapter' if phase == 'after' else Path(config['initial_adapter'])
    adapted, model, base_id, policy_id, prepared, forward = prepare_actor(config, output, adapter)
    ids = tuple(p['request_id'] for p in prepared)
    if phase == 'update':
        # Consume once even if export crashes after a successful optimizer step.
        # Recovery must use a new run root and an authenticated checkpoint.
        with (output/'update-claimed.json').open('x') as handle:
            json.dump(dict(policy_id=policy_id, request_ids=ids), handle)
        behavior_rows = []
        for i, p in enumerate(prepared):
            result = json.loads((output/f'rollout-before-{i}/result.json').read_text())
            identity = json.loads((output/f'rollout-before-{i}/request-identity.json').read_text())
            if result['request_id'] != p['request_id'] or identity['payload'] != p['identity']:
                raise ValueError('rollout exact input or policy identity mismatch')
            behavior_rows.append(result['candidate_logprobs'])
        behavior = tuple(torch.tensor([row[j] for row in behavior_rows], dtype=torch.float32)
                         for j in range(len(prepared[0]['candidates'])))
        rollout = record_rollout(policy_id, ids, behavior, config['samples_per_request'],
                                 torch.Generator().manual_seed(config['seed']))
        # Current temporal tasks are parallel. Do not invent a privileged Final
        # or process chain. All three reviewed labels contribute equally.
        targets = torch.tensor([p['targets'] for p in prepared], dtype=torch.long)
        rewards = (rollout.actions == targets[:,None,:]).float().mean(-1)
        torch.save(dict(actions=rollout.actions, old_joint_logprobs=rollout.old_joint_logprobs,
                        candidate_logprobs=rollout.candidate_logprobs, rewards=rewards,
                        policy_id=policy_id, request_ids=ids), output/'behavior.pt')
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = BF16MasterAdamW(params, lr=config['lr'])
        previous_updates = 0
        if config.get('initial_optimizer_checkpoint'):
            previous_updates = restore_span_optimizer(config['initial_optimizer_checkpoint'], model,
                                                       optimizer, base_id)['completed_updates']
        receipt = update_span_actor(rollout, forward, optimizer, params, rewards, policy_id, ids)
        atomic_json(output/'update.json', receipt)
        next_id = save_span_checkpoint(output/'policy-1', adapted, model, optimizer, base_id, previous_updates+1)
        atomic_json(output/'export.json', dict(previous_policy_id=policy_id, next_policy_id=next_id,
            reward='mean correctness of three parallel reviewed temporal decisions', optimizer=optimizer.receipt()))
        with torch.no_grad():
            expected = forward()
        atomic_json(output/'post-update-probabilities.json', [p.cpu().tolist() for p in expected])
        return
    with torch.no_grad():
        logp = forward()
    if phase == 'after':
        expected = json.loads((output/'post-update-probabilities.json').read_text())
        errors = [float((p.cpu()-torch.tensor(e)).abs().max()) for p,e in zip(logp,expected,strict=True)]
        atomic_json(output/'actor-reload.json', dict(maximum_logprob_error=max(errors), passed=max(errors)==0.))
        if max(errors) != 0.:
            raise ValueError('saved BF16 execution policy did not reload exactly')
    for i, p in enumerate(prepared):
        atomic_json(output/f'reference-{phase}-{i}.json', dict(input_request=p['bundle']['request'].model_input(),
            candidate_logprobs=[prob[i:i+1].cpu().tolist() for prob in logp]))


def orchestrate(config, output, config_path):
    if config.get('runtime_profile'):
        from multimodal_jev.span_runtime_profile import verify_suite_admission
        atomic_json(output/'runtime-admission.json', verify_suite_admission(config))
    def actor(phase):
        subprocess.run([sys.executable, __file__, '--config', str(config_path), '--output', str(output),
                        '--phase', phase], check=True)

    def rollout(phase):
        for i in range(2):
            adapter = output/'policy-1/adapter' if phase == 'after' else Path(config['initial_adapter'])
            reference = output/f'reference-{phase}-{i}.json'
            pins = dict(config['pins'])
            pins[str(reference)] = file_hash(reference)
            pins.update({str(p):file_hash(p) for p in adapter.iterdir() if p.is_file()})
            probe = dict(model=config['model_path'], adapter=str(adapter), reference=str(reference),
                         lora_execution_dtype='bfloat16', audio_preprocessing=config.get('audio_preprocessing'),
                         pins=pins, trace_layers=False)
            # A named profile requires a pinned multi-input acceptance above;
            # every current and newly exported policy is still gated below.
            if config.get('runtime_profile'):
                probe['runtime_profile'] = config['runtime_profile']
            path = output/f'rollout-{phase}-{i}-config.json'
            atomic_json(path, probe)
            subprocess.run([sys.executable, str(Path(__file__).with_name('probe_vllm_spans.py')),
                            '--config', str(path), '--output', str(output/f'rollout-{phase}-{i}')], check=True)
            result = json.loads((output/f'rollout-{phase}-{i}/result.json').read_text())
            if not result['complete_action_parity']['passed']:
                atomic_json(output/'admission-failure.json', dict(phase=phase, request_index=i,
                    optimizer_updates=int((output/'update.json').exists()), audit=result['complete_action_parity']))
                raise ValueError('real-audio actor/vLLM parity failed; policy cannot advance')

    actor('before')
    rollout('before')
    actor('update')
    actor('after')
    rollout('after')
    atomic_json(output/'READY.json', dict(status='passed', optimizer_updates=1,
        scope='One balanced real-audio verl update; actor reload and vLLM new-adapter parity passed. No held-out gain claim.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['before','update','after'])
    args = parser.parse_args()
    if args.phase is None:
        args.output.mkdir(parents=True, exist_ok=False)
    try:
        if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*')) != ['davinci4']:
            raise ValueError('isolated physical NPU 4 required')
        config = json.loads(args.config.read_text())
        if (config.get('lora_execution_dtype') != 'bfloat16'
                or config.get('reward_mode') != 'parallel_mean_correctness'
                or config.get('question_order') != ['date','time','timeofday']):
            raise ValueError('explicit BF16 parallel temporal pilot contract required')
        for path,digest in config['pins'].items():
            if file_hash(path) != digest:
                raise ValueError('pinned input changed: '+path)
        if args.phase:
            actor_phase(config, args.output, args.phase)
        else:
            orchestrate(config, args.output, args.config)
            atomic_json(args.output/'terminal.json', dict(exit_code=0, finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/f'{args.phase or "terminal"}-failure.json', dict(error=repr(exc)))
        if args.phase is None:
            atomic_json(args.output/'terminal.json', dict(exit_code=1, finished_unix=time.time(), error=repr(exc)))
        raise
