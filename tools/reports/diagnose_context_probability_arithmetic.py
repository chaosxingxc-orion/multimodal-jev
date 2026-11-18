"""Replay the failed formal step's inputs without any parameter update.

Compare actual actor logits/probabilities to archived vLLM behavior, then
separate execution error from sigmoid-to-log rounding. No threshold changes.
"""
import argparse
import json
from pathlib import Path


def probability_logs(logits):
    """Diagnostic candidate: use the sampler's exact FP32 probability/log path.

    Autograd crosses the small CPU decision tensor; there is no detach,
    straight-through estimator, clamping, or probability renormalization.
    Saturation remains an explicit failure, as in the existing sampler.
    """
    import torch
    p = logits.float().sigmoid().to('cpu')
    if not torch.isfinite(p).all() or ((p <= 0) | (p >= 1)).any():
        raise ValueError('finite interior actual probabilities required')
    return torch.stack((torch.log1p(-p), torch.log(p)), -1).to(logits.device)


def run(config, out):
    import torch
    import torch_npu
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.baseline import file_hash, atomic_json
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.local_checkpoint import commit_json, commit_file
    from multimodal_jev.context_policy_artifacts import read_execution_bundle
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.context_binary_policy import binary_logprobs
    from multimodal_jev.context_behavior_parity import compare
    from multimodal_jev.context_candidate_task import compile_candidates
    from multimodal_jev.cached_context_preparation import CachedContextPreparer, processor_cache_identity
    from multimodal_jev.encoded_audio_cache import EncodedAudioCache
    from multimodal_jev.context_audio_collate import forward_context_audio_batch
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.sft import initialize_adapter
    from multimodal_jev.context_decision_head import ContextDecisionHead
    from multimodal_jev.context_decision_model import ContextDecisionModel
    from multimodal_jev.context_policy_snapshot import load_policy
    from multimodal_jev.context_runtime_profile import install_actor, PROFILE
    from multimodal_jev.span_request_identity import tensor_identity
    require_training_storage(out)
    for name, digest in config['pins'].items():
        if file_hash(name) != digest:
            raise ValueError('diagnostic dependency changed: '+name)
    c = json.loads(Path(config['original_config']).read_text())
    for name, digest in c['pins'].items():
        if file_hash(name) != digest:
            raise ValueError('original executed package changed: '+name)
    if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*')) != ['davinci4']:
        raise ValueError('isolated physical card4 required')
    if checkpoint_identity(c['model'])[0] != c['base_id']:
        raise ValueError('actual base differs')
    bundle = read_execution_bundle(c['source_policy'], expected_policy_id=c['source_policy_id'],
        expected_contract=c['source_contract'], expected_base_id=c['base_id'])
    rollout = json.loads(Path(config['rollout']).read_text())
    batch = json.loads(Path(config['batch']).read_text())
    if (rollout['update'] != 0 or len(rollout['queries']) != 128
            or rollout['behavior_policy_id'] != c['source_policy_id']
            or batch['policy_id'] != c['source_policy_id']):
        raise ValueError('exact failed first full behavior batch required')
    release = Path(c['release'])
    if file_hash(release/'MANIFEST.json') != c['release_sha256']:
        raise ValueError('original learning release changed')
    manifest = json.loads((release/'MANIFEST.json').read_text())
    for name in ('learning.jsonl', 'tasks.json'):
        if file_hash(release/name) != manifest['files'][name]:
            raise ValueError('original learning inputs changed: '+name)
    population = {r['id']: r for r in
        map(json.loads, (release/'learning.jsonl').read_text().splitlines())}
    task = json.loads(Path(c['tasks']).read_text())['intent']
    definitions = [dict(id=v['id'], definition=v['description']) for v in task['schema']['candidates']]
    proc = Qwen3ASRProcessor.from_pretrained(bundle['processor'], local_files_only=True)
    def no_rebuild():
        raise ValueError('diagnostic requires existing exact learning cache')
    prepare = CachedContextPreparer(no_rebuild, EncodedAudioCache(c['encoded_audio_cache']),
        processor_sha256=processor_cache_identity(c['source_policy']), decision_token_id=151706,
        audio_token_id=151676, max_tokens=4096)
    items = []
    for metadata, record in zip(batch['records'], rollout['queries'], strict=True):
        row = population[metadata['id']]
        req = compile_candidates(row['audio'], definitions, task['thinking']['requirements'], semantics='exclusive_event')
        value = prepare((req, row['audio_sha256']))
        identity = dict(policy_id=c['source_policy_id'], positions=value.positions[0].tolist(),
            inputs={k:tensor_identity(v) for k,v in sorted(value.inputs.items())})
        if (not value.cache_hit or identity != record['behavior_record']['request_identity']
                or metadata['request_id'] != record['request_id']):
            raise ValueError('replayed input differs from actual failed run')
        items.append(dict(request_id=record['request_id'], policy_id=c['source_policy_id'],
                          inputs=value.inputs, positions=value.positions))
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch.manual_seed(317)
    torch.npu.manual_seed_all(317)
    torch.use_deterministic_algorithms(True)
    device = torch.device('npu', 0)
    _, base = load_runtime(c['model'], device, 'bfloat16')
    adapted = initialize_adapter(base, dict(lora_rank=4, lora_alpha=8, lora_execution_dtype='bfloat16')).eval()
    head = ContextDecisionHead(1024, 8, 1024, .1).to(device=device, dtype=torch.bfloat16).eval()
    model = ContextDecisionModel(adapted.get_base_model().model, head, 151706).eval()
    load_policy(c['source_policy'], model, expected_policy_id=c['source_policy_id'],
        expected_contract=c['source_contract'], expected_base_id=c['base_id'])
    handle, runtime = install_actor(adapted, model, PROFILE)
    commit_json(out/'runtime.json', runtime)
    collation = dict(decision_token_id=151706, policy_id=c['source_policy_id'],
        pad_token_id=proc.tokenizer.pad_token_id, audio_token_id=151676,
        audio_chunk_size=base.config.audio_config.n_window*2,
        audio_infer_window=base.config.audio_config.n_window_infer)
    reports = []
    for rank, indices in enumerate(batch['rank_indices']):
        for offset in range(0, len(indices), 8):
            chosen = indices[offset:offset+8]
            with torch.no_grad():
                logits = forward_context_audio_batch(model, [items[i] for i in chosen],
                    device=device, dtype=torch.bfloat16, **collation)
                prob = logits.sigmoid().cpu()
                stable = torch.stack(binary_logprobs(logits), 1).cpu()
            behavior = torch.tensor([rollout['queries'][i]['behavior_record']['prob'] for i in chosen])
            old = torch.stack((torch.log1p(-behavior), torch.log(behavior)), -1)
            z = logits.detach().requires_grad_(True)
            canonical = probability_logs(z)
            weights = torch.linspace(.1, 1., canonical.numel(), device=device).reshape_as(canonical)
            g_canonical = torch.autograd.grad((canonical*weights).sum(), z)[0].cpu()
            z2 = logits.detach().requires_grad_(True)
            analytic = torch.stack(binary_logprobs(z2), 1)
            g_stable = torch.autograd.grad((analytic*weights).sum(), z2)[0].cpu()
            before = compare(stable.flatten(1), old.flatten(1), c['behavior_bounds'])
            after = compare(canonical.detach().cpu().flatten(1), old.flatten(1), c['behavior_bounds'])
            row = dict(rank=rank, offset=offset, indices=chosen,
                probability_exact=torch.equal(prob, behavior), probability_max_abs=float((prob-behavior).abs().max()),
                original=before, candidate=after, gradient_finite=bool(torch.isfinite(g_canonical).all()),
                gradient_max_abs=float((g_canonical-g_stable).abs().max()))
            reports.append(row)
            commit_file(out/f'block-{rank}-{offset:02d}.pt', lambda f: torch.save(dict(
                logits=logits.detach().cpu(), actor_probability=prob, behavior_probability=behavior,
                stable_logs=stable, behavior_logs=old, candidate_logs=canonical.detach().cpu(),
                stable_gradient=g_stable, candidate_gradient=g_canonical), f))
            atomic_json(out/'progress.json', dict(blocks=len(reports), total=16))
    result = dict(policy_id=c['source_policy_id'], blocks=reports, recordings=128, updates=0,
        probabilities_exact=all(r['probability_exact'] for r in reports),
        original_passed=all(r['original']['passed'] for r in reports),
        candidate_passed=all(r['candidate']['passed'] for r in reports),
        gradients_finite=all(r['gradient_finite'] for r in reports),
        scope='Frozen real-audio single-NPU forward and small-logit autograd only. No FSDP/optimizer update or training admission.')
    commit_json(out/'result.json', result)
    return result


if __name__ == '__main__':
    import time
    from multimodal_jev.baseline import atomic_json
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(exist_ok=False)
    try:
        run(json.loads(Path(a.config).read_text()), out)
        atomic_json(out/'terminal.json', dict(exit_code=0, finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(out/'terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time()))
        raise
