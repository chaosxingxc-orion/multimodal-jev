#!/usr/bin/env python3
"""Collect two full scheduled audio batches in one actual vLLM engine.

Compares every distribution to accepted independent HF and every draw to the
existing artifact transport. Does not execute an optimizer or accept a trainer.
"""
import argparse
import json
import time
from pathlib import Path
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.vllm_spans import register_span_model
register_span_model()


def run(config, output):
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.simple_resident_batch import ResidentSimpleBatch
    from multimodal_jev.simple_next_behavior import next_batch
    from multimodal_jev.simple_rl_schedule import SimpleRLSchedule
    from multimodal_jev.slurp_official_training import load
    from multimodal_jev.span_resident_rollout import ResidentSpanRollout
    from multimodal_jev.span_parity import compare_span_policies
    from verify_slurp_simple_parity import verify
    files = {}
    def read(path):
        path = Path(path)
        files[str(path)] = file_hash(path)
        return json.loads(path.read_text())
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('resident probe dependency changed: '+path)
        files[path] = digest
    gates = [Path(config[k]) for k in ('reference_gate', 'behavior_gate')]
    accepted = [verify(g) for g in gates]
    if (accepted[0]['updates'] != 2 or accepted[1]['updates'] not in (1, 2)
            or accepted[1]['sampled_resident_parity'] is None
            or any(a['requests'] != 256 for a in accepted)):
        raise ValueError('accepted original CE and updated RL prefix256 required')
    for a in accepted: files.update(a['files'])
    policies = [read(g/'config.json') for g in gates]
    for field in ('model', 'base_id', 'runtime_profile', 'max_model_len', 'inputs', 'population'):
        if policies[0][field] != policies[1][field]:
            raise ValueError('independent comparison populations differ')
    fixed = policies[0]['base_id']+':'+policies[0]['adapter_id']
    current = policies[1]['base_id']+':'+policies[1]['adapter_id']
    state = read(policies[1]['checkpoint_evidence'])
    if state['reference_policy_id'] != fixed:
        raise ValueError('fixed reference lineage differs')
    train = read(Path(accepted[1]['training_root'])/'config.json')
    schedule = SimpleRLSchedule(Path(train['epoch_file']).parent,
        manifest_sha256=train['schedule_sha256'], release_sha256=train['release_sha256'])
    files.update(schedule.files)
    rows, task, _ = load(train['release'], train['release_sha256'], 'learning', purpose='optimize')
    population_check = schedule.validate_population(rows)
    population = {r['id']: r for r in rows}
    processor = Qwen3ASRProcessor.from_pretrained(train['model_path'], local_files_only=True)
    references = [read(g/'run/hf/result.json') for g in gates]
    vllm = [read(g/'run/vllm/result.json') for g in gates]
    indexed = [{r['request_id']: r for r in result['records']} for result in references]
    for policy in policies: policy['physical_devices'] = config['physical_devices']
    engine = ResidentSpanRollout(policies[0], policies[0]['adapter'])
    class RecordingEngine:
        calls = 0
        def rollout(self, policy, renderer):
            result = engine.rollout(policy, renderer)
            path = output/f'rollout-{self.calls}.json'
            atomic_json(path, result)
            files[str(path)] = file_hash(path)
            self.calls += 1
            return result
    recording_engine = RecordingEngine()
    collector = ResidentSimpleBatch(recording_engine, population, task, processor, policies[0], seed=train['seed'])
    reports = []
    try:
        # Explicit transport coordinates, not a claim of two optimizer updates.
        # Both scheduled batches use the SAME already accepted updated policy.
        for update in (0, 1):
            slots, cursor = schedule.at(update)
            items, receipt = collector.collect(slots, policy_config=policies[1], completed_updates=update)
            checks = []
            for item, sample in zip(items, receipt['sampling']['sampling'], strict=True):
                for policy_index, key, rid in (
                    (0, 'reference_policy', sample['reference_request_id']),
                    (1, 'behavior', sample['request_id'])):
                    actual = [p[0].tolist() for p in item[key]]
                    check = compare_span_policies(indexed[policy_index][rid]['candidate_logprobs'], actual)
                    if not check['passed']: raise ValueError('fresh scheduled behavior HF parity failed')
                    checks.append(check)
            if update == 1:
                replay, replay_receipt = next_batch(population, task, slots, processor,
                    behavior_result=vllm[1], reference_result=vllm[0], policy_id=current,
                    reference_policy_id=fixed, completed_updates=1, seed=train['seed'])
                if receipt['sampling'] != replay_receipt['sampling']:
                    raise ValueError('resident sampling differs from accepted artifact transport')
                for a, b in zip(items, replay, strict=True):
                    for key in ('actions', 'targets', 'supervised', 'span_weights'):
                        if not torch.equal(a[key], b[key]): raise ValueError('actor item differs: '+key)
            path = output/f'batch-{update}.json'
            atomic_json(path, receipt)
            files[str(path)] = file_hash(path)
            reports.append(dict(update=update, requests=len(items),
                maximum_logprob_error=max(c['maximum_logprob_error'] for c in checks),
                argmax_changes=sum(len(c['argmax_changed_spans']) for c in checks),
                engine_passes=receipt['engine_passes'], timings=receipt['timings']))
        return dict(passed=True, batches=reports, engine_calls=recording_engine.calls,
            population_check=population_check,
            fixed_reference_policy_id=fixed, behavior_policy_id=current, files=files,
            scope='Actual single resident vLLM collector on two complete scheduled batches; independent HF parity and second-batch exact sampling replay. No optimizer execution or full-trainer acceptance.')
    finally:
        engine.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    try:
        atomic_json(args.output/'result.json', run(json.loads(args.config.read_text()), args.output))
        atomic_json(args.output/'terminal.json', dict(exit_code=0, finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time()))
        raise
