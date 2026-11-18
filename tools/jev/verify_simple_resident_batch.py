#!/usr/bin/env python3
"""Rebuild scheduled sampling from raw resident outputs in a separate process."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.span_parity import compare_span_policies


def compare_rollout(result, accepted, request_ids):
    if (result.get('passed') is not True or result.get('reference_compared') is not False
            or any(result[k] != accepted[k] for k in ('base_id', 'adapter_id', 'backend'))
            or result['backend'] != 'vllm'):
        raise ValueError('actual reference-free same-policy vLLM output required')
    rows = {r['request_id']: r for r in result['records']}
    refs = {r['request_id']: r for r in accepted['records']}
    if (len(rows) != len(result['records']) or set(rows) != set(request_ids)
            or not set(rows) <= refs.keys()):
        raise ValueError('exact complete scheduled population required')
    worker = result['worker']
    if (len(worker) != 1 or worker[0]['registered_adapter_ids'] != [result['slot']]
            or worker[0]['lora_dtype'] != 'torch.bfloat16'
            or worker[0]['head_projection_dtype'] != 'torch.bfloat16'
            or not worker[0]['actual_lora_buffer_dtypes']
            or any(v != ['torch.bfloat16'] for v in worker[0]['actual_lora_buffer_dtypes'].values())):
        raise ValueError('actual resident adapter/BF16 worker required')
    checks = []
    for rid, row in rows.items():
        ref = refs[rid]
        if row['request_identity'] != ref['request_identity'] or row.get('parity') is not None:
            raise ValueError('scheduled input/policy identity differs')
        check = compare_span_policies(row['candidate_logprobs'], ref['candidate_logprobs'])
        if not check['passed']: raise ValueError('scheduled behavior probability differs')
        checks.append(check)
    return dict(requests=len(rows), maximum_logprob_error=max(c['maximum_logprob_error'] for c in checks))


def verify(root, observed_path):
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.simple_audio_batch import compile_batch, renderer_for_policy
    from multimodal_jev.simple_sampled_rollout import actor_batch
    from multimodal_jev.simple_rl_schedule import SimpleRLSchedule
    from multimodal_jev.slurp_official_training import load
    from verify_slurp_simple_parity import verify as verify_gate
    root = Path(root)
    files = {}
    def read(path):
        path = Path(path); files[str(path)] = file_hash(path)
        return json.loads(path.read_text())
    config, launch, observed = [read(p) for p in (root/'config.json', root/'launch.json', observed_path)]
    if (launch['config_sha256'] != file_hash(root/'config.json')
            or observed['container_id'] != launch['container_id'] or observed['image'] != launch['image']
            or observed['state']['Status'] != 'exited' or observed['state']['ExitCode'] != 0
            or read(root/'terminal.json')['exit_code'] != 0 or read(root/'run/terminal.json')['exit_code'] != 0):
        raise ValueError('exact successful resident collector process required')
    for path, digest in config['pins'].items():
        if file_hash(path) != digest: raise ValueError('collector dependency changed')
        files[path] = digest
    gates = [Path(config[k]) for k in ('reference_gate', 'behavior_gate')]
    accepted = [verify_gate(g) for g in gates]
    for report in accepted: files.update(report['files'])
    policies = [read(g/'config.json') for g in gates]
    fixed, behavior = [p['base_id']+':'+p['adapter_id'] for p in policies]
    train = read(Path(accepted[1]['training_root'])/'config.json')
    if read(policies[1]['checkpoint_evidence'])['reference_policy_id'] != fixed:
        raise ValueError('fixed reference lineage differs')
    schedule = SimpleRLSchedule(Path(train['epoch_file']).parent,
        manifest_sha256=train['schedule_sha256'], release_sha256=train['release_sha256'])
    files.update(schedule.files)
    rows, task, _ = load(train['release'], train['release_sha256'], 'learning', purpose='optimize')
    population_check = schedule.validate_population(rows)
    processor = Qwen3ASRProcessor.from_pretrained(train['model_path'], local_files_only=True)
    originals = [read(g/'run/vllm/result.json') for g in gates]
    checks = []
    for update in (0, 1):
        slots, _ = schedule.at(update)
        entries, renderer, compiled = compile_batch({r['id']: r for r in rows}, task, slots,
            processor, policy_id=behavior, global_batch=128)
        fixed_rows = renderer_for_policy(renderer, fixed)
        resident = [read(root/'run'/f'rollout-{update*2+i}.json') for i in (0, 1)]
        for result, reference, rendered in zip(resident, originals, (fixed_rows, renderer), strict=True):
            checks.append(compare_rollout(result, reference, [r[4] for r in rendered]))
        _, sampling = actor_batch(entries, resident[1], resident[0], policy_id=behavior,
            reference_policy_id=fixed, seed=train['seed'], update=update)
        receipt = read(root/'run'/f'batch-{update}.json')
        if (receipt['compiled'] != compiled or receipt['sampling'] != sampling
                or receipt['next_optimizer_update'] != update+1
                or receipt['engine_passes'] != 2 or receipt['independent_hf_passes'] != 0):
            raise ValueError('independently rebuilt scheduled sampling differs')
    report = read(root/'run/result.json')
    if (report['passed'] is not True or report['engine_calls'] != 4
            or report['population_check'] != population_check
            or report['fixed_reference_policy_id'] != fixed or report['behavior_policy_id'] != behavior):
        raise ValueError('resident collector summary differs')
    return dict(passed=True, container_id=launch['container_id'], batches=2, requests=256,
        checks=checks, population_check=population_check, files=files,
        scope='Independent raw resident output and full scheduled sampling reconstruction; no optimizer update or trainer acceptance.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--observed', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(a.output, verify(a.root, a.observed))
