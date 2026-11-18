"""Replay the accepted endpoint batch before any resumed optimizer update."""
import json
from pathlib import Path

from .baseline import atomic_json, file_hash
from .span_batch_training_config import batch_layout, reorder_global_items
from .simple_resident_acceptance import compare_updated_batch


def prepare(config, recovered, processor):
    """Recompile audio and independently replay the authenticated sampling join."""
    from .simple_rl_schedule import SimpleRLSchedule
    from .slurp_official_training import load
    from .simple_audio_batch import compile_batch
    from .simple_sampled_rollout import actor_batch
    updates = recovered['completed_updates']
    if type(updates) is not int or not 1 <= updates < 634:
        raise ValueError('bounded recovered endpoint required')
    root = Path(config['continuous_resume']['training_root']) / 'run'
    files = {}

    def read(path):
        path = Path(path)
        digest = file_hash(path)
        if recovered['files'].get(str(path)) != digest:
            raise ValueError('independently authenticated recovery evidence required: ' + str(path))
        files[str(path)] = digest
        return json.loads(path.read_text())

    step = updates - 1
    inputs = read(root / f'inputs-{step:04d}.json')
    behavior = read(root / f'behavior-{step:04d}.json')
    reference = read(root / f'reference-{step:04d}.json')
    rollout = read(root / f'rollout-after-{updates:04d}.json')
    original = [read(root / f'loop-rank-{r}' / f'after-{updates:04d}.json') for r in range(4)]
    schedule = SimpleRLSchedule(Path(config['epoch_file']).parent,
        manifest_sha256=config['schedule_sha256'], release_sha256=config['release_sha256'])
    files.update(schedule.files)
    population, task, _ = load(config['release'], config['release_sha256'], 'learning', purpose='optimize')
    schedule.validate_population(population)
    slots, _ = schedule.at(step)
    policy = inputs['sampling']['policy_id']
    entries, _, compiled = compile_batch({r['id']: r for r in population}, task, slots,
        processor, policy_id=policy, global_batch=128)
    items, sampling = actor_batch(entries, behavior, reference, policy_id=policy,
        reference_policy_id=recovered['reference_policy_id'], seed=config['seed'], update=step)
    ordered, permutation = reorder_global_items(items, config)
    layout = batch_layout(config)
    ids = [[items[i]['request_id'] for i in indices] for indices in permutation]
    if (inputs['compiled'] != compiled or inputs['sampling'] != sampling
            or inputs['next_optimizer_update'] != updates
            or inputs.get('candidate_batch_layout') != layout
            or inputs.get('candidate_rank_indices') != permutation
            or any([r['request_id'] for r in original[rank]] != ids[rank] for rank in range(4))):
        raise ValueError('recovery input, sampling or rank permutation differs')
    compare_updated_batch(original, rollout, sampling, policy_id=recovered['policy_id'], updates=updates)
    binding = dict(schema='jev-batched-stage2-restored-policy-v1', updates=updates,
        policy_id=recovered['policy_id'], reference_policy_id=recovered['reference_policy_id'],
        checkpoint_id=recovered['checkpoint_id'], schedule_sha256=config['schedule_sha256'],
        candidate_profile=config['candidate_profile'], batch_layout=layout,
        batch_order=config['batch_order'], candidate_rank_indices=permutation, files=files)
    return dict(items=ordered, sampling=sampling, rollout=rollout, original=original,
                rank_ids=ids, binding=binding)


def verify_reports(reports, prepared):
    """Recompute probabilities; a stored passed flag never admits recovery."""
    from .span_parity import compare_span_policies
    binding = prepared['binding']
    modes = (1, binding['batch_layout']['microbatch_per_rank'])
    if len(reports) != 4:
        raise ValueError('four complete recovery ranks required')
    for rank, report in enumerate(reports):
        if (report.get('rank') != rank or report.get('binding') != binding
                or not isinstance(report.get('modes'), list)
                or [m.get('microbatch') for m in report['modes']] != list(modes)):
            raise ValueError('exact recovery policy, cursor, layout and modes required')
        for mode in report['modes']:
            if [r['request_id'] for r in mode['records']] != prepared['rank_ids'][rank]:
                raise ValueError('complete ordered restored rank slots required')
    checks = []
    for index, micro in enumerate(modes):
        ranks = [report['modes'][index]['records'] for report in reports]
        check = compare_updated_batch(ranks, prepared['rollout'], prepared['sampling'],
            policy_id=binding['policy_id'], updates=binding['updates'])
        endpoint_checks = [compare_span_policies(row['candidate_logprobs'], old['candidate_logprobs'])
            for rank in range(4) for row, old in zip(ranks[rank], prepared['original'][rank], strict=True)]
        if not all(c['passed'] for c in endpoint_checks):
            raise ValueError('restored policy differs from pre-crash resident endpoint')
        checks.append(dict(check, microbatch=micro, endpoint_maximum_logprob_error=
                           max(c['maximum_logprob_error'] for c in endpoint_checks)))
    return dict(passed=True, updates=binding['updates'], policy_id=binding['policy_id'],
                checkpoint_id=binding['checkpoint_id'], comparisons=256, checks=checks)


def check_restored_policy(model, processor, config, recovered, root, *, rank, device, collation):
    """Run both modes collectively before returning control to the update loop."""
    import torch
    import torch.distributed as dist
    from .span_audio_collate import forward_span_audio_batch

    def gather(value):
        values = [None] * 4
        dist.all_gather_object(values, value)
        return values

    def collective(name, fn):
        error = None
        result = None
        try:
            result = fn()
        except Exception as exc:
            error = repr(exc)
        errors = gather(error)
        if any(errors):
            raise ValueError(name + ': ' + str(errors))
        return result

    prepared = collective('recovery batch replay', lambda: prepare(config, recovered, processor))
    payload = prepared['items'][rank * 32:(rank + 1) * 32]
    modes = []
    with torch.no_grad():
        for micro in (1, batch_layout(config)['microbatch_per_rank']):
            records = []
            for offset in range(0, 32, micro):
                group = payload[offset:offset + micro]
                def forward():
                    outputs = forward_span_audio_batch(model, group, device=device,
                                                        dtype=torch.bfloat16, **collation)
                    return [dict(request_id=item['request_id'],
                        candidate_logprobs=[p[0].cpu().tolist() for p in probs])
                        for item, probs in zip(group, outputs, strict=True)]
                records.extend(collective('restored forward probabilities', forward))
            modes.append(dict(microbatch=micro, records=records))
    report = dict(rank=rank, binding=prepared['binding'], modes=modes)
    collective('persist restored policy', lambda: atomic_json(
        Path(root) / f'rank-{rank}' / 'restored-policy.json', report))
    reports = gather(report)
    result = collective('independent restored probability comparison', lambda: verify_reports(reports, prepared))
    collective('persist recovery acceptance', lambda: atomic_json(
        Path(root) / f'rank-{rank}' / 'restored-policy-verified.json', result))
    return result
