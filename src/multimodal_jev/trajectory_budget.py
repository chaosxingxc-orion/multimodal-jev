"""Receipt-based RL cost audit; missing instrumentation never means zero cost."""
import json
import math
from pathlib import Path

from .baseline import file_hash
from .training_budget import sum_budgets

GENERATION_FIELDS = ('requests', 'responses', 'request_prompt_tokens',
                     'replicated_prompt_tokens', 'generated_tokens',
                     'length_terminated_responses', 'engine_generate_seconds')


def nonnegative(value, *, integer=True):
    valid = type(value) is int if integer else type(value) in (int, float)
    if not valid or not math.isfinite(value) or value < 0:
        raise ValueError('finite nonnegative budget value required')
    return value


def generation_counts(result):
    budget = result.get('generation_budget')
    if budget is None:
        return None
    counts = {key: nonnegative(budget[key], integer=not key.endswith('_seconds'))
              for key in GENERATION_FIELDS}
    prompts = budget['prompt_tokens_per_request']
    responses = budget['response_tokens_per_record']
    for value in prompts + responses:
        nonnegative(value)
    requests, samples = counts['requests'], counts['responses']
    if (requests < 1 or samples < 1 or samples % requests or
            len(prompts) != requests or len(responses) != samples or
            sum(prompts) != counts['request_prompt_tokens'] or
            sum(responses) != counts['generated_tokens'] or
            counts['replicated_prompt_tokens'] != sum(prompts) * (samples // requests) or
            counts['length_terminated_responses'] > samples or
            result['prompts'] != requests or result['rollouts'] != samples):
        raise ValueError('generation budget disagrees with receipt counts')
    return counts


def audit_trajectory(stages):
    """Audit explicit ordered stage paths, including legacy non-scheduler runs.

    Terminal receipts are application evidence, not a replacement for Docker
    observation. No missing terminal is interpreted as a dead process.
    """
    if not stages:
        raise ValueError('at least one iteration required')
    groups = {key: [] for key in ('training', 'generation', 'engine_load_seconds',
                                 'export_seconds')}
    entries = []
    seen = set()
    previous_actor = None
    previous_updates = None
    previous_export = None
    for index, paths in enumerate(stages):
        if set(paths) != {'rollout', 'actor', 'export'}:
            raise ValueError('rollout, actor and export paths required')
        results = {}
        evidence = {}
        for role, raw_path in paths.items():
            path = Path(raw_path).resolve()
            if path in seen:
                raise ValueError('duplicate stage would double-count cost')
            seen.add(path)
            terminal, result = path/'terminal.json', path/'result.json'
            item = {'path': str(path), 'state': 'unconfirmed', 'sha256': {}}
            for receipt in (terminal, result):
                if receipt.is_file():
                    item['sha256'][receipt.name] = file_hash(receipt)
            if terminal.is_file():
                state = json.loads(terminal.read_text())
                if type(state['exit_code']) is not int:
                    raise ValueError('integer terminal exit code required')
                if state['exit_code'] != 0:
                    item['state'] = 'failed'
                else:
                    if not result.is_file():
                        raise ValueError('successful terminal without result')
                    data = json.loads(result.read_text())
                    if role != 'export' and data.get('status') != 'passed':
                        raise ValueError('successful terminal with unsuccessful result')
                    item['state'] = 'completed'
                    results[role] = data
            evidence[role] = item
        rollout, actor, export = (results.get(role) for role in ('rollout', 'actor', 'export'))
        generation = generation_counts(rollout) if rollout else None
        load = rollout.get('engine_load_seconds') if rollout else None
        if load is not None:
            nonnegative(load, integer=False)
        training = actor.get('metrics', {}).get('training_budget') if actor else None
        if actor:
            if Path(actor['config']['rollout_run']).resolve() != Path(paths['rollout']).resolve():
                raise ValueError('actor references a different rollout')
            parent = actor['config'].get('parent_run')
            if index and (parent is None or Path(parent).resolve() != previous_actor):
                raise ValueError('actor parent does not match preceding iteration')
            updates = nonnegative(actor['updates'])
            updated = actor['metrics']['updated']
            if type(updated) is not bool:
                raise ValueError('boolean update outcome required')
            if previous_updates is not None and updates != previous_updates + int(updated):
                raise ValueError('cumulative optimizer updates disagree with iteration')
            if training is not None:
                training = sum_budgets([training])
                if training['optimizer_steps'] != int(updated):
                    raise ValueError('training budget disagrees with update outcome')
                if rollout and training['sequences'] != rollout['rollouts']:
                    raise ValueError('training sequence count disagrees with rollout')
            previous_updates = updates
        else:
            previous_updates = None
        if rollout and previous_export and rollout['policy_id'] != previous_export['policy_id']:
            raise ValueError('rollout did not use preceding exported policy')
        previous_actor = Path(paths['actor']).resolve()
        previous_export = export
        export_seconds = export.get('export_seconds') if export else None
        if export_seconds is not None:
            nonnegative(export_seconds, integer=False)
        values = dict(training=training, generation=generation,
                      engine_load_seconds=load, export_seconds=export_seconds)
        for key, value in values.items():
            groups[key].append(value)
        entries.append({'iteration': index, 'evidence': evidence,
                        'missing_accounting': [key for key, value in values.items() if value is None]})
    totals = {}
    for key, values in groups.items():
        known = [value for value in values if value is not None]
        if key == 'training':
            subtotal = sum_budgets(known) if known else None
        elif key == 'generation':
            subtotal = {field: sum(value[field] for value in known)
                        for field in GENERATION_FIELDS} if known else None
        else:
            subtotal = sum(known) if known else None
        totals[key] = {'observed_iterations': len(known), 'expected_iterations': len(stages),
                       'observed_subtotal': subtotal,
                       'total': subtotal if len(known) == len(stages) else None}
    return {'schema_version': 1, 'iterations': entries, 'costs': totals,
            'all_stages_completed': all(item['state'] == 'completed'
                                       for entry in entries for item in entry['evidence'].values()),
            'accounting_complete': all(value['total'] is not None for value in totals.values()),
            'scope': 'application receipt cost audit; excludes failed/unfinished stage costs; '
                     'not artifact-integrity, Docker-exit, FLOPs, hardware-utilization or matched-budget proof'}
