"""Whole-batch hard caps on effective training and generated tokens, not FLOPs."""
import json
from pathlib import Path

from .baseline import file_hash
from .policy_identity import checkpoint_identity
from .training_budget import batch_budget, sum_budgets

CONTRACT_FILES = ('audio_policy.py', 'task_schema.py', 'training_budget.py')
LIMIT_FIELDS = {'training_tokens', 'generated_tokens'}


def contract_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in CONTRACT_FILES}


def validate_training_reservation(observed, reserved):
    """Call before the first model forward, using actual processor tensors."""
    observed = sum_budgets([observed])
    reserved = sum_budgets([reserved])
    if any(observed[k] != reserved[k] for k in
           ('sequences', 'prompt_tokens', 'audio_frames', 'gradient_sequences',
            'no_grad_sequences', 'optimizer_steps')):
        raise ValueError('actual processor lengths or execution differ from reservation')
    if any(observed[k] > reserved[k] for k in observed):
        raise ValueError('actual response lengths exceed reservation')


class BudgetController:
    def __init__(self, limits, counts, *, samples=1, response_cap=None, no_grad_passes=0):
        if (set(limits) != LIMIT_FIELDS or
                any(type(v) is not int or v < 0 for v in limits.values())):
            raise ValueError('explicit nonnegative integer training/generated token limits required')
        if type(samples) is not int or samples < 1:
            raise ValueError('positive sample count required')
        if response_cap is not None and (type(response_cap) is not int or response_cap < 1):
            raise ValueError('positive response cap required')
        if type(no_grad_passes) is not int or no_grad_passes < 0:
            raise ValueError('nonnegative no-grad passes required')
        for row in counts.values():
            if any(type(row[k]) is not int or row[k] < 1
                   for k in ('prompt_tokens', 'gold_tokens', 'audio_frames')):
                raise ValueError('positive measured processor lengths required')
        self.limits = dict(limits)
        self.counts = counts
        self.samples = samples
        self.response_cap = response_cap
        self.no_grad_passes = no_grad_passes
        self.used = dict.fromkeys(LIMIT_FIELDS, 0)
        self.entries = []

    def reservation(self, ids):
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('complete nonempty unique batch IDs required')
        rows = [self.counts[uid] for uid in ids for _ in range(self.samples)]
        budget = batch_budget([r['prompt_tokens'] for r in rows],
            [r['gold_tokens'] if self.response_cap is None else self.response_cap for r in rows],
            [r['audio_frames'] for r in rows], gradient_passes=1,
            no_grad_passes=self.no_grad_passes, optimizer_steps=1)
        return {'training_tokens': budget['gradient_tokens'] + budget['no_grad_tokens'],
                'generated_tokens': 0 if self.response_cap is None else budget['response_tokens']}, budget

    def admit(self, ids):
        reserve, _ = self.reservation(ids)
        return all(self.used[k] + reserve[k] <= self.limits[k] for k in LIMIT_FIELDS)

    def consume(self, ids, training, generation=None):
        if not self.admit(ids):
            raise ValueError('batch exceeds reserved budget')
        reserve, upper = self.reservation(ids)
        training = sum_budgets([training])  # Reject missing/unknown accounting.
        for key in ('sequences', 'prompt_tokens', 'audio_frames'):
            if training[key] != upper[key]:
                raise ValueError('observed processor lengths differ from budget profile')
        response = training['response_tokens']
        if not training['sequences'] <= response <= upper['response_tokens']:
            raise ValueError('observed response lengths exceed reservation')
        tokens = training['prompt_tokens'] + response
        steps = training['optimizer_steps']
        if steps not in (0, 1):
            raise ValueError('one update or zero-advantage no-op required')
        expected = {'gradient_sequences': training['sequences'] * steps,
                    'gradient_tokens': tokens * steps,
                    'no_grad_sequences': training['sequences'] * self.no_grad_passes,
                    'no_grad_tokens': tokens * self.no_grad_passes}
        if any(training[k] != v for k, v in expected.items()):
            raise ValueError('inconsistent training execution counters')
        if self.response_cap is None:
            if generation is not None or training != upper:
                raise ValueError('SFT must execute its exact gold budget')
            generated = 0
        else:
            if generation is None:
                raise ValueError('RL generation accounting required')
            if (generation['responses'] != training['sequences'] or
                    generation['generated_tokens'] != response or
                    generation['replicated_prompt_tokens'] != training['prompt_tokens'] or
                    generation['requests'] != len(ids)):
                raise ValueError('generation and actor token counts differ')
            generated = generation['generated_tokens']
        spent = {'training_tokens': training['gradient_tokens'] + training['no_grad_tokens'],
                 'generated_tokens': generated}
        if any(spent[k] > reserve[k] for k in LIMIT_FIELDS):
            raise ValueError('observed cost exceeds pre-execution reservation')
        self.used = {k: self.used[k] + spent[k] for k in LIMIT_FIELDS}
        self.entries.append({'example_ids': list(ids), 'reservation': reserve, 'spent': spent})

    def report(self, next_ids=None):
        return {'limits': self.limits, 'used': self.used,
                'remaining': {k: self.limits[k] - self.used[k] for k in LIMIT_FIELDS},
                'iterations': self.entries,
                'next_reservation': self.reservation(next_ids)[0] if next_ids else None,
                'scope': 'effective unpadded training tokens (gradient plus no-grad) and generated '
                         'tokens; excludes generation prefill, loading, export and failed work; not FLOPs'}


def load_budget_controller(config, *, sft=False):
    spec = config.get('budget_control')
    if spec is None:
        return None
    if set(spec) != {'profile_path', 'profile_sha256', 'limits'}:
        raise ValueError('budget control requires pinned profile and explicit limits')
    path = Path(spec['profile_path'])
    if file_hash(path) != spec['profile_sha256']:
        raise ValueError('budget profile hash changed')
    profile = json.loads(path.read_text())
    if profile.get('contract_sha256') != contract_hashes():
        raise ValueError('budget processor contract changed or unrecorded')
    actor = config if sft else config['actor']
    manifest = config['train_manifest'] if sft else config['manifest']
    preflight = profile['preflight']
    if (file_hash(manifest) != preflight['manifest_sha256'] or
            checkpoint_identity(actor['model_path'])[0] != preflight['actor_base_policy_id']):
        raise ValueError('budget profile model/manifest mismatch')
    for audio, digest in preflight['audio_sha256'].items():
        if file_hash(audio) != digest:
            raise ValueError('budget profile audio changed')
    counts = profile['examples']
    wanted = {uid for ids in config['batches'] for uid in ids}
    if not wanted <= counts.keys():
        raise ValueError('budget profile missing scheduled example')
    if not sft and config.get('audio_input_mode') != 'actor_features':
        raise ValueError('budgeted RL requires actor_features parity contract')
    if not sft and max(counts[uid]['prompt_tokens'] for uid in wanted) + config['sampling']['max_tokens'] > config['engine']['max_model_len']:
        raise ValueError('budgeted response exceeds context')
    return BudgetController(spec['limits'], counts,
        samples=1 if sft else config['sampling']['n'],
        response_cap=None if sft else config['sampling']['max_tokens'],
        no_grad_passes=0 if sft else 1 + int(actor.get('reference') is not None))
