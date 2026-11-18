"""Full selection scoring on the admitted resident service at frozen RL steps.

An evaluation failure is terminal for this owner, never a partial score or a
retry. Inputs are streamed in bounded chunks; gold is joined after inference.
Execution/provenance audit of the containing training run remains separate.
"""
import json
from itertools import islice
from pathlib import Path

from .baseline import file_hash
from .local_checkpoint import commit_json
from .simple_final_metrics import score
from .simple_final_predictions import from_backend
from .simple_prepared_inputs import prepared
from .simple_selection_inputs import verify as verify_inputs
from .slurp_intent_structure import compile_intent

SELECTION_UPDATES = (0, 64, 128, 192, 256, 317, 384, 448, 512, 576, 634)


class ResidentSimpleSelection:
    def __init__(self, engine, processor, population, task, *, inputs, inputs_sha256,
                 release_sha256, output):
        self.engine = engine
        self.processor = processor
        self.population = population
        self.task = task
        self.output = Path(output)
        path = Path(inputs)
        if file_hash(path) != inputs_sha256:
            raise ValueError('pinned full selection inputs required')
        self.artifact = json.loads(path.read_text())
        if (len(population) != 9226 or any(row['role'] != 'selection'
                or row['known_pending_source'] for row in population)):
            raise ValueError('full admitted selection population required')
        self.input_check = verify_inputs(self.artifact, population, task, release_sha256)
        self.classes = list(compile_intent(population[0]['audio'], task, 'D').questions[0].values)
        self.inputs = str(path)
        self.inputs_sha256 = inputs_sha256
        self.release_sha256 = release_sha256
        self.phase = 'ready'
        self.last_update = None

    def restore_history(self, history, *, completed_updates):
        """Rescore accepted prior points without inference or rewriting them.

        Caller authenticates the stopped-prefix audit and supplies its policy
        identities. This binds all rows, inputs, raw scores and checkpoint bytes
        again before advancing the new owner's evaluation cursor.
        """
        if self.phase != 'ready' or self.last_update is not None:
            raise ValueError('fresh selection owner required for recovery')
        self.phase = 'restoring'
        try:
            if type(completed_updates) is not int or not 1 <= completed_updates < 634:
                raise ValueError('nonterminal recovery cursor required')
            due = [s for s in SELECTION_UPDATES if s <= completed_updates]
            if [entry.get('updates') for entry in history] != due:
                raise ValueError('all and only due selection points required')
            for entry in history:
                result = verify_result(entry['result'], entry['result_sha256'], self.population,
                                       self.task, release_sha256=self.release_sha256)
                raw = json.loads(Path(entry['result']).read_text())
                if (result['updates'] != entry['updates'] or result['policy_id'] != entry['policy_id']
                        or raw['inputs'] != self.inputs or raw['inputs_sha256'] != self.inputs_sha256):
                    raise ValueError('same-input audited selection history required')
        except BaseException:
            self.phase = 'failed'
            raise
        self.last_update = due[-1]
        self.phase = 'ready'
        return dict(passed=True, completed_updates=completed_updates,
                    last_selection_update=self.last_update, history=history)

    def evaluate(self, config, *, updates, checkpoint):
        for event in self.evaluate_steps(config, updates=updates, checkpoint=checkpoint):
            if event['done']:
                return event['receipt']

    def evaluate_steps(self, config, *, updates, checkpoint):
        """Yield between bounded batches so DP peers never wait a full eval.

        The trainer advances this iterator within a collective error boundary.
        Closing it early poisons this owner just like a worker failure.
        """
        if self.phase != 'ready':
            raise ValueError('ready selection owner required; failed evaluation cannot retry')
        self.phase = 'evaluating'
        try:
            result = yield from self._evaluate(config, updates, checkpoint)
        except BaseException:
            self.phase = 'failed'
            raise
        self.phase = 'ready'
        self.last_update = updates
        yield dict(done=True, receipt=result)

    def _evaluate(self, config, updates, checkpoint):
        expected_index = 0 if self.last_update is None else SELECTION_UPDATES.index(self.last_update) + 1
        if (type(updates) is not int or expected_index >= len(SELECTION_UPDATES)
                or updates != SELECTION_UPDATES[expected_index]):
            raise ValueError('exact preregistered selection step required')
        policy = config['base_id'] + ':' + config['adapter_id']
        # The pre-RL point binds the accepted final Stage1; later points bind
        # committed four-rank state. Do not let a policy name stand for a save.
        if (checkpoint.get('passed') is not True or checkpoint.get('policy_id') != policy
                or type(checkpoint.get('updates')) is not int or checkpoint['updates'] != updates
                or not checkpoint.get('files')):
            raise ValueError('same-policy accepted checkpoint evidence required')
        for path, digest in checkpoint['files'].items():
            if file_hash(path) != digest:
                raise ValueError('selection checkpoint evidence changed')
        if file_hash(self.inputs) != self.inputs_sha256:
            raise ValueError('selection inputs changed')
        dest = self.output / f'selection-{updates:04d}'
        dest.mkdir(parents=True, exist_ok=False)
        renderer = iter(prepared(config, self.artifact, processor=self.processor))
        predictions = []
        raw_files = {}
        offset = 0
        while True:
            chunk = list(islice(renderer, 128))
            if not chunk:
                break
            expected = self.artifact['records'][offset:offset + len(chunk)]
            if [row[0]['id'] for row in chunk] != [row['id'] for row in expected]:
                raise ValueError('selection renderer order or membership changed')
            result = self.engine.rollout(dict(config, population={'requests': len(chunk)}), chunk)
            if result.get('backend') != 'vllm':
                raise ValueError('resident vLLM selection result required')
            artifact = dict(self.artifact, records=expected, requests=len(chunk), recordings=len(chunk))
            converted = from_backend(artifact, result, policy_id=policy)
            path = dest / f'batch-{offset // 128:04d}.json'
            commit_json(path, result)
            raw_files[str(path)] = file_hash(path)
            predictions.extend(converted)
            offset += len(chunk)
            yield dict(done=False, completed=offset, total=9226)
        # score enforces exact full-population coverage and normalized finite
        # distributions. No metric is published until all 9226 are available.
        metrics = score(self.population, self.classes, predictions, policy_id=policy)
        result = dict(passed=True, updates=updates, policy_id=policy, backend='vllm',
            role='TRAIN_internal_selection_not_official_TEST', metrics=metrics,
            inputs=self.inputs, inputs_sha256=self.inputs_sha256,
            release_sha256=self.release_sha256, checkpoint=checkpoint, files=raw_files,
            scope='Complete same-policy resident selection metrics; independent run audit and HF parity separate.')
        commit_json(dest / 'result.json', result)
        return dict(passed=True, updates=updates, policy_id=policy, recordings=metrics['recordings'],
            result=str(dest / 'result.json'), result_sha256=file_hash(dest / 'result.json'),
            metrics={k: metrics[k] for k in ('accuracy', 'macro_F1_fixed60',
                       'macro_recall_supported', 'invalid_rate')})


def verify_result(path, digest, population, task, *, release_sha256):
    """Independently rescore every raw output from a pinned evaluation result.

    Caller separately authenticates population and containing process/checkpoint
    audit. A successful recomputation is not evidence of NPU execution.
    """
    path = Path(path)
    if file_hash(path) != digest:
        raise ValueError('pinned selection result required')
    result = json.loads(path.read_text())
    if (result.get('passed') is not True or result.get('backend') != 'vllm'
            or type(result.get('updates')) is not int or result['updates'] not in SELECTION_UPDATES
            or result.get('role') != 'TRAIN_internal_selection_not_official_TEST'
            or result['release_sha256'] != release_sha256
            or len(population) != 9226 or any(r['role'] != 'selection' or r['known_pending_source'] for r in population)
            or file_hash(result['inputs']) != result['inputs_sha256']):
        raise ValueError('complete admitted selection provenance required')
    artifact = json.loads(Path(result['inputs']).read_text())
    verify_inputs(artifact, population, task, release_sha256)
    checkpoint = result['checkpoint']
    if (checkpoint.get('passed') is not True or checkpoint.get('policy_id') != result['policy_id']
            or type(checkpoint.get('updates')) is not int or checkpoint['updates'] != result['updates']
            or not checkpoint.get('files')):
        raise ValueError('same-policy selection checkpoint required')
    for source, expected in checkpoint['files'].items():
        if file_hash(source) != expected:
            raise ValueError('selection checkpoint changed')
    paths = [path.parent / f'batch-{index:04d}.json' for index in range(73)]
    if set(result['files']) != {str(p) for p in paths}:
        raise ValueError('complete fixed73 selection batch files required')
    predictions = []
    for index, source in enumerate(paths):
        if file_hash(source) != result['files'][str(source)]:
            raise ValueError('selection raw output changed')
        rows = artifact['records'][index * 128:(index + 1) * 128]
        raw = json.loads(source.read_text())
        if raw.get('backend') != 'vllm':
            raise ValueError('resident vLLM output required')
        chunk = dict(artifact, records=rows, requests=len(rows), recordings=len(rows))
        predictions.extend(from_backend(chunk, raw, policy_id=result['policy_id']))
    classes = list(compile_intent(population[0]['audio'], task, 'D').questions[0].values)
    metrics = score(population, classes, predictions, policy_id=result['policy_id'])
    if result['metrics'] != metrics:
        raise ValueError('independently recomputed selection metrics differ')
    return dict(passed=True, updates=result['updates'], policy_id=result['policy_id'],
        metrics=metrics, result_sha256=digest, raw_batches=73,
        scope='Independent full9226 identity/probability/metric recomputation; execution audit separate.')
