"""Exercises the shared queue through planning, contention, replay and independent collection."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Event

import pytest

from multimodal_jev import evaluation_blocks as blocks
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import content_key
from multimodal_jev.local_checkpoint import commit_json
from tools.reports import evaluate_tpo_endpoints as endpoint
from tools.reports import tpo_evaluation_blocks as evaluation
from tools.reports.prepare_tpo_evaluation import block_queue


def test_shared_workers_process_other_datasets_without_waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(blocks, 'check_paths', lambda _: None)
    monkeypatch.setattr('multimodal_jev.local_checkpoint.check_paths', lambda _: None)
    large = [dict(id=f'a{i}', tokens=10, frames=i+10, width=2) for i in range(19)]
    small = [dict(id=f'b{i}', tokens=4, frames=2, width=2) for i in range(3)]
    args = dict(batch=2, batches_per_block=2, choice=False)
    a = list(blocks.make_blocks('large', large, ['policy'], partitions=3, **args))
    b = list(blocks.make_blocks('small', small, ['policy'], partitions=1, **args))
    for i in range(3):
        assert [uid for row in a if row['partition'] == i for uid in row['ids']] == [
            row['id'] for row in large[i::3]]
    plan = dict(namespace='ns', policies={'policy': 'frozen'}, choice=False)
    ordered = blocks.ordered_blocks([*a,*b], ['policy'])
    assert ordered[0]['dataset'] == 'large'
    manifest = dict(plan=plan, sha256=content_key(plan))
    held, drained = Event(), Event()

    def prediction(block):
        return {'prob': dict(policy_id='frozen', records=[
            dict(id=uid, prob=[0.1, 0.9]) for uid in block['ids']])}

    def slow():
        with blocks.claim(tmp_path, ordered[0]) as owns:
            assert owns
            held.set()
            assert drained.wait(10)
            blocks.publish(tmp_path, manifest, ordered[0], prediction(ordered[0]))

    def fast():
        assert held.wait(10)
        taken = []
        for block in ordered:
            with blocks.claim(tmp_path, block) as owns:
                if owns:
                    taken.append(block['dataset'])
                    blocks.publish(tmp_path, manifest, block, prediction(block))
        drained.set()
        return taken

    with ThreadPoolExecutor(max_workers=2) as pool:
        slow_worker, fast_worker = pool.submit(slow), pool.submit(fast)
        assert set(fast_worker.result()) == {'large', 'small'}
        slow_worker.result()
    assert len(list(tmp_path.glob('block-*.json'))) == len(ordered)
    with blocks.claim(tmp_path, ordered[0]) as owns:
        assert not owns


@pytest.fixture
def queued(tmp_path, monkeypatch):
    import torch

    for module in (blocks, evaluation):
        monkeypatch.setattr(module, 'check_paths', lambda _: None)
    monkeypatch.setattr('multimodal_jev.local_checkpoint.check_paths', lambda _: None)
    parent = tmp_path/'parent.json'
    commit_json(parent, {})
    roster = [dict(id=f'a{i}', tokens=10, frames=i+10, width=2) for i in range(10)]
    inputs = dict(manifest_sha256='manifest', inputs_sha256='index', seal_sha256='seal')
    monkeypatch.setattr(evaluation, 'roster', lambda _: (roster, inputs))
    monkeypatch.setattr(evaluation, 'read_verified', lambda _: (inputs, inputs))
    monkeypatch.setattr(endpoint, 'endpoint_config', lambda *args: args[-1])
    monkeypatch.setattr(endpoint, 'policy_sources',
                        lambda spec, c, arms: {a: (a, a) for a in arms})
    monkeypatch.setattr(endpoint, 'read_parent', lambda _: None)
    monkeypatch.setattr(endpoint, 'load', lambda _, **kwargs: (
        roster, ['one', 'two'], lambda a: (dict(value=int(a['id'][1:])), None)))
    monkeypatch.setattr('multimodal_jev.cached_context_preparation.processor_cache_identity', lambda _: 'processor')
    monkeypatch.setattr('multimodal_jev.context_policy_snapshot.load_policy',
                        lambda path, model, **_: model.update(current=path))
    loaded = []

    class Provider:
        @staticmethod
        def forward(model, items, policy_id, **_):
            assert model['current'] == policy_id
            return torch.tensor([[a['value'] + (policy_id == 'trained'), -a['value']] for a in items])

    def start(c, output):
        loaded.append(output)
        return Provider(), {'current': 'initial'}, 'cpu', None, {}, None
    monkeypatch.setattr(endpoint, 'start_model', start)

    spec = dict(parent=str(parent), parent_sha=file_hash(parent), io_workers=2, batch=2, devices=[0],
                endpoints={'demo':dict(cache={}, snapshot='snapshot', policy_id='initial',
                                       contract={}, base_id='base')},
                inferred_arms={'demo':['initial','trained']},
                queue=dict(root=str(tmp_path/'blocks'), namespace='execution', batches_per_block=2,
                           partitions={'demo':2}, prefetch=2),
                result_path=str(tmp_path/'planned.json'))
    return spec, roster, loaded


def test_resume_audits_complete_blocks_and_only_runs_missing_work(tmp_path, queued):
    spec, roster, loaded = queued
    info = evaluation.build(spec)
    assert info['blocks'] == 8
    worker = dict(spec, result_path=str(tmp_path/'worker.json'), output=str(tmp_path/'runtime'))
    result = evaluation.worker(worker)
    assert len(result['blocks']) == 8
    assert len(loaded) == 1  # Both policies use the same worker/model.
    initial = evaluation.publications(spec)
    records, pid = endpoint.collect_predictions((), 'initial.json', roster,
                                             publications=initial['demo','initial','prob'])
    assert pid == 'initial' and set(records) == {row['id'] for row in roster}

    prior = Path(spec['queue']['root'])
    (prior/'block-000001.json').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        evaluation.publications(spec)
    recovered = deepcopy(spec)
    recovered['result_path'] = str(tmp_path/'resume-plan.json')
    recovered['queue'].update(root=str(tmp_path/'recovered'), resume_from=str(prior))
    assert evaluation.build(recovered)['reused_blocks'] == 7
    worker = dict(recovered, result_path=str(tmp_path/'resume-worker.json'), output=str(tmp_path/'resume-runtime'))
    assert [r['block'] for r in evaluation.worker(worker)['blocks']] == ['block-000001']
    assert evaluation.publications(recovered) == initial

    damaged = prior/'block-000000.json'
    original = damaged.read_text()
    damaged.write_text(original.replace('"policy_id": "initial"', '"policy_id": "changed"'))
    rejected = dict(spec, result_path=str(tmp_path/'reject-plan.json'),
                    queue=dict(recovered['queue'], root=str(tmp_path/'rejected')))
    with pytest.raises(ValueError, match='content differs'):
        evaluation.build(rejected)


def test_full_choice_preserves_legacy_batches_and_pairs_predictions(tmp_path, queued, monkeypatch):
    from types import SimpleNamespace
    import torch
    from tools.reports import evaluate_choice_tasks as choice

    spec, roster, _ = queued
    spec['inferred_arms'] = {'demo': ['stage1']}
    spec['endpoints']['demo']['policy_id'] = 'stage1'
    spec.update(choice={'temperature': 1.0}, full_ontology_only=True)
    request = SimpleNamespace(questions=['one', 'two'])
    inputs = {item['id']: request for item in roster}
    encode = lambda item: (dict(value=int(item['id'][1:])), None)
    worker_batches = []

    class Compiler:
        def render(self, row, request, mode, indices):
            assert mode == 'open' and indices == [0, 1]
            row['choice'] = True

    class Provider:
        def choice_compiler(self, *_):
            return Compiler()

        def forward(self, model, items, **kwargs):
            if items[0].get('choice'):
                worker_batches.append([item['request_id'] for item in items])
                return torch.zeros((len(items), 3))
            return torch.zeros((len(items), 2))

    monkeypatch.setattr(endpoint, 'load', lambda _, **kwargs: (roster, ['one', 'two'], encode, inputs))
    monkeypatch.setattr(endpoint, 'start_model', lambda *args: (Provider(), None, 'cpu', None, {}, None))
    monkeypatch.setattr(endpoint, 'read_parent', lambda _: {'view_execution': {'max_input_tokens': 32}})
    evaluation.build(spec)
    evaluation.worker(dict(spec, output=str(tmp_path/'worker'), result_path=str(tmp_path/'done.json')))
    merged = evaluation.publications(spec)
    actual, pid = endpoint.collect_predictions((), 'stage1-choices.json', roster,
                                              publications=merged['demo', 'stage1', 'choices'])
    assert pid == 'stage1'
    expected_batches, expected = [], {}
    for part in range(2):
        rows = [(item, encode(item)[0]) for item in roster[part::2]]
        output = choice.full_ontology(rows, inputs, Compiler(), 2, 1.0,
                                      lambda batch, pid: torch.zeros((len(batch), 3)), pid)
        expected.update((row['id'], row) for row in output['records'])
        expected_batches.extend([row['id'] for row, _ in rows[i:i+2]] for i in range(0, len(rows), 2))
    assert sorted(worker_batches) == sorted(expected_batches)
    assert actual == expected


def test_variable_choice_cannot_silently_change_batch_dependent_proposals():
    config = dict(sample_blocks=dict(devices=list(range(8)), batches_per_block=4, prefetch=4))
    with pytest.raises(ValueError, match='legacy'):
        block_queue(config, {'choice': {'candidate_lengths': [3, 5]}}, {'demo': {'logical_partitions': 2}})
