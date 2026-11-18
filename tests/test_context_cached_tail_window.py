import ast
import copy
import json
from pathlib import Path

import pytest

from multimodal_jev.context_cached_tail_window import contract, select
from multimodal_jev.context_replica_schedule import rebatch, rank_indices
from tools.reports.context_cached_reference_trainer import generate as cached_generate
from tools.reports.context_cached_tail_trainer import generate


ROOT = Path(__file__).resolve().parents[1]


def config():
    stage = json.loads((ROOT/'configs/experiments/context-two-stage-v0.0.11.json').read_text())['stage2']
    return dict(stage, experiment_kind='cached_reference_tail_admission', start_update=0,
                stop_update=2, checkpoint_interval=1, engineering_prefix=True,
                training_dispatch_allowed=False)


@pytest.fixture(scope='module')
def schedule():
    c = config()
    population = [f'q{i}' for i in range(40513)]
    parent = [dict(epoch=e, index=0, slots=[dict(id=q, weight=1., padding=False)
              for q in population]) for e in range(2)]
    return set(population), rebatch(parent, population, c['batch_config'])


def test_full_schedule_real_tail_and_next_epoch_keep_identity_and_weights(schedule):
    population, batches = schedule
    c = config(); before = copy.deepcopy(batches)
    selected = select(batches, c, population)
    assert selected == [batches[316], batches[317]]
    assert [(b['epoch'], b['index']) for b in selected] == [(0, 316), (1, 0)]
    assert [sum(s['weight'] for s in b['slots']) for b in selected] == [65, 128]
    assert [sum(s['padding'] for s in b['slots']) for b in selected] == [63, 0]
    assert batches == before
    assert contract(c) == dict(schedule_indices=[316, 317], real_queries=[65, 128],
        padding_slots=[63, 0], diagnostic_updates=2, formal_history=False,
        optimizer_cursor='fresh_0_1', full_schedule_updates=634)
    # Real length bucketing must preserve all slots and the global denominator.
    for b in selected:
        ranks = rank_indices(b, [(100+i, 200+i) for i in range(128)], c['batch_config'])
        assert [len(r) for r in ranks] == [64, 64]
        assert sorted(i for r in ranks for i in r) == list(range(128))
        assert sum(b['slots'][i]['weight'] for r in ranks for i in r) == b['real_queries']


@pytest.mark.parametrize('fault', ['padding', 'duplicate', 'cursor', 'count', 'short'])
def test_complete_source_schedule_corruption_rejected(schedule, fault):
    population, original = schedule; batches = copy.deepcopy(original)
    if fault == 'padding': batches[316]['slots'][-1]['weight'] = 1.
    elif fault == 'duplicate': batches[0]['slots'][1]['id'] = batches[0]['slots'][0]['id']
    elif fault == 'cursor': batches[317]['epoch'] = 0
    elif fault == 'count': batches[316]['real_queries'] = 128
    else: batches = batches[316:318]
    with pytest.raises(ValueError): select(batches, config(), population)


@pytest.mark.parametrize('field,value', [
    ('start_update', 316), ('stop_update', 318), ('start_update', False),
    ('checkpoint_interval', True), ('experiment_kind', 'formal_source'),
    ('training_dispatch_allowed', True), ('resume_root', '/prior'),
    ('reference_cache', '/online'), ('stage', 'instruction'),
])
def test_diagnostic_cannot_claim_formal_or_resumed_history(field, value):
    c = config(); c[field] = value
    with pytest.raises(ValueError): contract(c)


def test_generated_trainer_preserves_shared_optimizer_and_reference_update_code():
    original = ast.parse(cached_generate(ROOT)); candidate = ast.parse(generate(ROOT))
    def optimizer_loop(tree):
        return next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                    and ast.unparse(node.iter) == 'range(start_update, stop_update)')
    assert ast.dump(optimizer_loop(original)) == ast.dump(optimizer_loop(candidate))
    compile(candidate, '<cached-tail-test>', 'exec')
    # Contract and schedule integration execute real helpers without NPU imports.
    run = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    names = [n for n in ast.walk(run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert sum(n.func.id == 'tail_contract' for n in names) == 1
    assert sum(n.func.id == 'select_tail' for n in names) == 1


def test_tail_output_audit_retains_shared_checks_and_rejects_missing_evidence(tmp_path):
    from tools.reports import audit_context_cached_training as audit
    namespace=dict(audit.shared.audit_outputs.__globals__,expected_contract=audit.expected_contract)
    auditor=audit.tail_auditor(namespace)
    package=tmp_path/'package';package.mkdir()
    (package/'config.json').write_text(json.dumps(config()))
    (package/'PREPARED.json').write_text('{}')
    with pytest.raises((KeyError,FileNotFoundError)):
        auditor(package,tmp_path/'collection',initial_policy=package/'initial')
    # Authenticated schedule selection is the actual shared helper, not a stub.
    assert auditor.__globals__['select_tail'] is select
