"""Real full-population scoring/binding; external processor/engine substituted."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev import simple_resident_selection as module
from test_simple_final_predictions import fixture as prediction_fixture


def fixture(tmp_path, monkeypatch, fault=None):
    # Host mount admission is tested separately. Exercise real atomic writes
    # here on pytest's tmpfs without weakening production filesystem checks.
    import multimodal_jev.local_checkpoint as storage
    monkeypatch.setattr(storage, 'check_paths', lambda paths: None)
    artifact, backend = prediction_fixture()
    prototype = artifact['records'][0]
    records = []
    population = []
    for i in range(9226):
        row = copy.deepcopy(prototype); row['id'] = row['recording_id'] = str(i)
        records.append(row)
        population.append(dict(id=str(i), role='selection', known_pending_source=False,
            audio='/a.wav', audio_sha256='filehash', intent=str(i % 60)))
    artifact.update(records=records, requests=9226, recordings=9226)
    inputs = tmp_path / 'inputs.json'; inputs.write_text(json.dumps(artifact))
    # Actual input compiler/independent task audit has its own tests. Here the
    # entire backend identity conversion and metric denominator remain real.
    monkeypatch.setattr(module, 'verify_inputs', lambda *args: dict(passed=True))
    monkeypatch.setattr(module, 'compile_intent', lambda *args: SimpleNamespace(
        questions=[SimpleNamespace(values=[str(i) for i in range(60)])]))
    def prepared(config, artifact, *, processor):
        assert processor is sentinel
        for row in artifact['records']:
            yield (dict(id=row['id']), None, None, None, None, None, processor)
    monkeypatch.setattr(module, 'prepared', prepared)
    sentinel = object(); calls = []
    class Engine:
        def rollout(self, config, chunk):
            calls.append([row[0]['id'] for row in chunk])
            assert len(chunk) <= 128 and config['population']['requests'] == len(chunk)
            assert all(set(row[0]) == {'id'} for row in chunk)
            outputs = []
            for item in chunk:
                row = copy.deepcopy(backend['records'][0]); row['id'] = item[0]['id']
                row['request_identity']['policy_id'] = config['base_id'] + ':' + config['adapter_id']
                row['request_id'] = hashlib.sha256(json.dumps(row['request_identity'],
                    sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                outputs.append(row)
            if fault == 'error': raise RuntimeError('worker failed')
            if fault == 'missing': outputs.pop()
            if fault == 'nan': outputs[0]['candidate_logprobs'][0][0] = float('nan')
            if fault == 'stale': outputs[0]['request_identity']['policy_id'] = 'stale'
            return dict(backend, backend='vllm', adapter_id=config['adapter_id'], records=outputs)
    owner = module.ResidentSimpleSelection(Engine(), sentinel, population, {}, inputs=inputs,
        inputs_sha256=file_hash(inputs), release_sha256='release', output=tmp_path / 'evaluation')
    evidence = tmp_path / 'checkpoint.json'; evidence.write_text('{}')
    checkpoint = dict(passed=True, policy_id='b:a', updates=0, files={str(evidence): file_hash(evidence)})
    return owner, dict(base_id='b', adapter_id='a'), checkpoint, calls


def test_complete_population_scored_once_in_73_bounded_batches(tmp_path, monkeypatch):
    owner, config, checkpoint, calls = fixture(tmp_path, monkeypatch)
    receipt = owner.evaluate(config, updates=0, checkpoint=checkpoint)
    assert len(calls) == 73 and len(calls[-1]) == 10
    assert [rid for batch in calls for rid in batch] == [str(i) for i in range(9226)]
    assert receipt['recordings'] == 9226
    # Uniform distribution uses deterministic first-candidate tie break.
    assert receipt['metrics']['accuracy'] == 154 / 9226
    result = json.loads(Path(receipt['result']).read_text())
    assert len(result['files']) == 73 and result['metrics']['per_class']['0']['correct'] == 154
    assert receipt['result_sha256'] == file_hash(receipt['result'])
    audit = module.verify_result(receipt['result'], receipt['result_sha256'], owner.population,
                                 owner.task, release_sha256='release')
    assert audit['metrics'] == result['metrics'] and audit['raw_batches'] == 73
    config['adapter_id'] = 'next'
    checkpoint.update(policy_id='b:next', updates=64)
    owner.evaluate(config, updates=64, checkpoint=checkpoint)
    assert owner.last_update == 64


def test_streamed_batches_yield_progress_and_early_close_poisons_owner(tmp_path, monkeypatch):
    owner, config, checkpoint, calls = fixture(tmp_path, monkeypatch)
    steps = owner.evaluate_steps(config, updates=0, checkpoint=checkpoint)
    assert next(steps) == dict(done=False, completed=128, total=9226)
    assert len(calls) == 1
    steps.close()
    assert owner.phase == 'failed' and not list((tmp_path / 'evaluation').rglob('result.json'))


def test_restore_evaluation_cursor_rescores_history_without_engine_calls(tmp_path, monkeypatch):
    owner, config, checkpoint, calls = fixture(tmp_path, monkeypatch)
    receipt = owner.evaluate(config, updates=0, checkpoint=checkpoint)
    history=[{k:receipt[k] for k in ('updates','policy_id','result','result_sha256')}]
    fresh = module.ResidentSimpleSelection(owner.engine,owner.processor,owner.population,owner.task,
        inputs=owner.inputs,inputs_sha256=owner.inputs_sha256,release_sha256=owner.release_sha256,
        output=tmp_path/'resumed-evaluation')
    before=len(calls);restored=fresh.restore_history(history,completed_updates=63)
    assert restored['last_selection_update']==0 and len(calls)==before
    checkpoint.update(updates=64)
    fresh.evaluate(config,updates=64,checkpoint=checkpoint)
    assert fresh.last_update==64 and len(calls)==before+73
    assert not (tmp_path/'resumed-evaluation/selection-0000').exists()
    assert Path(receipt['result']).exists()


@pytest.mark.parametrize('fault',['missing_due','policy','raw','repeat'])
def test_restore_selection_refuses_unverified_or_incomplete_history(tmp_path,monkeypatch,fault):
    owner, config, checkpoint, calls=fixture(tmp_path,monkeypatch)
    receipt=owner.evaluate(config,updates=0,checkpoint=checkpoint)
    history=[{k:receipt[k] for k in ('updates','policy_id','result','result_sha256')}]
    fresh=module.ResidentSimpleSelection(owner.engine,owner.processor,owner.population,owner.task,
        inputs=owner.inputs,inputs_sha256=owner.inputs_sha256,release_sha256=owner.release_sha256,
        output=tmp_path/'new')
    cursor=64 if fault=='missing_due' else 3
    if fault=='policy':history[0]['policy_id']='different'
    elif fault=='raw':
        path=Path(receipt['result']);path.write_text('changed')
    elif fault=='repeat':fresh=owner
    count=len(calls)
    with pytest.raises(ValueError):fresh.restore_history(history,completed_updates=cursor)
    assert len(calls)==count


@pytest.mark.parametrize('fault', ['metrics', 'raw_hash', 'missing_batch', 'raw_distribution'])
def test_independent_rescore_rejects_forged_reports_and_outputs(tmp_path, monkeypatch, fault):
    owner, config, checkpoint, _ = fixture(tmp_path, monkeypatch)
    receipt = owner.evaluate(config, updates=0, checkpoint=checkpoint)
    path = Path(receipt['result']); value = json.loads(path.read_text())
    if fault == 'metrics': value['metrics']['accuracy'] = 1.
    elif fault == 'missing_batch': value['files'].pop(next(iter(value['files'])))
    else:
        raw_path = Path(next(iter(value['files'])))
        raw = json.loads(raw_path.read_text()); raw['records'][0]['candidate_logprobs'][0][0] += 1
        raw_path.write_text(json.dumps(raw))
        if fault == 'raw_distribution': value['files'][str(raw_path)] = file_hash(raw_path)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        module.verify_result(path, file_hash(path), owner.population, owner.task, release_sha256='release')


@pytest.mark.parametrize('fault', ['error', 'missing', 'stale', 'nan'])
def test_failed_partial_or_changed_outputs_never_publish_metric(tmp_path, monkeypatch, fault):
    owner, config, checkpoint, calls = fixture(tmp_path, monkeypatch, fault)
    with pytest.raises((ValueError, RuntimeError)):
        owner.evaluate(config, updates=0, checkpoint=checkpoint)
    assert owner.phase == 'failed' and not list((tmp_path / 'evaluation').rglob('result.json'))
    count = len(calls)
    with pytest.raises(ValueError, match='cannot retry'):
        owner.evaluate(config, updates=0, checkpoint=checkpoint)
    assert len(calls) == count


@pytest.mark.parametrize('fault', ['skip_step', 'bool_step', 'stale_checkpoint', 'changed_checkpoint', 'changed_input'])
def test_invalid_step_or_provenance_rejected_before_rollout(tmp_path, monkeypatch, fault):
    owner, config, checkpoint, calls = fixture(tmp_path, monkeypatch)
    updates = 0
    if fault == 'skip_step': updates = 64
    elif fault == 'bool_step': updates = False
    elif fault == 'stale_checkpoint': checkpoint['policy_id'] = 'b:old'
    elif fault == 'changed_checkpoint': Path(next(iter(checkpoint['files']))).write_text('changed')
    else: Path(owner.inputs).write_text('changed')
    with pytest.raises(ValueError): owner.evaluate(config, updates=updates, checkpoint=checkpoint)
    assert not calls
