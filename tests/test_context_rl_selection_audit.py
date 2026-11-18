import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from tools.jev.audit_context_rl_selection import audit, validate_population
from tools.jev.context_rl_evaluation_plan import sha
from tools.jev.run_context_rl_selection import prepare
from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.slurp_training import content_hash


def population(n=3):
    task = dict(schema=dict(candidates=[dict(id=f'c{i}', description=f'candidate {i}') for i in range(60)]),
                thinking=dict(requirements=['Listen to the audio.']))
    rows, records, receipts = [], [], []
    candidates = [dict(id=x['id'], definition=x['description']) for x in task['schema']['candidates']]
    for i in range(n):
        asset = dict(id=str(i), audio=f'/fixture/{i}.wav', audio_sha256=f'{i:064x}', intent='c0')
        inputs = {'input_ids': dict(fixture=i)}
        identity = dict(policy_id='actual-export', inputs=inputs, positions=list(range(60)))
        rid = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        req = compile_candidates(asset['audio'], candidates, task['thinking']['requirements'], semantics='exclusive_event')
        pred = dict(id=str(i), policy_id='actual-export', request_id=rid, request_identity=identity,
                    prob=[.8] + [.01] * 59)
        receipt = dict(id=str(i), audio_sha256=asset['audio_sha256'], request_sha256=content_hash(req.model_input()),
                       engine_request_id=rid, inputs=inputs, positions=list(range(60)))
        rows.append(asset);records.append(pred);receipts.append(receipt)
    return rows, task, records, receipts


def test_population_provenance():
    validate_population(*population(), 'actual-export', expected_count=3)


@pytest.mark.parametrize('fault', ['subset', 'duplicate', 'order', 'audio', 'question', 'identity', 'policy'])
def test_population_rejects_mismatches(fault):
    rows, task, records, receipts = population()
    if fault == 'subset':records.pop()
    elif fault == 'duplicate':rows[1]['id'] = rows[0]['id']
    elif fault == 'order':records.reverse()
    elif fault == 'audio':receipts[0]['audio_sha256'] = 'wrong'
    elif fault == 'question':task['thinking']['requirements'] = ['Different instruction']
    elif fault == 'identity':receipts[0]['positions'] = [999]
    elif fault == 'policy':records[0]['policy_id'] = 'wrong'
    with pytest.raises(ValueError):validate_population(rows, task, records, receipts, 'actual-export', expected_count=3)


@pytest.fixture
def full_collection(tmp_path, monkeypatch):
    # Tensor/model execution is deliberately substituted. The real population
    # auditor and metrics evaluator run over all 9226 synthetic records.
    spec = importlib.util.spec_from_file_location('entry_fixture', Path(__file__).with_name('test_context_rl_selection_entry.py'))
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    args = module.sources(tmp_path)
    c = prepare(**args);c['release'] = '/fixture/release'
    c['rl_evaluation']['fixture_binding'] = 'fixed'
    root = tmp_path / 'evaluation-collected';root.mkdir();files = {}
    def put(name, value):
        p = root / name;p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value));files[name] = sha(p)
        return p
    config = put('config.json', c)
    _, _, hf, vl = module.parity()
    joint = dict(policy_id='actual-export', contract=c['source_contract'])
    put('evaluation/run/hf.json', hf);put('evaluation/run/vllm.json', dict(records=vl))
    put('evaluation/terminal.json', dict(exit_code=0))
    sc = dict(c, policy_id='actual-export', contract=c['source_contract'])
    sel = put('evaluation/selection-config.json', sc)
    put('evaluation/selection/runtime.json', dict(profile=c['source_contract']['runtime_profile']))
    put('evaluation/point-binding.json', dict(c['rl_evaluation'], inference_policy_id='actual-export',
        config_sha256=sha(config), selection_config_sha256=sha(sel), status='runtime_finished_independent_collection_required'))
    rows, task, records, receipts = population(9226)
    pred = put('evaluation/selection/predictions.json', dict(records=records))
    rp = put('evaluation/selection/input-receipts.json', receipts)
    for i, (p, r) in enumerate(zip(records, receipts)):
        put(f'evaluation/selection/record-{i:05d}.json', dict(prediction=p, input_receipt=r))
    import multimodal_jev.slurp_official_training as training
    import tools.jev.evaluate_context_event_records as evaluator
    import tools.jev.audit_context_rl_selection as auditor
    loader = lambda *a, **kw: (rows, task, {})
    monkeypatch.setattr(training, 'load', loader);monkeypatch.setattr(evaluator, 'load', loader)
    monkeypatch.setattr(auditor, 'authenticate_export', lambda *a: joint)
    result = evaluator.evaluate(c['release'], c['release_sha256'], pred, sha(pred), 'actual-export', 'selection')
    put('evaluation/selection/result.json', dict(result, input_receipts_sha256=sha(rp)))
    args.pop('template')
    args.update(collection=root, transport=dict(container_id='fixture-container',
        state=dict(Status='exited', ExitCode=0, OOMKilled=False), files=files,
        status='transport_complete_independent_audit_required'), expected_config_sha256=sha(config))
    return args, put


def test_full_audit_recomputes_9226_metrics(full_collection):
    args, put = full_collection
    result = audit(**args)
    assert result['passed'] and result['result']['metrics']['queries'] == 9226
    assert result['result']['metrics']['top1_event_accuracy'] == 1
    assert result['source_policy_id'] == 'rl-64'
    assert result['inference_policy_id'] == result['result']['policy_id'] == 'actual-export'
    put('evaluation/selection/runtime.json', dict(profile='wrong-runtime'))
    with pytest.raises(ValueError, match='runtime differs'):audit(**args)


@pytest.mark.parametrize('fault', ['wrong_point', 'false_metric', 'changed_record', 'live'])
def test_full_audit_rejects_bad_collection(full_collection, fault):
    args, put = full_collection;root = args['collection']
    if fault == 'wrong_point':args['step'] = 128
    elif fault == 'live':args['transport']['state']['Status'] = 'running'
    elif fault == 'changed_record':
        put('evaluation/selection/record-00000.json', {'wrong': True})
    else:
        name = 'evaluation/selection/result.json';r = json.loads((root / name).read_text())
        r['metrics']['top1_event_accuracy'] = .5;put(name, r)
    with pytest.raises(ValueError):audit(**args)
