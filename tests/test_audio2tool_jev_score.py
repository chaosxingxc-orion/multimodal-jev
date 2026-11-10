import json
import io
import importlib.util
from pathlib import Path

import pytest

from multimodal_jev import audio2tool_jev as jev
from multimodal_jev.audio2tool_inference import digest
from multimodal_jev.audio2tool_jev_score import decode, evaluate, score_bundle, summarize
from multimodal_jev.audio2tool_score import load_official
from multimodal_jev.audio2tool_targets import project


@pytest.fixture
def fixture(monkeypatch):
    path = Path(__file__).parents[1] / '.local/research/audio2tool-code-v1/source/audio_benchmark/evaluation/metrics.py'
    if not path.exists():
        pytest.skip('pinned upstream source unavailable')
    official = load_official(path)
    tools = [dict(tool_id=str(i), tool_name='t' + str(i), signature='t' + str(i) + '()',
                  description='fixture', argument_defaults='', argument_constraints='', domain='d', category='c')
             for i in range(152)]
    monkeypatch.setattr(jev, 'registry', lambda *args: tools)
    bank = jev.task_bank(b'fixture')

    def build(extras=None):
        native = project(dict(id=1, query_idx=1, tool_name='t0', extracted_params={'v': True},
                              expected_tool_call='t0(v=True)',
                              additional_tool_calls=['t1(v=None)', 't0(v=1)'] if extras is None else extras),
                         3, official, {t['tool_name'] for t in tools})
        request = dict(id=native['id'], tier=3, input=dict(task_bank_sha256=digest(bank)))
        gold = jev.target(request, native, bank, official)
        predictions = []
        for d in bank['decisions']:
            label = jev.answer(gold, d)
            value = label['value'] if label['status'] == 'available' else [{'unscored': 9}]
            predictions.append(dict(example_id=request['id'], decision_id=d['id'], request_sha256=digest(request),
                                    raw_text=json.dumps(value), error=None))
        return request, gold, predictions
    return bank, official, build


def change(predictions, key, raw, error=None):
    p = next(p for p in predictions if p['decision_id'] == key)
    p.update(raw_text=raw, error=error)


def test_typed_oracle_and_native_order(fixture):
    bank, official, build = fixture
    request, gold, predictions = build()
    result = evaluate([(request, gold, predictions)], bank, official)
    assert result['all']['decision_accuracy'] == dict(noul=1., typed_call_list=1., typed_tool_sequence=1.)
    assert result['native']['all']['full_population_strict_calls_exact'] == 1.
    # Python considers True==1; strict nested comparison must not.
    change(predictions, 'tool.0.calls', '[{"v":1},{"v":1}]')
    result = evaluate([(request, gold, predictions)], bank, official)
    assert result['all']['counts']['typed_call_list_correct'] == 151
    assert result['native']['all']['full_population_strict_calls_exact'] == 0.


def test_unknown_supervision_keeps_prediction_and_native_primary(fixture):
    bank, official, build = fixture
    request, gold, predictions = build(['t1'])
    result = evaluate([(request, gold, predictions)], bank, official)
    c = result['all']['counts']
    assert c['decisions'] == 457 and c['typed_call_list_available'] == 151 and c['typed_call_list_unavailable'] == 1
    assert result['native']['all']['full_record_metrics']['strict_primary_exact'] == 1
    assert result['native']['all']['full_population_strict_calls_exact'] is None
    change(predictions, 'tool.1.calls', 'null')
    scored = score_bundle(request, gold, bank, predictions, official)
    assert scored['decisions']['tool.1.calls']['correct'] is None
    assert scored['counts']['format_valid'] == 456 and not scored['native_assembly_valid']
    predictions[:] = [p for p in predictions if p['decision_id'] != 'tool.1.calls']
    with pytest.raises(ValueError, match='complete decision coverage'):
        score_bundle(request, gold, bank, predictions, official)


def test_noul_cannot_filter_or_repair_native(fixture):
    bank, official, build = fixture
    request, gold, predictions = build()
    change(predictions, 'tool.0.present', 'false')
    change(predictions, 'tool.0.absent', 'true')
    result = evaluate([(request, gold, predictions)], bank, official)
    assert result['all']['counts']['all_predictions_consistent'] == 0
    assert result['native']['all']['full_population_strict_calls_exact'] == 1


def test_native_format_failure_is_not_runtime_failure(fixture):
    bank, official, build = fixture
    request, gold, predictions = build()
    change(predictions, 'tool.0.calls', 'invalid JSON')
    result = evaluate([(request, gold, predictions)], bank, official)
    assert result['native']['all']['format_valid'] == 0
    assert result['native']['all']['runtime_errors'] == 0
    change(predictions, 'tool.0.calls', '', error='timeout')
    result = evaluate([(request, gold, predictions)], bank, official)
    assert result['native']['all']['runtime_errors'] == 1
    request, gold, predictions = build()
    change(predictions, 'tool.0.present', '', error='timeout')
    result = evaluate([(request, gold, predictions)], bank, official)
    assert result['all']['counts']['runtime_errors'] == 1
    assert result['native']['all']['runtime_errors'] == 0
    assert result['native']['all']['full_population_strict_calls_exact'] == 1


def test_absent_collapse_and_runtime_errors_not_true_negatives(fixture):
    bank, official, build = fixture
    request, gold, predictions = build()
    for p in predictions:
        p['raw_text'] = 'false' if p['decision_id'].endswith('.present') else (
            'true' if p['decision_id'].endswith('.absent') else '[]')
    scored = score_bundle(request, gold, bank, predictions, official)
    report = summarize(scored['counts'])
    assert report['decision_accuracy']['noul'] == 150 / 152
    for head in ('present', 'absent'):
        assert report['semantic_presence'][head]['recall'] == 0
        assert report['semantic_presence'][head]['f1'] == 0
        assert report['semantic_presence'][head]['balanced_accuracy'] == .5
    for p in predictions:
        p['error'] = 'timeout'
    report = evaluate([(request, gold, predictions)], bank, official)
    s = report['all']['semantic_presence']['present']
    assert s['invalid_positive'] == 2 and s['invalid_negative'] == 150
    assert s['tn'] == 0 and s['balanced_accuracy'] == 0
    assert report['all']['counts']['runtime_errors'] == 457
    assert report['native']['all']['full_population_strict_calls_exact'] == 0


@pytest.mark.parametrize('raw', ['"true"', '1', 'True', '```json\ntrue\n```', 'true false', 'null'])
def test_invalid_boolean_not_repaired(raw):
    assert not decode(dict(decision_type='noul'), raw)['valid']


@pytest.mark.parametrize('raw', ['[{"v":NaN}]', '[{"v":1e999}]', '[{"v":1,"v":2}]', '[null]', '{}'])
def test_invalid_argument_json_not_repaired(raw):
    assert not decode(dict(decision_type='typed_call_list'), raw)['valid']


@pytest.mark.parametrize('sequence', ['["t0","t1"]', '["t1","t0","t0"]', '["unknown"]'])
def test_sequence_multiplicity_order_and_registry(fixture, sequence):
    bank, official, build = fixture
    request, gold, predictions = build()
    change(predictions, 'tools.sequence', sequence)
    result = evaluate([(request, gold, predictions)], bank, official)
    assert result['all']['decision_accuracy']['typed_tool_sequence'] == 0
    assert result['native']['all']['full_population_strict_calls_exact'] == 0


@pytest.mark.parametrize('fault', ['duplicate', 'binding', 'missing', 'extra', 'false_error'])
def test_bad_prediction_ledger_rejected(fixture, fault):
    bank, official, build = fixture
    request, gold, predictions = build()
    if fault == 'duplicate': predictions.append(predictions[0])
    if fault == 'binding': predictions[0]['request_sha256'] = 'wrong'
    if fault == 'missing': predictions.pop()
    if fault == 'extra': predictions[0]['repaired_value'] = True
    if fault == 'false_error': predictions[0]['error'] = False
    with pytest.raises(ValueError): score_bundle(request, gold, bank, predictions, official)


def test_duplicate_sample_rejected(fixture):
    bank, official, build = fixture
    bundle = build()
    with pytest.raises(ValueError, match='duplicate release record'):
        evaluate([bundle, bundle], bank, official)


@pytest.mark.parametrize('fault', [None, 'short', 'extra', 'wrong_order', 'wrong_count', 'missing_gold'])
def test_stream_requires_full_declared_roster(fixture, fault):
    path = Path(__file__).parents[1] / 'tools/jev/score_audio2tool_jev.py'
    spec = importlib.util.spec_from_file_location('score_audio2tool_jev_cli', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    bank, official, build = fixture
    request, gold, predictions = build()
    if fault == 'short': predictions.pop()
    if fault == 'extra': predictions.append(predictions[0])
    if fault == 'wrong_order': predictions[0], predictions[1] = predictions[1], predictions[0]
    source = module.bundles(io.StringIO(json.dumps(request) + '\n'),
        io.StringIO('' if fault == 'missing_gold' else json.dumps(gold) + '\n'),
        io.StringIO(''.join(json.dumps(p) + '\n' for p in predictions)), bank, 2 if fault == 'wrong_count' else 1)
    if fault is None:
        assert evaluate(source, bank, official)['all']['counts']['records'] == 1
    else:
        with pytest.raises(ValueError): evaluate(source, bank, official)


def test_independent_control_builder_keeps_known_primary_and_null(fixture):
    path = Path(__file__).parents[1] / 'tools/jev/check_audio2tool_jev_score.py'
    spec = importlib.util.spec_from_file_location('check_audio2tool_jev_score_cli', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    bank, official, build = fixture
    request, gold, _ = build(['t0', 't1(v=None)', 't0(v=2)'])
    before = json.dumps(gold, sort_keys=True)
    for mode in module.MODES:
        predictions = module.controls(request, gold, bank, mode)
        report = evaluate([(request, gold, predictions)], bank, official)
        if mode == 'available_annotation_oracle':
            p = next(p for p in predictions if p['decision_id'] == 'tool.0.calls')
            assert json.loads(p['raw_text']) == [{'v': True}, {}, {'v': 2}]
            assert report['all']['decision_accuracy']['typed_call_list'] == 1
            assert report['native']['all']['full_record_metrics']['strict_primary_exact'] == 1
            assert report['native']['all']['full_record_metrics']['strict_ordered_names_exact'] == 1
            assert report['native']['all']['full_population_strict_calls_exact'] is None
        elif mode == 'always_absent':
            assert report['all']['decision_accuracy']['noul'] == 150 / 152
            assert report['all']['semantic_presence']['present']['recall'] == 0
        else:
            assert report['all']['counts']['runtime_errors'] == 457
            assert report['native']['all']['runtime_errors'] == 1
    assert json.dumps(gold, sort_keys=True) == before
