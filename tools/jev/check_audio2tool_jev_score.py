#!/usr/bin/env python3
"""Full-release synthetic scorer controls, never actor predictions or feedback."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from multimodal_jev.audio2tool_inference import digest, write_once
from multimodal_jev.audio2tool_jev import task_bank
from multimodal_jev.audio2tool_jev_score import evaluate
from multimodal_jev.audio2tool_score import load_official
from multimodal_jev.baseline import atomic_json, file_hash

MANIFEST_SHA = '56921792b06636d83bd6b8a21e49ae0f887f3de68a9813d75fe1b98d83c0f062'
MODES = ('always_absent', 'all_runtime_failures', 'available_annotation_oracle')


def controls(request, gold, bank, mode):
    """Build independently from native annotations, never using Jev answer().

    Unknown extra arguments get {} only in this synthetic prediction, never in
    gold. Preserve all known arguments and repetitions, including the primary.
    Unknown lists remain unscored by the real scorer.
    """
    if mode not in MODES:
        raise ValueError('registered analytic control required')
    target = gold['native_target']
    entries = [target['primary'], *target['additional_annotations']]
    calls = defaultdict(list)
    for entry in entries:
        calls[entry['name']].append({} if entry.get('argument_evidence') == 'unavailable' else entry['arguments'])
    predictions = []
    request_sha = digest(request)
    for d in bank['decisions']:
        kind = d['decision_type']
        if kind == 'noul':
            present = bool(calls[d['tool_name']]) if mode == 'available_annotation_oracle' else False
            value = present if d['id'].endswith('.present') else not present
        elif kind == 'typed_call_list':
            value = calls[d['tool_name']] if mode == 'available_annotation_oracle' else []
        else:
            value = [e['name'] for e in entries] if mode == 'available_annotation_oracle' else []
        predictions.append(dict(example_id=request['id'], decision_id=d['id'], request_sha256=request_sha,
            raw_text='' if mode == 'all_runtime_failures' else json.dumps(value, ensure_ascii=False),
            error='synthetic_control_failure' if mode == 'all_runtime_failures' else None))
    return predictions


def check_report(report, mode):
    c = report['all']['counts']; native = report['native']['all']
    if (c['records'], c['decisions'], c['noul_available'], c['typed_call_list_available'],
            c['typed_call_list_unavailable'], c['typed_tool_sequence_available']) != (
            30733, 14044981, 9342832, 4664793, 6623, 30733):
        raise ValueError('full decision coverage differs')
    if (native['records'], native['full_arguments_subset']['eligible_records'],
            native['full_arguments_subset']['unavailable_records'], native['full_population_strict_calls_exact']) != (
            30733, 25969, 4764, None):
        raise ValueError('native full/subset population differs')
    acc = report['all']['decision_accuracy']
    for head in ('present', 'absent'):
        s = report['all']['semantic_presence'][head]
        if s['tp'] + s['fn'] + s['invalid_positive'] != 37723 or s['tn'] + s['fp'] + s['invalid_negative'] != 4633693:
            raise ValueError('semantic polarity lost')
    if mode == 'always_absent':
        expected = dict(noul=4633693 / 4671416, typed_call_list=4633693 / 4664793, typed_tool_sequence=0.)
        if acc != expected or native['full_record_metrics']['strict_primary_exact'] != 0:
            raise ValueError('always-absent closed-form control differs')
        for s in report['all']['semantic_presence'].values():
            if (s['recall'], s['f1'], s['balanced_accuracy']) != (0., 0., .5):
                raise ValueError('always-absent semantic collapse hidden')
        if c['all_predictions_consistent'] != 30733:
            raise ValueError('semantic error must not be confused with internal inconsistency')
    elif mode == 'all_runtime_failures':
        if any(acc.values()) or c['runtime_errors'] != 14044981 or c['format_valid'] != 0:
            raise ValueError('runtime failures gained credit or disappeared')
        if native['runtime_errors'] != 30733 or any(native['full_record_metrics'].values()):
            raise ValueError('native runtime failure bridge differs')
        for s in report['all']['semantic_presence'].values():
            if (s['invalid_positive'], s['invalid_negative'], s['balanced_accuracy']) != (37723, 4633693, 0.):
                raise ValueError('invalid negative treated as true negative')
    elif mode == 'available_annotation_oracle':
        if any(v != 1 for v in acc.values()) or c['all_predictions_consistent'] != 30733:
            raise ValueError('available annotation identity control differs')
        if (native['full_record_metrics']['strict_primary_exact'] != 1
                or native['full_record_metrics']['strict_ordered_names_exact'] != 1
                or native['full_arguments_subset']['strict_full_calls_exact'] != 1):
            raise ValueError('known primary/ordered tools/complete subset lost')
        for s in report['all']['semantic_presence'].values():
            if (s['precision'], s['recall'], s['f1'], s['balanced_accuracy']) != (1., 1., 1., 1.):
                raise ValueError('oracle presence projection differs')
    else:
        raise ValueError('registered control required')


def run(args):
    start = time.monotonic()
    if file_hash(args.release / 'MANIFEST.json') != MANIFEST_SHA:
        raise ValueError('fixed full release required')
    manifest = json.loads((args.release / 'MANIFEST.json').read_text())
    pins = {'MANIFEST.json': MANIFEST_SHA}
    for name, key in [('task-bank.json', 'task_bank_file_sha256'),
                      ('external_test.requests.jsonl', 'requests_sha256'),
                      ('external_test.targets.jsonl', 'targets_sha256')]:
        pins[name] = manifest[key]
        if file_hash(args.release / name) != pins[name]:
            raise ValueError('registered full release bytes required')
    bank = json.loads((args.release / 'task-bank.json').read_text())
    if bank != task_bank(args.registry.read_bytes()):
        raise ValueError('unchanged full registry bank required')
    official = load_official(args.official_metrics)
    cases = {}
    for mode in MODES:
        def population():
            n = 0
            with (args.release / 'external_test.requests.jsonl').open() as requests, \
                    (args.release / 'external_test.targets.jsonl').open() as gold:
                for request_line, gold_line in zip(requests, gold, strict=True):
                    request, target = json.loads(request_line), json.loads(gold_line)
                    if n % 250 == 0:
                        atomic_json(args.output / 'progress.json', dict(mode=mode, completed_records=n, heartbeat_unix=time.time()))
                    yield request, target, controls(request, target, bank, mode)
                    n += 1
            if n != 30733:
                raise ValueError('full external-test population required')
        report = evaluate(population(), bank, official)
        check_report(report, mode)
        write_once(args.output / (mode + '.json'), report)
        cases[mode] = dict(report_sha256=file_hash(args.output / (mode + '.json')), all=report['all'], native=report['native']['all'])
    for name, pin in pins.items():
        if file_hash(args.release / name) != pin:
            raise ValueError('release changed during control')
    source = Path(__file__).resolve().parents[2]
    write_once(args.output / 'result.json', dict(status='synthetic_controls_passed', cases=cases,
        elapsed_seconds=time.monotonic()-start, release_sha256=pins,
        official_metrics_sha256=file_hash(args.official_metrics),
        source_sha256={name:file_hash(source / name) for name in ['tools/jev/check_audio2tool_jev_score.py',
            'src/multimodal_jev/audio2tool_jev_score.py', 'src/multimodal_jev/audio2tool_jev.py',
            'src/multimodal_jev/audio2tool_score.py']}, model_inference=False, predictions_written=False,
        scope='Three complete-roster synthetic scorer controls, generated in memory only. No model scores, actor feedback, teacher API, prompt selection or reward-weight tuning.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('release', 'registry', 'official-metrics', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        write_once(args.output / 'terminal.json', dict(exit_code=1, error=repr(exc)))
        raise
    write_once(args.output / 'terminal.json', dict(exit_code=0))
