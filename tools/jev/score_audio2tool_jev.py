#!/usr/bin/env python3
"""Full frozen Jev release scoring, without claiming actor provenance acceptance."""
import argparse
import json
from pathlib import Path

from multimodal_jev.audio2tool_inference import digest, write_once
from multimodal_jev.audio2tool_jev import task_bank
from multimodal_jev.audio2tool_jev_score import evaluate
from multimodal_jev.audio2tool_score import load_official
from multimodal_jev.baseline import file_hash

MANIFEST_SHA = '56921792b06636d83bd6b8a21e49ae0f887f3de68a9813d75fe1b98d83c0f062'


def bundles(request_file, gold_file, prediction_file, bank, expected_records):
    """Require release order, all decisions, no tail rows or invented failures."""
    count = 0
    for request_line in request_file:
        request = json.loads(request_line)
        gold_line = gold_file.readline()
        if not gold_line:
            raise ValueError('target population ended early')
        gold = json.loads(gold_line)
        predictions = []
        for decision in bank['decisions']:
            line = prediction_file.readline()
            if not line:
                raise ValueError('incomplete terminal prediction population')
            p = json.loads(line)
            if p.get('decision_id') != decision['id'] or p.get('example_id') != request['id']:
                raise ValueError('release-order decisions required')
            predictions.append(p)
        count += 1
        yield request, gold, predictions
    if count != expected_records or gold_file.readline() or prediction_file.readline():
        raise ValueError('full release population without extra rows required')


def run(args):
    if file_hash(args.release / 'MANIFEST.json') != MANIFEST_SHA:
        raise ValueError('fixed independently verified Jev release required')
    manifest = json.loads((args.release / 'MANIFEST.json').read_text())
    pins = {'MANIFEST.json': MANIFEST_SHA}
    for name, key in [('task-bank.json', 'task_bank_file_sha256'),
                      ('external_test.requests.jsonl', 'requests_sha256'),
                      ('external_test.targets.jsonl', 'targets_sha256')]:
        pins[name] = manifest[key]
        if file_hash(args.release / name) != pins[name]:
            raise ValueError('fixed complete release bytes required')
    bank = json.loads((args.release / 'task-bank.json').read_text())
    if bank != task_bank(args.registry.read_bytes()) or digest(bank) != manifest['identity']['task_bank_sha256']:
        raise ValueError('complete pinned registry bank required')
    prediction_sha = file_hash(args.predictions)
    official = load_official(args.official_metrics)
    with (args.release / 'external_test.requests.jsonl').open() as requests, \
            (args.release / 'external_test.targets.jsonl').open() as gold, args.predictions.open() as predictions:
        result = evaluate(bundles(requests, gold, predictions, bank, 30733), bank, official)
    for name, pin in pins.items():
        if file_hash(args.release / name) != pin:
            raise ValueError('release changed during scoring')
    if file_hash(args.predictions) != prediction_sha:
        raise ValueError('prediction ledger changed during scoring')
    c = result['all']['counts']
    if (c['records'], c['decisions'], c['noul_available'], c['typed_call_list_available'],
            c['typed_call_list_unavailable'], c['typed_tool_sequence_available']) != (
            30733, 14044981, 9342832, 4664793, 6623, 30733):
        raise ValueError('complete registered decision coverage required')
    result.update(predictions_sha256=prediction_sha, release_sha256=pins,
                  official_metrics_sha256=file_hash(args.official_metrics),
                  source_sha256={name:file_hash(Path(__file__).resolve().parents[2] / name)
                    for name in ['tools/jev/score_audio2tool_jev.py',
                                 'src/multimodal_jev/audio2tool_jev_score.py',
                                 'src/multimodal_jev/audio2tool_jev.py',
                                 'src/multimodal_jev/audio2tool_score.py']},
                  actor_execution_verified=False,
                  limitation='Full raw-ledger scoring only. Requires separate exact-container, prompt/audio, token decoding and runtime acceptance before reporting measured model performance.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('release', 'registry', 'official-metrics', 'predictions', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior score')
    write_once(args.output, run(args))
