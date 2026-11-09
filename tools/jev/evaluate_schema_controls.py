#!/usr/bin/env python3
"""Score explicit non-model controls on a frozen manifest; never report as inference."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.evaluate import evaluate
from multimodal_jev.policy_evaluation import prediction_record
from multimodal_jev.task_schema import JevExample
import multimodal_jev.evaluate as scorer


def run(manifest, output):
    output.mkdir(parents=True, exist_ok=False)
    examples = [JevExample.from_record(json.loads(line)) for line in manifest.open()]
    controls = {}
    for name in ('oracle', 'all_null', 'first_candidate_true', 'malformed'):
        records = []
        for ex in examples:
            if name == 'oracle':
                value = ex.target
            elif name == 'first_candidate_true':
                schema = ex.schema
                value = (schema.candidates[0].id if schema.candidates else
                         True if schema.decision_type == 'noul' else
                         schema.levels[0] if schema.levels else 'constant')
            else:
                value = None
            text = 'not JSON' if name == 'malformed' else json.dumps({'value': value})
            records.append(prediction_record(ex.example_id, text))
        predictions = output / (name + '.jsonl')
        predictions.write_text(''.join(json.dumps(r) + '\n' for r in records))
        controls[name] = evaluate(manifest, predictions)
    if controls['oracle']['by_variant']['all']['accuracy'] != 1:
        raise AssertionError('oracle must be perfectly correct')
    if controls['malformed']['by_variant']['all']['correct'] != 0:
        raise AssertionError('malformed output must never receive credit')
    for name in ('all_null', 'first_candidate_true'):
        if controls[name]['pairs']['noul']['both_correct'] != 0:
            raise AssertionError('constant binary policy cannot solve opposite questions')
    result = {'scope': 'scorer controls only; no model inference or efficacy evidence',
              'manifest': str(manifest), 'manifest_sha256': file_hash(manifest),
              'scorer_sha256': file_hash(Path(scorer.__file__)),
              'controls': controls}
    atomic_json(output / 'controls.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = run(args.manifest, args.output)
    print(json.dumps({name: {'accuracy': row['by_variant']['all']['accuracy'],
                            'noul_both_correct_rate': row['pairs']['noul']['both_correct_rate']}
                      for name, row in result['controls'].items()}, indent=2))
