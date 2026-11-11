#!/usr/bin/env python3
"""Run discriminating fixtures against the pinned upstream MAC-SLU scorer."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile

from multimodal_jev.baseline import atomic_json, file_hash

SCORER_SHA256 = '821b6389b618e474a915cf6f63f61defc868e2fec8d3df545e5ed77dc5b0f149'


def audit(path):
    if file_hash(path) != SCORER_SHA256:
        raise ValueError('official scorer revision differs')
    spec = importlib.util.spec_from_file_location('mac_slu_official_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frames = [dict(domain='音乐', intent='播放音乐', slots={'对象': '音乐'}),
              dict(domain='影视', intent='播放影视', slots={'对象': '影视'})]
    fixtures = [
        ('raw_mapping_silently_becomes_empty', [], {'意图1': {'音乐': [{'name': 'intent', 'value': '播放音乐'}]}},
         dict(overall_accuracy=1., intent_accuracy=1., slot_f1=1.)),
        ('frame_order_changes_overall', frames[::-1], frames,
         dict(overall_accuracy=0., intent_accuracy=1., slot_f1=1.)),
        ('slot_binding_swap_not_seen_by_slot_f1',
         [dict(frames[0], slots=frames[1]['slots']), dict(frames[1], slots=frames[0]['slots'])], frames,
         dict(overall_accuracy=0., intent_accuracy=1., slot_f1=1.)),
        ('record_ids_are_not_matched', frames, frames,
         dict(overall_accuracy=1., intent_accuracy=1., slot_f1=1.)),
    ]
    results = []
    with tempfile.TemporaryDirectory() as temp:
        pred, gold = Path(temp) / 'pred.jsonl', Path(temp) / 'gold.jsonl'
        for name, prediction, target, expected in fixtures:
            pred.write_text(json.dumps(dict(id='different-id', semantics=prediction)) + '\n')
            gold.write_text(json.dumps(dict(id='gold-id', semantics=target)) + '\n')
            observed = module.calculate_metrics(str(pred), str(gold))
            if any(observed[k] != value for k, value in expected.items()):
                raise ValueError('scorer behavior changed: ' + name)
            results.append(dict(name=name, observed=observed, expected=expected, passed=True))
    return dict(passed=True, revision='d54453f63e0dc57b17d3cb2cfea35fb7da6a4489',
                source_url='https://github.com/Gatsby-web/MAC_SLU/blob/d54453f63e0dc57b17d3cb2cfea35fb7da6a4489/metrics.py',
                scorer_sha256=SCORER_SHA256, fixtures=results,
                scope='Synthetic fixture execution of upstream code; no model evaluation or test labels. '
                      'Raw mapping labels require explicit conversion before this list-only scorer.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scorer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.scorer)
    atomic_json(a.output, result)
    print(json.dumps(result, ensure_ascii=False))
