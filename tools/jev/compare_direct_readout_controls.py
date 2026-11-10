#!/usr/bin/env python3
"""Audit paired actual/zero audio readouts; differences alone are not efficacy."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.direct_readout import decision


def load_run(root, condition):
    identity = json.loads((root / 'identity.json').read_text())
    result = json.loads((root / 'result.json').read_text())
    terminal = json.loads((root / 'terminal.json').read_text())
    assert terminal['exit_code'] == 0 and result['status'] == 'passed'
    assert result['audio_condition'] == condition and result['model_files_unchanged']
    assert result['weight_updates'] == 0
    assert file_hash(root / 'predictions.jsonl') == result['predictions_sha256']
    rows = [json.loads(line) for line in (root / 'predictions.jsonl').open()]
    indexed = {r['example_id']: r for r in rows}
    selected = json.loads((root / 'selection.json').read_text())['ids']
    assert len(indexed) == len(rows) == len(selected) == result['predictions'] == result['requests']
    assert set(indexed) == set(selected)
    for row in rows:
        assert row['audio_condition'] == condition and row['replicate'] == 0
        expected = decision(row['alternatives'], row['candidate_logits'])
        for key, value in expected.items():
            if key == 'probabilities':
                # CPU libm differs at float64 roundoff across host/container.
                # This audits serialization/recomputation, not HF/vLLM parity.
                assert all(math.isclose(x, y, rel_tol=1e-14, abs_tol=1e-15)
                           for x, y in zip(row[key], value, strict=True))
            else:
                assert row[key] == value
        assert json.loads(row['raw_text']) == {'value': row['value']}
    return identity, indexed


def compare(actual, zeros):
    ai, ar = load_run(actual, 'actual')
    zi, zr = load_run(zeros, 'zeros')
    for key in ('requests_sha256', 'instruction_sha256', 'models', 'runner_sha256', 'source_sha256'):
        assert ai[key] == zi[key], ('identity mismatch', key)
    ac, zc = dict(ai['config']), dict(zi['config'])
    assert ac.pop('audio_condition') == 'actual' and zc.pop('audio_condition') == 'zeros'
    assert file_hash(ac.pop('instruction')) == file_hash(zc.pop('instruction')) == ai['instruction_sha256']
    assert ac == zc and set(ar) == set(zr)
    assert file_hash(ac['requests']) == ai['requests_sha256']
    metrics = defaultdict(lambda: {'requests': 0, 'changed_values': 0, 'tv_sum': 0.,
                                   'max_probability_change': 0., 'actual_values': Counter(),
                                   'zeros_values': Counter(), 'actual_indices': Counter(),
                                   'zeros_indices': Counter()})
    for uid, a in ar.items():
        z = zr[uid]
        for key in ('request_sha256', 'prompt_sha256', 'source_group', 'split',
                    'alternatives', 'label_token_ids', 'input_tokens'):
            assert a[key] == z[key], ('pair mismatch', uid, key)
        # IDs are used only for offline grouping, never rendered to the actor.
        view = uid.split(':')[-1]
        m = metrics[view]
        av, zv = json.dumps(a['value']), json.dumps(z['value'])
        deltas = [abs(x-y) for x, y in zip(a['probabilities'], z['probabilities'], strict=True)]
        m['requests'] += 1
        m['changed_values'] += av != zv
        m['tv_sum'] += sum(deltas) / 2
        m['max_probability_change'] = max(m['max_probability_change'], max(deltas))
        m['actual_values'][av] += 1
        m['zeros_values'][zv] += 1
        m['actual_indices'][str(a['selected_index'])] += 1
        m['zeros_indices'][str(z['selected_index'])] += 1
    for m in metrics.values():
        m['mean_total_variation'] = m.pop('tv_sum') / m['requests']
    return {'actual_run': str(actual), 'zeros_run': str(zeros), 'requests': len(ar),
            'by_view': dict(metrics),
            'scope': 'TRAIN paired waveform sensitivity only; zero waveform is not a semantic unknown task; changes do not establish task understanding'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--actual', type=Path, required=True)
    p.add_argument('--zeros', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve comparison evidence')
    report = compare(args.actual, args.zeros)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
