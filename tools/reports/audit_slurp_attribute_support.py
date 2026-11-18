"""Read-only TRAIN support audit for label-projection questions, not semantic labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def audit(root):
    paths = [root / name for name in ('tasks.json', 'learning.jsonl', 'learning.audio.jsonl')]
    tasks = json.loads(paths[0].read_text())
    pairs = tasks['intent']['candidate_pairs']
    rows = [json.loads(line) for line in paths[1].read_text().splitlines()]
    audio = [json.loads(line) for line in paths[2].read_text().splitlines()]
    if len(rows) != 40513 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Expected unique full clean learning population')
    audio_by_id = {r['id']: r for r in audio}
    if len(audio_by_id) != len(audio) or set(audio_by_id) != {r['id'] for r in rows}:
        raise ValueError('Audio support population mismatch')
    for r in rows:
        if r['role'] != 'learning' or r['official_split'] != 'train':
            raise ValueError('Only learning TRAIN labels may inform this design')
        if pairs[r['intent']] != [r['scenario'], r['action']]:
            raise ValueError('Native attributes do not match frozen candidate mapping')
    if len(pairs) != 60 or len({tuple(v) for v in pairs.values()}) != 60:
        raise ValueError('Expected 60 uniquely paired native events')
    counts = Counter(r['intent'] for r in rows)
    if set(counts) != set(pairs):
        raise ValueError('Learning population must cover every final candidate')
    groups = defaultdict(set)
    for r in rows:
        groups[r['source_group']].add(r['intent'])
    if any(len(v) != 1 for v in groups.values()):
        raise ValueError('Source group contains conflicting event labels')
    result = {
        'schema': 'slurp-native-attribute-support-v1',
        'scope': 'TRAIN-only label projection analysis; no new semantic truth or training admission',
        'recordings': len(rows), 'source_groups': len(groups), 'final_candidates': len(pairs),
        'training_dispatch_allowed': False,
        'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'families': {},
    }
    for index, field in enumerate(('scenario', 'action')):
        menu = sorted({v[index] for v in pairs.values()})
        columns = []
        entropy = 0.0
        best_correct = 0
        for value in menu:
            candidates = sorted(k for k, v in pairs.items() if v[index] == value)
            support = [r for r in rows if r[field] == value]
            n = len(support)
            histogram = {k: counts[k] for k in candidates}
            conditional = -sum((c / n) * math.log2(c / n) for c in histogram.values() if c)
            entropy += n / len(rows) * conditional
            best_correct += max(histogram.values())
            columns.append({'value': value, 'positive_recordings': n,
                            'negative_recordings': len(rows) - n,
                            'positive_source_groups': len({r['source_group'] for r in support}),
                            'compatible_final_candidates': candidates,
                            'event_histogram': histogram})
        result['families'][field] = {
            'questions': len(menu), 'columns': columns,
            'conditional_event_entropy_bits': entropy,
            'oracle_attribute_train_majority_event_accuracy': best_correct / len(rows),
            'oracle_scope': 'Uses gold attribute and TRAIN frequency; not model performance or a deployed decoder',
        }
    slices = Counter()
    durations = defaultdict(list)
    for r in rows:
        a = sum(v[0] == r['scenario'] for v in pairs.values())
        b = sum(v[1] == r['action'] for v in pairs.values())
        label = ('both_attributes_ambiguous' if a > 1 and b > 1 else
                 'one_attribute_identifies_event' if a > 1 or b > 1 else
                 'either_attribute_identifies_event')
        slices[label] += 1
        durations[label].append(audio_by_id[r['id']]['seconds'])
    result['structural_slices'] = {
        name: {'recordings': n, 'mean_audio_seconds': sum(durations[name]) / n}
        for name, n in sorted(slices.items())
    }
    result['interpretation'] = [
        'All Q labels are deterministic projections of Final, not independent process supervision.',
        'Native action words have domain-dependent meanings; shared strings do not prove cross-dataset equivalence.',
        'Structural ambiguity and duration are diagnostics, not cognitive difficulty or multi-intent truth.',
        'Use a task-wide menu independent of per-query labels; no gold-dependent Q routing.',
        'Joint gold scenario/action uniquely identifies Final; this is an auxiliary decomposition control.',
        'No TEST or selection labels were read; no model or processor was executed.',
    ]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.release)
    report['tool_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('families', 'sources')}, indent=2))
