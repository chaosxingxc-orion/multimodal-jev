#!/usr/bin/env python3
"""Read-only full-learning contrast supervision audit; no training admission."""
import argparse
from collections import Counter
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.slurp_official_training import load, example
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.span_candidate_intervention import VARIANTS


def analyze(rows, task):
    pairs = task['candidate_pairs']
    counts = Counter(r['intent'] for r in rows)
    if (not rows or len({r['id'] for r in rows}) != len(rows)
            or any(r['role'] != 'learning' or r['known_pending_source']
                   or r['official_split'] != 'train'
                   or r['intent'] not in pairs
                   or pairs[r['intent']] != [r['scenario'], r['action']] for r in rows)):
        raise ValueError('unique admitted learning rows and exact native pair labels required')
    groups = {}
    for row in rows:
        label = groups.setdefault(row['source_group'], row['intent'])
        if label != row['intent']:
            raise ValueError('source group has conflicting intent')
    names = sorted(pairs)
    pools = {}
    for positive in names:
        scenario, action = pairs[positive]
        partition = dict(same_scenario=[], same_action=[], different_scenario_and_action=[])
        for negative in names:
            if negative == positive:
                continue
            ns, na = pairs[negative]
            if ns == scenario and na == action:
                raise ValueError('duplicate canonical intent pair')
            key = 'same_scenario' if ns == scenario else 'same_action' if na == action else 'different_scenario_and_action'
            partition[key].append(negative)
        flat = [n for values in partition.values() for n in values]
        if len(flat) != len(set(flat)) or set(flat) != set(names) - {positive}:
            raise ValueError('complete disjoint negative partition required')
        pools[positive] = partition

    # Every actual source, gold and audio identity is retained in this hash;
    # no raw audio/transcript or row-level business data is exported to docs.
    population_hash = content_hash(sorted((r['id'], r['source_group'], r['intent'], r['audio_sha256']) for r in rows))
    variants = []
    for variant in VARIANTS:
        moved = 0
        for positive in names:
            row = dict(id='schema-probe', audio='/schema-probe.wav', source_group='schema-probe',
                       intent=positive, scenario=pairs[positive][0], action=pairs[positive][1])
            native, _, _ = example(row, task, 'direct', 'native')
            request, target, binding = example(row, task, 'direct', variant)
            before = {c.id: c for c in native.questions[0].schema.candidates}
            after = {c.id: c for c in request.questions[0].schema.candidates}
            if (before != after or request.questions[0].schema.question != native.questions[0].schema.question
                    or request.requirements != native.requirements
                    or request.questions[0].values[target['targets'][0]] != positive
                    or sorted(binding['intent']['presented_to_canonical']) != list(range(len(names)))):
                raise ValueError('view changed semantics or failed target remapping')
            moved += target['targets'][0] != native.questions[0].values.index(positive)
        variants.append(dict(variant=variant, checked_classes=len(names), moved_gold_positions=moved,
                             unchanged_candidate_objects=True, input_selection_uses_gold=False))
    totals = {key: sum(counts[c] * len(pools[c][key]) for c in names)
              for key in next(iter(pools.values()))}
    return dict(recordings=len(rows), source_groups=len(groups), classes=len(names),
        learning_population_sha256=population_hash, class_counts=dict(sorted(counts.items())),
        negative_pools=pools, ordered_class_pairs=sum(len(v) for p in pools.values() for v in p.values()),
        recording_candidate_comparisons=sum(totals.values()), comparisons_by_partition=totals,
        classes_without_same_scenario_negatives=[c for c in names if not pools[c]['same_scenario']],
        classes_without_same_action_negatives=[c for c in names if not pools[c]['same_action']],
        candidate_view_checks=variants,
        evidence_basis='Official-clean exclusive I_class only; not I_present/S_present absence or individual semantic review.',
        difficulty_basis='Structural neighbors only; no model confusion or verified hard-negative claim.',
        loss_input_only=True, model_input_support='Full task candidate set, independent of gold; pools never select the input question/menu.',
        inference_invocations_per_view=1, training_dispatch_allowed=False,
        scope='Full learning metadata and schema/remapping audit. No processor/NPU execution, loss integration, or efficacy claim.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, task, _ = load(args.release, args.release_sha256, 'learning', purpose='optimize')
    result = analyze(rows, task)
    result.update(release=str(args.release.resolve()), release_sha256=args.release_sha256,
                  tasks_sha256=file_hash(args.release/'tasks.json'),
                  analyzer_sha256=file_hash(__file__))
    # Immutable output prevents silently replacing an earlier evidence artifact.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('recordings','source_groups','classes',
        'ordered_class_pairs','recording_candidate_comparisons','comparisons_by_partition')}, indent=2))


if __name__ == '__main__':
    main()
