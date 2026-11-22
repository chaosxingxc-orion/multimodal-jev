"""Review source rule coverage, paired Stage1 rankings and input-only transfer cards."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import re

from multimodal_jev.context_decision_contract import (
    ApplicabilityQuestion, ContextDecisionRequest, render_prefix_context_request,
)
from tools.reports.jev_train_confusions import pinned


def compare_views(views):
    populations = []
    contracts = []
    for root, digest in views:
        manifest = json.loads(pinned(root / 'MANIFEST.json', digest))
        contracts.append(manifest['contract'])
        rows = json.loads(pinned(root / 'selection.json', manifest['files']['selection.json']))['diagnostics']
        population = {r['id']: r for r in rows}
        if len(population) != len(rows) or any(r['ranking_correct'] != (r['gold_margin'] > 0) for r in rows):
            raise ValueError('invalid strict-ranking population')
        populations.append(population)
    left, right = populations
    first, second = contracts
    for key in ('rows_sha256', 'task_sha256'):
        if first[key] != second[key]:
            raise ValueError('source inputs differ')
    for key in ('release_sha256', 'candidate_ids'):
        if first['reference_identity'][key] != second['reference_identity'][key]:
            raise ValueError('source release or label order differs')
    if left.keys() != right.keys() or any(left[k]['intent'] != right[k]['intent'] for k in left):
        raise ValueError('unpaired source rows')
    transitions = Counter((left[k]['ranking_correct'], right[k]['ranking_correct']) for k in left)
    return dict(n=len(left), both_correct=transitions[True, True], gained=transitions[False, True],
                lost=transitions[True, False], neither=transitions[False, False],
                policies=[c['endpoint']['policy_id'] for c in contracts])


def coverage(bank, truth, labels):
    groups = [set(r['labels']) for r in bank['semantic_rules']]
    pairs = Counter((labels[r['gold']], labels[r['competitor']]) for r in truth.values() if r['stratum'] == 'error')
    covered = {p for p in pairs if any(set(p) <= group for group in groups)}
    return dict(errors=sum(pairs.values()), paired_by_rule=sum(pairs[p] for p in covered),
                directed_pairs=len(pairs), covered_pairs=len(covered),
                most_frequent_rival_in_rule=sum(any({c['id'], c['contrast_with']} <= group for group in groups)
                                                for c in bank['cards']))


def trial(panel, rule, definitions):
    original = ContextDecisionRequest('audio', tuple(ApplicabilityQuestion(**q) for q in panel['questions']),
                                     tuple(panel['requirements']), version=panel['version'])
    if definitions is not None and set(definitions) != {q.id for q in original.questions}:
        raise ValueError('definition inventory differs')
    questions = tuple(replace(q, definition=definitions[q.id]) for q in original.questions) if definitions else original.questions
    request = replace(original, requirements=(*original.requirements, rule), questions=questions)
    prefix, _, markers = render_prefix_context_request(request)
    rendered = json.loads(prefix.split('\n', 2)[-1])
    if (rendered['requirements'] != [*panel['requirements'], rule] or
            rendered['questions'] != [dict(id=q.id, definition=q.definition, question=q.question) for q in questions] or
            markers != render_prefix_context_request(original)[2]):
        raise ValueError('task question, descriptions or readouts changed unexpectedly')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('small', 'large', 'current', 'bank', 'truth', 'datasets', 'annotations'):
        p.add_argument('--' + key, required=True, type=Path)
    args = p.parse_args()
    project = Path(__file__).resolve().parents[2]
    draft = json.loads((project / 'experiments/knowledge-card-transfer-draft.json').read_bytes())
    bank = json.loads(pinned(args.bank, draft['bank_sha256']))
    truth = json.loads(pinned(args.truth, bank['pins']['truth']))
    labels = json.loads((args.small / 'MANIFEST.json').read_bytes())['contract']['reference_identity']['candidate_ids']
    if labels != [c['id'] for c in bank['cards']]:
        raise ValueError('reference label order differs')
    pins = draft['view_manifests']
    large = (args.large, pins['large_r4'])
    report = dict(stage1_r4=compare_views([(args.small, pins['small_r4']), large]),
                  stage1_r64_vs_r4=compare_views([(args.current, pins['small_r64']), large]),
                  rules=coverage(bank, truth, labels))
    seen = set()
    matches = defaultdict(set)
    for line in pinned(args.annotations, bank['pins']['annotations']).splitlines():
        annotation = json.loads(line)
        ids = {uid for r in annotation['recordings'] if (uid := 'slurp:' + r['file']) in truth}
        if not ids:
            continue
        label = annotation['scenario'] + '_' + annotation['action']
        if seen & ids or any(labels[truth[k]['gold']] != label for k in ids):
            raise ValueError('ambiguous TRAIN annotation')
        seen.update(ids)
        text = ' '.join(annotation['sentence'].casefold().split())
        for word, classes in draft['word_checks'].items():
            if label in classes and word in re.findall(r'[a-z]+', text):
                matches[word, label].add(text)
    if seen != truth.keys():
        raise ValueError('missing TRAIN annotation')
    report['distinct_train_texts'] = {w: {label: len(matches[w, label]) for label in classes}
                                      for w, classes in draft['word_checks'].items()}
    minds = project / 'configs/tasks/minds14'
    menu = json.loads((minds / 'native-descriptor-transfer-v1.json').read_bytes())
    provisional = json.loads((minds / 'definitions-candidate-v1.json').read_bytes())
    panel = dict(questions=[dict(**c, question=draft['minds_question'])
                            for c in menu['candidates']], requirements=menu['requirements'], version='banking-card-draft')
    trial(panel, draft['minds_rule'], {c['id']: c['description'] for c in provisional['candidates']})
    report['minds_trial_labels'] = len(panel['questions'])
    sources = json.loads((project / 'configs/portable/operations/semantic-transfer-sources.json').read_bytes())['sources']
    for name in ('sib', 'mmsu'):
        view = sources[name]['eval']
        root = args.datasets / view['release']
        path = 'inference/' + view['dataset'] + '/panels.json'
        manifest = json.loads((root / 'MANIFEST.json').read_bytes())
        panels = json.loads(pinned(root / path, manifest['files'][path]))
        for panel in panels.values():
            trial(panel, draft[name + '_rule'], draft.get(name + '_definitions'))
        report[name + '_trial_panels'] = len(panels)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
