"""Describe frozen Stage1 selection errors using predeclared native attributes.

No model inference, threshold search, target changes or process predictions.
Attribute correctness is a projection of Final top1, not a new head score.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from multimodal_jev.context_event_metrics import ContextEventMetrics
from tools.reports.build_context_attribute_candidate import digest

POLICY = '9164eddffd9e3fdb21c060a72827b1a4324f3d45e60eb0d491b589b194db0e66'
RELEASE = '38e4273d9328e4f02db081315e46f97a17628b974be7ba175254882e95421482'
ATTRIBUTE = '85d17ce7a12d2e39efb9bdc676cdb24b02f2531ccb4c93b7adeba303485d0710'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def analyze(proof, collection, release, attribute):
    proof, collection, release, attribute = map(Path, (proof, collection, release, attribute))
    evidence = read(proof)
    result = evidence['result']
    if (evidence['source_policy_id'] != POLICY or evidence['inference_policy_id'] != POLICY
            or result['policy_id'] != POLICY or result['role'] != 'selection'
            or result['release_sha256'] != RELEASE or not evidence['metric_comparison']['passed']
            or evidence['state']['Status'] != 'exited' or evidence['state']['ExitCode'] != 0
            or evidence['state']['OOMKilled'] or sha(release/'MANIFEST.json') != RELEASE
            or sha(attribute/'MANIFEST.json') != ATTRIBUTE):
        raise ValueError('Frozen successful Stage1 selection evidence and releases required')
    source = read(release/'MANIFEST.json'); manifest = read(attribute/'MANIFEST.json')
    for name in ('tasks.json', 'selection.jsonl'):
        if sha(release/name) != source['files'][name]:raise ValueError('Release payload changed')
    for name in ('catalog.json', 'inputs.jsonl', 'targets.jsonl'):
        if sha(attribute/name) != manifest['files'][name]:raise ValueError('Attribute payload changed')
    for name in ('selection/predictions.json', 'selection/result.json', 'selection-config.json'):
        if sha(collection/name) != evidence['files'][name]:raise ValueError('Collected inference changed')
    predictions = read(collection/'selection/predictions.json')['records']
    labels = rows(release/'selection.jsonl')
    inputs, targets = rows(attribute/'inputs.jsonl'), rows(attribute/'targets.jsonl')
    menu = read(attribute/'catalog.json')
    ids = result['metrics']['candidate_ids']
    pairs = read(release/'tasks.json')['intent']['candidate_pairs']
    if (ids != [r['id'] for r in menu['events']] or len(ids) != 60
            or len(predictions) != 9226 or len(labels) != 9226
            or len(inputs) != 9226 or len(targets) != 9226
            or len({p['id'] for p in predictions}) != 9226
            or result['predictions_sha256'] != sha(collection/'selection/predictions.json')):
        raise ValueError('Complete fixed-order population required')
    by_id = {r['id']:r for r in labels}
    attr = {r['id']:(i,r) for i,r in zip(inputs,targets,strict=True)}
    scenario_counts = Counter(p[0] for p in pairs.values())
    action_counts = Counter(p[1] for p in pairs.values())
    metrics = {}; counters = defaultdict(Counter); mistakes = Counter(); details = []
    for pred in predictions:
        row = by_id[pred['id']]; inp, target = attr[pred['id']]
        gold = row['intent']; p = pred['prob']
        structural = ('both_attributes_ambiguous' if min(scenario_counts[pairs[gold][0]],
                       action_counts[pairs[gold][1]]) > 1 else 'at_least_one_attribute_unique')
        if (row['role'] != 'selection' or target['id'] != inp['id']
                or row['audio_sha256'] != inp['audio_sha256']
                or target['source_group'] != row['source_group']
                or target['input_sha256'] != digest(inp)
                or target['structural_slice'] != structural
                or pred['policy_id'] != POLICY or pred['request_identity']['policy_id'] != POLICY
                or pred['request_id'] != hashlib.sha256(json.dumps(pred['request_identity'],
                    sort_keys=True,separators=(',',':')).encode()).hexdigest()
                or len(p) != 60 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p)):
            raise ValueError('Prediction/truth/native slice binding differs')
        top = ids[max(range(60), key=lambda i:p[i])]
        selected = [k for k,v in zip(ids,p,strict=True) if v >= .5]
        scenario_correct = pairs[top][0] == pairs[gold][0]
        action_correct = pairs[top][1] == pairs[gold][1]
        category = ('correct' if top == gold else 'scenario_only_wrong' if action_correct
                    else 'action_only_wrong' if scenario_correct else 'both_wrong')
        if top != gold:mistakes[(gold,top)] += 1
        for view in ('all',structural):
            if view not in metrics:metrics[view] = ContextEventMetrics(ids)
            metrics[view].add(row['id'],gold,p)
            count = counters[view]
            count[category] += 1
            count['empty'] += not selected
            count['singleton'] += len(selected) == 1
            count['multiple'] += len(selected) > 1
            count['top1_correct_but_empty'] += top == gold and not selected
            count['top1_correct_but_multiple'] += top == gold and len(selected) > 1
            count['gold_in_multiple'] += gold in selected and len(selected) > 1
        details.append(dict(id=row['id'],source_group=row['source_group'],gold=gold,top1=top,
            structural_slice=structural,error_category=category,selected_count=len(selected),
            set_f1=2*int(gold in selected)/(len(selected)+1)))
    scores = {k:v.result() for k,v in metrics.items()}
    original = result['metrics']; actual = scores['all']
    if actual['confusion'] != original['confusion']:raise ValueError('Original confusion differs')
    for key in ('query_macro_set_f1','exact_set','top1_event_accuracy','top1_macro_f1',
                'candidate_binary_nll','candidate_brier'):
        if not math.isclose(actual[key],original[key],rel_tol=1e-12,abs_tol=1e-15):
            raise ValueError('Original metric differs: '+key)
    # Independently aggregate native attributes directly from archived confusion.
    expected = Counter()
    for gold, entries in original['confusion'].items():
        for top,n in entries.items():
            category = ('correct' if top == gold else 'scenario_only_wrong' if pairs[top][1] == pairs[gold][1]
                        else 'action_only_wrong' if pairs[top][0] == pairs[gold][0] else 'both_wrong')
            expected[category] += n
    if any(counters['all'][k] != v for k,v in expected.items()):raise ValueError('Confusion projection differs')
    summary = {}
    for view,score in scores.items():
        n = score['queries']; count = counters[view]
        if sum(count[k] for k in ('correct','scenario_only_wrong','action_only_wrong','both_wrong')) != n:
            raise ValueError('Error partition incomplete')
        summary[view] = dict(recordings=n,source_groups=len({d['source_group'] for d in details
            if view == 'all' or d['structural_slice'] == view}),
            metrics={k:score[k] for k in ('query_macro_set_f1','exact_set','top1_event_accuracy',
                'top1_macro_f1','candidate_binary_nll','candidate_brier')},counts=dict(count),
            final_top1_projected_scenario_accuracy=(count['correct']+count['action_only_wrong'])/n,
            final_top1_projected_action_accuracy=(count['correct']+count['scenario_only_wrong'])/n)
    return dict(schema='context-stage1-native-structure-diagnostic-v1',policy_id=POLICY,
        proof=dict(path=str(proof),sha256=sha(proof)),
        predictions_sha256=sha(collection/'selection/predictions.json'),
        release_manifest_sha256=RELEASE,attribute_manifest_sha256=ATTRIBUTE,
        tool_sha256=sha(__file__),views=summary,
        top_confusions=[dict(gold=a,predicted=b,count=n) for (a,b),n in mistakes.most_common(20)],
        records=details,original_metrics_recomputed=True,independent_confusion_projection=True,
        scope='Descriptive Stage1 selection diagnostics; original simple template, not compact Direct/Q inference. Attribute accuracy projects Final top1; no process-head predictions, causal benefit, threshold search or TEST use.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('proof','collection','release','attribute','output'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    report = analyze(args.proof,args.collection,args.release,args.attribute)
    with Path(args.output).open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(views=report['views'],top_confusions=report['top_confusions'][:8]),indent=2))
