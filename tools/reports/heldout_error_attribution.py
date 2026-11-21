"""Describe paired audited endpoints; only the source TRAIN branch may fit a diagnostic cutoff."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from multimodal_jev.local_config import load_config
from tools.reports.summarize_audited_top1 import verified_rows


def checked(path, digest):
    b = Path(path).read_bytes()
    if hashlib.sha256(b).hexdigest() != digest:
        raise ValueError(f'Changed scoring input: {path}')
    return json.loads(b)


def train(c):
    root = Path(c['view'])
    view = checked(root/'MANIFEST.json', c['manifest_sha256'])
    data = checked(root/'selection.json', view['files']['selection.json'])
    rows = data['diagnostics']
    # Exact singleton iff best negative < cutoff <= gold. Smallest maximizing
    # cutoff breaks ties; no target, selection or TEST input is available here.
    counts = [sum(r['best_wrong_probability'] < t/100 <= r['gold_probability'] for r in rows)
              for t in range(1, 100)]
    cutoff = (1 + counts.index(max(counts))) / 100
    return dict(policy=view['contract']['endpoint']['policy_id'], n=len(rows), cutoff=cutoff,
                train_exact=max(counts), train_exact_at_half=counts[49],
                by_class=data['summary']['by_class'], buckets=data['summary']['buckets'])


def describe(c, source, seal_root, band, confident, *, observe=None):
    audit = checked(c['audit'], c['audit_sha256'])
    dataset = audit['dataset']
    manifest = checked(seal_root/'MANIFEST.json', audit['seal_sha256'])
    if c['dataset'] != dataset or c['step'] != audit['step']:
        raise ValueError('dataset or checkpoint mismatch')
    if (audit['status'] != 'complete_macro_evaluation_outputs_audited'
            or audit['target_tuning_allowed'] is not False or manifest['target_tuning_allowed'] is not False):
        raise ValueError('audited sealed test panel required')
    def item(rel):
        return checked(seal_root/rel, manifest['files'][rel])
    targets = item(f'scoring/{dataset}/targets.json')
    panel = item(f'inference/{dataset}/panel.json')
    roster = item(f'inference/{dataset}/rows.json')
    labels = panel['candidate_ids']
    if labels != audit['metrics']['candidate_ids'] or len(labels) != audit['decisions']:
        raise ValueError('candidate order mismatch')
    fields = {}
    if dataset == 'FSC_native31':
        for q in panel['questions']:
            fields[q['id']] = tuple(re.search(r'\b'+key+r': ([^.]+)\.', q['definition'])[1]
                                    for key in ('action', 'object', 'location'))
    preds = verified_rows(c['predictions'], sha256=audit['predictions_sha256'],
                          recordings=audit['recordings'], decisions=len(labels), policy_id=audit['policy_id'])
    pairs, breakdown, winners, mismatch, bins = Counter(), Counter(), Counter(), Counter(), {}
    examples, outcomes = {}, {}
    confusion = Counter()
    for pred, actual, asset in zip(preds, targets, roster, strict=True):
        uid, gold = actual['id'], actual['gold']
        if uid != pred['id'] or uid != asset['id'] or actual['audio_sha256'] != asset['audio_sha256']:
            raise ValueError('invalid sealed row')
        p = pred['prob']
        if observe is not None:
            observe(uid,gold,labels,p)
        g = labels.index(gold)
        order = sorted(range(len(p)), key=p.__getitem__, reverse=True)
        i = order[0]
        winner, mx, pg = labels[i], p[i], p[g]
        delta = pg - p[next(j for j in order if j != g)]
        ok = i == g
        outcomes[uid] = ok
        confusion[gold, winner] += 1
        winners[winner] += 1
        flags = ('right' if ok else 'wrong')
        selected = sum(v >= .5 for v in p)
        breakdown[flags + ('_empty' if selected == 0 else '_single' if selected == 1 else '_multiple')] += 1
        if sum(v >= source['cutoff'] for v in p) == 1 and pg >= source['cutoff']:
            breakdown['exact_at_source_cutoff'] += 1
        b = min(int(mx*10), 9)
        vals = bins.setdefault(b,[0,0,0.])
        vals[0] += 1
        vals[1] += ok
        vals[2] += mx
        if not ok:
            pairs[gold+' → '+winner] += 1
            kind = 'near' if -delta < band else 'separated'
            breakdown[kind] += 1
            breakdown['wrong_high'] += mx >= confident
            breakdown['wrong_gold_top3'] += g in order[:3]
            if fields:
                diff = '+'.join(key for key, a, z in zip(('action','object','location'), fields[gold], fields[winner]) if a != z)
                mismatch[diff] += 1
            if kind not in examples or (kind == 'separated' and mx >= confident and examples[kind]['max'] < confident):
                examples[kind] = dict(id=uid,gold=gold,winner=winner,p_gold=pg,max=mx)
    metrics = audit['metrics']
    if (any(confusion[k,l] != metrics['confusion'][k][l] for k in labels for l in labels)
            or breakdown['right_single']/len(outcomes) != metrics['exact_set']):
        raise ValueError('decomposition disagrees with audited outcomes')
    return dict(n=len(outcomes), policy=audit['policy_id'], step=audit['step'], counts=breakdown,
                bins=bins, top1_ece10=sum(abs(v[2]-v[1]) for v in bins.values())/len(outcomes),
                positive_recall_classes=sum(confusion[k,k]>0 for k in labels),
                winners=winners, pairs=pairs.most_common(15), differing_fields=mismatch,
                fsc_panel_fields=fields, examples=examples, accuracy=metrics['top1_event_accuracy'],
                macro_f1=metrics['top1_macro_f1'], brier=metrics['candidate_brier'],
                nll=metrics['candidate_binary_nll'], mean_top=sum(v[2] for v in bins.values())/len(outcomes)), outcomes


def run(c):
    source = train(c['train'])
    result = dict(train=source, datasets={})
    policies = set()
    for dataset, points in c['datasets'].items():
        records, paired = {}, []
        for name in ('stage1','stage2'):
            info, outcomes = describe(points[name], source, Path(c['seal']), c['margin'], c['high'])
            records[name] = info
            paired.append(outcomes)
        if paired[0].keys() != paired[1].keys() or records['stage1']['policy'] != source['policy']:
            raise ValueError('different paired endpoints or population')
        policies.add(records['stage2']['policy'])
        records['transitions'] = Counter(('right' if a else 'wrong')+'_'+('right' if paired[1][i] else 'wrong')
                                          for i,a in paired[0].items())
        result['datasets'][dataset] = records
    if len(policies) != 1:
        raise ValueError('different terminal policies')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config','local'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    config, provenance = load_config(args.config, args.local)
    print(json.dumps(dict(**run(config), provenance=provenance), ensure_ascii=False, indent=2))
