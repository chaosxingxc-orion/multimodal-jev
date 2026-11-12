#!/usr/bin/env python3
"""Recount all final-method cells without calling the production scorer.

Authenticates the completed comparison queue and exact source artifacts, then
checks every table count against saved per-request predictions and labels.
Adds descriptive fact/polarity diagnostics; no model calls or label edits.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def audit(comparison, queue):
    comparison, queue = Path(comparison), Path(queue)
    completed = read(queue/'run/completed.json')
    if read(queue/'run/terminal.json')['exit_code'] != 0 or completed['sha256'] != sha(comparison) or completed['result'] != str(comparison.resolve()):
        raise ValueError('completed exact comparison required')
    config = read(queue/'config.json')
    for p,h in config['pins'].items():
        if sha(p) != h: raise ValueError('comparison implementation changed: '+p)
    report = read(comparison)
    if not report['passed']: raise ValueError('comparison failed')
    pins = {str(comparison.resolve()):sha(comparison), str(queue/'run/completed.json'):sha(queue/'run/completed.json')}
    diagnostics = []
    for method,evidence in report['evidence'].items():
        root = Path(evidence['root']); study = read(root/'study.json')
        if sha(root/'study.json') != evidence['study_sha256']: raise ValueError('study changed')
        gold = {}; predictions = {}
        for role in ('learning','selection'):
            stage = root/('final-'+role); c = read(stage/'config.json')
            for p,h in c['pins'].items():
                if sha(p) != h: raise ValueError('evaluation asset changed: '+p)
            release = Path(c['release']); manifest = read(release/'MANIFEST.json')
            source = release/(role+'.jsonl')
            if sha(source) != manifest['files'][source.name]: raise ValueError('gold source changed')
            gr = rows(source); pr = rows(stage/'run/records.jsonl')
            if len({r['id'] for r in gr}) != len(gr) or len({r['id'] for r in pr}) != len(pr):
                raise ValueError('duplicate row')
            if {r['id'] for r in gr} != {r['id'] for r in pr}: raise ValueError('incomplete prediction roster')
            gold.update({r['id']:r for r in gr}); predictions.update({r['id']:r for r in pr})
            pins[str(source)] = sha(source)
            pins[str(stage/'run/records.jsonl')] = sha(stage/'run/records.jsonl')
        for cell, spec in study['evaluation_cells'].items():
            for arm in ('final_only','structured','reordered'):
                selected = [gold[uid] for uid in spec['request_ids'] if gold[uid]['arm'] == arm]
                by_rule = defaultdict(list); pairs = defaultdict(dict)
                correct = all_correct = true_pred = fact_correct = fact_total = 0
                for row in selected:
                    actual = predictions[row['id']]; pred = actual['predictions']; target = row['targets']
                    if actual['request_identity']['request'] != row['request'] or len(pred) != len(target):
                        raise ValueError('prediction input/shape differs')
                    if not all(type(x) is bool for x in pred+target) or not all(row['supervised']):
                        raise ValueError('complete Boolean diagnostic required')
                    correct += pred[-1] == target[-1]; all_correct += pred == target
                    true_pred += pred[-1]
                    fact_correct += sum(a==b for a,b in zip(pred[:-1],target[:-1],strict=True));fact_total += len(target)-1
                    provenance = row['provenance']; by_rule[provenance['rule_id']].append((target[-1],pred[-1]))
                    key = (provenance['recording_id'],provenance['complement_pair_id'])
                    polarity = provenance['rule_polarity']
                    if polarity in pairs[key]:raise ValueError('duplicate polarity')
                    pairs[key][polarity] = (target[-1],pred[-1])
                flips = both = same_false = same_true = 0
                for pair in pairs.values():
                    if set(pair) != {'positive','negative'}: raise ValueError('incomplete complement')
                    (ga,pa),(gb,pb) = pair['positive'],pair['negative']
                    if ga == gb: raise ValueError('noncomplementary gold')
                    flips += pa != pb; both += pa == ga and pb == gb
                    same_false += not pa and not pb; same_true += pa and pb
                ba = []
                for values in by_rule.values():
                    recalls = []
                    for label in (False,True):
                        supported = [p==g for g,p in values if g==label]
                        if not supported:raise ValueError('unsupported rule class')
                        recalls.append(sum(supported)/len(supported))
                    ba.append(sum(recalls)/2)
                expected = next(r for r in report['table'] if (r['method'],r['cell'],r['arm']) == (method,cell,arm))
                for key,value in [('requests',len(selected)),('final_accuracy',correct/len(selected)),
                                  ('all_span_accuracy',all_correct/len(selected)),('macro_rule_ba',sum(ba)/len(ba))]:
                    if abs(expected[key]-value)>1e-12:raise ValueError('recount mismatch: '+key)
                if expected['complement'] != dict(n=len(pairs),prediction_flips=flips,both_correct=both):
                    raise ValueError('complement table mismatch')
                diagnostics.append(dict(method=method,cell=cell,arm=arm,requests=len(selected),
                    final_correct=correct,final_true=true_pred,final_accuracy=correct/len(selected),
                    fact_correct=fact_correct,fact_decisions=fact_total,
                    fact_accuracy=fact_correct/fact_total if fact_total else None,
                    complement_pairs=len(pairs),both_correct=both,prediction_flips=flips,
                    same_false=same_false,same_true=same_true))
    return dict(passed=True,source_files=pins,cells_recounted=len(diagnostics),diagnostics=diagnostics,
        scope='Independent saved-row recount of all 36 final table cells and fact/polarity diagnostics. Does not independently rerun bootstrap, optimizer audit or inference; one-seed development results only.')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--comparison',type=Path,required=True);p.add_argument('--queue',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.comparison,a.queue)
    with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({'passed':r['passed'],'cells_recounted':r['cells_recounted']}))
