"""Enumerate all native top-1 confusions on the authenticated full source TRAIN reference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json


def pinned(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'changed input: {path.name}')
    return raw


def run(c, *, observe=None):
    root, source = Path(c['reference']), Path(c['release'])
    m = json.loads(pinned(root/'MANIFEST.json', c['reference_sha256']))
    identity = m['identity']
    release = json.loads(pinned(source/'MANIFEST.json', identity['release_sha256']))
    pop, records = (json.loads(pinned(root/k, m['files'][k])) for k in ('population.json','records.json'))
    rows = pinned(source/'learning.jsonl', release['files']['learning.jsonl'])
    labels = identity['candidate_ids']
    if len(set(labels)) != len(labels) or set(labels) != set(release['counts']['learning']['classes']):
        raise ValueError('candidate panel mismatch')
    seen = set()
    pairs, support = Counter(), Counter()
    ties = 0
    cases = []
    for line in rows.splitlines():
        r = json.loads(line)
        uid, gold = r['id'], r['intent']
        if (uid in seen or r['role'] != 'learning' or r['official_split'] != 'train'
                or r['known_pending_source'] is not False or gold != r['scenario']+'_'+r['action']):
            raise ValueError('unique native TRAIN rows required')
        seen.add(uid)
        record = records[pop[uid]]
        probs = record['prob']
        if (record['policy_id'] != identity['policy_id'] or record['request_id'] != pop[uid]
                or len(probs) != len(labels) or not all(type(x) in (float,int) and 0<=x<=1 for x in probs)):
            raise ValueError('reference identity or vector mismatch')
        peak = max(probs)
        competing = [label for label, probability in zip(labels, probs, strict=True) if probability == peak]
        winner = competing[0]
        if observe is not None:
            observe(uid, gold, labels, probs)
        ties += len(competing) > 1
        support[gold] += 1
        if gold != winner:
            pairs[gold,winner] += 1
        if 'output' in c and (competing != [gold]):
            kind = ('gold_tie' if gold in competing else 'same_scenario'
                    if any(label.startswith(r['scenario'] + '_') for label in competing) else 'cross_scenario')
            cases.append(dict(id=uid, audio_sha256=r['audio_sha256'], source_group=r['source_group'],
                              gold=gold, winner=winner, competing=competing, sampling_type=kind))
    if (seen != set(pop) or set(pop.values()) != set(records) or len(seen) != m['recordings']
            or dict(support) != release['counts']['learning']['classes']):
        raise ValueError('full learning population differs')
    return dict(schema='jev-native-train-confusions-v1', policy_id=identity['policy_id'],
                reference_sha256=c['reference_sha256'], release_sha256=identity['release_sha256'],
                recordings=len(seen), top1_errors=sum(pairs.values()), top1_ties=ties,
                support=support, pairs=[dict(gold=g, winner=w, count=n) for (g,w),n in pairs.most_common()],
                tie_rule='first candidate in frozen order', labels_source='clean native TRAIN',
                causal_diagnoses=0, model_calls=0,
                **(dict(strict_cases=sorted(cases, key=lambda row: row['id']),
                        strict_case_counts=dict(Counter(row['sampling_type'] for row in cases))) if 'output' in c else {}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    a = p.parse_args()
    cfg, provenance = load_config(a.config,a.local)
    report = dict(**run(cfg), provenance=provenance)
    if 'output' in cfg:
        path = Path(cfg['output'])
        if path.is_file():
            if json.loads(path.read_text()) != report:
                raise ValueError('existing collection differs from a fresh recomputation')
        else:
            commit_json(path, report)
        print(json.dumps({k: report[k] for k in ('recordings','strict_case_counts','reference_sha256')}))
    else:
        print(json.dumps(report, indent=2))
