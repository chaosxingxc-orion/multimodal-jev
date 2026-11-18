#!/usr/bin/env python3
"""Task-design audit of explicit harness policies; does not compile model inputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.slurp_official_training import load
from multimodal_jev.slurp_training import content_hash


FAMILIES = ('partition6', 'partition12', 'overlap6', 'availability6')


def policy(ontology, family, seed):
    """Input-only policy factory: no query, label or population is accepted."""
    if (family not in FAMILIES or type(seed) is not int or seed < 0
            or len(ontology) < 12 or len(set(ontology)) != len(ontology)
            or any(not isinstance(x, str) or not x for x in ontology)):
        raise ValueError('registered family, seed and unique complete ontology required')
    size = 12 if family == 'partition12' else 6
    order = sorted(ontology, key=lambda key: hashlib.sha256(
        json.dumps(['routing-policy-v1', seed, key], separators=(',', ':')).encode()).hexdigest())
    routes = [dict(id=f'route-{i:02d}', enabled=family != 'availability6' or i % 2 == 0,
                   accepts=[]) for i in range(size)]
    for index, event in enumerate(order):
        routes[index % size]['accepts'].append(event)
        if family == 'overlap6':
            routes[(index + 1) % size]['accepts'].append(event)
    for route in routes:
        route['accepts'].sort()
    return dict(family=family, seed=seed, routes=routes, fallback='no_route')


def target(event, ontology, caller_policy):
    """External supervision binding, never used to select the input policy."""
    routes = caller_policy['routes']
    ids = [r['id'] for r in routes]
    if (event not in ontology or not routes or len(ids) != len(set(ids)) or 'no_route' in ids
            or caller_policy['fallback'] != 'no_route'
            or any(type(r['enabled']) is not bool or not r['accepts']
                   or len(r['accepts']) != len(set(r['accepts']))
                   or not set(r['accepts']) <= set(ontology) for r in routes)):
        raise ValueError('known event, unique candidates and explicit valid rule table required')
    truth = {r['id']: r['enabled'] and event in r['accepts'] for r in routes}
    valid = [uid for uid, yes in truth.items() if yes]
    return dict(candidate_applicability=truth, valid_final=valid or ['no_route'],
                no_route=not valid, execution_success=None, semantic_unknown=None)


def analyze(rows, task, seed=317):
    ontology = sorted(task['candidate_pairs'])
    if (len(rows) != len({r['id'] for r in rows}) or not rows
            or any(r['role'] != 'learning' or r['known_pending_source']
                   or r['official_split'] != 'train' or r['intent'] not in ontology
                   or task['candidate_pairs'][r['intent']] != [r['scenario'], r['action']] for r in rows)):
        raise ValueError('complete admitted learning metadata required')
    counts = Counter(r['intent'] for r in rows)
    result = []
    for family in FAMILIES:
        spec = policy(ontology, family, seed)
        changed = policy(ontology, family, seed + 1)
        sizes = Counter(); per_route = Counter(); none = changed_set = 0
        for event, count in counts.items():
            truth = target(event, ontology, spec)
            sizes[len(truth['valid_final'])] += count
            none += count * truth['no_route']
            per_route.update({uid:count for uid, yes in truth['candidate_applicability'].items() if yes})
            changed_set += count * (set(truth['valid_final']) != set(target(event, ontology, changed)['valid_final']))
        result.append(dict(policy=spec, policy_sha256=content_hash(spec),
            valid_final_set_size_counts=dict(sorted(sizes.items())), no_route_recordings=none,
            positive_candidate_counts={r['id']:per_route[r['id']] for r in spec['routes']},
            total_candidate_boolean_decisions=len(rows)*len(spec['routes']),
            changed_valid_set_under_seed_plus_one=changed_set))
    return dict(recordings=len(rows), source_groups=len({r['source_group'] for r in rows}),
        ontology_size=len(ontology), policy_families=result, population_sha256=content_hash(
            sorted((r['id'], r['source_group'], r['intent'], r['audio_sha256']) for r in rows)),
        training_dispatch_allowed=False, new_independent_audio=0,
        scope='Full-learning symbolic routing supervision support only. Caller policies are generated without gold; targets use existing exclusive labels. No new semantic annotation, model execution, or agent success evidence.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--release-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows,task,_=load(args.release,args.release_sha256,'learning',purpose='optimize')
    report=analyze(rows,task)
    report.update(release=str(args.release.resolve()),release_sha256=args.release_sha256,
        analyzer_sha256=file_hash(__file__),contract_sha256=file_hash('configs/tasks/harness-routing-contract-candidate-v1.json'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps([dict(family=r['policy']['family'],no_route=r['no_route_recordings'],
        valid_set_sizes=r['valid_final_set_size_counts'],policy_change=r['changed_valid_set_under_seed_plus_one'])
        for r in report['policy_families']],indent=2))
