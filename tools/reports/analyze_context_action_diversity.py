"""Label-free exact action collision diagnostics; never changes sampling."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct


def distribution(prob, group=16):
    if not prob or group < 2 or any(not math.isfinite(p) or p < 0 or p > 1 for p in prob):
        raise ValueError('Finite Bernoulli probabilities and group >=2 required')
    return {
        'all_group_actions_identical_probability': math.prod(p**group+(1-p)**group for p in prob),
        'pairwise_action_difference_probability': 1-math.prod(p*p+(1-p)**2 for p in prob),
        'expected_pairwise_hamming_distance': sum(2*p*(1-p) for p in prob),
        'joint_entropy_nats': -sum(x*math.log(x) for p in prob for x in (p,1-p) if x),
        'coordinates_below_1e_4_or_above_1_minus_1e_4': sum(p < 1e-4 or p > 1-1e-4 for p in prob),
        'exact_endpoint_coordinates': sum(p in (0,1) for p in prob),
    }


def analyze(observation_path):
    raw=Path(observation_path).read_bytes();live=json.loads(raw)
    ranks=live['progress']['latest']
    if len(ranks)!=4 or {Path(r['path']).parent.name for r in ranks}!={f'rank-{i}' for i in range(4)}:
        raise ValueError('Four distinct ranks required')
    rollouts=[r['data']['rollout'] for r in ranks]
    if any(r!=rollouts[0] for r in rollouts[1:]):raise ValueError('Replicated global rollout differs across ranks')
    rollout=rollouts[0];queries=rollout['queries']
    if len(queries)!=128 or len({q['request_id'] for q in queries})!=128:raise ValueError('Unique global128 required')
    details=[]
    for q in queries:
        b,r=q['behavior_record'],q['reference_record']
        if b['policy_id']!=rollout['behavior_policy_id'] or r['policy_id']!=rollout['fixed_reference_policy_id']:
            raise ValueError('Policy mismatch')
        if any(b['request_identity'][k]!=r['request_identity'][k] for k in ('inputs','positions')):
            raise ValueError('Input mismatch')
        if len(b['prob'])!=60 or len(r['prob'])!=60 or q['rollout']['group_size']!=16:raise ValueError('Action contract mismatch')
        if hashlib.sha256(struct.pack('<60f',*b['prob'])).hexdigest()!=q['rollout']['probability']['sha256']:
            raise ValueError('Probability tensor mismatch')
        details.append({'request_id':q['request_id'],'behavior':distribution(b['prob']), 'reference':distribution(r['prob'])})
    summary={}
    for side in ('behavior','reference'):
        summary[side]={k:sum(d[side][k] for d in details)/len(details) for k in details[0][side]}
        summary[side]['queries_with_identical_group_probability_over_0_95']=sum(d[side]['all_group_actions_identical_probability']>.95 for d in details)
    return {'schema':'context-live-action-diversity-v1','observation':{'path':str(observation_path),'sha256':hashlib.sha256(raw).hexdigest()},
            'actor_update':ranks[0]['data']['update'],'behavior_update':rollout['update'],
            'behavior_policy_id':rollout['behavior_policy_id'],'reference_policy_id':rollout['fixed_reference_policy_id'],
            'global_queries':128,'rank_replicas_counted_once':True,'summaries':summary,'queries':details,
            'limits':['Live single TRAIN batch, not audited completed training or heldout efficacy.',
                      'No gold read: confidence cannot distinguish correct from incorrect certainty.',
                      'Identical actions imply identical deterministic rewards; converse is false. Collision is a lower bound on zero reward-variance probability.',
                      'Distribution expectations, not replay of actual sampled actions or gradient norms.',
                      'Reference comparison uses identical inputs; no across-batch learning trend inferred.']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--observation',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=analyze(a.observation);result['tool_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['summaries'],indent=2))
