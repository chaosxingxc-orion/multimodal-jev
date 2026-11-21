"""Enumerate misses on the authenticated full native TRAIN reference; no model calls."""
import argparse
from collections import Counter
import json

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.jev_train_confusions import run


def inspect_rank(uid,gold,labels,probs):
    idx=labels.index(gold); p=probs[idx]
    order=sorted(range(len(labels)),key=lambda i:-probs[i])
    best=1+sum(x>p for x in probs); worst=sum(x>=p for x in probs)
    return dict(id=uid,gold=gold,p_gold=p,rank=order.index(idx)+1,rank_best=best,rank_worst=worst,
                top3=[dict(label=labels[i],p=probs[i]) for i in order[:3]])


def collect(c, provenance):
    misses=[]; ties=[]; counts=Counter(); classes=Counter(); pairs=Counter()
    def visit(uid,gold,labels,probs):
        r=inspect_rank(uid,gold,labels,probs)
        counts[r['rank']]+=1
        if r['rank_best']<=3<r['rank_worst']:
            ties.append(r)
        if r['rank']>3:
            misses.append(r);classes[gold]+=1;pairs[gold,r['top3'][0]['label']]+=1
    verified=run(c,observe=visit)
    summary=dict(n=verified['recordings'],stable_order_misses=len(misses),
                 definitely_outside=sum(x['rank_best']>3 for x in misses),
                 boundary_ties=len(ties),ranks=dict(sorted(counts.items())),
                 by_gold=[dict(label=k,misses=n,total=verified['support'][k]) for k,n in classes.most_common()],
                 confusions=[dict(gold=k[0],winner=k[1],n=n) for k,n in pairs.most_common()])
    commit_json(c['output'],dict(summary=summary,policy_id=verified['policy_id'],reference_sha256=c['reference_sha256'],
                provenance=provenance,order='descending raw probability; frozen candidate order breaks ties',
                misses=misses,third_place_boundary=ties))
    return {k:v[:7] if isinstance(v,list) else v for k,v in summary.items()}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();print(json.dumps(collect(*load_config(a.config,a.local)),indent=2))
