"""Read-only reporting on audited held-out endpoints; never consumed by TRAIN."""
import argparse
from collections import Counter
import json
from pathlib import Path
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.heldout_error_attribution import describe
from tools.reports.stage1_top3_misses import inspect_rank

def run(c,provenance):
    result={}
    for name, endpoints in c['datasets'].items():
        pair={}; full_ranks={}
        for stage, endpoint in endpoints.items():
            misses=[]; ranks={}
            def collect(uid,gold,labels,p):
                r=inspect_rank(uid,gold,labels,p)
                ranks[uid]=(gold,r['rank'])
                if r['rank']>3:
                    misses.append(r)
            # describe also checks the whole confusion matrix; source cutoff is unused for ranking.
            info,_=describe(endpoint,{'cutoff':.5},Path(c['seal']),.1,.9,observe=collect)
            pair[stage]=dict(step=info['step'],policy=info['policy'],audit_sha256=endpoint['audit_sha256'],
                             n=info['n'],misses=misses)
            if len(ranks)!=info['n']:raise ValueError('unique complete endpoint required')
            full_ranks[stage]=ranks
        before,after=(full_ranks[k] for k in ('stage1','stage2'))
        if before.keys()!=after.keys() or any(before[i][0]!=after[i][0] for i in before):
            raise ValueError('paired recordings with identical gold required')
        deltas=Counter(before[i][1]-after[i][1] for i in before)
        pair['full_rank_changes']=dict(improved=sum(n for d,n in deltas.items() if d>0),
            unchanged=deltas[0],worsened=sum(n for d,n in deltas.items() if d<0),
            rank_gain_histogram=dict(sorted(deltas.items())))
        old,new=({r['id']:r['rank'] for r in pair[k]['misses']} for k in ('stage1','stage2'))
        common=old.keys() & new.keys()
        pair['changes']=dict(recovered=len(old.keys()-new.keys()),new_misses=len(new.keys()-old.keys()),
            still_out=len(common),still_out_rank_improved=sum(new[i]<old[i] for i in common),
            still_out_rank_unchanged=sum(new[i]==old[i] for i in common),
            still_out_rank_worsened=sum(new[i]>old[i] for i in common))
        result[name]=pair
    commit_json(c['output'],dict(role='test_reporting_only',provenance=provenance,datasets=result))
    return {k:v['changes'] for k,v in result.items()}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();print(json.dumps(run(*load_config(a.config,a.local)),indent=2))
