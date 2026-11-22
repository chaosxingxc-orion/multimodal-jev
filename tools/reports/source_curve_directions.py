"""Summarize the two-sided TRAIN error directions in an audited checkpoint curve."""
from collections import Counter
import argparse
import json
from pathlib import Path


def summarize(table):
    directions=table['directions']
    effects=Counter()
    for key,v in directions.items():
        a,b=key.split('->')
        rev=directions.get(b+'->'+a)
        if rev is None:
            effects['without_reverse_source_rows']+=v['n']
            continue
        if rev['n']!=v['reverse_n']:
            raise ValueError('reverse support mismatch')
        if a>b:
            continue
        net=[x['final_correct']-x['mid_correct'] for x in (v,rev)]
        kind=('tradeoff' if min(net)<0<max(net) else 'unchanged' if net==[0,0]
              else 'both_nonnegative' if min(net)>=0 else 'both_nonpositive')
        effects[kind+'_boundaries']+=1
        effects[kind+'_rows']+=v['n']+rev['n']
        effects[kind+'_gains']+=sum(max(0,x) for x in net)
        effects[kind+'_losses']+=sum(max(0,-x) for x in net)
    wrong=sum(v['n'] for v in directions.values())
    if wrong != sum(table[k]['points']['stage1']['n'] for k in table if k.startswith('error-q')):
        raise ValueError('source errors do not reconcile')
    return dict(source_errors=wrong,**effects)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt',type=Path);parser.add_argument('panel')
    a=parser.parse_args();r=json.loads(a.receipt.read_bytes())
    if r['status']!='audited_curve':
        raise ValueError('requires collected curve')
    print(json.dumps(summarize(r['sections'][a.panel]),sort_keys=True))
