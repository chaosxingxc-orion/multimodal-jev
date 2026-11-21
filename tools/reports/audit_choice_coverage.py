"""Count actual, non-padding TRAIN slates; exposure counts and recording counts differ."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse,json
from multimodal_jev.baseline import file_hash


def summarize(rows,batches,ids):
    cells=defaultdict(lambda:dict(hit=Counter(),miss=Counter(),positions=Counter(),proposals=Counter()))
    signatures=set(); population=set(); outcomes=defaultdict(set)
    for b in batches:
        if b['mode']=='noul':continue
        for s in b['slots']:
            if s['padding']:continue
            uid=s['id'];r=rows[uid]; gold=r['gold']; mode=b['mode']
            indices=s.get('slate', r.get('slate') if mode=='open' else list(range(len(ids))))
            allow=mode in ('open','choice')
            if not indices or len(indices)!=len(set(indices)) or not all(0<=i<len(ids) for i in indices):
                raise ValueError('invalid action support')
            hit=gold in indices
            if not hit and not allow: raise ValueError('unanswerable forced choice')
            cell=cells[len(indices),allow]
            cell['hit' if hit else 'miss'][gold]+=1
            cell['proposals'].update(indices)
            if hit:cell['positions'][indices.index(gold)+1]+=1
            signatures.add((uid,tuple(indices),allow));population.add(uid)
            if allow:outcomes[uid].add(hit)
    report={}
    for (k,allow),c in sorted(cells.items()):
        hits,misses=c['hit'],c['miss']
        report[f'k{k}_none{int(allow)}']=dict(k=k,allow_none=allow,hits=hits.total(),misses=misses.total(),
            gold_hit_classes=len(hits),gold_miss_classes=len(misses),positions=dict(sorted(c['positions'].items())),
            by_intent={tag:dict(hit=hits[i],miss=misses[i],offered=c['proposals'][i]) for i,tag in enumerate(ids)})
    return dict(choice_recordings=len(population),unique_recording_ordered_slates=len(signatures),
                recordings_with_both_none_targets=sum(len(x)==2 for x in outcomes.values()),cells=report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('view',type=Path);a=p.parse_args()
    m=json.loads((a.view/'MANIFEST.json').read_bytes())
    def read(n):
        if file_hash(a.view/n)!=m['files'][n]:raise ValueError('unverified view')
        return json.loads((a.view/n).read_bytes())
    print(json.dumps(summarize(read('rows.json'),read('batches.json'),m['candidate_ids']),indent=2))
