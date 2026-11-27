"""Fixed source TRAIN controls: reuse endpoint scores, never infer audio similarity from card scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from multimodal_jev.context_execution import map_io
from multimodal_jev.local_config import resolve
from multimodal_jev.speech_alignment_pairs import KINDS, Partners
from tools.reports.prepare_speech_pairs import compile_view, checked


def panel_scores(root, pop, workers, arm):
    def read(p):
        return json.loads(p.read_bytes())
    a=read(root/'audit.json')
    full=read(checked(root/'entry/full.json',a['full_sha256']))
    receipt=read(root/'collected/score/receipt.json')
    if not a['passed'] or not a['container_exited'] or any(receipt[k]!=a[k] for k in (
            'full_sha256','policy_id','policy_sha256','admission_sha256')):
        raise ValueError('invalid endpoint lineage')
    labels=full['labels']
    ids=[r['id'] for r in full['rows']]
    if (full['card_arm']!=arm or len(ids)!=len(pop) or set(ids)!=pop.keys() or labels!=a['labels'] or
            any((r['intent'],r['audio_sha256'])!=(pop[r['id']]['intent'],pop[r['id']]['audio_sha256'])
                for r in full['rows'])):
        raise ValueError('different TRAIN roster')
    ix={u:i for i,u in enumerate(ids)}
    matrix=np.full((len(ids),len(labels)),np.nan,dtype=np.float32)
    def load(job):
        path,sha=job
        rec=read(checked(path,sha))['payload']['records']
        return [ix[r['id']] for r in rec], np.asarray([r['prob'] for r in rec],dtype=np.float32)
    jobs=[(root/'collected/score'/f'worker-{i}'/b['name'],b['sha256'])
          for i,worker in enumerate(receipt['workers']) for b in worker['blocks']]
    blocks,_=map_io(load,jobs,workers=workers)
    for idx,v in blocks:
        if len(set(idx))!=len(idx) or np.isfinite(matrix[idx]).any() or v.shape!=(len(idx),len(labels)):
            raise ValueError('overlapping or incorrect endpoint block')
        matrix[idx]=v
    if not ((matrix>0)&(matrix<1)).all():
        raise ValueError('missing or invalid membership probability')
    return a,labels,ids,matrix


def audit(c, root):
    view, pop=compile_view(c)
    sampler=Partners(view,pop)
    a,labels,ids,prob=panel_scores(root,pop,c['io_workers'],sampler.arm)
    ix={u:i for i,u in enumerate(ids)}
    gold=np.array([labels.index(pop[u]['intent']) for u in ids])
    true=prob[np.arange(len(ids)),gold]
    wrong=prob.copy()
    wrong[np.arange(len(ids)),gold]=0
    gaps=true-wrong.max(1)
    tables={k:Counter(positive_anchors=0,full_menu_both_correct=0,anchor_wrong=0,
                      peer_restores_wrong=0,matched_panels=0,cards_switched=0,
                      cards_invariant=0,cards_both=0) for k in KINDS}
    panels=hashlib.sha256();covered=set();hard_set=set()
    for uid in sorted(ids):
        seen=set()
        for epoch in range(len(sampler.cycle)):
            k,p,n=sampler.draws(uid,epoch,c['seed'])
            if not k or k in seen:
                continue
            seen.add(k)
            panels.update(json.dumps([uid,k,p,n],separators=(',',':')).encode()+b'\n')
            t=tables[k]; i,j=ix[uid],ix[p]
            t['positive_anchors']+=1
            t['full_menu_both_correct']+=int(gaps[i]>0 and gaps[j]>0)
            t['anchor_wrong']+=int(gaps[i]<0)
            t['peer_restores_wrong']+=int(gaps[i]<0 and gaps[j]>0)
            if n is None:
                continue
            covered.add(uid)
            if gaps[i]<0:hard_set.add(uid)
            l=ix[n];g,h=gold[i],gold[l]
            invariant=prob[i,g]>prob[i,h] and prob[j,g]>prob[j,h]
            switch=prob[i,g]>prob[i,h] and prob[l,h]>prob[l,g]
            t['matched_panels']+=1;t['cards_invariant']+=int(invariant)
            t['cards_switched']+=int(switch);t['cards_both']+=int(invariant and switch)
    return dict(scope='official admitted TRAIN / endpoint rules / strict comparison, ties fail',
                rows=len(ids),classes=len(labels),source=view['source'],policy_id=a['policy_id'],
                panel_sha256=panels.hexdigest(),full_sha256=a['full_sha256'],seed=c['seed'],
                semantic_sampling=c['semantic_sampling'],strict_errors=int((gaps<0).sum()),
                eligible_anchors=len(covered),eligible_strict_error_anchors=len(hard_set),
                ambiguous_rows=sum(k<0 for k in view['rows'].values()),by_relation=tables,
                audio_retrieval=None,audio_retrieval_reason='endpoint archive contains membership probabilities only')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('recipe',type=Path);p.add_argument('macros',type=Path);p.add_argument('scores',type=Path)
    args=p.parse_args(); cfg=json.loads(args.recipe.read_bytes())
    inputs={k:cfg[k] for k in ('seed','bank','bank_sha256','annotations','release','semantic_sampling','io_workers')}
    inputs['mixed']={k:cfg['mixed'][k] for k in ('data','admission_sha256')}
    config,_=resolve(inputs,json.loads(args.macros.read_bytes())['macros'])
    print(json.dumps(audit(config,args.scores),indent=2,allow_nan=False))
