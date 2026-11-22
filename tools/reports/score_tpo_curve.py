"""Join the three source checkpoints by recording and report paired changes."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from multimodal_jev import evaluation_blocks as blocks
from multimodal_jev.context_execution import map_io
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.run_native_seal_endpoint import bound


def describe(probs, gold, groups):
    actual = np.arange(probs.shape[1]) == gold[:,None]
    win = probs.argmax(1)
    hit = win == gold
    pos = probs[actual]
    rival = np.where(actual,-1,probs).max(1)
    support = np.bincount(gold,minlength=probs.shape[1])
    count = np.bincount(win,minlength=len(support))
    hits = np.bincount(gold[hit],minlength=len(support))
    by_group = defaultdict(list)
    for i,g in enumerate(groups):
        by_group[g,gold[i]].append(i)
    repeated = [v for v in by_group.values() if len(v)>1]
    return dict(n=len(gold),correct=int(hit.sum()),accuracy=float(hit.mean()),
                macro_f1=float(np.divide(2*hits,support+count,out=np.zeros_like(hits,dtype=float),where=support+count>0).mean()),
                pos_brier=float(np.mean((1-pos)**2)),neg_brier=float(np.mean(probs[~actual]**2)),
                mean_top1=float(probs.max(1).mean()),mean_gold=float(pos.mean()),
                mean_rival=float(rival.mean()),mean_rank=float((1+(probs>pos[:,None]).sum(1)).mean()),
                all_correct_groups=sum(bool(hit[v].all()) for v in repeated),repeated_groups=len(repeated))


def run(spec):
    c=spec['curve']; tr,sl,mid=c['train'],c['selection'],c['mid']
    result=defaultdict(dict)
    root=Path(spec['queue']['root'])
    new=blocks.checked_plan(root,spec['queue']['namespace'])
    oldroot=Path(c['old_queue']);old=blocks.checked_plan(oldroot,c['old_namespace'])
    if old['sha256']!=c['old_plan']:raise ValueError('historical queue changed')
    def append(ds,arm,payload,pid):
        if payload['policy_id']!=pid:raise ValueError('policy mismatch')
        for row in payload['records']:
            uid=row['id']
            if uid in result[ds,arm]:raise ValueError('duplicate recording')
            result[ds,arm][uid]=row['prob']
    for src,m,selected in [(root,new,new['plan']['blocks']),
            (oldroot,old,[b for b in old['plan']['blocks'] if b['dataset']==sl and b['arm']=='tpo'])]:
        def visit(b):return blocks.read_block(src,m,b)['prob']
        pubs,_=map_io(visit,selected,workers=spec['io_workers'])
        for b,payload in zip(selected,pubs,strict=True):
            append(b['dataset'],b['arm'],payload,m['plan']['policies'][b['arm']])
    for shard in c['old_shards']:
        append(sl,'stage1',bound(shard),c['old_metrics']['stage1']['policy_id'])
    truth=bound(c['truth'])
    result[tr,'stage1']={u:r['prob'] for u,r in truth.items()}
    labels=c['labels'];idx={k:i for i,k in enumerate(labels)}
    cache=spec['endpoints'][sl]['cache'];seal=cache['seal'];base=Path(seal['root'])
    sm=bound(dict(path=base/'MANIFEST.json',sha256=seal['sha256']))
    def sealed(name):return bound(dict(path=base/name,sha256=sm['files'][name]))
    assert sealed(f'inference/{sl}/panel.json')['candidate_ids']==labels
    sel=sealed(f'scoring/{sl}/targets.json')
    goldrows={sl:{r['id']:(idx[r['gold']],r['group']) for r in sel},
              tr:{u:(r['gold'],r['group']) for u,r in truth.items()}}
    sections={};arms=['stage1',mid,'tpo']
    for ds,t in goldrows.items():
        uids=list(t);g=np.array([t[u][0] for u in uids]);groups=[t[u][1] for u in uids]
        p={}
        for a in arms:
            ix=result[ds,a]
            if ix.keys()!=t.keys():raise ValueError('incomplete curve')
            p[a]=np.array([ix[u] for u in uids])
            if p[a].shape!=(len(g),len(labels)) or not np.isfinite(p[a]).all():raise ValueError('score matrix differs')
        winners={a:v.argmax(1) for a,v in p.items()}
        hit={a:h==g for a,h in winners.items()}
        masks={'all':np.ones(len(g),bool)}
        if ds==tr:
            strata=[('error-q'+str(truth[u]['projected'])) if truth[u]['stratum']=='error' else truth[u]['stratum'] for u in uids]
            masks.update({k:np.array(strata)==k for k in set(strata)})
        table={}
        for k,mask in masks.items():
            ids=np.flatnonzero(mask)
            table[k]={'groups':len({groups[i] for i in ids}),
                      'points':{a:describe(p[a][mask],g[mask],[groups[i] for i in ids]) for a in arms},
                      'paired':{a+'-'+b:dict(fix=int((mask & ~hit[a] & hit[b]).sum()),
                           regression=int((mask & hit[a] & ~hit[b]).sum())) for a,b in
                           [('stage1',mid),(mid,'tpo'),('stage1','tpo')]}}
        if ds==sl:
            for arm in ('stage1','tpo'):
                proof=c['old_metrics'][arm];v=table['all']['points'][arm]
                if any(abs(v[k]-proof[key])>1e-10 for k,key in [('accuracy','top1_accuracy'),('macro_f1','top1_macro_f1'),
                            ('pos_brier','positive_brier'),('neg_brier','negative_brier')]):raise ValueError('endpoint metrics differ')
            from multimodal_jev.paired_event_decisions import compare
            def mapping(a):return {u:dict(zip(labels,row.tolist())) for u,row in zip(uids,p[a])}
            rule=spec['probability_panels'][sl]
            delta=compare(labels,sel,mapping(mid),mapping('tpo'),interval=rule['interval'],cluster=rule['cluster'])
            table['second_epoch_f1']={k:delta[k] for k in ('macro_f1_delta','macro_f1_delta_ci95')}
        else:
            directions=defaultdict(list)
            for i,u in enumerate(uids):
                if truth[u]['stratum']=='error':directions[(g[i],truth[u]['competitor'])].append(i)
            table['directions']={labels[a]+'->'+labels[b]: dict(n=len(v),groups=len({groups[i] for i in v}),
                   reverse_n=len(directions.get((b,a),[])),mid_correct=int(hit[mid][v].sum()),
                   final_correct=int(hit['tpo'][v].sum())) for (a,b),v in directions.items()}
        sections[ds]=table
    report=dict(status='audited_curve',plan=new['sha256'],policies=new['plan']['policies'],
                initial=c['old_metrics']['stage1']['policy_id'],sections=sections)
    commit_json(Path(spec['result_path']),report)
    return dict(status=report['status'])
