"""Matched full-ontology endpoint comparisons; frozen clusters and canonical tie rule."""
import numpy as np


def compare(ids, targets, left, right, *, interval, cluster, collect_ids=False):
    if interval.get('method')!='paired_cluster_percentile_bootstrap' or interval.get('confidence')!=.95:
        raise ValueError('registered paired percentile interval required')
    if type(interval['samples']) is not int or interval['samples']<100 or type(interval['seed']) is not int:
        raise ValueError('fixed replicates and seed required')
    if len(ids)<2 or len(set(ids))!=len(ids) or any(not isinstance(s,str) or not s for s in ids):
        raise ValueError('complete unique native ontology required')
    labels={k:i for i,k in enumerate(ids)}
    uids=[r['id'] for r in targets]
    if len(set(uids))!=len(uids) or set(uids)!=set(left) or set(uids)!=set(right) or not uids:
        raise ValueError('identical complete recording population required')
    if any(not isinstance(r[cluster],str) or not r[cluster] or r['gold'] not in labels for r in targets):
        raise ValueError('known gold and complete cluster binding required')
    clusters,group=np.unique([r[cluster] for r in targets],return_inverse=True)
    if len(clusters)<2:raise ValueError('independent clusters required')
    gold=np.array([labels[r['gold']] for r in targets]);n=len(uids);k=len(ids)
    mask=gold[:,None]==np.arange(k)
    counts=[];metrics=[];arrays=[];winners=[]
    def f1(tp,sup,ret):
        return np.divide(2*tp,sup+ret,out=np.zeros_like(tp,dtype=float),where=(sup+ret)>0).mean()
    for data in (left,right):
        if any(set(data[uid])!=set(ids) for uid in uids):raise ValueError('candidate identity differs')
        probs=np.array([[data[uid][s] for s in ids] for uid in uids])
        if probs.dtype.kind not in 'fi' or not np.isfinite(probs).all() or np.min(probs)<0 or np.max(probs)>1:
            raise ValueError('finite raw probabilities required')
        win=probs.argmax(axis=1);winners.append(win);arrays.append(probs)
        per=np.zeros((len(clusters),3,k),dtype=np.int64)
        np.add.at(per[:,0],(group,gold),1);np.add.at(per[:,1],(group,win),1)
        hit=win==gold;np.add.at(per[:,2],(group[hit],win[hit]),1)
        counts.append(per);sup,ret,tp=per.sum(axis=0)
        brier=np.square(probs-mask)
        with np.errstate(divide='ignore'):
            loss=-np.log(np.where(mask,probs,1-probs))
        def mean(v):return float(v.mean()) if np.isfinite(v).all() else None
        metrics.append(dict(top1_macro_f1=float(f1(tp,sup,ret)),top1_accuracy=float(hit.mean()),
                            predicted_classes=int((ret>0).sum()),top1_ties=int(((probs==probs.max(axis=1)[:,None]).sum(axis=1)>1).sum()),
                            nll=mean(loss),nll_positive=mean(loss[mask]),nll_negative=mean(loss[~mask]),
                            brier=mean(brier),brier_positive=mean(brier[mask]),brier_negative=mean(brier[~mask])))
    rng=np.random.default_rng(interval['seed']);samples=[]
    for _ in range(interval['samples']):
        mult=np.bincount(rng.integers(len(clusters),size=len(clusters)),minlength=len(clusters))
        scores=[]
        for per in counts:
            sup,ret,tp=np.tensordot(mult,per,axes=1)
            scores.append(f1(tp,sup,ret))
        samples.append(scores[1]-scores[0])
    lo,hi=map(float,np.quantile(samples,[.025,.975]))
    a,b=metrics
    left_hit,right_hit=winners[0]==gold,winners[1]==gold
    bins=dict(both_correct=left_hit&right_hit, fixed=~left_hit&right_hit,
              regressed=left_hit&~right_hit, both_wrong=~left_hit&~right_hit)
    return dict(recordings=n,candidates=k,clusters=len(clusters),cluster_key=cluster,interval=interval,
                tie_rule='first_in_native_order_after_semantic_alignment',left=a,right=b,
                macro_f1_delta=b['top1_macro_f1']-a['top1_macro_f1'],macro_f1_delta_ci95=[lo,hi],
                mean_abs_candidate_probability_shift=float(np.abs(arrays[1]-arrays[0]).mean()),
                **{name:int(mask.sum()) for name,mask in bins.items()},
                **({'recording_ids': {name:[uid for uid,matched in zip(uids,mask,strict=True) if matched]
                                      for name,mask in bins.items()}} if collect_ids else {}))
