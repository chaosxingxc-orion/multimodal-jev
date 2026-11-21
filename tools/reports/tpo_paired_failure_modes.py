"""Re-score paired sealed outputs and diagnose ranking vs probability drift, on CPU."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
import numpy as np


def read(path, expected=None):
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if expected and digest != expected:
        raise ValueError(f'content changed: {path}')
    return json.loads(raw), digest


def counts(mask):
    return int(np.count_nonzero(mask))


def check(got, want):
    for key, value in got.items():
        if not np.isclose(value, want[key], atol=1e-10, rtol=0):
            raise ValueError(f'{key} differs: {value} vs {want[key]}')


def scoring(win, y, k):
    acc = win == y
    den = np.bincount(win, minlength=k+1)[:k] + np.bincount(y, minlength=k)
    tp = np.bincount(y[acc], minlength=k)
    return counts(acc)/len(y), float(np.mean(2*tp/np.maximum(1,den)))


def dataset(args):
    ds, root, spec, lock, metrics = args
    seal = spec['endpoints'][ds]['cache']['seal']
    base = Path(seal['root'])
    manifest, _ = read(base/'MANIFEST.json', seal['sha256'])
    def sealed(rel):
        return read(base/rel, manifest['files'][rel])[0]
    ids = sealed(f'inference/{ds}/panel.json')['candidate_ids']
    truth = sealed(f'scoring/{ds}/targets.json')
    uids = [r['id'] for r in truth]
    if len(set(uids)) != len(uids) or len(set(ids)) != len(ids):
        raise ValueError('duplicate truth or candidates')
    y = np.array([ids.index(r['gold']) for r in truth]); n, k = len(y), len(ids)
    I = np.arange(n)
    gold = np.arange(k)[None,:] == y[:,None]
    out, data, hashes = {}, {}, {}
    for arm, pid in lock['arms'].items():
        vectors = []
        for suffix,field in (('.json','prob'),('-choices.json','logp')):
            files = [root/Path(shard).name/(arm+suffix) for shard in spec['shards'][ds]]
            lookup = {}
            with ThreadPoolExecutor(lock['io_workers']) as pool:
                for path,(obj,h) in zip(files,pool.map(read,files)):
                    hashes[str(path.relative_to(root))]=h
                    if obj['policy_id'] != pid: raise ValueError('wrong policy')
                    for row in obj['records']:
                        if row['id'] in lookup: raise ValueError('duplicate prediction')
                        lookup[row['id']] = row[field]
            if set(lookup)!=set(uids):raise ValueError('unpaired population')
            vectors.append(np.array([lookup[uid] for uid in uids]))
        p,z = vectors
        if p.shape!=(n,k) or z.shape!=(n,k+1) or not np.all((p>0)&(p<1)) or not np.isfinite(z).all():
            raise ValueError('invalid predictions')
        cp = np.exp(z)
        if not np.allclose(cp.sum(1),1,atol=1e-5,rtol=0):raise ValueError('invalid Choice normalization')
        win=p.argmax(1); pg=p[I,y]; neg=np.where(gold,-np.inf,p)
        ph=neg.max(1); margin=pg-ph; rank=1+(p>pg[:,None]).sum(1)
        pnll=-np.log(pg).mean()/k; nnll=-np.where(gold,0,np.log1p(-p)).sum(1).mean()/k
        pb=np.square(1-pg).mean()/k; nb=np.where(gold,0,p*p).sum(1).mean()/k
        acc,f1=scoring(win,y,k)
        check(dict(queries=n, top1_event_accuracy=acc,top1_macro_f1=f1,candidate_binary_nll=pnll+nnll,
                   candidate_brier=pb+nb),metrics[ds][arm])
        cw=z.argmax(1); cond=z[:,:k].argmax(1); ca,cf=scoring(cw,y,k)
        check(dict(queries=n,top1_accuracy=ca,top1_macro_f1=cf,false_none_rate=(cw==k).mean(),
                   categorical_nll=-z[I,y].mean(),categorical_brier=((cp[:,:k]-gold)**2).sum(1).mean()+(cp[:,k]**2).mean()),metrics[ds]['choices'][arm])
        t=lock['diagnostic_threshold']
        out[arm]=dict(n=n,k=k,nll_positive=pnll,nll_negative=nnll,brier_positive=pb,brier_negative=nb,
            median_gold=np.median(pg),median_hardest_negative=np.median(ph),median_signed_gap=np.median(margin),
            mean_negative=p[~gold].mean(),mean_false_positives_per_audio=((p>=t)&~gold).sum(1).mean(),
            missed_gold_under_threshold=counts(pg<t), multiple_selected=counts((p>=t).sum(1)>1),
            tied_top=counts((p==p.max(1)[:,None]).sum(1)>1),conditional_choice_accuracy=(cond==y).mean(),
            median_choice_none=np.median(cp[:,-1]))
        data[arm]=(win,pg,ph,margin,rank,cond)
    a,ag,ah,am,ar,_=data['prior']; b,bg,bh,bm,br,_=data['tpo']
    reg=(a==y)&(b!=y)
    out['paired']=dict(retained=counts((a==y)&(b==y)),fixed=counts((a!=y)&(b==y)),
            regressed=counts(reg),both_wrong=counts((a!=y)&(b!=y)),gold_rank_worse=counts(br>ar),
            gold_rank_improved=counts(br<ar),regressed_tied_gold=counts(reg&(bg==bh)),
            regressed_gold_down=counts(reg&(bg<ag)),regressed_negative_up=counts(reg&(bh>ah)),
            regressed_both_up=counts(reg&(bh>ah)&(bg>ag)),regressed_wrong_lead_ge_margin=counts(reg&(-bm>=lock['diagnostic_gap'])),
            regressed_gold_above_threshold=counts(reg&(bg>=lock['diagnostic_threshold'])),
            regressed_median_margin=[np.median(am[reg]),np.median(bm[reg])],
            regressed_median_gold=[np.median(ag[reg]),np.median(bg[reg])],
            regressed_median_hardest_negative=[np.median(ah[reg]),np.median(bh[reg])])
    pairs=Counter((ids[y[i]],ids[b[i]]) for i in np.flatnonzero(reg))
    examples=[]
    for (g,w),number in sorted(pairs.items(),key=lambda item:(-item[1],item[0]))[:5]:
        subset=[i for i in np.flatnonzero(reg) if ids[y[i]]==g and ids[b[i]]==w]
        i=sorted(subset,key=lambda i:(bm[i],uids[i]))[(number-1)//2]
        examples.append(dict(gold=g,winner=w,count=number,median_gap_example=uids[i],
            gold_prob=[ag[i],bg[i]],strongest_wrong_prob=[ah[i],bh[i]],gold_rank=[int(ar[i]),int(br[i])]))
    out['regressed_confusions']=examples
    out['shard_sha256']=hashes
    return ds,out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation',type=Path);parser.add_argument('lock',type=json.loads)
    args=parser.parse_args(); root=args.evaluation; lock=args.lock
    spec,_=read(root/'entry/score.json',lock['score_config_sha'])
    spec=spec['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    metrics,_=read(root/'metrics.json',lock['score_sha'])
    with ProcessPoolExecutor(lock['cpu_workers'],mp_context=get_context('spawn')) as pool:
        result=dict(pool.map(dataset,[(ds,root,spec,lock,metrics) for ds in spec['endpoints']]))
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
