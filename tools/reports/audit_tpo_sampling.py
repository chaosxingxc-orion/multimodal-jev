"""Measure TPO contrast coverage and active margins from TRAIN-only frozen scores."""
import argparse
from collections import Counter
import json
from pathlib import Path
from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config


def read(path):return json.loads(path.read_bytes())


def target_effect(rows, records, logq):
    """At the Stage1 starting point, dKL/dz = pi1 - q (per recording)."""
    import torch
    if sorted(r['index'] for r in rows.values())!=list(range(len(rows))):
        raise ValueError('unique complete target indices required')
    ordered=sorted(rows.values(),key=lambda r:r['index'])
    ref=torch.tensor([records[r['request_key']]['prob'] for r in ordered],dtype=torch.float64)
    gold=torch.tensor([r['gold'] for r in ordered])
    if ref.shape!=logq.shape or not ((ref>0)&(ref<1)).all():
        raise ValueError('complete interior Stage1 probability slate required')
    pi=torch.logit(ref).softmax(1)
    q=logq.double().exp()
    if not torch.isfinite(q).all() or not torch.allclose(q.sum(1),torch.ones_like(q[:,0]),atol=5e-6,rtol=0):
        raise ValueError('invalid target distribution')
    def margin(scores):
        mask=torch.nn.functional.one_hot(gold,scores.shape[1]).bool()
        return scores.gather(1,gold[:,None])[:,0]-scores.masked_fill(mask,-torch.inf).max(1).values
    before,after=margin(ref),margin(q)
    effective=(q-pi).abs().sum(1)/2
    result={}
    for name,mask in [('correct',before>0),('tied',before==0),('wrong',before<0)]:
        tv=effective[mask]
        result[name]=dict(rows=int(mask.sum()),target_gold_strictly_first=int(((after>0)&mask).sum()),
            mean_moved_ranking_mass=float(tv.mean()) if len(tv) else None)
    return result


def reward_counterfactual(rows, records, logq, settings):
    """Untrained TRAIN-only target feasibility; no held-out inputs or search."""
    import math
    import torch

    margin = math.log(settings['gold_odds_ratio'])
    cap = settings['max_correction_per_side']
    if not (0 < margin < math.inf and 0 < cap < math.inf):
        raise ValueError('finite positive margin and cap required')
    ordered = sorted(rows.values(), key=lambda row: row['index'])
    ref = torch.tensor([records[row['request_key']]['prob'] for row in ordered], dtype=torch.float64)
    gold = torch.tensor([row['gold'] for row in ordered])
    idx = torch.arange(len(gold))
    mask = torch.nn.functional.one_hot(gold, ref.shape[1]).bool()
    def gap(s):
        return s[idx,gold] - s.masked_fill(mask,-torch.inf).amax(1)
    base = logq.double()
    before = gap(base)
    shift = ((margin-before).clamp_min(0)/2).clamp_max(cap)
    ceiling = base.masked_fill(mask,-torch.inf).amax(1)-shift
    trial = torch.minimum(base,ceiling[:,None])
    trial[idx,gold] = base[idx,gold]+shift
    after = gap(trial)
    torch.testing.assert_close(after, before+2*shift, atol=1e-12, rtol=0)
    subset = gap(ref)<0
    return dict(settings=settings,rows=len(gold),strict_stage1_errors=int(subset.sum()),
                baseline_strict_fixes=int(((before>0)&subset).sum()),
                proposed_strict_fixes=int(((after>0)&subset).sum()),
                proposed_still_not_first=int(((after<=0)&subset).sum()),
                previously_strictly_correct_target_lost=int(((before>0)&(after<=0)).sum()),
                modified_rows=int((shift>0).sum()),cap_reached=int((shift==cap).sum()))


def measure(cfg, *, include_targets=False):
    root, ref = Path(cfg['output']), Path(cfg['reference'])
    m = read(root/'MANIFEST.json')
    rm = read(ref/'MANIFEST.json')
    if (file_hash(ref/'MANIFEST.json') != m['reference_sha256'] or
            file_hash(ref/'records.json') != rm['files']['records.json'] or
            any(file_hash(root/k)!=v for k,v in m['files'].items())):
        raise ValueError('training publication changed')
    rows=read(root/'rows.json')
    records=read(ref/'records.json')
    batches=read(root/'batches.json')
    gap=m['objective']['probability_gap']
    result=[]
    for e in range(m['schedule']['epochs']):
        counts=Counter(); golds=Counter()
        for b in batches:
            if b['epoch']!=e:continue
            for i,s in enumerate(b['slots']):
                if s['padding']:continue
                a=rows[s['id']]
                golds[s['id']]+=1
                if s['confusion']:
                    p=b['slots'][i^1]; other=rows[p['id']]
                    if (a['gold']==other['gold'] or a['audio_sha256']==other['audio_sha256'] or
                         not p['confusion'] or p['padding'] or not any(
                             x['ref_gap']<=0 and x['competitor']==y['gold']
                             for x,y in ((a,other),(other,a)))):
                        raise ValueError('unverified contrast edge')
                    scores,neg=(records[r['request_key']]['prob'] for r in (a,other))
                    counts['confusion_recordings']+=1
                    counts['active_cross_margin_directions']+=int(scores[a['gold']]-neg[a['gold']]<gap)
                    counts['reciprocal_pair_directions']+=int(all(x['ref_gap']<=0 and x['competitor']==y['gold']
                            for x,y in ((a,other),(other,a))))
                elif s['paired']:counts['invariant_recordings']+=1
                else:counts['unpaired_recordings']+=1
        if golds!=Counter(rows.keys()):raise ValueError('changed training exposure')
        result.append(dict(epoch=e,**counts))
    if include_targets:
        import torch
        logq = torch.load(root/'targets.pt',map_location='cpu',weights_only=True)
        report = dict(pair_sampling=result, at_stage1=target_effect(rows,records,logq))
        if 'reward_counterfactual' in cfg:
            report['reward_counterfactual'] = reward_counterfactual(rows,records,logq,cfg['reward_counterfactual'])
        return report
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('local')
    p.add_argument('--target-effect',action='store_true')
    a=p.parse_args(); print(json.dumps(measure(load_config(a.config,a.local)[0],include_targets=a.target_effect),indent=2))
