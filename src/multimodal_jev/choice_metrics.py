"""Score exact closed and open action distributions against an independent sealed roster."""
import math
from collections import defaultdict
from .context_event_metrics import ContextEventMetrics


def choice_metrics(assets, truth, predictions, candidate_ids):
    meter = ContextEventMetrics(candidate_ids)
    closed_nll = closed_brier = coverage_brier = 0.
    hits = rank_correct = false_accepts = open_correct = 0
    profile_strata = defaultdict(lambda: [0,0,0,0,0,0,0.])
    strata = defaultdict(lambda: [0,0,0,0,0,0,0.])
    for asset, label, row in zip(assets, truth, predictions, strict=True):
        if asset['id'] != row['id'] or any(asset[k] != label[k] for k in ('id', 'audio_sha256')):
            raise ValueError('Choice prediction roster differs')
        closed, scores, slate = row['closed'], row['open'], row['slate']
        if (len(closed) != len(candidate_ids) or len(scores) != len(slate)+1 or not slate
                or len(set(slate)) != len(slate) or not set(slate) <= set(candidate_ids)):
            raise ValueError('Choice distribution support differs')
        for probs in (closed,scores):
            if (not all(math.isfinite(x) and 0 <= x <= 1 for x in probs)
                    or abs(sum(probs)-1)>1e-5):
                raise ValueError('invalid probability distribution')
        gold=label['gold']
        meter.add(row['id'],gold,closed)
        winner=candidate_ids.index(gold)
        closed_nll += -math.log(closed[winner]) if closed[winner]>0 else math.inf
        closed_brier += sum((p-(i==winner))**2 for i,p in enumerate(closed))
        hit=gold in slate
        predicted=max(range(len(scores)),key=scores.__getitem__)
        hits+=hit
        rank_correct+=hit and slate[max(range(len(slate)),key=scores.__getitem__)]==gold
        false_accepts+=not hit and predicted<len(slate)
        open_correct+=(predicted==slate.index(gold) if hit else predicted==len(slate))
        residual=((1-scores[-1])-hit)**2
        coverage_brier+=residual
        cells=[strata[len(slate)]]
        if 'proposal' in row:cells.append(profile_strata[row['proposal']])
        for cell in cells:
            cell[0]+=1
            cell[1]+=hit
            cell[2]+=not hit and predicted<len(slate)
            cell[3]+=hit and predicted==len(slate)
            cell[4]+=hit and slate[max(range(len(slate)),key=scores.__getitem__)]==gold
            cell[5]+=predicted==len(slate)
            cell[6]+=residual
    num=len(assets)
    result=meter.result()
    def rates(groups):
        return {str(k):dict(queries=n,hits=h,misses=n-h,
        false_accept_rate=fa/(n-h) if n>h else None,false_none_rate=fr/h if h else None,
        conditional_accuracy=correct/h if h else None,none_predictions=none,coverage_brier=brier/n)
        for k,(n,h,fa,fr,correct,none,brier) in sorted(groups.items())}
    return dict(by_length=rates(strata),by_proposal=rates(profile_strata),queries=num,closed_macro_f1=result['top1_macro_f1'],closed_accuracy=result['top1_event_accuracy'],
        closed_nll=closed_nll/num,closed_brier=closed_brier/num,coverage_hits=hits,coverage_misses=num-hits,
        coverage_rate=hits/num,conditional_accuracy=rank_correct/hits if hits else None,
        false_accept_rate=false_accepts/(num-hits) if hits<num else None,open_accuracy=open_correct/num,
        coverage_brier=coverage_brier/num)


def full_ontology_metrics(assets, truth, predictions, candidate_ids):
    """One gold intent per audio; a winning NONE counts as an incorrect decision."""
    if not assets or not candidate_ids or len(set(candidate_ids))!=len(candidate_ids):
        raise ValueError('distinct native intents and nonempty population required')
    n=len(candidate_ids);counts=[[0,0,0] for _ in candidate_ids]
    correct=reject=0;nll=brier=0.
    for a,t,r in zip(assets,truth,predictions,strict=True):
        if a['id']!=r['id'] or any(a[k]!=t[k] for k in ('id','audio_sha256')):
            raise ValueError('Choice population differs')
        z=r['logp']
        if len(z)!=n+1 or any(not math.isfinite(x) or x>1e-6 for x in z):
            raise ValueError('invalid full action log probabilities')
        p=[math.exp(x) for x in z]
        if abs(math.fsum(p)-1)>1e-5:raise ValueError('unnormalized distribution')
        y=candidate_ids.index(t['gold']); win=max(range(n+1), key=z.__getitem__)
        nll-=z[y]; brier+=math.fsum((v-(i==y))**2 for i,v in enumerate(p))
        counts[y][0]+=1
        if win<n:counts[win][1]+=1
        if win==y:counts[y][2]+=1;correct+=1
        if win==n:reject+=1
    return dict(queries=len(assets),top1_macro_f1=math.fsum(2*tp/(a+b) if a+b else 0. for a,b,tp in counts)/n,
        top1_accuracy=correct/len(assets),categorical_nll=nll/len(assets),categorical_brier=brier/len(assets),
        false_none_rate=reject/len(assets))
