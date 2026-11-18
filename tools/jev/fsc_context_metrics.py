"""Native FSC probability scoring, separate from semantic-presence truth."""
import hashlib
import json

from multimodal_jev.context_event_metrics import ContextEventMetrics


def score(records, population, targets, candidate_ids, *, policy_id, input_receipts):
    """Caller authenticates policy/runtime, roster, labels and input receipts.

    No malformed/missing predictions are dropped. This function fails instead
    of publishing a complete-population score for a partial run.
    """
    ids=[r['id'] for r in population]
    if (not ids or len(set(ids))!=len(ids) or len(candidate_ids)!=31 or
            len(set(candidate_ids))!=31 or set(targets)!=set(ids) or
            len(records)!=len(ids) or len(input_receipts)!=len(ids)):
        raise ValueError('complete unique population, labels, inputs and native31 required')
    predicted={r['id']:r for r in records};receipts={r['id']:r for r in input_receipts}
    if set(predicted)!=set(ids) or set(receipts)!=set(ids):
        raise ValueError('prediction or receipt population differs')
    metrics=ContextEventMetrics(candidate_ids,threshold=.5)
    empty=multiple=0;by_speaker={};rows=[]
    for source in population:
        uid=source['id'];row=predicted[uid];receipt=receipts[uid]
        identity=row['request_identity']
        if (row['policy_id']!=policy_id or identity['policy_id']!=policy_id or
                identity['positions']!=receipt['positions'][0] or
                identity['inputs']!={k:v for k,v in receipt['tensors'].items() if k!='positions'} or
                source['audio_sha256']!=receipt['audio_sha256'] or len(identity['positions'])!=31):
            raise ValueError('policy or encoded input differs')
        digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if row['request_id']!=digest:raise ValueError('engine request digest differs')
        metrics.add(uid,targets[uid],row['prob'])
        selected={c for c,p in zip(candidate_ids,row['prob'],strict=True) if p>=.5}
        top1=candidate_ids[max(range(31),key=lambda j:row['prob'][j])]
        f1=2*int(targets[uid] in selected)/(len(selected)+1)
        empty+=not selected;multiple+=len(selected)>1
        speaker=source['speaker_id']
        if speaker not in by_speaker:by_speaker[speaker]=ContextEventMetrics(candidate_ids,threshold=.5)
        by_speaker[speaker].add(uid,targets[uid],row['prob'])
        rows.append(dict(id=uid,speaker_id=speaker,gold=targets[uid],top1=top1,
                         top1_correct=top1==targets[uid],set_f1=f1,selected_count=len(selected)))
    return dict(policy_id=policy_id,metrics=metrics.result(),
        empty_selection_rate=empty/len(ids),multiple_selection_rate=multiple/len(ids),
        per_speaker={k:v.result() for k,v in by_speaker.items()},records=rows,
        scope='Native exclusive command identity only; no semantic-presence or execution claim')


def compare(baseline,later):
    """Fixed paired descriptive difference; never chooses the target winner."""
    a=baseline['records'];b=later['records']
    if ([{k:r[k] for k in ('id','speaker_id','gold')} for r in a]!=
            [{k:r[k] for k in ('id','speaker_id','gold')} for r in b] or
            baseline['metrics']['candidate_ids']!=later['metrics']['candidate_ids'] or
            baseline['metrics']['threshold']!=later['metrics']['threshold']):
        raise ValueError('same target population, gold and definitions required')
    fields=('query_macro_set_f1','exact_set','top1_event_accuracy','top1_macro_f1',
            'candidate_binary_nll','candidate_brier','mean_selected_count')
    return dict(initial_policy_id=baseline['policy_id'],later_policy_id=later['policy_id'],
        delta={k:None if baseline['metrics'][k] is None or later['metrics'][k] is None else
               later['metrics'][k]-baseline['metrics'][k] for k in fields},
        top1_improved=sum(not x['top1_correct'] and y['top1_correct'] for x,y in zip(a,b)),
        top1_regressed=sum(x['top1_correct'] and not y['top1_correct'] for x,y in zip(a,b)),
        scope='Descriptive paired transfer difference for preselected source policies; no significance or checkpoint selection')
