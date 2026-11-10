"""Score separate Jev labels against complete frozen inference predictions."""
import json
from .answers import parse_answer, AnswerError


def score(requests, targets, predictions):
    by_id={r['id']:r for r in requests}
    gold={r['id']:r for r in targets}
    if not by_id or len(by_id)!=len(requests) or len(gold)!=len(targets) or not set(by_id)<=set(gold):
        raise ValueError('unique requests and covering targets required')
    found={}
    for row in predictions:
        key=(row['example_id'],row['replicate'])
        if key[0] not in by_id or key in found:raise ValueError('unknown or duplicate predictions')
        found[key]=row
    if not found:raise ValueError('predictions required')
    replicas={key[1] for key in found}
    if replicas!=set(range(max(replicas)+1)):raise ValueError('missing replicate')
    if set(found)!={(uid,i) for uid in by_id for i in replicas}:raise ValueError('incomplete prediction coverage')
    reports=[]
    for replicate in sorted(replicas):
        correct=valid=0;components={f:0 for f in ('action','object','location')};has_native=True
        for uid,request in by_id.items():
            row=found[uid,replicate]
            try:answer=parse_answer(row['raw_text']);parsed=True
            except AnswerError:answer={'value':None};parsed=False
            value=answer['value'];target=gold[uid]['target']
            ok=parsed and type(value) is type(target) and value==target
            correct+=ok;valid+=parsed
            labels=gold[uid]['metadata'].get('native_labels')
            if not labels or gold[uid]['metadata'].get('variant')!='native':
                has_native=False;continue
            candidates=request['input']['schema']['candidates']
            desc=next((c['description'] for c in candidates if c['id']==value),'') if parsed else ''
            decoded=dict(p.split('=',1) for p in desc.split('; ') if '=' in p)
            for field in components:components[field]+=decoded.get(field)==labels[field]
        reports.append({'replicate':replicate,'examples':len(by_id),'correct':correct,
            'accuracy':correct/len(by_id),'format_valid':valid,
            'fsc_native_component_accuracy':{f:v/len(by_id) for f,v in components.items()} if has_native else None})
    return {'replicates':reports,'predictions':len(found),'coverage':1.,
        'scope':'complete selected request population; TRAIN probe is not official test performance'}
