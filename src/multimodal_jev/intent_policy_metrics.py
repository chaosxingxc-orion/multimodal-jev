"""Finite decision metrics with explicit observed-label and split scope."""
from collections import Counter
import math


def score(items,records,wire_arm):
    """Score a fixed complete population, with no threshold/template selection.

    `items` carry authenticated requests and targets outside model input.
    `records` carry full candidate probabilities and exact request identities.
    This function does not promote a learning population to held-out evidence.
    """
    if wire_arm not in ('D','Q') or not items:raise ValueError('fixed nonempty population required')
    if len({i['id'] for i in items})!=len(items) or len({r['id'] for r in records})!=len(records):
        raise ValueError('unique population required')
    predictions={r['id']:r for r in records}
    if set(predictions)!={i['id']+'::'+wire_arm for i in items}:raise ValueError('exact prediction population required')
    gold_count=Counter();pred_count=Counter();correct_count=Counter();confusion=Counter();components=Counter()
    rows=[];probability_sum=0.;weighted_reward=0.;weight_sum=0.;compatible=0
    final_schema=items[0]['request'].questions[-1].schema
    for item in items:
        request=item['request'];target=item['target'];record=predictions[item['id']+'::'+wire_arm]
        import json
        if record['request_identity']['request']!=json.loads(json.dumps(request.model_input())):
            raise ValueError('prediction input identity differs')
        if request.questions[-1].schema!=final_schema:raise ValueError('final ontology differs across rows')
        probs=record['candidate_logprobs']
        if len(probs)!=len(request.questions) or len(probs)!=len(target['targets']):raise ValueError('complete Span support required')
        predicted=[];truth=[];gold_probs=[]
        for j,(q,logp) in enumerate(zip(request.questions,probs,strict=True)):
            if (len(logp)!=len(q.values) or any(type(p) not in (int,float) or not math.isfinite(p) for p in logp)
                    or abs(sum(math.exp(p) for p in logp)-1)>1e-5):raise ValueError('normalized full candidate support required')
            y=target['targets'][j]
            if type(y)!=int or not 0<=y<len(q.values) or not target['supervised'][j]:raise ValueError('complete reviewed metric supervision required')
            best=max(range(len(logp)),key=lambda k:logp[k])
            predicted.append(q.values[best]);truth.append(q.values[y]);gold_probs.append(math.exp(logp[y]))
            components[q.id]+=best==y
        gold=truth[-1];prediction=predicted[-1];gold_count[gold]+=1;pred_count[prediction]+=1
        correct_count[gold]+=gold==prediction;confusion[(gold,prediction)]+=1
        probability_sum+=gold_probs[-1]
        w=target['request_weight']
        if type(w) not in (int,float) or not math.isfinite(w) or w<=0:raise ValueError('positive finite request weight required')
        weight_sum+=w;weighted_reward+=w*gold_probs[-1]
        relation=None
        if wire_arm=='Q':
            if [q.id for q in request.questions]!=['scenario','action','intent']:raise ValueError('canonical structure required')
            relation=predicted[0]+'_'+predicted[1]==prediction;compatible+=relation
        rows.append(dict(id=item['id'],gold=truth,prediction=predicted,correct=gold==prediction,
            gold_probability=gold_probs,compatible=relation))
    labels=[c.id for c in final_schema.candidates];observed=sorted(gold_count)
    f1={label:2*correct_count[label]/(gold_count[label]+pred_count[label])
        if gold_count[label]+pred_count[label] else 0. for label in labels}
    return dict(recordings=len(items),final_accuracy=sum(correct_count.values())/len(items),
        final_macro_recall_observed=sum(correct_count[c]/gold_count[c] for c in observed)/len(observed),
        final_macro_f1_observed=sum(f1[c] for c in observed)/len(observed),
        observed_classes=len(observed),candidate_classes=len(labels),unobserved_classes=sorted(set(labels)-set(observed)),
        component_accuracy={k:v/len(items) for k,v in components.items()},
        final_expected_reward=probability_sum/len(items),weighted_final_expected_reward=weighted_reward/weight_sum,
        component_final_compatibility=compatible/len(items) if wire_arm=='Q' else None,
        confusion=[dict(gold=g,prediction=p,count=n) for (g,p),n in sorted(confusion.items())],
        records=rows,tie_policy='first supplied candidate',
        scope='Metrics for supplied population only. Macro averages cover observed gold classes; no claim on absent classes or held-out improvement.')
