"""Matched SLURP intent/type evaluation with explicit partial-label masks.

This scorer does not approve annotations. A caller must supply an authenticated
reviewed evaluation release; native annotation availability is not approval.
"""
from collections import Counter, defaultdict


def ontology(taxonomy, folds, omni_intents):
    pairs = [tuple(p) for p in taxonomy['scenario_action_pairs']]
    if len(pairs) != len(set(pairs)):
        raise ValueError('duplicate intent pair')
    intents = [s+'_'+a for s, a in pairs]
    if len(intents) != len(set(intents)) or set(intents) != set(omni_intents):
        raise ValueError('canonical intent mapping differs from omni')
    types = taxonomy['entity_types']
    folded = [t for f in folds for t in f['types']]
    if len(types) != len(set(types)) or len(folded) != len(set(folded)) or set(types) != set(folded):
        raise ValueError('folds must cover business types exactly once')
    if len({f['fold_id'] for f in folds}) != len(folds):
        raise ValueError('duplicate fold identity')
    aliases = defaultdict(set)
    seen = set()
    for row in taxonomy['intent_alias_map']:
        pair = row['scenario'], row['action']
        if pair not in pairs or pair in seen:
            raise ValueError('invalid intent alias pair')
        seen.add(pair)
        for alias in row['raw_names']:
            aliases[alias].add('_'.join(pair))
    if seen != set(pairs):
        raise ValueError('missing intent alias mapping')
    return dict(schema='slurp-jev-matched-ontology-v1',
        intents=[dict(id=s+'_'+a, scenario=s, action=a) for s,a in sorted(pairs)],
        slot_types=sorted(types), type_folds=folds,
        ambiguous_raw_aliases={k:sorted(v) for k,v in sorted(aliases.items()) if len(v)>1},
        raw_alias_count=len(aliases),
        raw_alias_policy='Use authenticated scenario/action pair; never resolve ambiguous raw alias alone.',
        semantic_definitions_approved=False,
        scope='Identifier and candidate inventory only; no annotation approval or semantic definition completeness claim.')


def ratio(a, b):
    return a/b if b else None


def prf(tp, fp, fn):
    # Empty prediction precision is zero, as is F1 when no TP/FP/FN exist.
    # Gold-support counts remain explicit; macro-F1 uses supported classes only.
    return dict(precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def score(gold, predictions, bank):
    """Full roster scoring; invalid predictions never count as correct empty.

    Gold rows: id, source_group, intent (canonical ID or None), types (an
    explicit mapping of every business type to bool or None). None means
    unreviewed/unknown supervision, never a model action or negative label.
    Predictions: id, intent_ranking, types. Intent/type validity is separate.
    Missing rows stay in the roster. Unknown/duplicate prediction IDs reject
    the artifact. Duplicate/unknown labels or extra fields invalidate a row.
    """
    intents = [r['id'] for r in bank['intents']]
    types = bank['slot_types']
    if not intents or not types or len(set(intents)) != len(intents) or len(set(types)) != len(types):
        raise ValueError('unique nonempty ontology required')
    gold_index = {}
    for row in gold:
        if set(row) != {'id','source_group','intent','types'} or not isinstance(row['id'],str) or not row['id']:
            raise ValueError('exact gold schema required')
        if not isinstance(row['source_group'],str) or not row['source_group'] or row['id'] in gold_index:
            raise ValueError('unique gold IDs and source groups required')
        if row['intent'] is not None and row['intent'] not in intents:
            raise ValueError('unknown gold intent')
        if not isinstance(row['types'],dict) or set(row['types']) != set(types):
            raise ValueError('explicit supervision status for every type required')
        if any(v is not None and type(v) is not bool for v in row['types'].values()):
            raise ValueError('gold types must be bool or masked None')
        gold_index[row['id']] = row
    if not gold_index:
        raise ValueError('nonempty evaluation roster required')
    pred_index = {}
    for row in predictions:
        if not isinstance(row,dict) or row.get('id') not in gold_index or row['id'] in pred_index:
            raise ValueError('unknown or duplicate prediction ID')
        pred_index[row['id']] = row
    intent_counts = {k:Counter(tp=0,fp=0,fn=0,support=0) for k in intents}
    type_counts = {k:Counter(tp=0,fp=0,fn=0,tn=0,invalid=0,support=0,reviewed=0) for k in types}
    totals = Counter(); rows = []
    for uid, target in gold_index.items():
        pred = pred_index.get(uid)
        fields_valid = pred is not None and set(pred) == {'id','intent_ranking','types'}
        ranking = pred.get('intent_ranking') if fields_valid else None
        selected = pred.get('types') if fields_valid else None
        intent_valid = (type(ranking) is list and 1 <= len(ranking) <= len(intents)
                        and all(type(x) is str and x in intents for x in ranking)
                        and len(set(ranking)) == len(ranking))
        type_valid = (type(selected) is list and all(type(x) is str and x in types for x in selected)
                      and len(set(selected)) == len(selected))
        totals['missing_predictions'] += pred is None
        totals['invalid_intent_predictions'] += not intent_valid
        totals['invalid_type_predictions'] += not type_valid
        true_intent = target['intent']
        top1 = top3 = None
        if true_intent is not None:
            totals['intent_reviewed'] += 1
            intent_counts[true_intent]['support'] += 1
            top1 = bool(intent_valid and ranking[0] == true_intent)
            top3 = bool(intent_valid and true_intent in ranking[:3])
            totals['intent_top1_correct'] += top1
            totals['intent_top3_correct'] += top3
            if top1: intent_counts[true_intent]['tp'] += 1
            else:
                intent_counts[true_intent]['fn'] += 1
                if intent_valid: intent_counts[ranking[0]]['fp'] += 1
        labels = target['types']; chosen = set(selected) if type_valid else set()
        observed = {t:v for t,v in labels.items() if v is not None}
        local = Counter()
        for name, value in observed.items():
            c = type_counts[name]; c['reviewed'] += 1; c['support'] += value
            if not type_valid:
                c['invalid'] += 1; local['invalid'] += 1
                # Missed known positives are false negatives. Unknown model
                # outputs on known negatives are failures, never true negatives.
                if value: c['fn'] += 1; local['fn'] += 1
                continue
            key = 'tp' if value and name in chosen else 'fn' if value else 'fp' if name in chosen else 'tn'
            c[key] += 1; local[key] += 1
        totals['reference_unmatched_types'] += local['fp']
        totals['reviewed_type_decisions'] += len(observed)
        totals['type_correct_decisions'] += local['tp']+local['tn']
        complete = len(observed) == len(types)
        exact = covered = empty_correct = None
        if complete:
            positives = {t for t,v in observed.items() if v}
            totals['complete_type_samples'] += 1
            exact = bool(type_valid and chosen == positives)
            totals['type_set_exact'] += exact
            if positives:
                totals['complete_positive_samples'] += 1
                covered = bool(type_valid and positives <= chosen)
                totals['positive_complete'] += covered
            else:
                totals['complete_empty_samples'] += 1
                empty_correct = bool(type_valid and not chosen)
                totals['empty_correct'] += empty_correct
                totals['empty_false_positive'] += bool(type_valid and chosen)
                totals['empty_invalid'] += not type_valid
        rows.append(dict(id=uid,source_group=target['source_group'],intent_valid=intent_valid,
            types_valid=type_valid,intent_top1=top1,intent_top3=top3,
            reviewed_type_decisions=len(observed),type_counts=dict(local),
            complete_type_supervision=complete,type_set_exact=exact,
            positive_complete=covered,empty_correct=empty_correct))
    for c in intent_counts.values(): c.update(prf(c['tp'],c['fp'],c['fn']))
    for c in type_counts.values(): c.update(prf(c['tp'],c['fp'],c['fn']))
    aggregate = {k:sum(c[k] for c in type_counts.values()) for k in ('tp','fp','fn','tn','invalid')}
    supported_intents = [c['f1'] for c in intent_counts.values() if c['support']]
    supported_types = [c['f1'] for c in type_counts.values() if c['support']]
    return dict(schema='slurp-jev-matched-score-v1',samples=len(gold_index),counts=dict(totals),
        intent=dict(top1=ratio(totals['intent_top1_correct'],totals['intent_reviewed']),
            top3=ratio(totals['intent_top3_correct'],totals['intent_reviewed']),
            macro_f1_supported=ratio(sum(supported_intents),len(supported_intents)),
            supported_classes=len(supported_intents),per_class=intent_counts),
        types=dict(counts=aggregate,micro=prf(aggregate['tp'],aggregate['fp'],aggregate['fn']),
            macro_f1_supported=ratio(sum(supported_types),len(supported_types)),supported_types=len(supported_types),
            set_exact=ratio(totals['type_set_exact'],totals['complete_type_samples']),
            positive_complete=ratio(totals['positive_complete'],totals['complete_positive_samples']),
            empty_false_positive_rate=ratio(totals['empty_false_positive'],totals['complete_empty_samples']),
            empty_failure_rate=ratio(totals['empty_false_positive']+totals['empty_invalid'],totals['complete_empty_samples']),
            decision_coverage=totals['reviewed_type_decisions']/(len(gold_index)*len(types)),
            invalid_output_rate=totals['invalid_type_predictions']/len(gold_index),
            mean_reviewed_reference_unmatched=totals['reference_unmatched_types']/len(gold_index),per_type=type_counts),
        records=rows,
        conventions=dict(masked_labels='Excluded, never fabricated negatives; complete-set metrics require every label reviewed.',
            invalid_outputs='Invalid positive predictions count FN; invalid negatives are failures, not TN. Empty FPR excludes invalids; empty failure includes them. Always report invalid counts alongside F1.',
            macro='Mean over classes with positive gold support; all ontology classes retained in per-class output.',
            scope='Scoring only, no semantic admission or statistical superiority claim.'))
