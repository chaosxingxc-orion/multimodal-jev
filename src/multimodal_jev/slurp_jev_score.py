"""Full-bundle Jev metrics with strict coverage and native reconstruction."""
from collections import Counter
import hashlib
import json

from .answers import parse_answer, AnswerError
from .consumption import canonical
from .slurp_jev_full import expand_bundle, reconstruct
from .task_schema import DecisionSchema


def score_bundle(bundle, target, bank, predictions):
    requests = {r['id']: r for r in expand_bundle(bundle, bank)}
    if (target['id'] != bundle['id'] or target['bundle_sha256'] != hashlib.sha256(canonical(bundle)).hexdigest()
            or set(target['values']) != set(bank['tasks'])):
        raise ValueError('target/bundle/task identity changed')
    found = {}
    for row in predictions:
        uid = row['example_id']
        if uid not in requests or uid in found or type(row['replicate']) is not int or row['replicate'] != 0:
            raise ValueError('unknown/duplicate prediction or invalid replicate')
        if row['request_sha256'] != hashlib.sha256(canonical(requests[uid])).hexdigest():
            raise ValueError('prediction/request identity changed')
        found[uid] = row
    if set(found) != set(requests): raise ValueError('complete bundle predictions required')
    values = {}; outcomes = {}; valid = {}; counts = Counter()
    for key, definition in bank['tasks'].items():
        schema = DecisionSchema.from_dict(definition); gold = target['values'][key]
        if not schema.accepts(gold): raise ValueError('invalid typed target')
        try:
            value = parse_answer(found[bundle['id']+'::'+key]['raw_text'])['value']; parsed = True
        except AnswerError:
            value = None; parsed = False
        valid[key] = parsed and schema.accepts(value)
        values[key] = value
        outcomes[key] = valid[key] and type(value) is type(gold) and value == gold
        counts['decisions'] += 1; counts['format_valid'] += parsed; counts['schema_valid'] += valid[key]
    counts['recordings'] = 1
    for view in ('native','permutation','rebind'):
        counts['intent_'+view+'_correct'] = outcomes['intent.'+view]
    for view in ('permutation','rebind'):
        counts['intent_'+view+'_both_correct'] = outcomes['intent.native'] and outcomes['intent.'+view]
    counts['intent_all_three_correct'] = all(outcomes['intent.'+v] for v in ('native','permutation','rebind'))
    for entity in bank['entity_types']:
        prefix = 'entity.'+entity
        gold = target['values'][prefix+'.values']
        counts['presence_pairs'] += 1
        counts['presence_both_correct'] += outcomes[prefix+'.present'] and outcomes[prefix+'.absent']
        subset = 'present' if gold else 'absent'
        counts[subset+'_type_pairs'] += 1
        counts[subset+'_type_both_correct'] += outcomes[prefix+'.present'] and outcomes[prefix+'.absent']
        counts['slot_queries'] += 1; counts['slot_list_exact'] += outcomes[prefix+'.values']
        if gold:
            counts['nonempty_slot_queries'] += 1
            counts['nonempty_slot_list_exact'] += outcomes[prefix+'.values']
        if len(gold)>1:
            counts['multiple_slot_queries'] += 1
            counts['multiple_slot_list_exact'] += outcomes[prefix+'.values']
        # Invalid lists are all wrong even for empty gold; native recon below
        # marks the whole composite invalid if any required extraction fails.
    required = ['intent.native'] + ['entity.'+t+'.values' for t in bank['entity_types']]
    native_valid = all(valid[k] for k in required)
    counts['native_reconstruction_valid'] = native_valid
    if native_valid:
        # Auxiliary view errors do not invalidate native reconstruction.
        completed = dict(values)
        # Reconstruction reads only native intent and .values. Replace
        # auxiliary fields with fixed legal constants; never use gold.
        for key in completed:
            if key not in required:
                definition = bank['tasks'][key]
                completed[key] = False if definition['decision_type']=='noul' else definition['candidates'][0]['id']
        native = reconstruct(completed, bank)
    else:
        native = None
    counts['all_decisions_correct'] = all(outcomes.values())
    return dict(counts), native


def summarize(counts):
    denominators = {
        **{'intent_'+v+'_accuracy':'recordings' for v in ('native','permutation','rebind')},
        'intent_permutation_both_correct_rate':'recordings',
        'intent_rebind_both_correct_rate':'recordings',
        'intent_all_three_correct_rate':'recordings',
        'all_decisions_correct_rate':'recordings',
        'native_reconstruction_valid_rate':'recordings',
        'presence_both_correct_rate':'presence_pairs',
        'present_type_both_correct_rate':'present_type_pairs',
        'absent_type_both_correct_rate':'absent_type_pairs',
        'slot_list_exact_rate':'slot_queries',
        'nonempty_slot_list_exact_rate':'nonempty_slot_queries',
        'multiple_slot_list_exact_rate':'multiple_slot_queries'}
    rates = {}
    for key, denominator in denominators.items():
        numerator = key.replace('_accuracy','_correct') if key.endswith('_accuracy') else key.removesuffix('_rate')
        rates[key] = counts.get(numerator,0)/counts[denominator] if counts.get(denominator) else None
    return {'counts':dict(counts),'rates':rates,
            'scope':'correlated derived decisions; absent entities dominate aggregate accuracy; native SLU metrics require official reconstruction scoring'}
