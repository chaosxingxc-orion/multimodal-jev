"""Label-free native SLURP decision requests for matched Jev evaluation.

Identifier descriptions and draft type definitions are input engineering only.
They do not grant semantic training admission or imply baseline accuracy.
"""
from .span_contract import SpanQuestion, SpanRequest
from .task_schema import Candidate, DecisionSchema

REQUIREMENTS = (
    'Use the complete spoken request as evidence for every supplied question.',
    'Use the supplied candidate definitions; do not assume an unspoken value or role.',
    'Fill the provided typed spans in one forward pass without generating text.',
)


def compile_requests(audio_request, bank, definitions, requirements=REQUIREMENTS):
    """One 60-way intent request plus ten independent multi-Span type folds.

    Only a clean-source request is accepted. No target, transcript, gold intent,
    examples, review labels or predictions are consumed. All folds are fixed
    before inference; none depends on intent or another fold's answer.
    """
    if set(audio_request) != {'id','official_split','role','source_group','audio_sha256','input'}:
        raise ValueError('exact label-free clean-source request required')
    if set(audio_request['input']) != {'audio'}:
        raise ValueError('only audio belongs in per-record model input')
    if bank.get('schema') != 'slurp-jev-matched-ontology-v1':
        raise ValueError('matched ontology required')
    types = bank['slot_types']; folds = bank['type_folds']; intents = bank['intents']
    if len(types)!=55 or len(set(types))!=55 or len(intents)!=60 or len({r['id'] for r in intents})!=60:
        raise ValueError('locked 60-intent/55-type inventory required')
    flattened = [t for fold in folds for t in fold['types']]
    if len(folds)!=10 or len({f['fold_id'] for f in folds})!=10 or sorted(flattened)!=sorted(types):
        raise ValueError('ten disjoint complete type folds required')
    if any(r['id'] != r['scenario']+'_'+r['action'] for r in intents):
        raise ValueError('canonical intent identity differs')
    if definitions.get('schema') != 'slurp-type-definition-candidates-v1':
        raise ValueError('explicit versioned type definitions required')
    entries = definitions['types']
    if len(entries)!=55 or {r['entity_type'] for r in entries}!=set(types):
        raise ValueError('exact type definition coverage required')
    by_type = {r['entity_type']:r for r in entries}
    audio = audio_request['input']['audio']
    intent = SpanQuestion('intent', DecisionSchema(
        id='slurp-intent-60-v1', decision_type='choice',
        question='Which listed scenario and action best describe the spoken request?',
        candidates=tuple(Candidate(r['id'], 'Scenario: '+r['scenario']+'; action: '+r['action']+'.') for r in intents),
        allow_unknown=False))
    requests = [dict(id=audio_request['id']+'::intent', task='intent', fold=None,
                     request=SpanRequest(audio,(intent,),'intent',tuple(requirements)))]
    for fold in folds:
        questions = []
        for name in fold['types']:
            definition = by_type[name]['definition']
            if not isinstance(definition,str) or not definition.strip():
                raise ValueError('nonempty type definition required')
            questions.append(SpanQuestion(name,DecisionSchema(id='slurp-type:'+name,decision_type='noul',
                question='Does the audio express this slot type? '+name+': '+definition,allow_unknown=False)))
        if not questions:raise ValueError('nonempty fold required')
        requests.append(dict(id=audio_request['id']+'::'+fold['fold_id'],task='types',fold=fold['fold_id'],
            request=SpanRequest(audio,tuple(questions),questions[-1].id,tuple(requirements))))
    return requests


def decode_probabilities(compiled, candidate_logprobs):
    """Typed candidate readout to scorer format, no free-form text decoding.

    Exact ties use the fixed candidate order. Returns evidence for that choice;
    an invalid/missing fold rejects aggregation, never becomes empty output.
    """
    import math
    if len(compiled)!=11 or len(candidate_logprobs)!=11:
        raise ValueError('complete one-intent and ten-fold prediction required')
    if compiled[0]['task']!='intent' or any(r['task']!='types' for r in compiled[1:]):
        raise ValueError('fixed request order required')
    ranked = None; selected = []; seen = set(); ties = 0
    for entry, probabilities in zip(compiled,candidate_logprobs,strict=True):
        questions = entry['request'].questions
        if len(probabilities)!=len(questions):raise ValueError('one distribution per Span required')
        for question, logp in zip(questions,probabilities,strict=True):
            if (len(logp)!=len(question.values) or any(type(v) not in (int,float) or not math.isfinite(v) for v in logp)
                    or abs(sum(math.exp(v) for v in logp)-1)>1e-5):
                raise ValueError('finite normalized full-candidate log probabilities required')
            order = sorted(range(len(logp)),key=lambda i:(-logp[i],i))
            ties += logp.count(logp[order[0]]) > 1
            if entry['task']=='intent':
                if ranked is not None or question.id!='intent':raise ValueError('single intent Span required')
                ranked=[question.values[i] for i in order]
            else:
                if question.id in seen or question.values!=(False,True):raise ValueError('unique Boolean type Span required')
                seen.add(question.id)
                if question.values[order[0]]:selected.append(question.id)
    if len(seen)!=55 or ranked is None or len(ranked)!=60:raise ValueError('full task support required')
    return dict(intent_ranking=ranked,types=sorted(selected)),dict(tied_spans=ties,tie_policy='first_fixed_candidate')
