"""Dataset-independent candidate definitions for a shared probability head.

This bridge has two explicit target semantics: exclusive benchmark event identity
or caller-supplied candidate applicability. It never turns an exclusive intent
label into multi-intent semantic-presence annotation. Input compilation takes no
gold labels; target binding is a separate operation with request identity.
"""
from .context_decision_contract import ApplicabilityQuestion, ContextDecisionRequest
from .slurp_training import content_hash


def compile_candidates(audio, candidates, requirements, *, semantics, history=()):
    if semantics not in ('exclusive_event', 'candidate_applicability'):
        raise ValueError('explicit target semantics required')
    if not candidates or any(set(c) != {'id','definition'} for c in candidates):
        raise ValueError('explicit candidate IDs and definitions required')
    if len({c['id'] for c in candidates}) != len(candidates):
        raise ValueError('candidate IDs must be unique')
    if semantics == 'exclusive_event':
        rule = ('The benchmark assigns exactly one event under the supplied ontology and annotation rules. '
                'For each candidate judge whether it is that assigned event. This is not semantic mention detection.')
        question = 'Is the event assigned to this audio under the supplied annotation rules this candidate?'
    else:
        rule = ('Judge each candidate independently under its supplied applicability definition. '
                'Several candidates or no candidate may apply. Do not force probabilities to sum to one.')
        question = 'Does this candidate apply to the audio under its definition and supplied caller rules?'
    return ContextDecisionRequest(audio,
        tuple(ApplicabilityQuestion(c['id'],c['definition'],question) for c in candidates),
        (*tuple(requirements),rule),tuple(history),version='jev-context-'+semantics+'-v1')


def bind_candidate_targets(request, *, semantics, exclusive_event=None, applicability=None):
    """Map existing annotation only; None stays unsupervised, including all-null.

An exclusive label supports negatives only under exclusive_event semantics, not
semantic applicability. Explicit applicability requires every candidate key and
bool/None values. No-routes/unknown/execution outcomes are never fabricated.
"""
    if request.version != 'jev-context-'+semantics+'-v1':
        raise ValueError('request and target semantics differ')
    ids = [q.id for q in request.questions]
    if semantics == 'exclusive_event':
        if applicability is not None or (exclusive_event is not None and exclusive_event not in ids):
            raise ValueError('known exclusive label or explicit missing annotation required')
        labels = [None if exclusive_event is None else uid == exclusive_event for uid in ids]
    elif semantics == 'candidate_applicability':
        if (exclusive_event is not None or not isinstance(applicability,dict) or
                set(applicability) != set(ids) or
                any(value is not None and type(value) is not bool for value in applicability.values())):
            raise ValueError('explicit complete candidate map with bool/null labels required')
        labels = [applicability[uid] for uid in ids]
    else:
        raise ValueError('known target semantics required')
    return dict(request_sha256=content_hash(request.model_input()),semantics=semantics,
                candidate_ids=ids,labels=labels,supervised=[v is not None for v in labels])
