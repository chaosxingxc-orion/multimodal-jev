#!/usr/bin/env python3
"""Prepare one explicit revision of rejected Omni proposals, without held-out data."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.teacher_candidates import strict_json, load_proposal
from configure_omni_teacher import configure

CORRECTION = '''
Revision after explicit rejection: neither prior candidate was evaluated.
Do not copy rejected semantics. Preserve this type table exactly:
choice = listed candidate ID string; noul = JSON boolean; score = listed number;
open_slot = one string; open_slots = list of strings, retaining order and duplicates.
For EVERY type, null remains available iff allow_unknown is true and the task's
evidence/valid-answer conditions warrant it. Never ban null globally for noul,
score or slots. A decidably false proposition is false, not null. Empty open_slots
list means no occurrences, not unknown. Do not create synonyms or define none;
the current schema is authoritative. Do not introduce task-specific fields.
Candidate evidence_ids MUST be from the supplied representative_evidence_ids;
diagnosis IDs may be from all_group_summary. Copy IDs exactly.
Greedy decoding is fixed. Distinguish aggregate evidence from representative
examples; do not generalize all zero outputs are fenced without full evidence.
Keep diagnoses concise (at most 3), return 1 or 2 distinct generic instructions
of at most1500 characters, and reserve enough output for final JSON.
'''


def prepare(source, output, executor):
    result=json.loads((source/'response/result.json').read_text())
    review=json.loads((source/'rejection-review.json').read_text())
    old=json.loads((source/'prepared.json').read_text())
    if (result['state']!='response_received' or result['finish_reasons']!=['stop']
            or result['response_sha256']!=file_hash(source/'response/response.sse')
            or review['response_sha256']!=result['response_sha256']
            or review['evaluation_eligible']!=[]
            or review['selection_or_test_items_read_for_review'] is not False):
        raise ValueError('complete bound rejected response and TRAIN-only review required')
    messages_path=source/'messages.private.json'
    if file_hash(messages_path)!=old['messages_sha256']:
        raise ValueError('prior feedback changed')
    messages=json.loads(messages_path.read_text());context=strict_json(messages[1]['content'])
    if (context['schema']!='jev-generative-audio-ace-feedback-v1'
            or context['split']!='train' or context['selection_or_test_feedback'] is not False
            or context['parent_prompt_sha256']!=review['parent_prompt_sha256']):
        raise ValueError('same Omni TRAIN feedback required')
    _, bound_context, _, proposal = load_proposal(source)
    if bound_context != context:
        raise ValueError('configured prior feedback differs from revision source')
    context['previous_proposal']=proposal
    context['local_rejection_feedback']=dict(candidates=review['candidates'],diagnosis_corrections=review['diagnosis_corrections'])
    context['representative_evidence_ids']=[r['evidence_id'] for r in context['representatives']]
    raw=canonical(context)
    if len(raw)>150000:
        raise ValueError('bounded feedback exceeded')
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    path=output/'messages.private.json'
    path.write_bytes(canonical([dict(role='system',content=messages[0]['content']+CORRECTION),
                               dict(role='user',content=raw.decode())]));path.chmod(0o600)
    receipt={**old,'messages_sha256':file_hash(path),'context_bytes':len(raw),
        'review_sha256':file_hash(source/'rejection-review.json'),'previous_usage':result['usage'],
        'source_request':str(source),'scope':'explicit semantic correction of rejected proposals, same TRAIN data; no automatic replay'}
    atomic_json(output/'prepared.json',receipt)
    configure(output,source/'executor-config.private.json',executor)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','output','executor'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output,a.executor),indent=2))
