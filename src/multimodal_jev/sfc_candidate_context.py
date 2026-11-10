"""Bind CPU-only prompt checks to exact staged candidate bytes."""
import hashlib
from pathlib import Path

from .baseline import file_hash
from .sfc_prompt_candidates import check_instruction
from .sfc_jev_runtime import INSTRUCTION
from .teacher_candidates import strict_json


def load_candidate(staging,staging_hash,candidate_id):
    root=Path(staging).resolve();path=root/'staged.json'
    if file_hash(path)!=staging_hash:raise ValueError('candidate staging changed')
    receipt=strict_json(path.read_text())
    if (receipt.get('schema') not in ('sfc-jev-local-candidate-staging-v1','sfc-jev-teacher-wire-staging-v1')
            or candidate_id not in ('p1','p2')
            or receipt.get('parent_prompt_sha256')!=hashlib.sha256(INSTRUCTION.encode()).hexdigest()):
        raise ValueError('registered SFC candidate required')
    matches=[v for v in receipt['candidates'] if v['id']==candidate_id]
    if len(matches)!=1:raise ValueError('one staged candidate required')
    instruction=root/(candidate_id+'.txt');text=instruction.read_text();check_instruction(text)
    if file_hash(instruction)!=matches[0]['instruction_sha256'] or len(text)!=matches[0]['characters']:
        raise ValueError('staged instruction changed')
    if receipt['schema']=='sfc-jev-teacher-wire-staging-v1':
        # Real teacher candidates are staged directly under their request root.
        # Recheck persisted request/SSE custody rather than trusting a label.
        from .sfc_teacher_wire import load_proposal
        from .sfc_prompt_candidates import validate
        context,payload,evidence=load_proposal(root.parent,receipt['intent_sha256'])
        candidates=validate(payload,context)
        expected=[dict(id=c['id'],instruction_sha256=hashlib.sha256(c['instruction'].encode()).hexdigest(),
                       characters=len(c['instruction'])) for c in candidates]
        if (any(receipt.get(k)!=v for k,v in evidence.items())
                or evidence['output_budget_exceeded'] is True or receipt.get('wire_authenticated') is not True
                or receipt['candidates']!=expected
                or receipt['parent_prompt_sha256']!=context['parent_prompt_sha256']
                or text!=next(c['instruction'] for c in candidates if c['id']==candidate_id)):
            raise ValueError('candidate differs from verified teacher response')
    return text,dict(kind='staged-candidate-context-only',candidate_id=candidate_id,
        staging=str(root),staging_sha256=staging_hash,instruction_sha256=file_hash(instruction),
        parent_prompt_sha256=receipt['parent_prompt_sha256'],semantic_review=receipt['semantic_review'],
        candidate_accepted=False,
        scope='CPU input feasibility only; does not approve semantics, teacher provenance or inference.')
