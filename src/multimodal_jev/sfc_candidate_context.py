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
    if (receipt.get('schema')!='sfc-jev-local-candidate-staging-v1'
            or candidate_id not in ('p1','p2')
            or receipt.get('parent_prompt_sha256')!=hashlib.sha256(INSTRUCTION.encode()).hexdigest()):
        raise ValueError('registered SFC candidate required')
    matches=[v for v in receipt['candidates'] if v['id']==candidate_id]
    if len(matches)!=1:raise ValueError('one staged candidate required')
    instruction=root/(candidate_id+'.txt');text=instruction.read_text();check_instruction(text)
    if file_hash(instruction)!=matches[0]['instruction_sha256'] or len(text)!=matches[0]['characters']:
        raise ValueError('staged instruction changed')
    return text,dict(kind='staged-candidate-context-only',candidate_id=candidate_id,
        staging=str(root),staging_sha256=staging_hash,instruction_sha256=file_hash(instruction),
        parent_prompt_sha256=receipt['parent_prompt_sha256'],semantic_review=receipt['semantic_review'],
        candidate_accepted=False,
        scope='CPU input feasibility only; does not approve semantics, teacher provenance or inference.')
