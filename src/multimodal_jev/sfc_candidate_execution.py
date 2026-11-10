"""Bind candidate inference to teacher provenance and an explicit semantic review.

Review is a recorded engineering assessment, not a user permission request or a
performance acceptance. TEST candidate selection requires a separate protocol.
"""
import hashlib
from pathlib import Path

from .baseline import file_hash
from .sfc_candidate_context import load_candidate
from .sfc_jev_runtime import INSTRUCTION
from .teacher_candidates import strict_json

REVIEW_CHECKS = ('generic_instruction_only', 'audio_direct', 'history_and_audio_order',
                'function_environment_unchanged', 'decision_output_contracts',
                'no_noul_gating', 'types_nan_duplicates_and_order', 'no_sample_answers')


def execution_prompt(cfg):
    if cfg.get('prompt_id') == 'parent':
        if 'candidate' in cfg or cfg.get('instruction_sha256') != hashlib.sha256(INSTRUCTION.encode()).hexdigest():
            raise ValueError('original parent instruction required')
        return INSTRUCTION, None
    spec = cfg.get('candidate')
    if (cfg.get('prompt_id') not in ('p1','p2') or type(spec) is not dict
            or set(spec) != {'staging','staging_hash','candidate_id','review','review_sha256'}
            or spec['candidate_id'] != cfg['prompt_id']):
        raise ValueError('explicit reviewed candidate specification required')
    text, identity = load_candidate(**{k:spec[k] for k in ('staging','staging_hash','candidate_id')})
    staged = strict_json((Path(spec['staging'])/'staged.json').read_text())
    if staged['schema'] != 'sfc-jev-teacher-wire-staging-v1' or staged.get('wire_authenticated') is not True:
        raise ValueError('actual authenticated teacher response required for candidate inference')
    if file_hash(spec['review']) != spec['review_sha256']:
        raise ValueError('candidate semantic review changed')
    review = strict_json(Path(spec['review']).read_text())
    if (set(review) != {'schema','candidate_id','staging_sha256','instruction_sha256',
                       'decision','reviewer','rationale','checks'}
            or review['schema'] != 'sfc-jev-candidate-semantic-review-v1'
            or review['candidate_id'] != cfg['prompt_id'] or review['decision'] != 'eligible_for_evaluation'
            or review['staging_sha256'] != spec['staging_hash']
            or review['instruction_sha256'] != identity['instruction_sha256']
            or cfg.get('instruction_sha256') != identity['instruction_sha256']
            or set(review['checks']) != set(REVIEW_CHECKS)
            or any(v is not True for v in review['checks'].values())
            or any(type(review[k]) is not str or not review[k].strip() for k in ('reviewer','rationale'))):
        raise ValueError('explicit semantic eligibility review for exact candidate required')
    return text, identity


def context_spec(cfg):
    if cfg.get('prompt_id') == 'parent':
        return None
    execution_prompt(cfg)
    return {k:cfg['candidate'][k] for k in ('staging','staging_hash','candidate_id')}
