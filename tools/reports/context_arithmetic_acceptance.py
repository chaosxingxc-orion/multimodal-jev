"""Rebind affected-path evidence and reuse unchanged checkpoint-frequency logic.

Numerical repair requires new continuous/replay runs. The unchanged save/export
branch has already been matched on NPU at intervals1/64; reuse that evidence
only after byte-level comparison of the trainer outside the declared numeric
contract insertion. No new grant of formal training is made by this function.
"""
import json
from pathlib import Path
from tools.jev.context_formal_acceptance import verify as original_accept
from tools.jev.context_rl_evaluation_plan import sha
from tools.jev.context_replay_audit import verify_update
from tools.reports.audit_context_arithmetic_candidate import audit
from tools.reports.prepare_context_arithmetic_candidate import transform_trainer
from tools.reports.context_probability_policy_v2 import PROFILE


def verify(continuous, fresh_replay, previous_acceptance):
    old_spec=json.loads(Path(previous_acceptance).read_text())
    old=original_accept(**old_spec)
    roots={};reports={};configs={};bindings={}
    for role,entry in [('continuous',continuous),('fresh_replay',fresh_replay)]:
        spec={k:entry[k] for k in ('collection','transport','launch','package','initial_policy','schedule')}
        for k in ('transport','launch'):spec[k]=json.loads(Path(spec[k]).read_text())
        actual=audit(**spec)
        archived=json.loads(Path(entry['report']).read_text())
        if actual!=archived:raise ValueError('new numeric evidence differs from independent re-audit')
        root=Path(entry['collection']);c=json.loads((root/'config.json').read_text())
        if c['start_update']!=(0 if role=='continuous' else 1) or c['checkpoint_interval']!=1:
            raise ValueError('matched continuous and fresh-process replay required')
        roots[role]=root;reports[role]=actual;configs[role]=c
        bindings[role]={k:dict(path=entry[k],sha256=sha(entry[k])) for k in ('report','transport','launch')}
    if reports['continuous']['actor_container']==reports['fresh_replay']['actor_container']:
        raise ValueError('distinct actual containers required')
    shared=('source_policy_id','source_contract','base_id','runtime_profile','release_sha256',
            'schedule_sha256','algorithm','behavior_bounds','binary_policy_arithmetic')
    if any(configs['continuous'][k]!=configs['fresh_replay'][k] for k in shared):
        raise ValueError('new numerical runtime/replay source mismatch')
    candidate=Path(continuous['package'])
    old_package=Path(old_spec['candidate_package'])
    rel='source/tools/jev/train_context_online_candidate.py'
    if (candidate/rel).read_text()!=transform_trainer((old_package/rel).read_text()):
        raise ValueError('checkpoint/export/update code changed beyond declared numeric identity')
    # The original experiment used an earlier policy; never claim its weights
    # or updates were reproduced by the repaired Stage1-endpoint candidate.
    replay=verify_update(roots['continuous']/'run',roots['fresh_replay']/'run')
    return dict(status='arithmetic_affected_paths_accepted_no_dispatch',binary_policy_arithmetic=PROFILE,
        source_policy_id=configs['continuous']['source_policy_id'],bindings=bindings,replay=replay,
        reused_checkpoint_frequency=dict(specification=previous_acceptance,
            specification_sha256=sha(previous_acceptance),evidence=old,
            qualification='Earlier-policy NPU interval comparison reused only for byte-identical checkpoint/export logic; not same-policy numerical trajectory evidence.'),
        unchanged_checkpoint_trainer_sha256=sha(old_package/rel),
        candidate_trainer_sha256=sha(candidate/rel),
        scope='Affected continuous/backward/refresh and fresh replay re-audited; old interval lifecycle evidence reused with exact transform check. Formal publication and efficacy separate.')
