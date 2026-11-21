"""Independent fresh-process cached GRPO result and exact replay audit."""
import copy
import inspect
from pathlib import Path
from types import FunctionType

from tools.reports import audit_context_cached_training as cached
from tools.reports import prepare_context_cached_recovery as preparation
from tools.reports.context_infonce_recovery import compare as compare_state
from tools.jev.context_replay_audit import require_exact


def compare(continuous,replayed,*,world_size=2):
    """Call only after both independent output audits; no fresh-NPU claim here."""
    import torch
    continuous,replayed=Path(continuous),Path(replayed)
    result=compare_state(continuous,replayed,world_size=world_size)
    require_exact(torch.load(continuous/'groups-0001.pt',map_location='cpu',weights_only=True),
                  torch.load(replayed/'groups-0001.pt',map_location='cpu',weights_only=True),'sampled_groups')
    prior=cached.read(continuous/'checkpoint-0001-committed.json')['joint_policy']['policy_id']
    if cached.read(replayed/'batch-0001.json')['policy_id']!=prior:
        raise ValueError('replay behavior must be the committed first-step policy')
    for name in ('rollout-0001.json',):
        left=cached.read(continuous/name);right=cached.read(replayed/name)
        for key in ('queries','behavior_policy_id','fixed_reference_policy_id','runtime_identity'):
            require_exact(left[key],right[key],'rollout/'+key)
    return dict(result,exact_sampled_groups=True,exact_reference_and_behavior=True)


def replay_auditor(namespace,prior):
    source=inspect.getsource(cached.shared.audit_outputs)
    edits=[
        ("if experiment_window(c) != 'engineering_prefix' or start != 0:",
         "if experiment_window(c) != 'engineering_prefix' or start != 1:"),
        ("or ready['policy_id'] != (initial['policy_id'] if view else c['source_policy_id'])",
         "or ready['policy_id'] != replay_prior['policy_id']"),
        ("current = c['source_policy_id']; prior = initial['tensors']; reference_seen = {}",
         "current = replay_prior['policy_id']; prior = replay_prior['tensors']; reference_seen = {}"),
    ]
    for before,after in edits:
        if source.count(before)!=1:raise ValueError('shared replay audit anchor changed')
        source=source.replace(before,after)
    scope=dict(namespace,replay_prior=prior)
    exec(compile(source,'<cached-recovery-output-audit>','exec'),scope)
    return scope['audit_outputs']


def audit_outputs(package,collection,*,reference_root):
    from multimodal_jev.context_policy_artifacts import read_execution_bundle
    package,collection=Path(package),Path(collection);config=cached.read(package/'config.json')
    if (config['start_update']!=1 or config['stop_update']!=2 or config['experiment_kind']!='engineering_prefix'
            or config['engineering_prefix'] is not True or config['training_dispatch_allowed'] is not False
            or config['resume_root']!=cached.read(package/'PREPARED.json')['remote_root']+'/resume'):
        raise ValueError('exact step1 to step2 cached replay required')
    commit=cached.read(package/'resume/checkpoint-0001-committed.json')
    prior=read_execution_bundle(package/'resume/policy-0001',expected_policy_id=commit['joint_policy']['policy_id'],
        expected_contract=config['source_contract'],expected_base_id=config['base_id'])
    verifier=cached.ReferenceConsumption(config['reference_dataset'],reference_root)
    receipt=cached.read(collection/'run/rollout-0001.json')
    if receipt['update']!=1:raise ValueError('replay rollout cursor differs')
    verifier.check(receipt,cached.read(collection/'run/batch-0001.json'))
    def transition(actual,seen,*,current,reference,replicas):
        if (actual!=receipt or current!=prior['policy_id'] or reference!=verifier.policy or replicas!=2):
            raise ValueError('replay must preserve Stage1 reference and step1 behavior')
        return []
    namespace=dict(cached.shared.audit_outputs.__globals__,expected_contract=cached.expected_contract,
                   reference_transition=transition)
    result=replay_auditor(namespace,prior)(package,collection,initial_policy=package/'initial/policy')
    result.update(status='cached_recovery_outputs_audited',consumed_reference_queries=verifier.samples,
        online_reference_calls=0,reference_policy_switches=0,
        reference_manifest_sha256=config['reference_dataset']['manifest_sha256'])
    return result


def audit(c,provenance):
    spec=c['training_package'];parent=spec['parent_audit']
    # Replays source/parity/reference and original training evidence each time.
    _,_,_,run,parent_report,_,_=preparation.inputs(spec,provenance)
    bound=copy.deepcopy(c)
    bound['training_package']['reference_job']=parent['training_package']['reference_job']
    auditor=FunctionType(cached.audit.__code__,dict(cached.audit.__globals__,
        audit_package=preparation.audit,audit_outputs=audit_outputs),'audit_recovery_transport')
    result=auditor(bound,provenance)
    if result['container_id']==parent_report['container_id']:
        raise ValueError('fresh container required for recovery acceptance')
    result.update(replay=compare(run,Path(c['collection'])/'run'),
        status='finite_cached_reference_recovery_audited',fresh_process_npu_resume_verified=True,
        formal_training_admitted=False,training_dispatch_allowed=False)
    return result
