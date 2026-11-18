"""Shared four-card lifecycle and independent label-free transfer audit."""
import argparse
import json
from pathlib import Path
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_runtime_profile import PROFILE
from multimodal_jev.context_replica_macro_rollout import plan_rollouts
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports import context_train_parity_operations as lifecycle
from tools.reports import audit_context_train_parity as transport
from tools.reports import prepare_context_replica_transfer as preparation
from tools.reports import context_replica_transfer_client as client
from tools.reports.prepare_context_replica_training_candidate import rebase
from tools.reports.audit_context_replica_training import audit_journal
from tools.reports.context_release_deployment import audit as audit_release


def read(p):
    return json.loads(Path(p).read_text())


def validate(c):
    spec = c['transfer_package']
    preparation.audit(spec, {'source':'transfer_lifecycle'})
    fn = FunctionType(lifecycle.validate.__code__, dict(lifecycle.validate.__globals__, audit_package=lambda *a:None))
    result = fn(dict(c, parity_package=spec))
    audit_release(lifecycle.command(c, result[0], result[1], producer=True), result[1])
    return result


def launch(c, provenance):
    if c.get('transfer_dispatch_allowed') is not True:
        raise ValueError('explicit full transfer authorization required')
    fn = FunctionType(lifecycle.launch.__code__, dict(lifecycle.launch.__globals__, validate=validate))
    result = fn(dict(c, finite_inference_dispatch_allowed=True), provenance)
    return dict(result, status='transfer_collected_independent_audit_required')


def local_config(package):
    package = Path(package)
    m = read(package/'PREPARED.json')
    return rebase(read(package/'parity-macro.json'), m['remote_root'], str(package.resolve()))


def audit_outputs(package, collection):
    package, collection = Path(package), Path(collection)
    m = read(package/'PREPARED.json')
    c = read(package/'parity-macro.json')
    pool = c['macro_client']
    if (read(collection/'hf-process/terminal.json')['exit_code'] != 0
            or read(collection/'hf/hf-runtime.json')['profile'] != PROFILE):
        raise ValueError('successful owned HF with declared runtime required')
    supervision = read(collection/'supervision/terminal.json')
    if (supervision.get('passed') is not True or supervision.get('actor_started') is not True
            or supervision.get('actor_exit_code') != 0 or supervision.get('forced_cleanup') is not False
            or supervision.get('child_exit_codes') != [0,0,0]
            or [s['name'] for s in supervision['services']] != [s['name'] for s in m['inference_services']]
            or any(s.get('exit_code') != 0 for s in supervision['services'])):
        raise ValueError('owned full transfer client and services must complete')
    for binding in m['inference_services']:
        name = binding['name']
        service = read(package/m['layout']['services'][name])
        p = service['policy']
        ready = read(collection/f'service-{name}/ready.json')
        expected = dict(policy_id=c['policy_id'], contract=c['contract'], physical_devices=[binding['device']],
            transport_protocol='bulk_dual_logits_candidate_v1', model_forward_mode=pool['runtime_identity']['profile'],
            max_batch_queries=p['max_batch_queries'], max_batch_tokens=p['max_batch_tokens'],
            entrypoint_sha256=service['pins'][service['source_root']+'/tools/reports/serve_context_dual_candidate.py'])
        if (any(ready.get(k) != v for k,v in expected.items())
                or read(collection/f'service-{name}/terminal.json')['exit_code'] != 0
                or p['max_decisions'] != m['decisions']):
            raise ValueError('actual transfer service differs: '+name)
    # Source score JSON contains remote paths. Rebind only the read operation,
    # preserving hashes and the source-binding path recorded by both clients.
    local = local_config(package)
    remote_root = Path(m['remote_root'])
    def bound(entry):
        p = Path(entry['path'])
        if not p.is_relative_to(remote_root/'source-evidence'):
            raise ValueError('packaged source evidence required')
        return client.read_bound(dict(entry, path=str(package/p.relative_to(remote_root))))
    source_binding = FunctionType(client.source_binding.__code__,
        dict(client.source_binding.__globals__, read_bound=bound))
    local['source_evaluation'] = c['source_evaluation']
    # The generic auditor has its own source_binding global, so bind it too.
    generic = FunctionType(client.shared_audit.audit_outputs.__code__,
        dict(client.shared_audit.audit_outputs.__globals__, source_binding=source_binding))
    from types import SimpleNamespace
    fn = FunctionType(client.audit_outputs.__code__, dict(client.audit_outputs.__globals__,
        source_binding=source_binding, shared_audit=SimpleNamespace(audit_outputs=generic)))
    journal = collection/'transfer-journal'/pool['attempt_id']
    proof = fn(local, collection/'evaluation', journal, collection/'hf')
    dispatch = []
    inference = collection/'evaluation/inference'
    for index, entry in enumerate(read(inference/'result.json')['batches']):
        records = [{k:v for k,v in row.items() if k != 'id'} for row in read(inference/entry['path'])['predictions']]
        dispatch.append(audit_journal(journal/f'evaluation-{index:08d}', records, policy=c['policy_id'],
            client=pool, planner=lambda costs,width:plan_rollouts(costs,width,replicas=2)))
    return dict(proof, status='complete_replica_transfer_audited', dispatch_audit=dispatch,
        actor_updates=0, full_population_parity=False, training_dispatch_allowed=False)


def audit(c, provenance):
    fn = FunctionType(transport.audit.__code__, dict(transport.audit.__globals__,
        audit_package=preparation.audit, audit_outputs=audit_outputs))
    result = fn(dict(c, parity_package=c['transfer_package']), provenance)
    result['scope'] = 'Full native label-free predictions and per-policy HF panel parity; target scoring remains separate'
    return result


def run(c, provenance):
    if c.get('transfer_dispatch_allowed') is not True:
        raise ValueError('explicit full transfer authorization required')
    root = Path(c['operations_root'])/'finalization'
    check_paths({'transfer_finalization':root})
    root.mkdir(parents=True, exist_ok=False)
    commit_json(root/'config.json', c)
    commit_json(root/'provenance.json', provenance)
    try:
        result = launch(c, provenance)
        if result['status'] != 'transfer_collected_independent_audit_required':
            raise ValueError('full transfer collection required')
        stem = Path(c['operations_root'])/c['tag']
        bound = dict(transfer_package=c['transfer_package'], deployment=c,
            **{k:str(stem)+suffix for k,suffix in (
                ('launch','-launch.json'), ('inspect','-inspect.json'), ('wait','-wait.json'),
                ('transport','-collected.transport.json'), ('collection','-collected'), ('archive','-collected.tar'))})
        commit_json(root/'audit-config.json', bound)
        proof = audit(bound, provenance)
        if (proof['status'] != 'complete_replica_transfer_audited'
                or proof['runtime_provenance_verified'] is not True or proof['target_scored'] is not False):
            raise ValueError('independent complete runtime/output audit required')
        commit_json(root/'audit.json', proof)
        commit_json(root/'terminal.json', dict(exit_code=0, status='transfer_collected_and_audited', target_scored=False))
        return proof
    except BaseException as exc:
        commit_json(root/'terminal.json', dict(exit_code=1, error=repr(exc), restart=False, preserve_handles=True))
        raise


if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    p.add_argument('--action', choices=['validate','run','audit'], default='validate')
    a = p.parse_args()
    c, provenance = load_config(a.config, a.local)
    if a.action == 'run': result = run(c, provenance)
    elif a.action == 'audit': result = audit(c, provenance)
    else:
        validate(c)
        result = dict(status='transfer_lifecycle_validated_not_executed', npu_executed=False)
    print(json.dumps(result, indent=2))
