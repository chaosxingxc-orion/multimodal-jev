"""InfoNCE source provenance for the shared selection runtime and scorer.

The runtime template supplies code only. Its historical policy, training audit,
and data release never establish the origin of a new evaluation policy.
"""
import argparse
import json
from pathlib import Path
import shutil

from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_preparation import processor_cache_identity
from multimodal_jev.context_policy_artifacts import (
    export_native_adapter, read_execution_bundle, verify_native_adapter,
)
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports import context_selection_macro_operations as shared
from tools.reports.context_infonce_source_plan import build
from tools.reports.context_replica_training_operations import release_evidence
from tools.reports.evaluate_context_selection_macro import inputs
from tools.reports.context_selection_audited_cache import load_evidence
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.selection_macro_cache_operations import audit_receipts, IMAGE

from multimodal_jev.stage_experiment_contract import source_curve

# Only unversioned historical plans use this frozen design.
_LEGACY = Path(__file__).resolve().parents[2]/'configs/experiments/context-two-stage-v0.0.11.json'
STEPS = json.loads(_LEGACY.read_bytes())['stage1']['evaluation_steps']


def read(path):
    return json.loads(Path(path).read_text())


def validate_plan(plan):
    curve = plan.get('source_curve')
    if curve is None:
        curve = source_curve(read(_LEGACY))
    steps = curve['steps']
    if (not steps or len(steps)<2 or steps[0]!=0 or steps!=sorted(set(steps))
            or any(type(i) is not int for i in steps)):
        raise ValueError('ordered preregistered source curve required')
    experiment = plan['experiment']
    evaluation = experiment['evaluation']
    primary='query_macro_set_f1'
    if 'stage1_layout' in plan:
        from multimodal_jev.stage2_dataset_release import require_endpoint_layout
        require_endpoint_layout(plan,plan['stage1_layout'])
        primary='top1_macro_f1'
    if (plan['schema'] != 'jev-infonce-source-evaluation-plan-v1'
            or plan['status'] != 'audited_source_policies_bound_evaluation_pending'
            or plan['algorithm'] != 'candidate_infonce'
            or plan['source_audit']['status'] != 'formal_infonce_artifacts_audited'
            or plan['source_audit']['world_size'] != curve['world_size']
            or plan['source_audit']['real_exposures'] != curve['real_exposures']
            or len(plan['source_audit']['updates']) != steps[-1]
            or [p['step'] for p in plan['checkpoints']] != steps
            or any(type(p['step']) is not int for p in plan['checkpoints'])
            or plan['checkpoints'][0]['policy_id'] != plan['initial_policy_id']
            or plan['reference_endpoint'] != dict(step=steps[-1], policy_id=plan['checkpoints'][-1]['policy_id'])
            or experiment['task_stage']['updates'] != steps[-1]
            or experiment['selection_recordings'] != curve['selection_recordings']
            or evaluation['checkpoints'] != steps
            or evaluation['primary'] != primary
            or evaluation['set_threshold'] != .5
            or not {'candidate_binary_nll','candidate_brier','top1_event_accuracy','top1_macro_f1'} <= set([evaluation['primary'],*evaluation['secondary']])
            or plan['target_tuning_allowed'] is not False
            or plan['model_selection_performed'] is not False):
        raise ValueError('complete declared InfoNCE source and preregistered curve required')
    return plan


def configured_steps(config):
    if file_hash(config['source_plan'])!=config['source_plan_sha256']:
        raise ValueError('source plan changed')
    return validate_plan(read(config['source_plan']))['experiment']['evaluation']['checkpoints']


def reconstruct_plan(config):
    if file_hash(config['source_plan']) != config['source_plan_sha256']:
        raise ValueError('source plan changed')
    plan = validate_plan(read(config['source_plan']))
    actual = build(config['source_finalization'], plan['config_provenance'])
    if actual != plan:
        raise ValueError('source plan differs from independently audited InfoNCE training')
    return actual


def prepare(config, provenance):
    return prepare_point(config, provenance, reconstruct_plan(config))


def prepare_point(c, provenance, plan):
    """Seal a point after the public entry independently reconstructs its plan."""
    validate_plan(plan)
    points = [p for p in plan['checkpoints'] if p['step'] == c['source_step']]
    if type(c['source_step']) is not int or len(points) != 1:
        raise ValueError('one registered InfoNCE checkpoint required')
    point = points[0]
    template = Path(c['runtime_template']); source = template/'package'
    receipt = read(template/'PREPARED.json')
    if (verify(source) != c['runtime_manifest_sha256']
            or receipt['files_sha256'] != c['runtime_manifest_sha256'] or receipt['image'] != IMAGE
            or receipt['status'] != 'macro_selection_prepared_only'):
        raise ValueError('sealed shared selection runtime template required')
    snapshot = Path(point['snapshot'])
    bundle = read_execution_bundle(snapshot, expected_policy_id=point['policy_id'],
        expected_contract=plan['source_contract'], expected_base_id=plan['base_id'])
    if (file_hash(snapshot/'MANIFEST.json') != point['manifest_sha256']
            or processor_cache_identity(snapshot) != receipt['runtime_identity']['processor_sha256']):
        raise ValueError('actual source policy or processor differs')
    training = read(Path(plan['source_operations']['package'])/'config.json')
    if (file_hash(Path(plan['source_operations']['package'])/'config.json') != plan['training_config_sha256']
            or c['release'] != training['release']
            or c['release_sha256'] != training['release_sha256']
            or c['release_sha256'] != plan['experiment']['release_sha256']):
        raise ValueError('exact formal source release required')
    # Rebind label-free inputs to the current release rather than the runtime
    # template's old policy/release. The independent cache receipts remain exact.
    evaluation = read(source/'evaluation.json')
    evaluation.update(release=c['release'], release_sha256=c['release_sha256'])
    evidence = source/'evaluation-inputs'
    binding = read(evidence/'binding/binding.json')
    for key, name in [('cache_audit','original-cache-audit.json'), ('cache_receipts','original-cache-stream.jsonl')]:
        path = evidence/name
        if file_hash(path) != binding[key]['sha256']:
            raise ValueError('template cache evidence differs')
        evaluation[key] = dict(path=str(path), sha256=file_hash(path))
    population, _, candidates = inputs(evaluation)
    load_evidence(evaluation, population, release_manifest=read(Path(c['release'])/'MANIFEST.json'),
                  processor_sha256=processor_cache_identity(snapshot))
    proof = audit_receipts(c['binding_root'], c['cache_collection'], c['binding_sha256'])
    if (read(Path(c['binding_root'])/'population.json') != population
            or read(Path(c['binding_root'])/'candidate-ids.json') != candidates
            or file_hash(evidence/'binding/binding.json') != c['binding_sha256']
            or file_hash(evidence/'input-receipts.json') != file_hash(Path(c['cache_collection'])/'input-receipts.json')
            or any(read(evidence/'cache-verification.json').get(k) != v for k,v in proof.items())):
        raise ValueError('complete independently verified selection inputs required')
    output = Path(c['output']); remote = Path(c['remote_root'])
    forbidden = [Path(receipt['remote_root']), Path(plan['source_operations']['remote_root'])]
    if (not remote.is_absolute() or '..' in remote.parts or ':' in str(remote)
            or len(str(remote/'ipc').encode())+8 > 107
            or any(remote.is_relative_to(p) or p.is_relative_to(remote) for p in forbidden)
            or output.exists() or output.resolve().is_relative_to(template.resolve())):
        raise ValueError('fresh isolated selection package and remote root required')
    check_paths({'selection_package': output})
    output.mkdir(parents=True); dest = output/'package'; shutil.copytree(source, dest)
    for directory in ('policy','adapter'):
        shutil.rmtree(dest/directory)
    shutil.copytree(snapshot, dest/'policy')
    export_native_adapter(dest/'adapter', bundle)
    verify_native_adapter(dest/'adapter', bundle)
    # Runtime files remain byte-identical. Historical ancillary artifacts are
    # inert and retained under the template manifest, never used as admission.
    (dest/'FILES.json').unlink()
    commit_json(dest/'SOURCE-PLAN.json', plan)
    (dest/'TRAINING-AUDIT.json').unlink()
    commit_json(dest/'TRAINING-AUDIT.json', plan['source_audit'])
    root = str(remote/'package'); endpoints = [dict(socket=str(remote/f'ipc/r{i}.sock'),
        exchange=str(remote/f'ipc/exchange-{i}')) for i in range(2)]
    pins = {root+'/'+str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*')
            if p.is_file() and p.name not in ('service-0.json','service-1.json','evaluation.json')}
    for i in range(2):
        service = read(source/f'service-{i}.json')
        service.update(source_root=root, source_policy=root+'/policy', pins=pins,
            dual_runtime_manifest=root+'/RUNTIME-SOURCES.json', **dict(socket_path=endpoints[i]['socket'],
            exchange=endpoints[i]['exchange']))
        service['policy'].update(snapshot=root+'/policy', adapter=root+'/adapter',
            policy_id=point['policy_id'], contract=plan['source_contract'], base_id=plan['base_id'],
            max_batch_queries=32,max_batch_tokens=131072,max_decisions=60)
        (dest/f'service-{i}.json').write_text(json.dumps(service, indent=2)+'\n')
    evaluation.update(snapshot=root+'/policy',policy_id=point['policy_id'],contract=plan['source_contract'],
        base_id=plan['base_id'],encoded_audio_cache=c['cache_root'],output=str(remote/'evaluation'),pins=pins)
    evaluation['macro_client'].update(journal_root=str(remote/'journal'),replicas=endpoints)
    for key, name in [('cache_audit','original-cache-audit.json'), ('cache_receipts','original-cache-stream.jsonl')]:
        evaluation[key]['path']=root+'/evaluation-inputs/'+name
    (dest/'evaluation.json').write_text(json.dumps(evaluation,indent=2)+'\n')
    commit_json(dest/'FILES.json',{str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()})
    result = dict(receipt, status='macro_source_point_prepared_only', files_sha256=verify(dest),
        policy_id=point['policy_id'],policy_role='source_curve_point',source_step=point['step'],
        source_plan_sha256=file_hash(dest/'SOURCE-PLAN.json'),remote_root=str(remote),ipc_root=str(remote/'ipc'),
        parent_manifest_sha256=plan['source_operations']['manifest_sha256'],
        parent_audit_sha256=file_hash(dest/'TRAINING-AUDIT.json'),config_provenance=provenance,
        runtime_template_manifest_sha256=c['runtime_manifest_sha256'],source_changes=[],
        service_batch_queries=32,service_batch_tokens=131072,numerical_admission_pending=True,
        scope='InfoNCE policy with reused selection runtime; prepared only, numerical parity and actual evaluation pending')
    commit_json(output/'PREPARED.json',result)
    return result


def validate(c):
    receipt=shared.validate(c,source_curve=True,source_kind='infonce')
    for i in range(2):
        policy=read(Path(c['prepared'])/f'package/service-{i}.json')['policy']
        if (policy['max_batch_queries']!=32 or policy['max_batch_tokens']!=131072
                or policy['max_decisions']!=60):
            raise ValueError('declared batch32 complete source ontology required')
    return receipt


def command(c,role):
    argv=shared.command(c,role)
    if role in ('preflight','client'):
        config=read(Path(c['prepared'])/'package/evaluation.json')
        mount=release_evidence(c,config)
        if mount:
            origin,destination=mount
            if Path(c['remote_root']).is_relative_to(destination) or Path(destination).is_relative_to(c['remote_root']):
                raise ValueError('release evidence must not shadow evaluation runtime')
            argv[argv.index('--entrypoint'):argv.index('--entrypoint')]=['-v',origin+':'+destination+':ro']
    return argv


def audit_outputs(package,collection):
    return shared.audit_outputs(package,collection,source_curve=True)


def audit_collected(config):
    from tools.reports.audit_context_macro_training import audit_collected as reaudit
    return reaudit(config,validator=validate,command_builder=command,output_auditor=audit_outputs)


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    # Dispatch is deliberately absent until the new policy's HF/vLLM numerical
    # admission is bound to this package. A packaging check is not that gate.
    parser.add_argument('--action',choices=['prepare','validate','audit'],default='validate')
    args=parser.parse_args();c,p=load_config(args.config,args.local)
    if args.action=='prepare': result=prepare(c,p)
    elif args.action=='validate': result=validate(c)
    else: result=audit_collected(c)
    print(json.dumps(result,indent=2))
