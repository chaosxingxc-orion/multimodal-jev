"""Seal source-frozen native transfer on the unchanged four-card model runtime."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_preparation import processor_cache_identity
from multimodal_jev.context_policy_artifacts import read_execution_bundle, export_native_adapter
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_cached_grpo_transfer_source import admit, seal
from tools.reports.prepare_context_stage2_runtime import audit as audit_parent
from tools.reports.prepare_context_replica_training_candidate import rebase
from tools.reports.prepare_context_replica_selection import runner as selection_runner
from tools.reports.context_device_visibility import visibility
from tools.reports.context_replica_training_operations import path
from tools.reports import context_replica_transfer_client as client
from tools.reports.context_replica_transfer_auxiliary import prepare as auxiliary


def read(p):
    return json.loads(Path(p).read_text())


def inputs(c, provenance):
    source = admit(c['source_admission'])
    if c['parent'] != source['source_operations']['training_package']['parent']:
        raise ValueError('actual formal training parent required')
    proof = audit_parent(c['parent'], provenance)
    parent = Path(c['parent']['output'])
    m = read(parent/'PREPARED.json')
    training = read(parent/'config.json')
    point = source['point']
    bundle = read_execution_bundle(point['snapshot'], expected_policy_id=point['policy_id'],
        expected_contract=training['source_contract'], expected_base_id=training['base_id'])
    if (source['training']['source_contract'] != training['source_contract']
            or source['training']['base_id'] != training['base_id']
            or source['training']['release_sha256'] != training['release_sha256']
            or source['training']['dual_runtime_identity'] != training['dual_runtime_identity']
            or file_hash(Path(point['snapshot'])/'MANIFEST.json') != point['manifest_sha256']
            or processor_cache_identity(point['snapshot']) != training['dual_runtime_identity']['processor_sha256']):
        raise ValueError('same admitted model, source and runtime required')
    # Targets are opened only after complete source/weights admission.
    full, selected, evidence = client.inputs(c)
    if (full['processor_sha256'] != training['dual_runtime_identity']['processor_sha256']
            or (c['dataset'] == 'SLURP_clean_TEST60'
                and c['source_release'] != dict(path=training['release'], sha256=training['release_sha256']))):
        raise ValueError('same processor and actual source TEST release required')
    return source, proof, parent, m, training, point, bundle, full, selected, evidence


def runner(m):
    source = selection_runner(m)
    old = "if c['expected_population']!=9226 or c['source_step'] not in [0,*range(32,318,32),318]:"
    new = ("if c['dataset'] not in {'FSC_native31':3793,'MINDS_native14':8168,'SLURP_clean_TEST60':12991} "
        "or c['expected_population']!={'FSC_native31':3793,'MINDS_native14':8168,'SLURP_clean_TEST60':12991}[c['dataset']]:")
    if source.count(old) != 1:
        raise ValueError('shared full population guard changed')
    source = source.replace(old, new).replace('selection-client.py', 'transfer-client.py')
    source = source.replace('REPLICA_SELECTION_CLIENT_CONFIG', 'REPLICA_TRANSFER_CLIENT_CONFIG')
    source = source.replace('complete TRAIN endpoint reference only', 'complete native held-out population only')
    compile(source, '<replica-transfer-runner>', 'exec')
    return source


def configured(c, training, m, point, remote, source_entry, count):
    value = rebase(training, m['remote_root'], remote)
    value.update(source_policy_id=point['policy_id'], encoded_audio_cache=c['portable_full_inputs']['cache_root'],
        training_dispatch_allowed=False, runtime_preparation_only=True)
    value.pop('reference_cache', None)
    value.pop('reference_dataset', None)
    evaluate = copy.deepcopy(value)
    evaluate.update(dataset=c['dataset'], source_evaluation=source_entry,
        policy_role=c['source_admission']['policy_role'], snapshot=value['source_policy'],
        policy_id=point['policy_id'], contract=value['source_contract'], macro_queries=128,
        expected_population=count, source_step=point['step'], max_model_len=4096,
        portable_full_inputs=dict(c['portable_full_inputs'], root=remote+'/full-inputs'),
        portable_parity_inputs=dict(c['portable_parity_inputs'], root=remote+'/parity-inputs'))
    if c['dataset'] == 'SLURP_clean_TEST60':
        evaluate['source_release'] = c['source_release']
    evaluate['macro_client'].update(attempt_id='transfer-v1', journal_root=remote+'/transfer-journal')
    visibility(m)
    target = m['training_devices'][0]
    mounted = sorted(m['training_devices']+[v['device'] for v in m['inference_services']])
    evaluate.update(physical_devices=[target], process_device_scope=dict(
        kind='same_container_runtime_visibility_v1', mounted_physical=mounted,
        visible_ordinal=str(mounted.index(target))))
    return value, evaluate


def prepare(c, provenance):
    source, proof, parent, m, training, point, bundle, full, selected, evidence = inputs(c, provenance)
    output = Path(c['output'])
    remote = path(c['remote_root'])
    old = m['remote_root']
    protected = [parent, Path(point['snapshot']), Path(c['source_admission']['source_evaluation']['path']),
        Path(c['source_admission']['source_curve_config']['source_plan']),
        Path(c['portable_full_inputs']['root']), Path(c['portable_parity_inputs']['root'])]
    if (output.exists() or any(output.resolve().is_relative_to(p.resolve())
            or p.resolve().is_relative_to(output.resolve()) for p in protected)
            or Path(remote).is_relative_to(old) or Path(old).is_relative_to(remote)
            or any(Path(remote).is_relative_to(c[key]['cache_root'])
                or Path(c[key]['cache_root']).is_relative_to(remote)
                for key in ('portable_full_inputs', 'portable_parity_inputs'))):
        raise ValueError('fresh disjoint transfer package required')
    check_paths({'transfer_package': output})
    shutil.copytree(parent, output)
    (output/'PREPARED.json').unlink()
    for name in ('policy', 'adapter'):
        shutil.rmtree(output/'initial'/name)
    shutil.copytree(point['snapshot'], output/'initial/policy')
    export_native_adapter(output/'initial/adapter', bundle)
    (output/'initial/endpoint-commit.json').unlink()
    commit_json(output/'TRANSFER-SOURCE-ADMISSION.json', source)
    commit_json(output/'TRANSFER-PARENT-AUDIT.json', proof)
    commit_json(output/'TRANSFER-INPUT-COVERAGE.json', evidence)
    shutil.copytree(c['portable_full_inputs']['root'], output/'full-inputs')
    shutil.copytree(c['portable_parity_inputs']['root'], output/'parity-inputs')
    source_entry = seal(source, output/'source-evidence', remote+'/source-evidence')
    auxiliary(output, output/'RUNTIME-SOURCES.json')
    (output/'run.py').write_text(runner(m))
    (output/'parity-template.json').write_text('"${REPLICA_TRANSFER_CLIENT_CONFIG}"\n')
    config, evaluate = configured(c, training, m, point, remote, source_entry, len(full['population']))
    names = {'config.json', 'parity-hf.json', 'parity-macro.json', 'parity-hf-local.json', *m['layout']['services'].values()}
    external = {k:v for k,v in training['pins'].items() if not Path(k).is_relative_to(old)}
    pins = dict(external, **{remote+'/'+str(p.relative_to(output)):file_hash(p) for p in output.rglob('*')
        if p.is_file() and str(p.relative_to(output)) not in names})
    config['pins'] = pins
    evaluate['pins'] = pins
    hf = dict(evaluate, output=remote+'/hf')
    macro = dict(evaluate, output=remote+'/evaluation')
    configs = {'config.json': config, 'parity-hf.json': hf, 'parity-macro.json': macro,
        'parity-hf-local.json': dict(schema='jev-local-macros-v1', macros=dict(REPLICA_TRANSFER_CLIENT_CONFIG=hf))}
    for name in m['layout']['services'].values():
        service = rebase(read(parent/name), old, remote)
        service['pins'] = pins
        service['policy'].update(policy_id=point['policy_id'], max_decisions=len(full['candidate_ids']))
        configs[name] = service
    for name, value in configs.items():
        (output/name).write_text(json.dumps(value, indent=2)+'\n')
    if {str(p.relative_to(output/'source')):file_hash(p) for p in (output/'source').rglob('*.py')} != read(parent/'RUNTIME-SOURCES.json'):
        raise ValueError('transfer changed sealed model runtime')
    receipt = dict(m, status='replica_transfer_prepared_only', remote_root=remote,
        source_policy_id=point['policy_id'], source_step=point['step'], policy_role=evaluate['policy_role'],
        parent_sha256=c['parent']['manifest_sha256'], dataset=c['dataset'], recordings=len(full['population']),
        decisions=len(full['candidate_ids']), actor_updates=0, numerical_admission=False,
        training_dispatch_allowed=False, config_provenance=provenance)
    receipt['layout'] = copy.deepcopy(m['layout'])
    receipt['layout']['configs'] += ['parity-hf.json', 'parity-macro.json', 'parity-hf-local.json',
        'parity-template.json', 'transfer-client.py', 'TRANSFER-SOURCE-ADMISSION.json',
        'TRANSFER-PARENT-AUDIT.json', 'TRANSFER-INPUT-COVERAGE.json']
    receipt['layout']['directories'] = ['hf', 'hf-process', 'evaluation', 'transfer-journal', 'supervision',
        *['service-'+s['name'] for s in m['inference_services']]]
    receipt['files'] = {str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file()}
    commit_json(output/'PREPARED.json', receipt)
    return dict(status=receipt['status'], manifest_sha256=file_hash(output/'PREPARED.json'),
        npu_executed=False, training_dispatch_allowed=False)


def audit(c, provenance):
    root = Path(c['output'])
    manifest = root/'PREPARED.json'
    if root.is_symlink() or file_hash(manifest) != c['manifest_sha256']:
        raise ValueError('transfer manifest differs')
    m = read(manifest)
    if (any(p.is_symlink() for p in root.rglob('*'))
            or {str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file() and p != manifest} != m['files']):
        raise ValueError('complete immutable transfer package required')
    check_paths({'reconstruction_parent': root.parent})
    with tempfile.TemporaryDirectory(prefix='transfer-reconstruct-', dir=root.parent) as tmp:
        spec = dict(c, output=str(Path(tmp)/'package'))
        reconstructed = prepare(spec, m['config_provenance'])
        if read(Path(spec['output'])/'PREPARED.json') != m or reconstructed['manifest_sha256'] != c['manifest_sha256']:
            raise ValueError('transfer differs from independent reconstruction')
    return dict(status='replica_transfer_package_audited_not_executed', dataset=m['dataset'],
        policy_id=m['source_policy_id'], policy_role=m['policy_role'], recordings=m['recordings'],
        manifest_sha256=c['manifest_sha256'], runtime_identity=m['dual_runtime_identity'],
        numerical_admission=False, npu_executed=False, provenance=provenance)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    p.add_argument('--action', choices=['prepare', 'audit'], default='prepare')
    a = p.parse_args()
    c, provenance = load_config(a.config, a.local)
    print(json.dumps((prepare if a.action == 'prepare' else audit)(c, provenance), indent=2))
