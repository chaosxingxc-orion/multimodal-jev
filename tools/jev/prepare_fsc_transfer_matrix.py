#!/usr/bin/env python3
"""Freeze the complete seven-policy FSC matrix after actual source acceptance.

Preparation only: no target-label reads, inference, Docker creation or dispatch.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.container_pins import validate_bind_pins
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.fsc_span_transfer import compile_request
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read, completed
from resolve_fsc_transfer_sources import resolve


def audit_inputs(root, task_path, population_path, image):
    root = Path(root); launch = read(root/'launch.json'); cfg = read(root/'config.json')
    if launch['image'] != image or launch['physical_devices'] != [] or not completed(launch['container_id'], image):
        raise ValueError('actual completed CPU processor required')
    check_pins({str(root/'config.json'): launch['config_sha256'], str(root/'source-manifest.json'): launch['source_manifest_sha256']})
    check_pins(cfg['pins'])
    if read(root/'run/terminal.json')['exit_code'] != 0: raise ValueError('CPU processing failed')
    artifact = read(root/'run/result.json'); task = read(task_path); population = read(population_path)
    if (artifact['passed'] is not True or artifact['role'] != 'frozen_transfer_evaluation_inputs'
            or artifact['task_sha256'] != file_hash(task_path) or artifact['roster_sha256'] != file_hash(population_path)
            or file_hash(cfg['task']) != file_hash(task_path) or cfg['roster'] != str(population_path)
            or artifact['recordings'] != population['recordings'] or artifact['requests'] != 6*population['recordings']):
        raise ValueError('complete frozen evaluation CPU inputs required')
    rows = {r['id']: r for r in artifact['records']}
    if len(rows) != len(artifact['records']) or len(rows) != artifact['requests']: raise ValueError('distinct full CPU population required')
    expected = set()
    for source in population['records']:
        if file_hash(source['audio']) != source['audio_sha256']: raise ValueError('evaluation audio changed')
        features = None
        for template in ('direct', 'structured'):
            for variant in ('native', 'permutation', 'rebind'):
                uid = source['id']+'::'+template+'::'+variant; expected.add(uid); row = rows[uid]
                request, binding = compile_request(source['audio'], task, template, variant=variant, seed=317)
                if (row['request'] != json.loads(json.dumps(request.model_input())) or row['binding'] != binding
                        or row['audio_sha256'] != source['audio_sha256'] or not 0 < row['input_tokens'] <= 4096):
                    raise ValueError('full configured CPU request differs')
                current = {k: row[k] for k in ('input_features', 'input_features_mask', 'audio_processing')}
                if features is not None and features != current: raise ValueError('template changes audio evidence')
                features = current
    if rows.keys() != expected: raise ValueError('CPU view population differs')
    return population


def protocol(root, selection, task, population):
    root = Path(root)
    keys = ['step0']+[f'{a}-step{s}' for a in ('D', 'Q', 'P') for s in (6, 12)]
    if ([r['key'] for r in selection['resolved_sources']] != keys or selection['target_feedback_used'] is not False
            or selection['checkpoints'] != [{k: r[k] for k in ('key', 'adapter_id')} for r in selection['resolved_sources']]):
        raise ValueError('complete fixed seven-policy source selection required')
    entries = [dict(key=r['key'], adapter_id=r['adapter_id'], root=str(root/('gate-'+r['key']))) for r in selection['resolved_sources']]
    return dict(schema='fsc-span-transfer-evaluation-v1',
        selection_rule='source_only_frozen_before_target_inference', target_feedback_used_for_selection=False,
        binding_seed=317, bootstrap_seed=941, bootstrap_replicates=2000,
        image=selection['image'], base_id=selection['base_id'], runtime_profile=selection['runtime_profile'],
        task=str(task), task_sha256=file_hash(task), population=str(population), population_sha256=file_hash(population),
        source_selection=str(root/'source-selection.json'), source_selection_sha256=file_hash(root/'source-selection.json'),
        checkpoints=entries, scope='Exploratory D/Q/P fixed midpoint/endpoint transfer; historical FSC exposure disclosed; no target training or selection.')


def prepare(config, root):
    root = Path(root)
    if root.exists(): raise FileExistsError('immutable matrix root already exists')
    check_pins(config['pins'])
    for key in ('resolution_config', 'task', 'population'):
        if config['pins'].get(config[key]) != file_hash(config[key]): raise ValueError('fixed matrix preparation inputs required')
    # Resolve all real source identities before any matrix artifact is created.
    selection = resolve(read(config['resolution_config']))
    population_path = Path(config['population']); task_path = Path(config['task'])
    population = audit_inputs(config['processor_root'], task_path, population_path, selection['image'])
    template = Path(config['gate_template']); base = read(template/'config.json'); plan = read(template/'prepared.json')
    if (plan['image'] != selection['image'] or plan['physical_devices'] != [4]
            or base['base_id'] != selection['base_id'] or base['runtime_profile'] != selection['runtime_profile']):
        raise ValueError('accepted TP1 runtime template required')
    for name in ('config.json', 'prepared.json', 'source-manifest.json'):
        if config['pins'].get(str(template/name)) != file_hash(template/name): raise ValueError('gate template must be pinned')
    base_id, model_files = checkpoint_identity(base['model'])
    if base_id != selection['base_id']: raise ValueError('base weights differ')
    root.mkdir(); shutil.copyfile(task_path, root/'task.json'); shutil.copyfile(population_path, root/'population.json')
    atomic_json(root/'source-selection.json', selection)
    locked = protocol(root, selection, root/'task.json', root/'population.json'); atomic_json(root/'protocol.json', locked)
    repo = Path(__file__).resolve().parents[2]; gates = []
    for entry, source in zip(locked['checkpoints'], selection['resolved_sources'], strict=True):
        gate = Path(entry['root']); gate.mkdir()
        for folder in ('src', 'tools'):
            shutil.copytree(repo/folder, gate/'source'/folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        manifest = {str(p.relative_to(gate/'source')): file_hash(p) for p in (gate/'source').rglob('*') if p.is_file()}
        atomic_json(gate/'source-manifest.json', manifest)
        cfg = {k: copy.deepcopy(v) for k, v in base.items() if k != 'pins'}
        cfg.update(adapter=source['adapter'], adapter_id=source['adapter_id'],
            inputs=str(Path(config['processor_root'])/'run/result.json'),
            population=dict(recordings=population['recordings'], requests=6*population['recordings']))
        adapter_id, adapter_files = checkpoint_identity(source['adapter'])
        if adapter_id != source['adapter_id']: raise ValueError('selected adapter changed during preparation')
        pins = {str(Path(base['model'])/p): h for p, h in model_files.items()}
        pins.update({str(Path(source['adapter'])/p): h for p, h in adapter_files.items()})
        pins.update({str(gate/'source'/p): h for p, h in manifest.items()})
        for path in (root/'protocol.json', root/'task.json', root/'population.json', root/'source-selection.json', Path(cfg['inputs'])):
            pins[str(path)] = file_hash(path)
        cfg['pins'] = pins; atomic_json(gate/'config.json', cfg)
        argv = [a.replace(str(template), str(gate)) for a in plan['create_argv']]
        argv[argv.index('--name')+1] = 'multimodal-jev-'+root.name+'-'+entry['key'].lower()
        # The tiny engineering template's 1h limit is insufficient for 22,758
        # full HF and vLLM requests. Fixed wall limit, not a restart allowance.
        if argv.count('3600s') != 1: raise ValueError('known gate timeout template required')
        argv[argv.index('3600s')] = '172800s'
        atomic_json(gate/'bind-pins.json', validate_bind_pins(argv, pins))
        prepared = dict(root=str(gate), image=selection['image'], physical_devices=[4], create_argv=argv,
            config_sha256=file_hash(gate/'config.json'), source_manifest_sha256=file_hash(gate/'source-manifest.json'),
            status='prepared_not_dispatched', scope='Full official FSC matrix member; no target-driven selection or partial scores.')
        atomic_json(gate/'prepared.json', prepared); gates.append(str(gate))
    receipt = dict(root=str(root), protocol_sha256=file_hash(root/'protocol.json'), gates=gates,
        recordings=population['recordings'], requests_per_policy=6*population['recordings'], policies=len(gates),
        status='prepared_not_dispatched')
    atomic_json(root/'prepared.json', receipt); return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(prepare(read(args.config), args.root), indent=2))
