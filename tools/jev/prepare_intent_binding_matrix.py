#!/usr/bin/env python3
"""Prepare fixed step0/step4 candidate diagnostics without taking NPU resources."""
import argparse
import copy
import json
from pathlib import Path
import shutil
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.container_pins import validate_bind_pins
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.span_candidate_intervention import VARIANTS, from_input, intervene
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read, completed
from queue_intent_finite_npu import verify_final


def audit_inputs(root, image):
    root = Path(root); launch = read(root/'launch.json'); config = read(root/'config.json')
    if (launch['image'] != image or launch['physical_devices'] != []
            or not completed(launch['container_id'], image)):
        raise ValueError('actually completed CPU inputs required')
    check_pins({str(root/'config.json'): launch['config_sha256'],
                str(root/'source-manifest.json'): launch['source_manifest_sha256']})
    check_pins(config['pins'])
    if read(root/'run/terminal.json')['exit_code'] != 0: raise ValueError('CPU failed')
    artifact = read(root/'run/result.json'); source = read(config['inputs'])
    rows = {r['id']: r for r in artifact['records']}; expected = set()
    if not artifact['passed'] or artifact['requests'] != 712 or len(rows) != 712 or artifact['recordings'] != 89:
        raise ValueError('complete intervention population required')
    for old in source['records']:
        for variant in VARIANTS:
            uid = old['id']+'::'+variant; expected.add(uid); row = rows[uid]
            req, binding = intervene(from_input(old['request']), variant)
            if row['request'] != json.loads(json.dumps(req.model_input())) or row['binding'] != binding:
                raise ValueError('input-only transformation differs')
            for key in ('audio_sha256', 'audio_processing', 'input_features', 'input_features_mask'):
                if row[key] != old[key]: raise ValueError('audio evidence differs')
            if not 0 < row['input_tokens'] <= 4096: raise ValueError('context overflow')
            if variant == 'native':
                for key in ('input_ids', 'positions', 'candidate_mapping'):
                    if row[key] != old[key]: raise ValueError('native inputs differ')
    if set(rows) != expected: raise ValueError('incomplete source/view coverage')
    return artifact


def prepare(config, root):
    root = Path(root)
    if root.exists(): raise FileExistsError('immutable matrix exists')
    check_pins(config['pins'])
    image = config['image']; processor = Path(config['processor_root'])
    required = [processor/'config.json', processor/'launch.json', processor/'run/result.json']
    required += [Path(config['gate_template'])/n for n in ('config.json', 'prepared.json', 'source-manifest.json')]
    required += [Path(e['gate'])/n for e in config['sources'] for n in ('config.json', 'launch.json', 'run/READY.json')]
    if any(config['pins'].get(str(p)) != file_hash(p) for p in required):
        raise ValueError('fixed preparation sources required')
    audit_inputs(processor, image)
    entries = config['sources']
    if [e['key'] for e in entries] != ['step0', 'step4']: raise ValueError('fixed step0/step4 required')
    identities = []; base_id = None
    for entry in entries:
        gate = Path(entry['gate']); launch = read(gate/'launch.json'); cfg = read(gate/'config.json')
        if not completed(launch['container_id'], image) or launch['image'] != image:
            raise ValueError('accepted source gate required')
        check_pins({str(gate/'config.json'): launch['config_sha256'],
                    str(gate/'source-manifest.json'): launch['source_manifest_sha256']})
        check_pins(cfg['pins']); ready = read(gate/'run/READY.json')
        if (not ready['passed'] or ready['adapter_id'] != cfg['adapter_id'] or ready['requests'] != 178
                or ready['result_sha256'] != file_hash(gate/'run/vllm/result.json')
                or read(gate/'run/terminal.json')['exit_code'] != 0): raise ValueError('source readiness differs')
        aid, files = checkpoint_identity(cfg['adapter'])
        if aid != cfg['adapter_id']: raise ValueError('adapter changed')
        if entry['key'] == 'step4':
            verify_final(gate)
            verified = read(gate/'verified.json')
            if verified['updates'] != 4 or verified['arm'] != 'D': raise ValueError('accepted D step4 required')
        if base_id is not None and base_id != cfg['base_id']: raise ValueError('base differs')
        base_id = cfg['base_id']; identities.append((entry, cfg, files))
    template = Path(config['gate_template']); plan = read(template/'prepared.json')
    base = read(template/'config.json')
    if plan['image'] != image or plan['physical_devices'] != [4]: raise ValueError('fixed TP1 template required')
    root.mkdir(); protocol = dict(schema='intent-binding-diagnostic-v1', image=image,
        base_id=base_id, runtime_profile=base['runtime_profile'], inputs=str(processor/'run/result.json'),
        inputs_sha256=file_hash(processor/'run/result.json'), variants=list(VARIANTS),
        checkpoints=[dict(key=e['key'], root=str(root/('gate-'+e['key'])), adapter_id=c['adapter_id']) for e,c,_ in identities],
        scope='Label-free paired input diagnostics, all views reported; no target selection or efficacy claim.')
    atomic_json(root/'protocol.json', protocol)
    repo = Path(__file__).resolve().parents[2]
    for entry, cfg, adapter_files in identities:
        gate = root/('gate-'+entry['key']); gate.mkdir()
        for folder in ('src', 'tools'):
            shutil.copytree(repo/folder, gate/'source'/folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        manifest = {str(p.relative_to(gate/'source')): file_hash(p) for p in (gate/'source').rglob('*') if p.is_file()}
        atomic_json(gate/'source-manifest.json', manifest)
        target = {k: copy.deepcopy(v) for k,v in base.items() if k != 'pins'}
        target.update(adapter=cfg['adapter'], adapter_id=cfg['adapter_id'], inputs=protocol['inputs'],
                      population=dict(recordings=89, requests=712))
        bid, model_files = checkpoint_identity(target['model'])
        if bid != base_id or target['runtime_profile'] != cfg['runtime_profile']: raise ValueError('runtime differs')
        pins = {str(Path(target['model'])/p): h for p,h in model_files.items()}
        pins.update({str(Path(cfg['adapter'])/p): h for p,h in adapter_files.items()})
        pins.update({str(gate/'source'/p): h for p,h in manifest.items()})
        for p in (root/'protocol.json', processor/'run/result.json'): pins[str(p)] = file_hash(p)
        target['pins'] = pins; atomic_json(gate/'config.json', target)
        argv = [a.replace(str(template), str(gate)) for a in plan['create_argv']]
        argv[argv.index('--name')+1] = 'multimodal-jev-intent-binding-'+entry['key']+'-v1'
        atomic_json(gate/'bind-pins.json', validate_bind_pins(argv, pins))
        atomic_json(gate/'prepared.json', dict(root=str(gate), image=image, physical_devices=[4], create_argv=argv,
            config_sha256=file_hash(gate/'config.json'), source_manifest_sha256=file_hash(gate/'source-manifest.json'),
            status='prepared_not_dispatched', dispatch_after='Existing training and FSC matrix; resource recheck required'))
    receipt = dict(root=str(root), protocol_sha256=file_hash(root/'protocol.json'), policies=2,
                   requests_per_policy=712, status='prepared_not_dispatched')
    atomic_json(root/'prepared.json', receipt); return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); p.add_argument('--root', type=Path, required=True)
    a = p.parse_args(); print(json.dumps(prepare(read(a.config), a.root), indent=2))
