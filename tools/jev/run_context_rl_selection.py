#!/usr/bin/env python3
"""Evaluate one artifact-bound formal RL point using existing audio runtimes.

This entry runs inside an isolated evaluation container. It does not allocate
devices, admit training, select checkpoints or rewrite inference policy IDs.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys

if __package__:
    from .context_rl_evaluation_plan import build, sha
else:
    from context_rl_evaluation_plan import build, sha


def prepare(*, template, step, **plan_arguments):
    """Rebuild the complete plan from collected sources, never trust a saved plan."""
    plan = build(**plan_arguments)
    if type(step) is not int or step not in [p['step'] for p in plan['checkpoints']]:
        raise ValueError('predeclared evaluation point required')
    point = next(p for p in plan['checkpoints'] if p['step'] == step)
    source = Path(point['collection']) / point['policy_relative_path']
    manifest = json.loads((source / 'MANIFEST.json').read_text())
    experiment = plan_arguments['experiment']
    if (template['release_sha256'] != experiment['release_sha256']
            or template['base_id'] != manifest['contract']['base_id']
            or template.get('diagnostic_update') is not False
            or template.get('assembled_execution') is not None
            or template['context_runtime_profile'] != manifest['contract']['runtime_profile']):
        raise ValueError('evaluation template changes fixed population or policy execution')
    diagnostic_keys = ('bind_exclusive_event_contract', 'diagnostic_packed_alignment',
                       'diagnostic_backbone_alignment', 'capture_hidden', 'parity_microbatch',
                       'reference_from_current_inputs', 'isolate_head_batch', 'batch_runtime')
    if any(template.get(k) not in (None, False) for k in diagnostic_keys):
        raise ValueError('diagnostic execution overrides forbidden in formal evaluation')
    if (len(template['audio']) != 256
            or len({r['id'] for r in template['audio']}) != 256):
        raise ValueError('complete distinct learning parity panel required')
    c = copy.deepcopy(template)
    # Drop all former source pins. Policy assets are bound below; other runtime
    # pins remain fixed. A shared parent path is never implicitly remapped.
    previous = c.get('source_policy')
    if previous:
        c['pins'] = {p: h for p, h in c['pins'].items()
                     if not Path(p).is_relative_to(Path(previous))}
    c.update(source_policy=str(source.resolve()), source_policy_id=point['policy_id'],
             source_contract=manifest['contract'], checkpoint_step=step)
    c['pins'][str((source / 'MANIFEST.json').resolve())] = sha(source / 'MANIFEST.json')
    for asset, digest in manifest['files'].items():
        c['pins'][str((source / asset).resolve())] = digest
    # Retain the original collection proofs/configs/commits as runtime inputs.
    # A prepared point is not permission to swap the source after preparation.
    roots = [plan_arguments['stage1_root'], plan_arguments['rl_root']]
    binding_files = {}
    for root in roots:
        path = Path(root) / 'config.json'
        binding_files[str(path.resolve())] = sha(path)
    commit = Path(point['collection']) / f"run/checkpoint-{point['source_step']:04d}-committed.json"
    binding_files[str(commit.resolve())] = point['commit_sha256']
    c['rl_evaluation'] = dict(run_id=plan['run_id'], algorithm=plan['algorithm'],
        step=step, source_stage=point['source_stage'], source_step=point['source_step'],
        initial_policy_id=plan['initial_policy_id'], source_policy_id=point['policy_id'],
        binding_files=binding_files, selection_recordings=9226, threshold=.5,
        status='prepared_only_no_runtime_admission')
    return c


def validate_parity(config, joint, hf, vl):
    if joint['contract'] != config['source_contract']:
        raise ValueError('evaluation execution contract changed')
    if not len(hf) == len(vl) == len(config['audio']) == 256:
        raise ValueError('complete parity population required')
    for x, y, asset in zip(hf, vl, config['audio'], strict=True):
        if (x['id'] != y['id'] or x['id'] != asset['id']
                or x['inputs'] != y['request_identity']['inputs']
                or x['positions'][0] != y['request_identity']['positions']
                or y['request_identity']['policy_id'] != joint['policy_id']):
            raise ValueError('parity policy/input binding differs')
        if (len(x['prob']) != 60 or x['prob'] != y['prob']
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                       for v in x['prob'])):
            raise ValueError('finite exact 60-probability parity required')


def authenticate_export(config, output):
    """Read actual tensors; matching step numbers/metadata cannot replace this."""
    import torch
    from multimodal_jev.context_policy_artifacts import read_execution_bundle, verify_native_adapter
    joint = json.loads((output / 'joint.json').read_text())
    source = read_execution_bundle(config['source_policy'],
        expected_policy_id=config['source_policy_id'], expected_contract=config['source_contract'],
        expected_base_id=config['base_id'])
    exported = read_execution_bundle(output / 'policy', expected_policy_id=joint['policy_id'],
        expected_contract=config['source_contract'], expected_base_id=config['base_id'])
    if (source['tensors'].keys() != exported['tensors'].keys()
            or any(not torch.equal(v, exported['tensors'][k]) for k, v in source['tensors'].items())):
        raise ValueError('evaluation differs from collected training weights')
    for name, digest in json.loads((Path(config['source_policy']) / 'MANIFEST.json').read_text())['files'].items():
        if name.startswith('processor/') and sha(output / 'policy' / name) != digest:
            raise ValueError('evaluation processor differs from training')
    verify_native_adapter(output / 'adapter', exported)
    return joint


def execute(config_path, output):
    from multimodal_jev.local_checkpoint import commit_json
    from multimodal_jev.local_runtime_storage import require_training_storage
    require_training_storage(output)
    c = json.loads(Path(config_path).read_text())
    b = c['rl_evaluation']
    if (b['source_policy_id'] != c['source_policy_id'] or b['step'] != c['checkpoint_step']
            or b['algorithm'] not in ('grpo_v1', 'rloo_v1')
            or b['selection_recordings'] != 9226 or b['threshold'] != .5
            or c['diagnostic_update'] is not False or c.get('assembled_execution') is not None):
        raise ValueError('prepared RL point identity changed')
    for name, digest in {**c['pins'], **b['binding_files']}.items():
        if sha(name) != digest:
            raise ValueError('prepared evaluation dependency changed: ' + name)
    output = Path(output)
    output.mkdir(exist_ok=False)
    run = output / 'run'
    code = 1
    try:
        engine = Path(__file__).with_name('probe_context_engine_audio.py')
        evaluator = Path(__file__).with_name('evaluate_context_selection_vllm.py')
        for phase in ('hf', 'vllm'):
            with (output / f'{phase}.log').open('w') as log:
                subprocess.run([sys.executable, str(engine), '--config', str(config_path),
                    '--output', str(run), '--phase', phase], stdout=log, stderr=subprocess.STDOUT, check=True)
        joint = authenticate_export(c, run)
        validate_parity(c, joint, json.loads((run / 'hf.json').read_text()),
                        json.loads((run / 'vllm.json').read_text())['records'])
        selection = dict(c, snapshot=str(run / 'policy'), policy_id=joint['policy_id'],
                         contract=joint['contract'], adapter=str(run / 'adapter'))
        selection_path = output / 'selection-config.json'
        commit_json(selection_path, selection)
        with (output / 'selection.log').open('w') as log:
            subprocess.run([sys.executable, str(evaluator), '--config', str(selection_path),
                '--output', str(output / 'selection')], stdout=log, stderr=subprocess.STDOUT, check=True)
        commit_json(output / 'point-binding.json', dict(b, inference_policy_id=joint['policy_id'],
            config_sha256=sha(config_path), selection_config_sha256=sha(selection_path),
            source_tensors_exact=True, parity_queries=256,
            status='runtime_finished_independent_collection_required'))
        code = 0
    finally:
        commit_json(output / 'terminal.json', dict(exit_code=code))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare', action='store_true', help='config is a preparation specification; no NPU work')
    args = p.parse_args()
    if args.prepare:
        spec = json.loads(args.config.read_text())
        parameters = dict(spec)
        for name in ('template', 'experiment', 'stage1_proof', 'rl_proof'):
            parameters[name] = json.loads(Path(spec[name]).read_text())
        config = prepare(**parameters)
        for name in ('template', 'experiment', 'stage1_proof', 'rl_proof'):
            config['rl_evaluation']['binding_files'][str(Path(spec[name]).resolve())] = sha(spec[name])
        with args.output.open('x') as stream:
            json.dump(config, stream, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        execute(args.config.resolve(), args.output.resolve())
