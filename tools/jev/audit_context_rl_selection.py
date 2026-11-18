#!/usr/bin/env python3
"""Independently audit a collected RL selection point; no remote dispatch.

The collection root contains config.json and evaluation/ from the point runner.
A transport receipt binds bytes and an actual terminal container inspection.
Formal training collections are revalidated independently of runtime claims.
"""
import argparse
import json
from pathlib import Path

if __package__:
    from .context_rl_evaluation_plan import build, bound, sha
    from .run_context_rl_selection import authenticate_export, validate_parity
else:
    from context_rl_evaluation_plan import build, bound, sha
    from run_context_rl_selection import authenticate_export, validate_parity


def validate_population(rows, task, records, receipts, policy_id, *, expected_count=9226):
    from multimodal_jev.context_candidate_task import compile_candidates
    from multimodal_jev.slurp_training import content_hash
    if (len(rows) != expected_count or len({r['id'] for r in rows}) != expected_count
            or len(records) != expected_count or len(receipts) != expected_count):
        raise ValueError('complete distinct selection population required')
    candidates = [dict(id=x['id'], definition=x['description']) for x in task['schema']['candidates']]
    if len(candidates) != 60:
        raise ValueError('complete 60-candidate ontology required')
    for asset, pred, receipt in zip(rows, records, receipts, strict=True):
        request = compile_candidates(asset['audio'], candidates, task['thinking']['requirements'],
                                     semantics='exclusive_event')
        if (asset['id'] != pred['id'] or asset['id'] != receipt['id']
                or asset['audio_sha256'] != receipt['audio_sha256']
                or receipt['request_sha256'] != content_hash(request.model_input())
                or receipt['engine_request_id'] != pred['request_id']
                or pred['policy_id'] != policy_id
                or pred['request_identity'] != dict(policy_id=policy_id, inputs=receipt['inputs'],
                                                     positions=receipt['positions'])):
            raise ValueError('selection audio/request/policy provenance differs')


def audit(*, collection, transport, expected_config_sha256, step, **plan_arguments):
    root = Path(collection)
    state = transport['state']
    if (transport.get('status') != 'transport_complete_independent_audit_required'
            or not transport['container_id'] or state['Status'] != 'exited'
            or state['ExitCode'] != 0 or state['OOMKilled']):
        raise ValueError('successful actual terminal collection required')
    # All evaluated weights, adapter, records and receipts must be authenticated,
    # including assets accessed indirectly by the tensor loader.
    for name in transport['files']:
        bound(root, transport, name)
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('collection symlink forbidden')
        if path.is_file() and str(path.relative_to(root)) not in transport['files']:
            raise ValueError('unbound collected artifact')
    def read(name):
        return json.loads(bound(root, transport, name).read_text())
    config_path = bound(root, transport, 'config.json')
    if sha(config_path) != expected_config_sha256:
        raise ValueError('launch config identity differs')
    c = read('config.json')
    plan = build(**plan_arguments)
    points = [p for p in plan['checkpoints'] if p['step'] == step]
    if type(step) is not int or len(points) != 1:
        raise ValueError('predeclared RL point required')
    point = points[0]
    binding = c['rl_evaluation']
    expected = dict(run_id=plan['run_id'], algorithm=plan['algorithm'], step=step,
        source_stage=point['source_stage'], source_step=point['source_step'],
        initial_policy_id=plan['initial_policy_id'], source_policy_id=point['policy_id'],
        selection_recordings=9226, threshold=.5)
    if (any(binding.get(k) != v for k, v in expected.items())
            or c['source_policy_id'] != point['policy_id'] or c['checkpoint_step'] != step
            or c['release_sha256'] != plan_arguments['experiment']['release_sha256']
            or c['diagnostic_update'] is not False or c.get('assembled_execution') is not None):
        raise ValueError('evaluation does not belong to the formal RL point')
    if read('evaluation/terminal.json')['exit_code'] != 0:
        raise ValueError('evaluation runner failed')
    # Translate only the source filesystem location for local tensor reading.
    # Source and inference IDs remain distinct and predictions are never relabeled.
    local = dict(c, source_policy=str(Path(point['collection']) / point['policy_relative_path']))
    joint = authenticate_export(local, root / 'evaluation/run')
    validate_parity(c, joint, read('evaluation/run/hf.json'), read('evaluation/run/vllm.json')['records'])
    point_receipt = read('evaluation/point-binding.json')
    if (any(point_receipt.get(k) != v for k, v in binding.items() if k != 'status')
            or point_receipt['inference_policy_id'] != joint['policy_id']
            or point_receipt['config_sha256'] != expected_config_sha256
            or point_receipt['selection_config_sha256'] != sha(bound(root, transport, 'evaluation/selection-config.json'))):
        raise ValueError('point execution receipt differs')
    selection = read('evaluation/selection-config.json')
    if (selection['policy_id'] != joint['policy_id'] or selection['contract'] != c['source_contract']
            or selection['rl_evaluation'] != binding or selection['release'] != c['release']
            or selection['release_sha256'] != c['release_sha256']):
        raise ValueError('selection config differs from authenticated point')
    runtime = read('evaluation/selection/runtime.json')
    if runtime['profile'] != c['source_contract']['runtime_profile']:
        raise ValueError('selection runtime differs from training contract')
    from multimodal_jev.slurp_official_training import load
    if __package__:
        from .evaluate_context_event_records import evaluate
    else:
        from evaluate_context_event_records import evaluate
    rows, task, _ = load(c['release'], c['release_sha256'], 'selection', purpose='evaluate')
    records = read('evaluation/selection/predictions.json')['records']
    receipts = read('evaluation/selection/input-receipts.json')
    validate_population(rows, task, records, receipts, joint['policy_id'])
    for i, (pred, receipt) in enumerate(zip(records, receipts, strict=True)):
        if read(f'evaluation/selection/record-{i:05d}.json') != dict(prediction=pred, input_receipt=receipt):
            raise ValueError('durable record differs from final predictions')
    predictions = bound(root, transport, 'evaluation/selection/predictions.json')
    result = evaluate(c['release'], c['release_sha256'], predictions, sha(predictions), joint['policy_id'], 'selection')
    reported = read('evaluation/selection/result.json')
    if (any(reported.get(k) != v for k, v in result.items())
            or reported['input_receipts_sha256'] != sha(bound(root, transport, 'evaluation/selection/input-receipts.json'))):
        raise ValueError('independently recomputed selection metrics differ')
    return dict(passed=True, **expected, container_id=transport['container_id'], state=state,
        inference_policy_id=joint['policy_id'], source_tensors_exact=True, parity_queries=256,
        result=result, files=transport['files'],
        scope='Collected weights/processor/adapter, parity, complete population and metrics audited. '
              'Saved input receipts checked; processor tensors not independently regenerated. '
              'Does not prove causal RL gain, complete curve or TEST performance.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    for name in ('transport', 'experiment', 'stage1_proof', 'rl_proof'):
        spec[name] = json.loads(Path(spec[name]).read_text())
    report = audit(**spec)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
