"""Bounded sequential Span updates with a full-version gate at every boundary.

This controls the tested single-rank tools, not a distributed verl trainer.
An exclusive new output root is required; recover using the last accepted
checkpoint in a new root, never by replaying an uncertain optimizer step.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

from .baseline import atomic_json, file_hash
from .policy_identity import checkpoint_identity
from .span_runtime_profile import verify_suite_admission


def read(path):
    return json.loads(Path(path).read_text())


def check_pins(pins):
    for path, digest in pins.items():
        if file_hash(path) != digest:
            raise ValueError('pinned chain input changed: ' + path)


def checkpoint_receipt(path, pins=None):
    path = Path(path)
    required = [path/'CHECKPOINT.json', path/'optimizer.pt',
                *sorted(p for p in (path/'adapter').rglob('*') if p.is_file())]
    if pins is not None:
        for p in required:
            if pins.get(str(p)) != file_hash(p):
                raise ValueError('chain checkpoint must be pinned: ' + str(p))
    receipt = read(path/'CHECKPOINT.json')
    adapter, files = checkpoint_identity(path/'adapter')
    if (receipt.get('schema') != 'jev-span-checkpoint-v1'
            or receipt['adapter_id'] != adapter or receipt['adapter_files'] != files
            or receipt['optimizer_sha256'] != file_hash(path/'optimizer.pt')
            or type(receipt['completed_updates']) is not int or receipt['completed_updates'] < 1):
        raise ValueError('chain checkpoint identity or update count mismatch')
    return receipt, {str(p): file_hash(p) for p in required}


def require_ready(output):
    if read(output/'terminal.json')['exit_code'] != 0 or read(output/'READY.json')['status'] != 'passed':
        raise ValueError('child did not publish a successful terminal and READY')


def run_chain(config, output, tool_directory):
    """Execute exactly num_updates, advancing only after full 48-input parity."""
    output, tool_directory = Path(output), Path(tool_directory)
    count = config['num_updates']
    if type(count) is not int or count < 1:
        raise ValueError('positive integer num_updates required')
    step = dict(config['step'])
    step['pins'] = dict(step['pins'])
    check_pins(step['pins'])
    identity = config['suite_identity']
    if step['pins'].get(identity) != file_hash(identity):
        raise ValueError('pinned suite input identity required')
    checkpoint = Path(step['initial_optimizer_checkpoint'])
    if (checkpoint/'adapter').resolve() != Path(step['initial_adapter']).resolve():
        raise ValueError('initial chain adapter and optimizer differ')
    previous, _ = checkpoint_receipt(checkpoint, step['pins'])
    cursor = previous.get('training_state')
    if not cursor or step.get('legacy_selection_evidence'):
        raise ValueError('chain requires a checkpoint with native training cursor')
    start = cursor['next_batch_index']
    if (type(start) is not int or start != cursor['batch_index'] + 1
            or start != step['batch_index'] or start + count > cursor['epoch_batches']):
        raise ValueError('chain must continue next pair within fixed balanced epoch')
    for key in ('seed', 'manifest_sha256', 'question_order'):
        if cursor[key] != step[key]:
            raise ValueError('chain learning contract changed')
    # Verify initial policy/48-input acceptance before launching any child.
    verify_suite_admission(step)
    if previous['base_id'] != checkpoint_identity(step['model_path'])[0]:
        raise ValueError('chain checkpoint base differs')
    initial_updates = previous['completed_updates']
    versions = []

    def progress(stage, index):
        atomic_json(output/'progress.json', dict(stage=stage, batch_index=index,
            accepted_updates=len(versions), requested_updates=count, updated_unix=time.time()))

    def child(script, child_config, destination):
        path = destination.with_suffix('.config.json')
        atomic_json(path, child_config)
        # Persistent per-child output survives interruption of the controller.
        with destination.with_suffix('.log').open('x') as log:
            subprocess.run([sys.executable, str(tool_directory/script), '--config', str(path),
                            '--output', str(destination)], stdout=log, stderr=subprocess.STDOUT, check=True)
        require_ready(destination)

    for offset in range(count):
        index = start + offset
        step['batch_index'] = index
        check_pins(step['pins'])
        progress('training', index)
        training = output/f'update-{initial_updates + offset + 1:04d}'
        child('train_span_rl_step.py', step, training)
        checkpoint = training/'policy-1'
        receipt, pins = checkpoint_receipt(checkpoint)
        selection = read(training/'selection.json')
        state = receipt.get('training_state', {})
        if (receipt['completed_updates'] != previous['completed_updates'] + 1
                or receipt['base_id'] != previous['base_id'] or state != selection
                or state.get('batch_index') != index or state.get('next_batch_index') != index + 1):
            raise ValueError('child checkpoint did not advance exactly one balanced update')
        for key in ('schema', 'seed', 'schedule_sha256', 'epoch_batches', 'manifest_sha256', 'question_order'):
            if state.get(key) != cursor.get(key):
                raise ValueError('child changed balanced learning contract')
        step['pins'].update(pins)
        progress('full_version_acceptance', index)
        suite = output/f'acceptance-{receipt["completed_updates"]:04d}'
        suite_config = dict(model=step['model_path'], adapter=str(checkpoint/'adapter'),
            identity=identity, runtime_profile=step['runtime_profile'], pins=dict(step['pins']))
        child('check_span_policy_version.py', suite_config, suite)
        acceptance = suite/'vllm/result.json'
        step.update(initial_adapter=str(checkpoint/'adapter'), initial_optimizer_checkpoint=str(checkpoint),
                    runtime_acceptance=str(acceptance))
        step['pins'][str(acceptance)] = file_hash(acceptance)
        # Independently recompute every gate and check the actual new identity.
        verify_suite_admission(step)
        ready = read(suite/'READY.json')
        if (ready['base_id'] != receipt['base_id'] or ready['adapter_id'] != receipt['adapter_id']
                or ready['recordings'] != 48 or ready['result_sha256'] != file_hash(acceptance)):
            raise ValueError('suite READY differs from current checkpoint or full results')
        version = dict(completed_updates=receipt['completed_updates'], checkpoint=str(checkpoint),
            adapter_id=receipt['adapter_id'], optimizer_sha256=receipt['optimizer_sha256'],
            checkpoint_receipt_sha256=file_hash(checkpoint/'CHECKPOINT.json'),
            runtime_acceptance=str(acceptance), acceptance_sha256=file_hash(acceptance),
            next_batch_index=state['next_batch_index'])
        versions.append(version)
        atomic_json(output/'accepted-versions.json', versions)
        atomic_json(output/'last-accepted.json', version)
        previous = receipt
        progress('version_accepted', index)
    atomic_json(output/'READY.json', dict(status='passed', optimizer_updates=count,
        initial_completed_updates=initial_updates, completed_updates=previous['completed_updates'],
        latest=versions[-1], scope='Sequential real verl/vLLM BF16 lifecycle; full 48-input gate per version. '
        'No distributed, uninterrupted-resume-equivalence or held-out performance claim.'))
