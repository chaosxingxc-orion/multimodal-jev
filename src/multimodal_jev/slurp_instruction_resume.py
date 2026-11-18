"""Instruction continuation metadata independent of the old 89-source RL path.

Host caller must observe exact completed containers. This module authenticates
immutable acceptance artifacts and the full-data cursor; no PID/marker shortcuts.
"""
import json
from pathlib import Path
from .baseline import file_hash

FIELDS = ('model_path','base_id','release','release_sha256','schedule_sha256',
          'global_batch','lora_rank','lora_alpha','lora_execution_dtype','lr','seed',
          'runtime_profile','context_limit')


def validate_lineage(previous, current, completed_updates, total_updates):
    if (type(completed_updates) is not int or type(total_updates) is not int
            or not 1 <= completed_updates < total_updates
            or current.get('start_update') != completed_updates
            or current.get('total_updates') != total_updates):
        raise ValueError('exact bounded full-schedule continuation required')
    for key in FIELDS:
        if key not in previous or previous[key] != current.get(key):
            raise ValueError('instruction continuation changed '+key)
    for key, default in (('task_mode', 'instruction_views'), ('loss_reduction', 'legacy_weight_mean'),
                         ('microbatch_per_rank', 1)):
        if previous.get(key, default) != current.get(key, default):
            raise ValueError('instruction continuation changed '+key)
    previous_start = previous.get('start_update', 0)
    if previous_start+previous['updates'] != completed_updates:
        raise ValueError('predecessor completed update count differs')
    if (type(current.get('updates')) is not int or current['updates'] < 1
            or completed_updates+current['updates'] > total_updates):
        raise ValueError('continuation exceeds frozen epoch budget')
    # Naming repairs live outside the immutable training root and must be bound
    # by the independently verified state, not silently treated as new weights.
    expected = Path(current.get('accepted_adapter', str(Path(current['previous_root'])/'run'/f'adapter-step{completed_updates:04d}')))
    if Path(current['initial_adapter']) != expected:
        raise ValueError('exact predecessor BF16 adapter required')


def authenticate(config):
    """Require independently verified state and separate HF/vLLM probability gate."""
    def pinned(path):
        path = Path(path)
        if config['pins'].get(str(path)) != file_hash(path):
            raise ValueError('unpinned continuation evidence: '+str(path))
        return json.loads(path.read_text())
    root = Path(config['previous_root']); gate = Path(config['previous_gate'])
    previous = pinned(root/'config.json')
    launch = pinned(root/'launch.json')
    observed = pinned(gate/'training-observed.json')
    if (observed['container_id'] != launch['container_id'] or observed['image'] != launch['image']
            or observed['state']['Status'] != 'exited' or observed['state']['ExitCode'] != 0):
        raise ValueError('successful exact predecessor observation required')
    verified = pinned(gate/'verified.json')
    state = pinned(gate/'state-verified.json')
    if (not verified['passed'] or not state['passed'] or verified['training_root'] != str(root)
            or verified['state_sha256'] != file_hash(gate/'state-verified.json')
            or verified['updates'] != config['start_update']
            or state['updates'] != config['start_update']
            or not verified['full_candidate_parity'] or not verified['exact_master_export']):
        raise ValueError('independent instruction state and full-candidate gate required')
    if (state['training_root'] != str(root) or state['adapter_id'] != verified['adapter_id']
            or state['adapter'] != config['initial_adapter']
            or config.get('accepted_adapter') != state['adapter'] or not state['exact_peft_reload']
            or not state['exact_master_export']):
        raise ValueError('exact independently loaded predecessor adapter required')
    validate_lineage(previous, config, verified['updates'], config['total_updates'])
    for path, digest in {**state['files'], **verified['files']}.items():
        if config['pins'].get(path) != digest or file_hash(path) != digest:
            raise ValueError('accepted predecessor artifact changed')
    for rank in range(4):
        path = str(root/'run'/f'rank-{rank}'/f'checkpoint-{config["start_update"]:04d}.pt')
        if path not in state['files']:
            raise ValueError('all rank optimizer/master shards required')
    for name in ('adapter_config.json','adapter_model.safetensors'):
        if str(Path(state['adapter'])/name) not in state['files']:
            raise ValueError('accepted adapter configuration and weights must be pinned')
    return dict(previous_root=str(root), previous_gate=str(gate),
                completed_updates=verified['updates'], adapter_id=verified['adapter_id'],
                verified_sha256=file_hash(gate/'verified.json'))


def schedule_cursor(completed_updates, epoch_updates):
    """Next batch; an epoch boundary maps to the next epoch, never repeats data."""
    if (type(completed_updates) is not int or completed_updates < 0 or not epoch_updates
            or any(type(n) is not int or n <= 0 for n in epoch_updates)
            or completed_updates > sum(epoch_updates)):
        raise ValueError('bounded integer schedule cursor required')
    remaining = completed_updates
    for epoch, count in enumerate(epoch_updates):
        if remaining < count:
            return epoch, remaining
        remaining -= count
    return len(epoch_updates), 0


def load_full_schedule(config):
    """Authenticate both full epochs and completed processor preflight."""
    from .slurp_training import content_hash
    root=Path(config['schedule'])
    if file_hash(root/'MANIFEST.json') != config['schedule_sha256']:
        raise ValueError('frozen full schedule required')
    manifest=json.loads((root/'MANIFEST.json').read_text())
    if (manifest['release'] != config['release'] or manifest['manifest_sha256'] != config['release_sha256']
            or manifest['global_batch'] != config['global_batch'] or manifest['world_size'] != 4):
        raise ValueError('schedule training factors differ')
    batches=[];epoch_updates=[]
    for index,epoch in enumerate(manifest['epochs']):
        path=root/f'epoch-{index}.json'
        if epoch['epoch']!=index or file_hash(path)!=epoch['file_sha256']:
            raise ValueError('ordered exact epoch files required')
        data=json.loads(path.read_text())
        if len(data['batches'])!=epoch['updates'] or content_hash(data['batches'])!=epoch['batches_sha256']:
            raise ValueError('complete epoch batches required')
        epoch_updates.append(epoch['updates'])
        batches.extend((index,i,slots) for i,slots in enumerate(data['batches']))
    if len(batches)!=config['total_updates']:
        raise ValueError('full stage budget differs')
    proof_path=Path(config['processor_acceptance'])
    if config['pins'].get(str(proof_path))!=file_hash(proof_path):
        raise ValueError('pinned full processor acceptance required')
    proof=json.loads(proof_path.read_text())
    processor_root=proof_path.parent.parent
    # New local-storage runs keep their original output layout. Explicit paths
    # avoid copying receipts into a fabricated legacy directory structure.
    evidence_paths = config.get('processor_evidence_paths')
    required_receipts = ('launch.json','observed-final.json','config.json','run/terminal.json')
    if evidence_paths is not None and set(evidence_paths) != set(required_receipts):
        raise ValueError('complete explicit processor receipt paths required')
    receipts={}
    for name in required_receipts:
        path=Path(evidence_paths[name]) if evidence_paths is not None else processor_root/name
        if config['pins'].get(str(path))!=file_hash(path):
            raise ValueError('pinned actual processor completion required')
        receipts[name]=json.loads(path.read_text())
    observed=receipts['observed-final.json'];launch=receipts['launch.json']
    if (observed['container_id']!=launch['container_id'] or observed['image']!=launch['image']
            or observed['state']['Status']!='exited' or observed['state']['ExitCode']!=0
            or receipts['run/terminal.json']['exit_code']!=0
            or launch['config_sha256']!=file_hash(
                Path(evidence_paths['config.json']) if evidence_paths is not None else processor_root/'config.json')):
        raise ValueError('successful exact full processor observation required')
    processor_config=receipts['config.json']
    if (processor_config['schedule_sha256']!=config['schedule_sha256']
            or processor_config['model']!=config['model_path']
            or processor_config['context_limit']!=config['context_limit']):
        raise ValueError('processor configuration differs from continued model/schedule')
    if (not proof['passed'] or proof['schedule_sha256']!=config['schedule_sha256']
            or proof['release_sha256']!=config['release_sha256']
            or proof['requests']!=manifest['total_nonpadding_recording_uses']
            or proof['context_limit']!=config['context_limit'] or proof['context_overflow']):
        raise ValueError('full input population not accepted')
    if config.get('task_mode') == 'direct_final':
        audit_path = Path(config['processor_ledger_acceptance'])
        if config['pins'].get(str(audit_path)) != file_hash(audit_path):
            raise ValueError('pinned independent full processor ledger audit required')
        audit = json.loads(audit_path.read_text())
        if (audit.get('passed') is not True
                or audit.get('independently_matched_every_schedule_coordinate') is not True
                or audit.get('result_sha256') != file_hash(proof_path)
                or audit.get('records_sha256') != proof['records_sha256']
                or audit.get('requests') != proof['requests']
                or audit.get('container_id') != observed['container_id']
                or audit.get('image') != observed['image']):
            raise ValueError('independent complete processor ledger differs')
    return batches,epoch_updates,manifest
