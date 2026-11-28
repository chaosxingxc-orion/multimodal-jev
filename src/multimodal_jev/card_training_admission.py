"""Bind formal graph execution and variable epoch endpoints to audited evidence."""
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash


def admit_graph(config, pin):
    receipt_path = Path(pin['path'])
    if file_hash(receipt_path) != pin['sha256']:
        raise ValueError('finite graph receipt changed')
    receipt = json.loads(receipt_path.read_bytes())
    path = receipt_path.with_name('config.json')
    if file_hash(path) != receipt['config_sha256']:
        raise ValueError('finite graph configuration changed')
    finite = json.loads(path.read_bytes())
    if (receipt['engineering_prefix'] is not True or finite['engineering_prefix'] is not True
            or finite['training_dispatch_allowed'] is not False or finite['parity_passes'] < 1
            or receipt['updates'] != finite['stop_update'] or finite['start_update'] != config['start_update']
            or receipt['training_graph'] != config['training_graph']
            or finite.get('speech_partners') != config.get('speech_partners')
            or any(config.get(k) != finite.get(k) for k in ('training_graph', 'physical_devices', 'source_policy_id',
                   'instruction_objective', 'token_buckets', 'training_acoustic_engine', 'packed_acoustics',
                   'class_logit_ema', 'instance_negative_manifest', 'instance_epoch', 'schedule_sha256', 'stage1_layout', 'parity_probability_atol'))):
        raise ValueError('formal graph differs from finite admission')


def completed_update(train, proof):
    if 'token_buckets' not in train:
        return train['stop_update']
    ends = proof.get('epoch_end_updates')
    if (not isinstance(ends, list) or not ends or len(ends) != (1 if 'instance_epoch' in train else train['batch_config']['epochs'])
            or any(type(x) is not int or x < 1 for x in ends)
            or sorted(set(ends)) != ends or ends[-1] != proof['updates']):
        raise ValueError('missing or inconsistent completed bucket epochs')
    return ends[-1]
