"""Bind the card TPO runtime to the audited endpoint and its own frozen targets."""
import json
from pathlib import Path
import sys

from multimodal_jev.baseline import file_hash
from multimodal_jev.card_device_config import tpo_schedule
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.card_scoring_benchmark import pinned


def bind(c):
    proof_path = Path(c['endpoint'])
    proof = json.loads(proof_path.read_bytes())
    source = Path(c['source'])
    manifest = pinned(dict(path=source, sha256=proof['manifest_sha256']))
    training = pinned(dict(path=proof_path.with_name('training.json'), sha256=proof['config_sha256']))
    targets = json.loads(Path(c['targets']).read_bytes())
    if (proof['passed'] is not True or proof['container_exited'] is not True
            or not targets['source_policy'] == manifest['policy_id'] == proof['policy_id']
            or targets['source_policy_sha256'] != proof['manifest_sha256']
            or targets['card_input']['checkpoint_sha256'] != proof['checkpoint_sha256']
            or targets['reference_sha256'] != file_hash(c['scores'])
            or manifest['contract'] != training['source_contract']):
        raise ValueError('TPO requires the audited endpoint and its matching fixed reference')
    parent = pinned(c['runtime'])
    if 'training_layout' in c:
        layout = c['training_layout']
        schedule = tpo_schedule(layout, targets['schedule'])
        pool = layout['training_devices']
        parent.update(physical_devices=pool, global_batch=schedule['global_batch'], microbatch=schedule['microbatch'])
        parent['batch_config'].update(training_devices=len(pool), microbatch=schedule['microbatch'],
                                      per_dp_replica_batch=layout['per_dp_replica_batch'])
    parent.update(model=training['model'], base_id=training['base_id'],
                  source_contract=manifest['contract'], source_policy_id=manifest['policy_id'],
                  stage_lora_transition=training['stage_lora_transition'],
                  release_sha256=targets['release_sha256'],
                  training_dataset=dict(manifest_sha256=targets['view_sha256']),
                  reference_dataset=dict(manifest_sha256=targets['reference_sha256']),
                  card_caches=dict(parent['card_caches'], real=training['encoded_audio_cache']))
    if 'probability_max_abs' in c:
        bound = c['probability_max_abs']
        if not 0 < bound <= parent['behavior_bounds']['probability_max_abs']:
            raise ValueError('release may only tighten the probability admission bound')
        parent['behavior_bounds']['probability_max_abs'] = bound
    if 'recovery' in c:
        parent['recovery'] = c['recovery']
    commit_json(Path(c['output']), parent)


if __name__ == '__main__':
    bind(load_config(*sys.argv[1:])[0])
