"""Read an externally accepted, complete sampling population without targets.

The coordinator pins the independent acceptance file. This revalidates its
artifact bindings and raw records, not live Docker state or source provenance.
"""
import json
from pathlib import Path

from .baseline import file_hash
from .omni_sampling_acceptance import audit_records, read_json
from .sfc_reward_batch import digest


def load_accepted_records(root, acceptance, acceptance_hash, rows, checks, tokenizer):
    root, acceptance = Path(root), Path(acceptance)
    if file_hash(acceptance) != acceptance_hash:
        raise ValueError('registered independent sampling acceptance required')
    accepted = read_json(acceptance)
    launch = read_json(root/'launch.json')
    plan = read_json(root/'preparation.json')
    config = read_json(root/'config.json')
    run = root/'run'
    identity = read_json(run/'identity.json')
    if (accepted['container_id'] != launch['container_id']
            or accepted['image'] != plan['image']
            or accepted['source_commit'] != launch['source_commit']
            or launch['source_commit'] != plan['source_commit']
            or file_hash(root/'config.json') != plan['config_sha256']
            or identity['config'] != config or identity['inputs_sha256'] != digest(rows)
            or accepted['result_sha256'] != file_hash(run/'result.json')
            or accepted['records_sha256'] != file_hash(run/'records.jsonl')
            or accepted['weight_updates'] != 0 or accepted['responses'] != 156):
        raise ValueError('accepted sampling artifact binding changed')
    audited = audit_records(run, rows, checks, tokenizer,
        policy_sha256=identity['policy_sha256'],
        instruction_sha256=identity['instruction_sha256'],
        sampling=identity['effective_sampling'])
    if any(accepted.get(k) != v for k, v in audited.items()):
        raise ValueError('accepted raw sampling evidence changed')
    with (run/'records.jsonl').open() as handle:
        records = [json.loads(line) for line in handle]
    # Bind again after reading; never return a file changed during validation.
    if file_hash(run/'records.jsonl') != accepted['records_sha256']:
        raise ValueError('sampling records changed while reading')
    return records, identity
