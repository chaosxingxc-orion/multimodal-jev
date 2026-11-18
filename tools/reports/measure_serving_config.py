"""Time one serving-config arm over a fixed record set; independent per arm.

Dev-state ``v0.0.1`` timing harness. Each arm constructs the owner with one
declared service policy, runs one warmup plus two timed passes plus one reversed
pass over the exact same CPU rows, and requires every pass to return identical
records (order/capacity invariance). It writes a small digest-based report; the
full record tensors are not retained. No optimizer and no training admission.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.context_batched_service_candidate import score_rows
# This module and its owner are shipped beside the runtime sources (/build), so
# they are imported as top-level modules rather than under ``tools.reports``.
from context_serving_config_owner_candidate import owner_class, PROFILE

CONFIG_FIELDS = ('max_batch_queries', 'max_batch_tokens', 'gpu_memory_utilization',
                 'kv_cache_memory_bytes', 'max_model_len')


def resolve_policy(base, arm):
    if not set(CONFIG_FIELDS) <= set(arm) or set(arm) - set(CONFIG_FIELDS) != {'physical_device'}:
        raise ValueError('explicit complete serving fields and one physical device required')
    policy = dict(base)
    policy.update({k: arm[k] for k in CONFIG_FIELDS})
    policy['physical_devices'] = [arm['physical_device']]
    policy['candidate_runtime'] = PROFILE
    policy.pop('output_protocol', None)
    return policy


@contextmanager
def fresh_audio_pass(owner, index):
    """Unique MM cache namespace per pass; tensors and policy IDs stay intact."""
    engine = owner.engine

    class FreshAudioView:
        def encode(self, prompts, **kwargs):
            fresh = []
            for prompt in prompts:
                uuids = prompt['multi_modal_uuids']
                if set(uuids) != {'audio'} or len(uuids['audio']) != 1:
                    raise ValueError('one explicit audio cache identity required')
                fresh.append(dict(prompt, multi_modal_uuids={
                    'audio': [f'serving-config-pass-{index}:' + uuids['audio'][0]]}))
            return engine.encode(fresh, **kwargs)

    owner.engine = FreshAudioView()
    try:
        yield
    finally:
        owner.engine = engine


def record_digest(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def run(config, output, arm):
    import torch
    torch.set_num_threads(1)
    from multimodal_jev.local_runtime_storage import require_training_storage
    from tools.reports.context_bulk_rpc_candidate import validate_rows
    output = Path(output)
    require_training_storage(output)
    output.mkdir(exist_ok=False)
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('frozen dependency changed: ' + path)
    if file_hash(config['inputs']) != config['input_sha256']:
        raise ValueError('fixed record payload changed')
    rows = torch.load(config['inputs'], map_location='cpu', weights_only=True)
    n = config['probe_rows']
    if not isinstance(rows, list) or len(rows) != n or n < 2:
        raise ValueError('declared complete probe record count required')
    # The wire contract bounds one message at 128 rows; validate in those chunks.
    for start in range(0, n, 128):
        validate_rows(rows[start:start + 128])
    policy = resolve_policy(config['base_policy'], config['arms'][arm])
    owner = None
    try:
        owner = owner_class(config['source_root'])(policy)
        policy_id = owner.bundle['policy_id']
        commit_json(output / 'startup.json', dict(runtime=owner.runtime,
            candidate=getattr(owner, 'segmented_receipt', None)))

        def timed_pass(order, index):
            records, call_sizes = [], []
            torch.npu.synchronize()
            started = time.perf_counter()
            with fresh_audio_pass(owner, index):
                for start in range(0, n, 128):
                    indices = order[start:start + 128]
                    values, plan = score_rows(owner, [rows[i] for i in indices],
                        policy_id=policy_id, audio_token_id=config['audio_token_id'])
                    if len(values) != len(indices):
                        raise ValueError('incomplete model batch')
                    records.extend(values)
                    call_sizes.append(dict(batch_queries=plan['batch_queries'],
                        batch_tokens=plan['batch_tokens']))
            torch.npu.synchronize()
            return records, time.perf_counter() - started, call_sizes

        passes = []
        for index in range(4):
            order = list(range(n))
            if index == 3:
                order.reverse()
            records, elapsed, call_sizes = timed_pass(order, index)
            if not 0 < elapsed < float('inf') or owner.bundle['policy_id'] != policy_id:
                raise ValueError('finite elapsed time and unchanged policy required')
            canonical = [None] * n
            for slot, record in zip(order, records, strict=True):
                canonical[slot] = record
            digest = record_digest(canonical)
            if index and digest != passes[0]['records_sha256']:
                raise ValueError('repeat or reversed-order invariance failed')
            value = dict(kind='warmup' if index == 0 else 'reversed' if index == 3 else 'timed',
                order='reversed' if index == 3 else 'forward', seconds=elapsed,
                records_sha256=digest, calls=len(call_sizes), plan=call_sizes)
            commit_json(output / f'pass-{index}.json', value)
            passes.append(value)
        timed = [p['seconds'] for p in passes[1:3]]
        mean = sum(timed) / 2
        result = dict(arm=arm, policy_id=policy_id, input_sha256=config['input_sha256'],
            rows=n, executed_policy=policy, passes=passes, mean_seconds=mean,
            rows_per_second=n / mean, records_sha256=passes[0]['records_sha256'],
            training_dispatch_allowed=False,
            scope='serving-stage wall time over fixed records; engine startup excluded; no training')
        commit_json(output / 'measurement.json', result)
        owner.close(); owner = None
        commit_json(output / 'terminal.json', dict(exit_code=0))
        return result
    except BaseException as exc:
        commit_json(output / 'terminal.json', dict(exit_code=1, error=repr(exc)))
        raise
    finally:
        if owner is not None:
            owner.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--arm', required=True)
    a = p.parse_args()
    print(json.dumps(run(json.loads(Path(a.config).read_text()), a.output, a.arm), indent=2))
