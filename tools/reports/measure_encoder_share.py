"""Split serving wall time into audio-encoder vs text-backbone cost.

Dev-state ``v0.0.1`` diagnostic. The engine keys its multimodal encoder cache by
``multi_modal_uuids``. Passing a fresh namespace forces the audio encoder to run
again; reusing the same namespace makes the second call a cache hit, so the
difference between the two is exactly the audio-encoder share. Every pass must
return bit-identical records, so the comparison cannot hide a semantic change.

No optimizer, no training admission, no policy change.
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
from context_serving_config_owner_candidate import owner_class, PROFILE

CONFIG_FIELDS = ('max_batch_queries', 'max_batch_tokens', 'gpu_memory_utilization',
                 'kv_cache_memory_bytes', 'max_model_len')


def resolve_policy(base, arm):
    policy = dict(base)
    policy.update({k: arm[k] for k in CONFIG_FIELDS})
    policy['physical_devices'] = [arm['physical_device']]
    policy['candidate_runtime'] = PROFILE
    policy.pop('output_protocol', None)
    return policy


@contextmanager
def audio_namespace(owner, tag):
    """One stable multimodal cache namespace for the whole pass."""
    engine = owner.engine

    class NamespacedAudio:
        def encode(self, prompts, **kwargs):
            tagged = []
            for prompt in prompts:
                uuids = prompt['multi_modal_uuids']
                if set(uuids) != {'audio'} or len(uuids['audio']) != 1:
                    raise ValueError('one explicit audio cache identity required')
                tagged.append(dict(prompt, multi_modal_uuids={
                    'audio': [tag + ':' + uuids['audio'][0]]}))
            return engine.encode(tagged, **kwargs)

    owner.engine = NamespacedAudio()
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
    rows = torch.load(config['inputs'], map_location='cpu', weights_only=True)
    n = config['probe_rows']
    if not isinstance(rows, list) or len(rows) != n or n < 2:
        raise ValueError('declared probe record count required')
    for start in range(0, n, 128):
        validate_rows(rows[start:start + 128])
    policy = resolve_policy(config['base_policy'], config['arms'][arm])
    owner = None
    try:
        owner = owner_class(config['source_root'])(policy)
        policy_id = owner.bundle['policy_id']
        commit_json(output / 'startup.json', dict(runtime=owner.runtime))

        def one_pass(tag):
            records = []
            torch.npu.synchronize()
            started = time.perf_counter()
            with audio_namespace(owner, tag):
                for start in range(0, n, 128):
                    values, _ = score_rows(owner, rows[start:start + 128],
                        policy_id=policy_id, audio_token_id=config['audio_token_id'])
                    if len(values) != min(128, n - start):
                        raise ValueError('incomplete model batch')
                    records.extend(values)
            torch.npu.synchronize()
            return records, time.perf_counter() - started

        # Pass 0 warms kernels. Pass 1 and 3 encode audio; pass 2 reuses the
        # namespace from pass 1, so its audio encoder work is a cache hit.
        plan = [('warmup', 'share-a'), ('fresh', 'share-b'), ('cached', 'share-b'),
                ('fresh', 'share-c')]
        passes = []
        for kind, tag in plan:
            records, elapsed = one_pass(tag)
            digest = record_digest(records)
            if passes and digest != passes[0]['records_sha256']:
                raise ValueError('record identity changed across encoder-cache states')
            value = dict(kind=kind, namespace=tag, seconds=elapsed, records_sha256=digest)
            commit_json(output / f'pass-{len(passes)}.json', value)
            passes.append(value)
        fresh = [p['seconds'] for p in passes if p['kind'] == 'fresh']
        cached = [p['seconds'] for p in passes if p['kind'] == 'cached'][0]
        fresh_mean = sum(fresh) / len(fresh)
        result = dict(arm=arm, rows=n, passes=passes, fresh_mean_seconds=fresh_mean,
            cached_seconds=cached, audio_encoder_seconds=fresh_mean - cached,
            audio_encoder_share=(fresh_mean - cached) / fresh_mean,
            records_sha256=passes[0]['records_sha256'],
            training_dispatch_allowed=False,
            scope='audio-encoder share of serving wall time; startup excluded; no training')
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