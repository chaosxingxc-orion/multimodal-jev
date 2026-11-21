"""Macro192 connection and service binding, retaining the existing dual owner.

Only request transport capacity changes. Internal model batch/token limits,
runtime source validation, policy activation and numerical profile are reused.
No launcher or NPU capacity admission is supplied by this component.
"""
import copy
import math

from multimodal_jev.context_logits_wire import validate_logits_record
from multimodal_jev.span_rollout_rpc import send, receive
from tools.reports.context_dual_runtime_candidate import verify_runtime
from tools.reports.context_replica_bulk_rpc import BulkContextEngineClient, MAX_ROWS


HELLO = 'replica-macro-runtime-hello-v1'


def shutdown_timeout(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('finite positive shutdown deadline required')
    return value


def configure_shutdown(owner_class, seconds):
    timeout = shutdown_timeout(seconds)

    def factory(root):
        class ConfiguredOwner(owner_class(root)):
            engine_shutdown_timeout_seconds = timeout
        return ConfiguredOwner
    return factory


def hello(runtime_identity, bundle):
    return dict(protocol=HELLO, max_request_queries=MAX_ROWS,
        runtime_identity=runtime_identity, policy_id=bundle['policy_id'],
        contract=bundle['contract'])


class BoundReplicaClient(BulkContextEngineClient):
    def __init__(self, *args, runtime_identity, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            if receive(self.socket) != hello(runtime_identity, self.bundle):
                raise ValueError('macro192 service runtime, capacity or policy differs')
            self.runtime_identity = copy.deepcopy(runtime_identity)
        except BaseException:
            self.phase = 'failed'; self.socket.close(); raise

    def score_many(self, rows, *, audio_token_id):
        try:
            records = super().score_many(rows, audio_token_id=audio_token_id)
            for record in records:
                validate_logits_record(record)
            return records
        except BaseException:
            self.phase = 'failed'; raise


def service_namespace(config):
    from multimodal_jev.cached_context_preparation import processor_cache_identity
    from tools.reports.serve_context_dual_candidate import service_namespace as original
    from tools.reports.context_replica_batched_service import serve_connection
    identity = verify_runtime(config)
    if processor_cache_identity(config['policy']['snapshot']) != identity['processor_sha256']:
        raise ValueError('service startup processor source differs')
    if config['policy']['max_batch_queries'] not in (32, 48):
        raise ValueError('explicit service batch32/48 required')
    namespace = original()
    scope = config['policy'].get('process_device_scope')
    module = config['policy'].get('device_scope_module')
    if scope is not None:
        from tools.reports.context_dual_rollout_candidate import owner_class
        if module is None:
            namespace['owner_class'] = lambda root: owner_class(root, shared_scope=True)
        else:
            namespace['owner_class'] = lambda root: owner_class(root, scope_module=module)
    elif module is not None:
        raise ValueError('device scope module requires an explicit process device scope')
    actor_order = config.get('lora_arithmetic')
    if actor_order is not None and (actor_order != 'actor_order_bf16_qv_v1' or 'attention_execution' not in config):
        raise ValueError('explicit actor BF16 Q/V arithmetic requires bounded prefill')
    if 'attention_execution' in config:
        if config['attention_execution'] != dict(kind='bounded_prefill_v1', max_requests=8,
                                                acoustic_batch_chunks=32):
            raise ValueError('explicit bounded attention8/acoustic32 execution required')
        from tools.reports.context_bounded_rollout_candidate import owner_class as bounded_owner
        namespace['owner_class'] = lambda root: bounded_owner(root,
            shared_scope=scope is not None and module is None, scope_module=module, actor_order=actor_order is not None)
    if 'engine_shutdown_timeout_seconds' in config:
        namespace['owner_class'] = configure_shutdown(
            namespace['owner_class'], config['engine_shutdown_timeout_seconds'])
    def bound(connection, engine, exchange):
        send(connection, hello(identity, engine.bundle))
        return serve_connection(connection, engine, exchange)
    namespace['serve_connection'] = bound
    return namespace


def main():
    import argparse
    import json
    import time
    from pathlib import Path
    from multimodal_jev.baseline import atomic_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(exist_ok=False)
    try:
        service_namespace(config)['run'](config, args.output)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', dict(exit_code=1, error=repr(exc),
                                                    finished_unix=time.time()))
        raise
    atomic_json(args.output/'terminal.json', dict(exit_code=0, finished_unix=time.time()))


if __name__ == '__main__':
    main()
