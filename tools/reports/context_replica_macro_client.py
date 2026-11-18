"""Connect explicitly configured, already started services.

Candidate for matched topology comparison; no deployment or device admission.

The launcher remains responsible for image/device/source admission. Connection
hello binds the declared runtime and policy; it is not remote attestation.
"""
import copy
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

from multimodal_jev.context_replica_macro_rollout import MacroRolloutPool, write
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_replica_runtime import BoundReplicaClient


def connect_pool(config, *, bundle):
    required = {'schema', 'attempt_id', 'journal_root', 'runtime_identity',
                'replicas', 'timeout_seconds', 'rollout_queries', 'max_queries'}
    if set(config) != required or config['schema'] != 'jev-replica-macro-client-v1':
        raise ValueError('explicit macro client configuration required')
    attempt = config['attempt_id']; replicas = config['replicas']
    timeout = config['timeout_seconds']
    if (not isinstance(attempt, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', attempt)
            or not isinstance(replicas, list) or not 1 <= len(replicas) <= 8
            or any(not isinstance(r, dict) or set(r) != {'socket', 'exchange'} for r in replicas)
            or type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0
            or type(config['max_queries']) is not int or not 1 <= config['max_queries'] <= 4096
            or type(config['rollout_queries']) is not int
            or not 1 <= config['rollout_queries'] <= config['max_queries']):
        raise ValueError('bounded batch, timeout, attempt and one to eight replicas required')
    for r in replicas:
        if any(not isinstance(v, str) or not v or '${' in v for v in r.values()):
            raise ValueError('resolved socket and exchange paths required')
    if (len({str(Path(r['socket']).resolve()) for r in replicas}) != len(replicas)
            or len({str(Path(r['exchange']).resolve()) for r in replicas}) != len(replicas)):
        raise ValueError('independent replica sockets and exchanges required')
    root = Path(config['journal_root']) / attempt
    check_paths({'macro_attempt': root})
    root.mkdir(parents=True, exist_ok=False)  # Never silently reuse a previous attempt.
    clients = []; errors = []
    try:
        write(root/'CONNECTING.json', dict(schema=config['schema'], attempt_id=attempt,
            runtime_identity=config['runtime_identity'], policy_id=bundle['policy_id']))
        with ThreadPoolExecutor(max_workers=len(replicas)) as executor:
            futures = [executor.submit(BoundReplicaClient, r['socket'], r['exchange'],
                bundle=copy.deepcopy(bundle), runtime_identity=config['runtime_identity'],
                timeout=timeout) for r in replicas]
            for future in futures:
                try: clients.append(future.result())
                except BaseException as exc: errors.append(exc)
        if errors: raise errors[0]
        pool = MacroRolloutPool(clients, root, runtime_identity=config['runtime_identity'],
            rollout_queries=config['rollout_queries'], max_queries=config['max_queries'])
        write(root/'CONNECTED.json', dict(policy_id=pool.bundle['policy_id'], replicas=len(replicas)))
        return pool
    except BaseException as exc:
        for client in clients:
            try: client.close()
            except Exception: pass
        write(root/'CONNECTION_FAILED.json', dict(error=repr(exc)))
        raise


def load_pool(template, local, *, bundle):
    config, provenance = load_config(template, local)
    return connect_pool(config, bundle=bundle), provenance
