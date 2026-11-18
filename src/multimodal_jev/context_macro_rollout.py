"""Whole-rollout ownership and complete macro-batch delivery over two clients.

Clients implement authenticated score_many/switch. This layer does not start
engines, sample actions, update an actor, retry ambiguous work, or prove NPU parity.
Journal directories are immutable attempts on validated local storage.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import re
import threading
import time

import torch

from .context_logits_wire import validate_logits_record
from .context_rollout_batch import bind_rollout
from .local_checkpoint import commit_file
from .local_runtime_storage import check_paths
from .span_request_identity import tensor_identity


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write(path, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    commit_file(path, lambda stream: stream.write(raw))


def plan_rollouts(costs, rollout_queries):
    """Longest-first chunks assigned as indivisible jobs to the lighter replica."""
    if (not costs or type(rollout_queries) is not int or rollout_queries < 1
            or any(type(x) is not int or x <= 0 for x in costs)):
        raise ValueError('positive integral costs and rollout size required')
    order = sorted(range(len(costs)), key=lambda i: (-costs[i], i))
    loads = [0, 0]; jobs = []
    for start in range(0, len(order), rollout_queries):
        indices = order[start:start + rollout_queries]
        replica = min(range(2), key=lambda i: (loads[i], i))
        cost = sum(costs[i] for i in indices)
        jobs.append(dict(rollout_id=f'rollout-{len(jobs):04d}', replica_id=replica,
                         indices=indices, estimated_cost=cost))
        loads[replica] += cost
    return jobs


class MacroRolloutPool:
    def __init__(self, clients, journal_root, *, runtime_identity,
                 rollout_queries=32, max_queries=128):
        if (len(clients) != 2 or clients[0] is clients[1]
                or type(max_queries) is not int or max_queries < 1
                or type(rollout_queries) is not int or not 1 <= rollout_queries <= max_queries
                or not isinstance(runtime_identity, dict) or not runtime_identity):
            raise ValueError('two independent clients and bounded macro batch required')
        # Each IPC client must own a different payload namespace and connection.
        if (hasattr(clients[0], 'exchange') and hasattr(clients[1], 'exchange')
                and Path(clients[0].exchange).resolve() == Path(clients[1].exchange).resolve()):
            raise ValueError('replicas require independent exchange directories')
        self.clients = tuple(clients)
        self.runtime_identity = copy.deepcopy(runtime_identity)
        self.bundle = copy.deepcopy(clients[0].bundle)
        self.root = Path(journal_root)
        check_paths({'macro_journal': self.root})
        self.root.mkdir(parents=True, exist_ok=True)
        self.rollout_queries = rollout_queries; self.max_queries = max_queries
        self.phase = 'ready'; self.sequence = 0; self.last_receipt = None
        self._lock = threading.Lock()
        self._check_clients(self.bundle['policy_id'])

    def _check_clients(self, policy_id):
        for client in self.clients:
            if (client.phase != 'ready' or client.bundle['policy_id'] != policy_id
                    or client.bundle['contract'] != self.bundle['contract']
                    or getattr(client, 'runtime_identity', None) != self.runtime_identity):
                raise ValueError('replicas must acknowledge the same policy/contract/runtime')

    @staticmethod
    def _inputs(rows):
        identities = []; costs = []
        keys = {'input_ids', 'attention_mask', 'input_features', 'input_features_mask'}
        for row in rows:
            if (not isinstance(row, dict) or set(row) != {'inputs', 'positions'}
                    or not isinstance(row['inputs'], dict) or set(row['inputs']) != keys
                    or any(not isinstance(t, torch.Tensor) or t.device.type != 'cpu'
                           for t in [*row['inputs'].values(), row['positions']])):
                raise ValueError('only complete label-free CPU rows accepted')
            positions = row['positions']; mask = row['inputs']['attention_mask']
            if (positions.ndim != 2 or positions.shape[0] != 1 or positions.shape[1] < 1
                    or mask.ndim != 2 or mask.shape[0] != 1):
                raise ValueError('one query and complete candidate positions per row required')
            identities.append({k: tensor_identity(v) for k, v in
                               sorted({**row['inputs'], 'positions': positions}.items())})
            # Scheduling estimate only, not engine memory admission or measured FLOPs.
            costs.append(max(1, int(mask.sum()) + int(row['inputs']['input_features_mask'].sum())))
        return identities, costs

    def score_macro(self, batch_id, rows, *, audio_token_id):
        with self._lock:
            if self.phase != 'ready': raise ValueError('pool is not ready')
            if (not isinstance(batch_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', batch_id)
                    or not isinstance(rows, list) or not 1 <= len(rows) <= self.max_queries):
                raise ValueError('unique safe batch id and complete bounded rows required')
            policy = self.bundle['policy_id']; self._check_clients(policy)
            identities, costs = self._inputs(rows)
            jobs = plan_rollouts(costs, self.rollout_queries)
            root = self.root / batch_id
            root.mkdir(exist_ok=False)  # Duplicate/failed attempts require reconciliation.
            self.phase = 'executing'; started = time.monotonic()
            manifest = dict(batch_id=batch_id, policy_id=policy,
                            runtime_identity=self.runtime_identity, inputs=identities,
                            expected_indices=list(range(len(rows))), jobs=jobs)
            try:
                write(root/'INPUT.json', manifest)
                def run_replica(replica):
                    client = self.clients[replica]; completed = []
                    for job in jobs:
                        if job['replica_id'] != replica: continue
                        begin = time.monotonic()
                        indices = job['indices']
                        records = client.score_many([rows[i] for i in indices], audio_token_id=audio_token_id)
                        if not isinstance(records, list) or len(records) != len(indices):
                            raise ValueError('incomplete rollout response')
                        for index, record in zip(indices, records, strict=True):
                            validate_logits_record(record)
                            bind_rollout(record, **rows[index], policy_id=policy, group_size=2, seed=0)
                        if client.bundle['policy_id'] != policy:
                            raise ValueError('policy changed inside rollout')
                        result = dict(**job, policy_id=policy, records=records,
                                      seconds=time.monotonic()-begin)
                        write(root/(job['rollout_id']+'.json'), result)
                        completed.append(result)
                    return completed
                # One thread per connection; a rollout never migrates between replicas.
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(run_replica, i) for i in range(2)]
                    completed = [result for future in futures for result in future.result()]
                self._check_clients(policy)
                after, _ = self._inputs(rows)
                if after != identities: raise ValueError('input mutated during macro batch')
                ordered = [None] * len(rows); hashes = {}
                for result in completed:
                    hashes[result['rollout_id']] = digest(result)
                    for i, record in zip(result['indices'], result['records'], strict=True):
                        if ordered[i] is not None: raise ValueError('duplicate query result')
                        ordered[i] = record
                if any(record is None for record in ordered): raise ValueError('incomplete macro batch')
                receipt = dict(batch_id=batch_id, policy_id=policy, status='complete_macro_inference',
                               input_sha256=digest(manifest), rollout_hashes=hashes,
                               completed_indices=list(range(len(rows))), result_sha256=digest(ordered),
                               seconds=time.monotonic()-started, actor_update_performed=False)
                write(root/'COMPLETE.json', receipt)
                self.last_receipt = receipt; self.phase = 'ready'
                return ordered
            except BaseException as exc:
                self.phase = 'failed'
                write(root/'FAILED.json', dict(error=repr(exc), actor_consumption_allowed=False))
                raise

    def score_many(self, rows, *, audio_token_id):
        # Session adapter; callers must serialize requests/activation as before.
        batch_id = f'batch-{self.sequence:08d}'
        result = self.score_macro(batch_id, rows, audio_token_id=audio_token_id)
        self.sequence += 1
        return result

    def switch(self, *, snapshot, policy_id, adapter):
        with self._lock:
            if self.phase != 'ready': raise ValueError('quiescent complete pool required')
            self._check_clients(self.bundle['policy_id']); self.phase = 'activating'
            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(c.switch, snapshot=snapshot, policy_id=policy_id,
                                               adapter=adapter) for c in self.clients]
                    receipts = [f.result() for f in futures]
                if any(r.get('policy_id') != policy_id or r.get('head_marker_exact') is not True
                       or r.get('native_loaded_tensors') != 112 for r in receipts):
                    raise ValueError('incomplete replica activation')
                self._check_clients(policy_id)
                self.bundle = copy.deepcopy(self.clients[0].bundle); self.phase = 'ready'
                return dict(policy_id=policy_id, head_marker_exact=True, native_loaded_tensors=112,
                            replica_receipts=receipts)
            except BaseException:
                self.phase = 'failed'; raise

    def close(self):
        with self._lock:
            errors = []
            for client in self.clients:
                try: client.close()
                except BaseException as exc: errors.append(exc)
            self.phase = 'closed'
            if errors: raise errors[0]
