"""Versioned serving-config owner: expose gpu_memory_utilization as a config field.

Dev-state candidate ``v0.0.1`` for the 1000-record serving comparison. It is the
same frozen switch/activation lifecycle as ``context_batched_owner_candidate``
(``SOURCE_SHA`` re-verified, in-memory transform only), with one added change:

* ``gpu_memory_utilization`` is read from the declared config instead of the
  hardcoded ``.4`` so larger KV budgets can be exercised;
* ``max_num_seqs`` follows ``max_batch_queries`` and ``max_num_batched_tokens``
  follows ``max_batch_tokens`` exactly as the already-adopted batched owner;
* the existing batched worker extension is reused unchanged.

No in-place source edit and no training release. This owner exists only to time
the serving stage; it is not a formal or effect-bearing candidate.
"""
import hashlib
from pathlib import Path

from tools.reports.context_batched_request_candidate import encode_many

SOURCE_SHA = 'beb755c48b4307a0754204a3aaae0fff5aba30187f1d70957894b208cb1c0813'
PROFILE = 'context-serving-config-candidate-v1'


def transformed_source(source_root):
    """Re-verify the pinned lifecycle and apply the serving-config union."""
    path = Path(source_root) / 'tools/jev/context_switch_rollout.py'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA:
        raise ValueError('frozen switching owner source changed')
    source = data.decode()
    replacements = [
        ('max_num_seqs=1,seed=317,',
         "max_num_seqs=config['max_batch_queries'],max_num_batched_tokens=config['max_batch_tokens'],seed=317,"),
        ("worker_extension_cls='context_switch_worker.SwitchingContextWorker',",
         "worker_extension_cls='tools.reports.context_batched_worker_extension_candidate.BatchedContextWorkerCandidate',"),
        ('gpu_memory_utilization=.4', "gpu_memory_utilization=config['gpu_memory_utilization']"),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('unexpected frozen owner construction')
        source = source.replace(before, after)
    return source


def owner_class(source_root):
    namespace = {'__name__': __name__ + '.derived'}
    exec(compile(transformed_source(source_root), '<serving-config-owner-candidate>', 'exec'), namespace)
    base = namespace['SwitchingContextRollout']

    class ServingConfigOwnerCandidate(base):
        def __init__(self, config):
            if (config.get('candidate_runtime') != PROFILE
                    or type(config.get('max_batch_queries')) is not int
                    or not 1 <= config['max_batch_queries'] <= 128
                    or type(config.get('max_batch_tokens')) is not int
                    or config['max_batch_tokens'] < config['max_model_len']
                    or type(config.get('gpu_memory_utilization')) not in (int, float)
                    or not 0.0 < float(config['gpu_memory_utilization']) < 1.0
                    or type(config.get('kv_cache_memory_bytes')) is not int
                    or config['kv_cache_memory_bytes'] < 1
                    or config['contract']['runtime_profile'] != namespace['PROFILE']):
                raise ValueError('explicit bounded serving-config owner config required')
            self.max_batch_queries = config['max_batch_queries']
            self.max_batch_tokens = config['max_batch_tokens']
            super().__init__(config)
            try:
                receipts = self.engine.llm_engine.collective_rpc('install_jev_segmented_candidate',
                                                                 args=(self.bundle['policy_id'],))
                if (len(receipts) != 1 or receipts[0]['policy_id'] != self.bundle['policy_id']
                        or receipts[0]['layers'] <= 0 or receipts[0]['production_promoted']
                        or receipts[0]['training_dispatch_allowed']):
                    raise ValueError('segmented worker installation receipt mismatch')
                self.segmented_receipt = receipts[0]
            except BaseException:
                self.close()
                raise

        def score_many(self, rows, *, audio_token_id):
            from vllm import PoolingParams
            with self._transaction:
                if self.phase != 'ready':
                    raise ValueError('ready quiescent batch owner required')
                self.phase = 'evaluating'
                try:
                    result = encode_many(self, rows, audio_token_id=audio_token_id, params_class=PoolingParams)
                except BaseException:
                    self.phase = 'failed'
                    raise
                self.phase = 'ready'
                return result

    return ServingConfigOwnerCandidate
