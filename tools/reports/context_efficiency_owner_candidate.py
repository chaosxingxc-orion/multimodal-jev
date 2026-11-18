"""Versioned engine-efficiency owner: token budget plus KV memory utilization.

The frozen switch/activation lifecycle is read, re-hashed and transformed in
memory only. Three replacements are applied as one union:

* ``max_num_seqs``/token budget come from the declared service config;
* the existing batched worker extension is reused unchanged;
* ``gpu_memory_utilization`` grows from .4 to .6 so the 65536-token KV budget
  fits on the card.

No in-place source edit, task-definition change or training release is
performed. Dispatch stays blocked until the v9 formal run releases the cards.
"""
import ast
import hashlib
from pathlib import Path

from tools.reports.context_batched_request_candidate import encode_many

SOURCE_SHA = 'beb755c48b4307a0754204a3aaae0fff5aba30187f1d70957894b208cb1c0813'
RUNTIME = 'context-efficiency-candidate-v1'
GPU_MEMORY_UTILIZATION = .6


def transformed_source(source_root):
    """Re-verify the pinned lifecycle and apply the three-replacement union."""
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
        ('gpu_memory_utilization=.4', 'gpu_memory_utilization=.6'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('unexpected frozen owner construction')
        source = source.replace(before, after)
    return source


def generate(source_root):
    """Packager entrypoint: the transformed source, syntax-checked."""
    source = transformed_source(source_root)
    ast.parse(source)
    return source


def owner_class(source_root):
    namespace = {'__name__': __name__ + '.derived'}
    exec(compile(transformed_source(source_root), '<efficiency-owner-candidate>', 'exec'), namespace)
    base = namespace['SwitchingContextRollout']

    class EfficiencyContextOwnerCandidate(base):
        def __init__(self, config):
            if (config.get('candidate_runtime') != RUNTIME
                    or type(config.get('max_batch_queries')) is not int
                    or not 1 <= config['max_batch_queries'] <= 128
                    or type(config.get('max_batch_tokens')) is not int
                    or config['max_batch_tokens'] < config['max_model_len']
                    or config['contract']['runtime_profile'] != namespace['PROFILE']):
                raise ValueError('explicit bounded efficiency owner config required')
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

    return EfficiencyContextOwnerCandidate
