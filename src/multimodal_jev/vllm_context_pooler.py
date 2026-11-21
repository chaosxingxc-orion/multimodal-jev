"""Candidate vLLM 0.30 pooler for the shared history-attending head.

Not registered as a deployable model: authenticated joint loading, LoRA/marker
switching and real NPU parity remain separate admission requirements.
"""
import torch
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate

from .context_pooling import ContextHistoryReadout


class ContextProbabilityPooler(Pooler):
    def __init__(self, head, decision_token_id, policy_id, profiling_decisions=60, *, return_logits=False):
        super().__init__()
        if type(profiling_decisions) is not int or profiling_decisions <= 0:
            raise ValueError('positive profiling decision budget required')
        self.readout = ContextHistoryReadout(head, decision_token_id, policy_id, return_logits=return_logits)
        self.profiling_decisions = profiling_decisions
        self.memory_profiling = False

    def get_supported_tasks(self):
        return {'token_classify'}

    def get_pooling_updates(self, task):
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(self, hidden_states, pooling_metadata):
        cursor = pooling_metadata.get_pooling_cursor()
        if cursor.is_partial_prefill() or not all(cursor.get_finished_mask()):
            raise ValueError('complete uncached prefill required for context decisions')
        lengths = cursor.num_scheduled_tokens_cpu.tolist()
        prompts = pooling_metadata.get_prompt_token_ids_cpu()
        # vLLM uses int32 token buffers (including startup profiling), while
        # the backend-independent readout uses int64. Widen only integer IDs;
        # never silently round floating-point or accept boolean token buffers.
        if any(p.dtype not in (torch.int32, torch.int64) for p in prompts):
            raise ValueError('vLLM prompt IDs must be int32 or int64')
        prompts = [p.to(dtype=torch.long) for p in prompts]
        extras = [p.extra_kwargs for p in pooling_metadata.pooling_params]
        if self.memory_profiling:
            # Explicit startup-only mode: exercise the real branch over all
            # history. Never infer profiling merely from absent user metadata.
            if (len(prompts) != len(lengths) or len(extras) != len(lengths) or
                    any(extra for extra in extras) or
                    any(not bool((p == 0).all()) for p in prompts)):
                raise ValueError('profiling mode accepts synthetic zero prompts only')
            # vLLM's startup dummy prompts are one rectangular int32 block,
            # while scheduled lengths can differ in the last request because
            # of the token remainder. The dummy rows are verified all-zero
            # above, so rebuild exact-length synthetic prompts from the
            # cursor instead of trusting the block width.
            prompts = [torch.zeros(n, dtype=torch.long) for n in lengths]
            extras = []
            for prompt in prompts:
                if not len(prompt):
                    raise ValueError('nonempty profiling prompt required')
                count = min(len(prompt), self.profiling_decisions)
                prompt[-count:] = self.readout.decision_token_id
                extras.append(dict(policy_id=self.readout.policy_id,
                    marker_positions=list(range(len(prompt)-count, len(prompt))),
                    expected_input_ids=prompt.tolist()))
        return self.readout(hidden_states, lengths, prompts, extras)
