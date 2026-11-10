"""Complete Omni audio groups at AsyncLLM boundary, not a verl Ray integration.

Caller pins engine/image/weights, authenticates learning inputs and waveform
hashes, and persists a dispatch intent before calling. An instance never reuses
an engine request ID. Recovery across processes requires the caller's ledger.
"""
import asyncio
import hashlib

from .omni_rollout_records import capture, sampling_kwargs, validate_sampling
from .sfc_jev_runtime import vllm_request
from .sfc_reward_batch import digest, valid_hash


class OmniAsyncSampler:
    def __init__(self, engine, processor, *, policy_sha256, instruction_sha256,
                 max_model_len=4096, sampling_params_factory=None):
        if not valid_hash(policy_sha256) or not valid_hash(instruction_sha256):
            raise ValueError('fixed policy and instruction identity required')
        if type(max_model_len) is not int or max_model_len < 1:
            raise ValueError('positive context limit required')
        config = engine.model_config
        if (config.logprobs_mode != 'raw_logprobs' or config.generation_config != 'vllm'
                or config.max_model_len != max_model_len):
            raise ValueError('raw model probabilities and exact engine context required')
        if sampling_params_factory is None:
            from vllm import SamplingParams
            sampling_params_factory = SamplingParams
        self.engine, self.processor = engine, processor
        self.policy_sha256, self.instruction_sha256 = policy_sha256, instruction_sha256
        self.max_model_len, self.params_factory = max_model_len, sampling_params_factory
        self.dispatched = set()
        self.last_output = None
        self._lock = asyncio.Lock()

    async def sample(self, row, decision_id, prepared, waves, *, request_id,
                     group_size=2, seed=17, max_tokens=1024):
        async with self._lock:
            if type(request_id) is not str or not request_id or request_id in self.dispatched:
                raise ValueError('new explicit engine request ID required; no replay')
            self.last_output = None
            if (prepared['request_sha256'] != digest(row)
                    or prepared['input_ids_sha256'] != digest(prepared['input_ids'].tolist())):
                raise ValueError('bound actor request and expanded tokens required')
            if len(prepared['input_ids']) + max_tokens > self.max_model_len:
                raise ValueError('complete requested output must fit; no budget clamping')
            request = vllm_request(self.processor, row, decision_id, waves)
            if hashlib.sha256(request['prompt'].encode()).hexdigest() != prepared['prompt_sha256']:
                raise ValueError('raw rollout prompt differs from actor preparation')
            params = self.params_factory(**sampling_kwargs(group_size, seed, max_tokens))
            validate_sampling(params, group_size, seed, max_tokens)
            config = self.engine.model_config
            if (config.logprobs_mode != 'raw_logprobs' or config.generation_config != 'vllm'
                    or config.max_model_len != self.max_model_len):
                raise ValueError('engine probability/context contract changed')
            self.dispatched.add(request_id)
            finished = False
            try:
                stream = self.engine.generate(prompt=request, sampling_params=params, request_id=request_id)
                async for output in stream:
                    self.last_output = output
                    if output.request_id != request_id:
                        raise ValueError('async response request identity differs')
                    finished = output.finished is True
                if not finished:
                    raise ValueError('async stream ended without completed group')
                return capture(self.last_output, row, decision_id, prepared, self.processor.tokenizer,
                    policy_sha256=self.policy_sha256, instruction_sha256=self.instruction_sha256,
                    group_size=group_size, max_tokens=max_tokens)
            except BaseException as exc:
                if not finished:
                    try:
                        await self.engine.abort(request_id)
                    except BaseException as abort_error:
                        exc.add_note('Abort also failed: '+repr(abort_error))
                raise
