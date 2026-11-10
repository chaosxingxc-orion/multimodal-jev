"""Audio boundary for vLLM AsyncLLM; not yet a native verl Ray server."""
import asyncio
import copy
import time
from uuid import uuid4

from .audio_policy import hf_prompt, vllm_request
from .rollout import decode_outputs, generation_budget, validate_sampling


class AsyncAudioRollout:
    """One immutable policy and serialized admission; groups remain complete.

    Uses raw prompt text and exact actor features, leaving placeholder expansion
    to vLLM once. Async streaming is internal; only complete groups are returned.
    """

    def __init__(self, engine, processor, policy_id, max_model_len):
        if not policy_id or type(max_model_len) is not int or max_model_len < 1:
            raise ValueError('policy identity and positive context limit required')
        self.engine, self.processor, self.policy_id = engine, processor, policy_id
        self.max_model_len = max_model_len
        self.last_generation_budget = None
        self.last_request_ids = None
        self._lock = asyncio.Lock()

    async def sample(self, examples, audios, config):
        from vllm import SamplingParams
        from multimodal_jev.vllm_audio_features import enable_feature_passthrough

        async with self._lock:
            self.last_generation_budget = None
            self.last_request_ids = None
            validate_sampling(config)
            if not examples or len(examples) != len(audios):
                raise ValueError('aligned nonempty examples/audio required')
            if len({ex.example_id for ex in examples}) != len(examples):
                raise ValueError('duplicate request identity')
            # Admission is whole-group and occurs before the first engine call.
            requests, expected_prompts = [], []
            for ex, audio in zip(examples, audios):
                prompt_ids = hf_prompt(self.processor, ex, audio)['input_ids'][0].tolist()
                if len(prompt_ids) + config['max_tokens'] > self.max_model_len:
                    raise ValueError('expanded prompt and requested output exceed context; no silent truncation')
                requests.append(vllm_request(self.processor, ex, audio, 'actor_features'))
                expected_prompts.append(prompt_ids)
            enable_feature_passthrough()
            outputs, request_ids = [], []
            started = time.monotonic()
            for request, expected in zip(requests, expected_prompts):
                request_id = uuid4().hex
                request_ids.append(request_id)
                final = None
                completed = False
                try:
                    stream = self.engine.generate(prompt=request,
                        sampling_params=SamplingParams(**copy.deepcopy(config), logprobs=0),
                        request_id=request_id)
                    async for output in stream:
                        if output.request_id != request_id:
                            raise ValueError('async engine response identity mismatch')
                        final = output
                    if final is None or not final.finished:
                        raise ValueError('async engine ended without completed request')
                    completed = True
                    if list(final.prompt_token_ids or []) != expected:
                        raise ValueError('async engine/actor expanded prompt IDs differ')
                    # Validate each completed group before admitting the next.
                    decode_outputs([final], [examples[len(outputs)]], self.processor, self.policy_id, config['n'])
                    outputs.append(final)
                finally:
                    if not completed:
                        await self.engine.abort(request_id)
            records = decode_outputs(outputs, examples, self.processor, self.policy_id, config['n'])
            self.last_generation_budget = generation_budget(outputs, records, config['n'], time.monotonic()-started)
            self.last_request_ids = request_ids
            return records


def create_async_engine(checkpoint, engine_options):
    """Separate engine with the same explicit feature/generation contract."""
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from .vllm_audio_features import enable_feature_passthrough

    allowed = {'dtype', 'max_model_len', 'gpu_memory_utilization', 'tensor_parallel_size',
               'enforce_eager', 'max_num_seqs', 'seed'}
    if set(engine_options)-allowed:
        raise ValueError('unexpected async engine configuration')
    enable_feature_passthrough()
    args = AsyncEngineArgs(model=str(checkpoint), tokenizer=str(checkpoint), trust_remote_code=False,
        generation_config='vllm', limit_mm_per_prompt={'audio': 1}, enable_mm_embeds=True, **engine_options)
    return AsyncLLM.from_engine_args(args)
