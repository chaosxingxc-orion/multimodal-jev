"""vLLM offline audio sampling with explicit policy and response identities.

The engine is created from a checkpoint for one policy version. Recreate it after
an update; silently keeping an old engine is not supported. No HF fallback.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
from .audio_policy import hf_prompt, vllm_request


@dataclass(frozen=True)
class Rollout:
    example_id: str
    replicate: int
    policy_id: str
    token_ids: tuple[int,...]
    log_probs: tuple[float,...]
    text: str
    finish_reason: str


def validate_sampling(config):
    # Raw-policy actor logprobs are comparable only to this distribution in v1.
    # Temperature/top-k changes require applying the identical transform to actor logits.
    if config.get('temperature')!=1.0 or config.get('top_p')!=1.0 or config.get('top_k')!=-1:
        raise ValueError('v1 requires temperature=1, top_p=1, top_k=-1 for actor probability parity')
    if config.get('repetition_penalty',1.0)!=1.0 or config.get('presence_penalty',0.0)!=0.0 or config.get('frequency_penalty',0.0)!=0.0:
        raise ValueError('v1 disallows probability-changing penalties')
    if type(config.get('n')) is not int or config['n']<2:
        raise ValueError('GRPO requires n>=2')
    if type(config.get('max_tokens')) is not int or config['max_tokens']<1:
        raise ValueError('positive generation limit required')
    allowed={'temperature','top_p','top_k','n','max_tokens','seed','repetition_penalty','presence_penalty','frequency_penalty'}
    if set(config)-allowed:raise ValueError('unsupported sampling parameters')


def decode_outputs(outputs,examples,processor,policy_id,n):
    if not policy_id:raise ValueError('policy identity required')
    if len(outputs)!=len(examples):raise ValueError('request output count mismatch')
    records=[]
    for request,example in zip(outputs,examples):
        if not request.finished:raise ValueError('request has not finished')
        if len(request.outputs)!=n:raise ValueError('incomplete rollout group')
        indices={response.index for response in request.outputs}
        if indices!=set(range(n)):raise ValueError('duplicate or missing replicate index')
        for response in sorted(request.outputs,key=lambda r:r.index):
            ids=list(response.token_ids)
            if not ids or response.logprobs is None or len(response.logprobs)!=len(ids):
                raise ValueError('nonempty aligned token logprobs required')
            logps=[]
            for token,distribution in zip(ids,response.logprobs):
                if distribution is None or token not in distribution:raise ValueError('sampled token logprob missing')
                logp=float(distribution[token].logprob)
                if not math.isfinite(logp) or logp>1e-5:raise ValueError('invalid token logprob')
                logps.append(logp)
            if response.finish_reason not in {'stop','length'}:raise ValueError('unsupported rollout termination')
            records.append(Rollout(example.example_id,response.index,policy_id,tuple(ids),tuple(logps),
                processor.tokenizer.decode(ids,skip_special_tokens=True),response.finish_reason))
    return records


def generation_budget(outputs,records,n,elapsed_seconds):
    """Count engine-observed tokens, without assuming prefill/cache compute."""
    if not math.isfinite(elapsed_seconds) or elapsed_seconds<0:
        raise ValueError('finite nonnegative generation duration required')
    if type(n) is not int or n<2 or not outputs or len(records)!=len(outputs)*n:
        raise ValueError('complete generation groups required for accounting')
    lengths=[]
    for request in outputs:
        ids=getattr(request,'prompt_token_ids',None)
        if ids is None or len(ids)==0:
            raise ValueError('engine prompt token IDs required for accounting')
        lengths.append(len(ids))
    response_lengths=[len(record.token_ids) for record in records]
    return {'requests':len(outputs),'responses':len(records),
            'prompt_tokens_per_request':lengths,
            'request_prompt_tokens':sum(lengths),
            'replicated_prompt_tokens':sum(lengths)*n,
            'generated_tokens':sum(response_lengths),
            'response_tokens_per_record':response_lengths,
            'length_terminated_responses':sum(r.finish_reason=='length' for r in records),
            'engine_generate_seconds':elapsed_seconds,
            'scope':'observed engine token counts; replicated prompts are exposures, not measured prefill work or FLOPs'}


class VllmAudioRollout:
    def __init__(self,checkpoint,processor,policy_id,engine_options,audio_input_mode='waveform'):
        from vllm import LLM
        if not policy_id:raise ValueError('policy identity required')
        if audio_input_mode not in {'waveform','actor_features'}:
            raise ValueError('unsupported audio_input_mode')
        self.audio_input_mode=audio_input_mode
        self._closed=False
        self.last_generation_budget=None
        self.last_prompt_identity=None
        self.max_model_len=engine_options.get('max_model_len')
        self.policy_id=policy_id;self.processor=processor
        if set(engine_options)-{'dtype','max_model_len','gpu_memory_utilization','tensor_parallel_size','enforce_eager','max_num_seqs','seed'}:
            raise ValueError('unexpected engine configuration')
        self.engine=LLM(model=checkpoint,tokenizer=checkpoint,trust_remote_code=False,
                        generation_config='vllm',limit_mm_per_prompt={'audio':1},
                        enable_mm_embeds=audio_input_mode=='actor_features',**engine_options)

    def sample(self,examples,audios,config,audio_input_mode=None):
        from vllm import SamplingParams
        if self._closed:
            raise RuntimeError('rollout engine is closed')
        self.last_generation_budget=None
        self.last_prompt_identity=None
        if audio_input_mode is None:audio_input_mode=self.audio_input_mode
        if audio_input_mode!=self.audio_input_mode:
            raise ValueError('audio_input_mode differs from engine initialization')
        if audio_input_mode=='actor_features':
            from .vllm_audio_features import enable_feature_passthrough
            enable_feature_passthrough()
        validate_sampling(config)
        if not examples or len(examples)!=len(audios):raise ValueError('aligned examples/audio required')
        if len({ex.example_id for ex in examples})!=len(examples):raise ValueError('duplicate request identity')
        requests=[vllm_request(self.processor,ex,audio,audio_input_mode) for ex,audio in zip(examples,audios)]
        expected_prompts=[hf_prompt(self.processor,ex,audio)['input_ids'][0].tolist()
                          for ex,audio in zip(examples,audios)]
        if self.max_model_len is not None and any(
                len(ids)+config['max_tokens']>self.max_model_len for ids in expected_prompts):
            raise ValueError('expanded prompt and requested output exceed context; no silent truncation')
        params=SamplingParams(**config,logprobs=0)
        started=time.monotonic()
        outputs=self.engine.generate(requests,params,use_tqdm=False)
        elapsed=time.monotonic()-started
        if len(outputs)!=len(expected_prompts):
            raise ValueError('request output count mismatch')
        for output,expected in zip(outputs,expected_prompts):
            if list(getattr(output,'prompt_token_ids',None) or [])!=expected:
                raise ValueError('engine/actor expanded prompt IDs differ')
        records=decode_outputs(outputs,examples,self.processor,self.policy_id,config['n'])
        self.last_generation_budget=generation_budget(outputs,records,config['n'],elapsed)
        self.last_prompt_identity=[{'example_id':ex.example_id,'prompt_token_ids':ids}
                                   for ex,ids in zip(examples,expected_prompts)]
        return records

    def close(self):
        """Release the pinned v1 engine before Python joins child processes.

        Explicit lifecycle for isolated offline samplers; do not rely on weakref
        finalizers racing multiprocessing's interpreter-exit handler.
        """
        if not self._closed:
            self.engine.llm_engine.engine_core.shutdown(timeout=30.0)
            self._closed=True
