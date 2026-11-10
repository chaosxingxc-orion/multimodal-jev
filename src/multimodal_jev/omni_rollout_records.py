"""Strict vLLM sampling-to-SFC record bridge, without inference dispatch.

The initial contract uses temperature 1 and no distribution truncation or
penalties. With engine raw_logprobs this matches the model policy evaluated by
the actor at temperature 1. Engine/image/weights must be pinned by the caller.
"""
import math

from .sfc_reward_batch import digest, valid_hash


def sampling_kwargs(group_size, seed, max_tokens=1024):
    if type(group_size) is not int or group_size < 2:
        raise ValueError('at least two sampling replicates required')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('registered nonnegative 32-bit seed required')
    if type(max_tokens) is not int or not 1 <= max_tokens <= 1024:
        raise ValueError('registered response budget in [1,1024] required')
    return dict(n=group_size, seed=seed, temperature=1., top_p=1., top_k=0, min_p=0.,
                presence_penalty=0., frequency_penalty=0., repetition_penalty=1.,
                max_tokens=max_tokens, min_tokens=0, logprobs=0, prompt_logprobs=None,
                stop=None, stop_token_ids=None, ignore_eos=False,
                skip_special_tokens=True, spaces_between_special_tokens=False,
                include_stop_str_in_output=False, detokenize=True)


def capture(response, request, decision_id, prepared, tokenizer, *, policy_sha256,
            instruction_sha256, group_size, max_tokens=1024):
    """Capture one complete K-output request, using engine indices as replicates.

    Caller must use sampling_kwargs and explicitly set engine
    logprobs_mode='raw_logprobs', generation_config='vllm', with no additional
    logits processors, grammar, allowed-token masks or adapter. Do not infer
    these settings from a response object: they are absent from that object.
    """
    sampling_kwargs(group_size, 0, max_tokens)
    if not valid_hash(policy_sha256) or not valid_hash(instruction_sha256):
        raise ValueError('pinned policy and instruction hashes required')
    if decision_id not in {d['id'] for d in request['input']['decisions']}:
        raise ValueError('declared decision required')
    expected = prepared['input_ids'].tolist()
    if (prepared['request_sha256'] != digest(request)
            or prepared['input_ids_sha256'] != digest(expected)
            or not valid_hash(prepared['prompt_sha256'])):
        raise ValueError('prepared prompt identity changed')
    if (response.finished is not True or response.prompt_token_ids is None
            or list(response.prompt_token_ids) != expected or len(response.outputs) != group_size):
        raise ValueError('complete response with identical expanded prompt required')
    found = {}
    for answer in response.outputs:
        rep = answer.index
        ids = list(answer.token_ids)
        if (type(rep) is not int or not 0 <= rep < group_size or rep in found
                or not ids or len(ids) > max_tokens
                or any(type(t) is not int or not 0 <= t < len(tokenizer) for t in ids)
                or answer.finish_reason not in ('stop', 'length')
                or type(answer.text) is not str):
            raise ValueError('unique registered replicate and valid sampled tokens required')
        if tokenizer.decode(ids, skip_special_tokens=True) != answer.text:
            raise ValueError('sampled text differs from complete response tokens')
        if answer.logprobs is None or len(answer.logprobs) != len(ids):
            raise ValueError('one logprob entry per sampled token required')
        logps = []
        for token, values in zip(ids, answer.logprobs, strict=True):
            if values is None or token not in values:
                raise ValueError('sampled token missing from logprob entry')
            value = values[token].logprob
            if type(value) not in (float, int) or not math.isfinite(value) or value > 1e-5:
                raise ValueError('finite nonpositive sampled-token logprob required')
            logps.append(float(value))
        found[rep] = dict(example_id=request['id'], decision_id=decision_id,
            request_sha256=digest(request), replicate=rep, policy_sha256=policy_sha256,
            instruction_sha256=instruction_sha256,
            prompt_sha256=prepared['prompt_sha256'], input_ids_sha256=prepared['input_ids_sha256'],
            raw_text=answer.text, token_ids=ids, log_probs=logps, finish_reason=answer.finish_reason)
    return [found[rep] for rep in range(group_size)]
