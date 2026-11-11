"""Explicit, versioned BF16 complete-prefill alignment profile.

This profile keeps native vLLM LoRA execution. It is restricted to TP1,
one uncached complete sequence, no quantization or decoding. A profile name
does not grant training admission: every request/version still needs parity.
"""

PROFILE = 'qwen3-asr-bf16-hf-order-v1'
RPCS = (
    'align_jev_internal_format',
    'align_jev_audio_attention',
    'align_jev_language_norms',
    'align_jev_language_qkv',
    'align_jev_language_mlp',
    'align_jev_language_rotary',
    'align_jev_language_attention',
)


def install_rollout_profile(engine, name):
    if name != PROFILE:
        raise ValueError('unsupported Span runtime profile')
    receipts = {}
    for rpc in RPCS:
        receipts[rpc] = engine.llm_engine.collective_rpc(rpc)
    return dict(profile=name, operations=receipts,
                scope='BF16 TP1 single complete uncached sequence; parity required per policy/request')


def configure_actor_profile(name):
    if name != PROFILE:
        raise ValueError('unsupported Span runtime profile')
    import torch
    import torch_npu
    torch.npu.config.allow_internal_format = False
    if torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT') != b'disable':
        raise ValueError('actor internal-format configuration failed')
    return dict(profile=name, internal_format='disable', attention='HF eager',
                lora_execution_dtype='bfloat16', head_dtype='bfloat16')


def verify_suite_admission(config):
    """Authenticate the acceptance artifact and independently recompute gates.

The caller pins the source/runtime image separately. This is a prerequisite
for the bounded pilot, never a substitute for before/after version gates.
"""
    import json
    from pathlib import Path
    from .baseline import file_hash
    from .policy_identity import checkpoint_identity
    from .span_parity import compare_span_policies
    if config.get('runtime_profile') != PROFILE:
        raise ValueError('unsupported Span runtime profile')
    path = config.get('runtime_acceptance')
    if not path or config.get('pins',{}).get(path) != file_hash(path):
        raise ValueError('pinned runtime acceptance required')
    result = json.loads(Path(path).read_text())
    if (result.get('runtime_profile') != PROFILE or result.get('passed') is not True
            or result.get('base_id') != checkpoint_identity(config['model_path'])[0]
            or result.get('adapter_id') != checkpoint_identity(config['initial_adapter'])[0]):
        raise ValueError('runtime acceptance policy identity differs')
    rows = result.get('records',[])
    if len(rows)!=48 or len({r['id'] for r in rows})!=48 or len({r['request_id'] for r in rows})!=48:
        raise ValueError('complete 48-record acceptance required')
    signatures = {tuple(r['labels']) for r in rows if all(r['supervised'])}
    if len(signatures)!=8 or any(len(s)!=3 or any(type(v)!=bool for v in s) for s in signatures):
        raise ValueError('all eight reviewed Boolean signatures required')
    for row in rows:
        if not compare_span_policies(row['hf_candidate_logprobs'],row['candidate_logprobs'])['passed']:
            raise ValueError('runtime acceptance candidate parity failed')
    return dict(profile=PROFILE, acceptance_sha256=file_hash(path), records=len(rows))
