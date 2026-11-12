"""Explicit, versioned BF16 complete-prefill alignment profile.

This profile keeps native vLLM LoRA execution. It is restricted to TP1,
one uncached complete sequence, no quantization or decoding. A profile name
does not grant training admission: every request/version still needs parity.
"""

PROFILE = 'qwen3-asr-bf16-hf-order-v1'
SHORT_AUDIO_PROFILE = 'qwen3-asr-bf16-hf-order-short-audio-v2'
RPCS = (
    'align_jev_internal_format',
    'align_jev_audio_attention',
    'align_jev_language_norms',
    'align_jev_language_qkv',
    'align_jev_language_mlp',
    'align_jev_language_rotary',
    'align_jev_language_attention',
)


def initialize_verl_preserving_qwen3():
    """Import verl before model construction and undo its three NPU patches.

    The pinned verl package changes HF Qwen3 class methods on first import.
    Restoring the exact original objects keeps the accepted actor equations;
    all other verl initialization and its real loss functions remain active.
    Reject an already patched process instead of guessing original functions.
    """
    import importlib
    from transformers.models.qwen3 import modeling_qwen3 as qwen3
    targets=((qwen3.Qwen3RMSNorm,'forward'),(qwen3.Qwen3MLP,'forward'),
             (qwen3,'apply_rotary_pos_emb'))
    return preserve_qwen3_import(targets, lambda:importlib.import_module('verl.trainer.ppo.core_algos'))


def preserve_qwen3_import(targets, importer):
    originals=[getattr(owner,name) for owner,name in targets]
    if any(getattr(fn,'__module__','').startswith('verl.') for fn in originals):
        raise ValueError('initialize Span verl before importing patched Qwen3 runtime')
    changed=[]
    try:
        importer()
        for (owner,name),before in zip(targets,originals,strict=True):
            after=getattr(owner,name)
            if after is not before:
                if getattr(after,'__module__','')!='verl.models.transformers.npu_patch':
                    raise ValueError('unexpected Qwen3 runtime mutation during verl import')
                changed.append(dict(owner=getattr(owner,'__name__',type(owner).__name__),attribute=name,
                                    replacement_module=after.__module__))
    finally:
        for (owner,name),before in zip(targets,originals,strict=True):
            setattr(owner,name,before)
    return dict(restored=changed,scope='Original HF Qwen3 norm/MLP/rotary retained; real verl loss imported before forward')


def install_rollout_profile(engine, name):
    if name not in (PROFILE,SHORT_AUDIO_PROFILE):
        raise ValueError('unsupported Span runtime profile')
    receipts = {}
    operations=RPCS+('align_jev_short_audio_padding',) if name==SHORT_AUDIO_PROFILE else RPCS
    for rpc in operations:
        receipts[rpc] = engine.llm_engine.collective_rpc(rpc)
    return dict(profile=name, operations=receipts,
                scope='BF16 TP1 single complete uncached sequence; parity required per policy/request')


def configure_actor_profile(name):
    if name not in (PROFILE,SHORT_AUDIO_PROFILE):
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
