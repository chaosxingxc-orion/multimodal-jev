"""Named context probability runtime, binding previously measured components.

Installation is not training admission. New policies still require exact input,
probability, optimizer and distributed recovery checks.
"""
from .span_runtime_profile import SHORT_AUDIO_PROFILE, install_rollout_profile

PROFILE = 'qwen3-asr-context-bf16-packed-row-v1'
NATIVE_DIAGNOSTIC = 'context-vllm-native-bf16-diagnostic-v1'


def install_actor(adapted, model, name):
    if name != PROFILE:
        raise ValueError('explicit context packed-row profile required')
    from .span_batch_candidate_runtime import install_candidate
    from .context_row_head import install_row_head
    handle, components = install_candidate(adapted, model)
    components['head'] = install_row_head(model.decision_head)
    return handle, dict(profile=PROFILE, components=components,
        scope='Complete BF16 prefills; batched backbone, shared per-request head; admission separate')


def install_rollout(engine, name):
    if name != PROFILE:
        raise ValueError('explicit context packed-row profile required')
    receipt = install_rollout_profile(engine, SHORT_AUDIO_PROFILE)
    rpc = engine.llm_engine.collective_rpc
    attention = rpc('align_jev_candidate_packed_attention')
    projection = rpc('align_jev_candidate_fixed_base')
    if (len(attention)!=1 or len(projection)!=1 or attention[0]['layers']<=0 or
            projection[0]['language_qkv']!=attention[0]['layers'] or
            attention[0]['attention']!='TND_sparse3_inner2' or
            projection[0]['tile_rows']!=128 or projection[0]['tile_batch']!=8):
        raise ValueError('complete TP1 packed/fixed projection installation required')
    return dict(profile=PROFILE, alignment=receipt, attention=attention, projection=projection,
        head='one complete request per engine call', training_dispatch_allowed=False)
