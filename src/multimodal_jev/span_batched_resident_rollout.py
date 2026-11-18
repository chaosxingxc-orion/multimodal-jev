"""Candidate resident rollout using the jointly verified batch execution kernels.

Independent opt-in owner; the original engine/profile stays unchanged. Startup
receipts are not probability acceptance or authorization for RL training.
"""
from .span_resident_rollout import ResidentSpanRollout, adapter_layout
from .policy_identity import checkpoint_identity
from .span_batch_training_config import PROFILE
from .span_runtime_profile import SHORT_AUDIO_PROFILE, install_rollout_profile
from .span_device_scope import validate_tp1


def validate_candidate(config):
    if (config.get('candidate_profile') != PROFILE
            or config.get('runtime_profile') != SHORT_AUDIO_PROFILE
            or config.get('rollout_candidate_component') != 'full_native_lora'):
        raise ValueError('explicit complete candidate resident profile required')


def create_engine(config, targets):
    from .local_ipc import require_local_ipc
    require_local_ipc()
    import torch
    from vllm import LLM
    from .vllm_spans import register_span_model
    from .vllm_audio_features import enable_feature_passthrough
    register_span_model(); enable_feature_passthrough()
    packed = sorted({t.replace('model.language_model.', 'language_model.model.', 1).rsplit('.', 1)[0]+'.qkv_proj' for t in targets})
    return LLM(model=config['model'], tokenizer=config['model'], runner='pooling',
        hf_overrides={'architectures': ['Qwen3ASRForSpanDecisions'], 'head_dtype': 'model'},
        pooler_config={'task': 'token_classify', 'tok_pooling_type': 'ALL', 'use_activation': False},
        dtype='bfloat16', max_model_len=4096, gpu_memory_utilization=.3,
        tensor_parallel_size=1, enforce_eager=True, max_num_seqs=1, seed=17,
        worker_extension_cls='multimodal_jev.span_batch_rollout_candidate.BatchCandidateWorkerExtension',
        enable_chunked_prefill=False, enable_prefix_caching=False, enable_mm_embeds=True,
        limit_mm_per_prompt={'audio': 1}, enable_lora=True, max_loras=1, max_cpu_loras=1,
        max_lora_rank=8, lora_dtype=torch.bfloat16, lora_target_modules=packed)


class BatchedResidentSpanRollout(ResidentSpanRollout):
    def __init__(self, config, initial_adapter):
        validate_candidate(config)
        validate_tp1(config)
        if checkpoint_identity(config['model'])[0] != config['base_id']:
            raise ValueError('base identity differs')
        fields = ('model', 'base_id', 'runtime_profile', 'max_model_len',
                  'candidate_profile', 'rollout_candidate_component')
        self.config = {k: config[k] for k in fields}
        self.config['physical_devices'] = list(config['physical_devices'])
        self.targets = adapter_layout(initial_adapter)
        self.slots = {}; self.phase = 'starting'
        self.engine = create_engine(config, self.targets)
        try:
            base = install_rollout_profile(self.engine, config['runtime_profile'])
            attention = self.engine.llm_engine.collective_rpc('align_jev_candidate_packed_attention')
            projections = self.engine.llm_engine.collective_rpc('align_jev_candidate_fixed_base')
            if (len(attention) != 1 or len(projections) != 1
                    or attention[0].get('attention') != 'TND_sparse3_inner2'
                    or type(attention[0].get('layers')) is not int or attention[0]['layers'] < 1
                    or projections[0].get('language_qkv') != attention[0]['layers']
                    or not projections[0].get('parallel_projections')
                    or projections[0].get('tile_rows') != 128 or projections[0].get('tile_batch') != 8
                    or projections[0].get('lora') != 'native_vllm_unchanged'):
                raise ValueError('complete candidate worker installation required')
            self.runtime_receipt = dict(base=base, candidate_profile=PROFILE,
                attention=attention, projections=projections,
                scope='Candidate initialized; requires independent same-policy probability acceptance.')
        except BaseException:
            self.phase = 'failed'; self.close(); raise
        self.phase = 'ready'
