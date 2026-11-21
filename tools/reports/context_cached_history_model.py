"""Isolated cache-enabled pooling candidate; not admitted for formal scoring."""
from types import FunctionType
import torch
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.qwen3_asr import Qwen3ASRMultiModalProcessor,Qwen3ASRProcessingInfo,Qwen3ASRDummyInputsBuilder
from multimodal_jev import vllm_context_model as parent
from multimodal_jev.context_prefix_history import PrefixHistory
from multimodal_jev.context_logits_wire import install_logits_readout
from multimodal_jev.vllm_audio_features import enable_feature_passthrough


def validate(config):
    if (not config.cache_config.enable_prefix_caching or config.scheduler_config.enable_chunked_prefill
            or config.model_config.dtype!=torch.bfloat16 or config.quant_config is not None
            or config.parallel_config.tensor_parallel_size!=1 or config.parallel_config.pipeline_parallel_size!=1
            or config.lora_config is None or not config.model_config.enforce_eager):
        raise ValueError('native BF16 eager TP1 prefix-enabled complete suffix candidate required')
    spec=config.model_config.hf_config.jev_context_policy
    if set(spec)!={'snapshot','policy_id','contract','base_id','profiling_decisions'}:raise ValueError('explicit joint policy required')
    return spec


@MULTIMODAL_REGISTRY.register_processor(Qwen3ASRMultiModalProcessor,info=Qwen3ASRProcessingInfo,dummy_inputs=Qwen3ASRDummyInputsBuilder)
class Qwen3ASRForContextDecisions(parent.Qwen3ASRForContextDecisions):
    def __init__(self,*,vllm_config,prefix=''):
        # Preserve original weight/head registration; only the candidate
        # runtime validator differs. The frozen parent file remains intact.
        fn=FunctionType(parent.Qwen3ASRForContextDecisions.__init__.__code__,
            dict(parent.Qwen3ASRForContextDecisions.__init__.__globals__,validate_context_runtime=validate),
            closure=parent.Qwen3ASRForContextDecisions.__init__.__closure__)
        fn(self,vllm_config=vllm_config,prefix=prefix)
        self.pooler=CachedHistoryPooler(self.pooler,vllm_config.cache_config.block_size or 128)


class CachedHistoryPooler(torch.nn.Module):
    def __init__(self,original,block_size):
        super().__init__();self.original=original
        # Parameter names must stay pooler.readout.head.* for LoRA/model loader.
        self.readout=original.readout;del original._modules['readout'];object.__setattr__(original,'readout',self.readout)
        self.history=PrefixHistory(block_size=block_size,max_bytes=512*1024**2)
        self.memory_profiling=True;self.profiling_decisions=original.profiling_decisions
        self.last_stats={}

    def get_supported_tasks(self):return self.original.get_supported_tasks()
    def get_pooling_updates(self,task):return self.original.get_pooling_updates(task)

    def forward(self,hidden_states,pooling_metadata):
        if self.memory_profiling:
            self.original.memory_profiling=True;return self.original(hidden_states,pooling_metadata)
        self.original.memory_profiling=False
        cursor=pooling_metadata.get_pooling_cursor()
        if not all(cursor.get_finished_mask()):raise ValueError('chunking not admitted')
        lengths=cursor.prompt_lens_cpu.tolist();scheduled=cursor.num_scheduled_tokens_cpu.tolist()
        prompts=[p.long() for p in pooling_metadata.get_prompt_token_ids_cpu()];extras=[p.extra_kwargs for p in pooling_metadata.pooling_params]
        outputs=[];clean=[];hits=[]
        for suffix,n,prompt,extra in zip(hidden_states.split(scheduled),lengths,prompts,extras,strict=True):
            if set(extra)!={'policy_id','marker_positions','expected_input_ids','history_binding'}:raise ValueError('cache request binding required')
            binding=extra['history_binding']
            if binding['policy_id']!=self.readout.policy_id or extra['policy_id']!=self.readout.policy_id:raise ValueError('policy namespace changed')
            computed=n-len(suffix);hits.append(computed)
            outputs.append(self.history.restore(suffix,prompt.tolist(),computed,binding))
            clean.append({k:v for k,v in extra.items() if k!='history_binding'})
        self.last_stats=dict(cached_tokens=hits,scheduled_tokens=scheduled,history_bytes=self.history.bytes,history_hit_tokens=self.history.hit_tokens)
        return self.readout(torch.cat(outputs,0),lengths,prompts,clean)


def register():
    from vllm import ModelRegistry
    ModelRegistry.register_model('Qwen3ASRForContextDecisions','tools.reports.context_cached_history_model:Qwen3ASRForContextDecisions')
    enable_feature_passthrough()

register()

class CachedHistoryWorker:
    def finish_jev_cached_history(self,policy):
        m=self.get_model()
        if m._jev_policy_id!=policy or not m._jev_base_weights_loaded:raise ValueError('loaded policy required')
        m.pooler.memory_profiling=False
        receipt=install_logits_readout(m.pooler.readout)
        return dict(receipt,prefix_caching=self.vllm_config.cache_config.enable_prefix_caching,block_size=m.pooler.history.block_size)

    def jev_cached_history_stats(self):return self.get_model().pooler.last_stats

    def verify_jev_cached_adapter(self, policy):
        from context_switch_worker import verify_loaded_lora
        from multimodal_jev.context_policy_artifacts import read_execution_bundle
        m=self.get_model();spec=self.vllm_config.model_config.hf_config.jev_context_policy
        bundle=read_execution_bundle(spec['snapshot'],expected_policy_id=policy,
                    expected_contract=spec['contract'],expected_base_id=spec['base_id'])
        # Native vLLM installs the adapter on the first actual encode request.
        receipt,count=verify_loaded_lora(m,bundle,1)
        return dict(receipt,native_loaded_tensors=count,base_weights_merged=False)

    def align_jev_cached_history(self, policy):
        """Retain actor arithmetic outside attention; native paged KV stays live.

        No monkey patch of the attention implementation or scheduler. Admission
        depends on cold/warm/frozen comparisons after this component install.
        """
        from types import MethodType
        import torch_npu
        from multimodal_jev.span_audio_attention_alignment import install_eager_audio_attention
        from multimodal_jev.span_audio_padding import install_short_audio_padding
        from multimodal_jev.span_norm_alignment import install_hf_order_language_norm
        from multimodal_jev.span_rotary_alignment import install_hf_order_rotary
        from multimodal_jev.span_fixed_rollout_projection import install_fixed_rollout_projections
        from tools.reports.context_segmented_audio_candidate import segmented_audio_forward
        m = self.get_model()
        if m._jev_policy_id != policy or m.pooler.memory_profiling:
            raise ValueError('loaded fixed policy required before arithmetic alignment')
        original_attention = [layer.self_attn.forward for layer in m.language_model.model.layers]
        torch.npu.config.allow_internal_format = False
        if torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT') != b'disable':
            raise ValueError('internal format alignment failed')
        result = dict(audio=install_eager_audio_attention(m),
                      norms=install_hf_order_language_norm(m),
                      rotary=install_hf_order_rotary(m),
                      projection=install_fixed_rollout_projections(m),
                      padding=install_short_audio_padding(m))
        original_audio = m.audio_tower.forward
        def audio_forward(tower, input_features, feature_lens, aftercnn_lens):
            return segmented_audio_forward(original_audio, input_features, feature_lens, aftercnn_lens)
        m.audio_tower.forward = MethodType(audio_forward, m.audio_tower)
        if [layer.self_attn.forward for layer in m.language_model.model.layers] != original_attention:
            raise ValueError('native attention was changed')
        return dict(result, attention='native_paged_kv', formal_admitted=False)
