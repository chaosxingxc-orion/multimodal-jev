"""Candidate resident context engine; versioned separately from frozen Stage1.

The initialization is pinned to the fixed engine implementation. Only the
worker extension changes. All requests and switches share one process lock.
"""
import threading
import torch
from multimodal_jev.context_resident_rollout import FixedContextRollout
from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.context_runtime_profile import PROFILE,NATIVE_DIAGNOSTIC,install_rollout
from context_policy_activation import prepare_switch


class SwitchingContextRollout(FixedContextRollout):
    def __init__(self, config):
        from multimodal_jev.local_ipc import require_local_ipc
        from multimodal_jev.span_device_scope import validate_tp1
        from multimodal_jev.vllm_audio_features import enable_feature_passthrough
        from multimodal_jev.vllm_context_model import register_context_model
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        require_local_ipc()
        validate_tp1(config)
        if checkpoint_identity(config['model'])[0] != config['base_id']:
            raise ValueError('actual base differs from joint policy base')
        bundle = read_execution_bundle(config['snapshot'],expected_policy_id=config['policy_id'],
            expected_contract=config['contract'],expected_base_id=config['base_id'])
        verify_native_adapter(config['adapter'],bundle)
        profile = config['contract']['runtime_profile']
        if profile not in (NATIVE_DIAGNOSTIC, PROFILE):
            raise ValueError('explicit candidate native runtime identity required')
        self.bundle = bundle
        self.engine = None
        self.phase = 'starting'
        self.max_model_len = config['max_model_len']
        self.max_decisions = config['max_decisions']
        if (type(self.max_model_len) is not int or self.max_model_len < 1 or
                type(self.max_decisions) is not int or self.max_decisions < 1):
            raise ValueError('positive context and decision capacity required')
        self.lora = LoRARequest('context-'+bundle['policy_id'],1,str(config['adapter']))
        specs = bundle['contract']['lora']
        packed = sorted({n.replace('language_model.','language_model.model.',1).rsplit('.',1)[0]+'.qkv_proj'
                         for n in specs})
        register_context_model()
        enable_feature_passthrough()
        memory_options = {}
        if 'kv_cache_memory_bytes' in config:
            budget = config['kv_cache_memory_bytes']
            if type(budget) is not int or budget < 1:
                raise ValueError('positive explicit KV cache byte budget required')
            memory_options['kv_cache_memory_bytes'] = budget
        try:
            self.engine = LLM(model=config['model'],tokenizer=bundle['processor'],runner='pooling',
                hf_overrides=dict(architectures=['Qwen3ASRForContextDecisions'],head_dtype='model',
                    jev_context_policy=dict(snapshot=str(config['snapshot']),policy_id=bundle['policy_id'],
                        contract=bundle['contract'],base_id=config['base_id'],profiling_decisions=self.max_decisions)),
                pooler_config=dict(task='token_classify',tok_pooling_type='ALL',use_activation=False),
                dtype='bfloat16',max_model_len=self.max_model_len,gpu_memory_utilization=.4,
                tensor_parallel_size=1,enforce_eager=True,max_num_seqs=1,seed=317,
                worker_extension_cls='context_switch_worker.SwitchingContextWorker',
                enable_chunked_prefill=False,enable_prefix_caching=False,enable_mm_embeds=True,
                limit_mm_per_prompt={'audio':1},enable_lora=True,max_loras=1,max_cpu_loras=1,
                max_lora_rank=max(8,max(s['rank'] for s in specs.values())),
                lora_dtype=torch.bfloat16,lora_target_modules=packed,**memory_options)
            self.startup = self.engine.llm_engine.collective_rpc('finish_jev_context_startup',
                                                                  args=(bundle['policy_id'],))
            if (len(self.startup) != 1 or self.startup[0]['policy_id'] != bundle['policy_id'] or
                    self.startup[0]['head_dtype'] != 'torch.bfloat16' or self.startup[0]['profiling']):
                raise ValueError('worker context startup differs')
            self.runtime = (install_rollout(self.engine, profile) if profile == PROFILE
                            else dict(profile=profile, training_dispatch_allowed=False))
        except BaseException:
            self.phase = 'failed'
            self.close()
            raise
        self._transaction=threading.Lock()
        self._slot=1
        self.phase = 'ready'


    def switch(self,*,snapshot,policy_id,adapter):
        from vllm.lora.request import LoRARequest
        with self._transaction:
            if self.phase!='ready':raise ValueError('ready quiescent owner required')
            self.phase='switching'
            try:
                bundle=prepare_switch(self.bundle,snapshot=snapshot,policy_id=policy_id,
                    adapter=adapter,base_id=self.bundle['contract']['base_id'])
                self._slot+=1
                replies=self.engine.llm_engine.collective_rpc('switch_jev_context_policy',
                    args=(self.bundle['policy_id'],str(snapshot),policy_id,str(adapter),self._slot))
                if (len(replies)!=1 or replies[0]['policy_id']!=policy_id or replies[0]['slot']!=self._slot or
                        not replies[0]['head_marker_exact'] or replies[0]['native_loaded_tensors']!=112):
                    raise ValueError('worker joint activation receipt differs')
                self.bundle=bundle
                self.lora=LoRARequest('context-'+policy_id,self._slot,str(adapter))
                self.phase='ready'
                return replies[0]
            except BaseException:
                self.phase='failed';raise

    def score(self,inputs,positions,*,audio_token_id):
        with self._transaction:
            return super().score(inputs,positions,audio_token_id=audio_token_id)
