"""Single authenticated context policy served by a native vLLM engine.

Candidate owner: no hot switching, labels, text decoding or implicit fallback.
The caller compares actual audio probabilities before admitting training.
"""
import hashlib
import json

import torch

from .context_policy_artifacts import read_execution_bundle, verify_native_adapter
from .policy_identity import checkpoint_identity
from .span_request_identity import tensor_identity
from .context_runtime_profile import PROFILE, NATIVE_DIAGNOSTIC, install_rollout


class FixedContextRollout:
    architecture = 'Qwen3ASRForContextDecisions'
    worker_extension = 'multimodal_jev.context_worker_extension.ContextBootstrapWorkerExtension'
    profiles = (NATIVE_DIAGNOSTIC, PROFILE)
    return_logits = False
    modality_limits = {'audio': 1}

    def prepare_adapter(self, config, bundle):
        verify_native_adapter(config['adapter'], bundle)
        return config['adapter']

    def register_model(self):
        from .vllm_audio_features import enable_feature_passthrough
        from .vllm_context_model import register_context_model
        register_context_model()
        enable_feature_passthrough()

    def packed_targets(self, specs):
        return sorted({n.replace('language_model.','language_model.model.',1).rsplit('.',1)[0]+'.qkv_proj'
                       for n in specs})

    def runtime_receipt(self, profile):
        return (install_rollout(self.engine, profile) if profile == PROFILE
                else dict(profile=profile, training_dispatch_allowed=False))

    def __init__(self, config):
        from .local_ipc import require_local_ipc
        from .span_device_scope import validate_tp1
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        require_local_ipc()
        validate_tp1(config)
        if checkpoint_identity(config['model'])[0] != config['base_id']:
            raise ValueError('actual base differs from joint policy base')
        bundle = read_execution_bundle(config['snapshot'],expected_policy_id=config['policy_id'],
            expected_contract=config['contract'],expected_base_id=config['base_id'])
        profile = config['contract']['runtime_profile']
        if profile not in self.profiles:
            raise ValueError('explicit candidate native runtime identity required')
        self.bundle = bundle
        self.engine = None
        self.phase = 'starting'
        self.max_model_len = config['max_model_len']
        self.max_decisions = config['max_decisions']
        if (type(self.max_model_len) is not int or self.max_model_len < 1 or
                type(self.max_decisions) is not int or self.max_decisions < 1):
            raise ValueError('positive context and decision capacity required')
        adapter = self.prepare_adapter(config, bundle)
        self.lora = LoRARequest('context-'+bundle['policy_id'],1,str(adapter))
        specs = bundle['contract']['lora']
        self.register_model()
        try:
            self.engine = LLM(model=config['model'],tokenizer=bundle['processor'],runner='pooling',
                hf_overrides=dict(architectures=[self.architecture],head_dtype='model',
                    jev_context_policy=dict(snapshot=str(config['snapshot']),policy_id=bundle['policy_id'],
                        contract=bundle['contract'],base_id=config['base_id'],profiling_decisions=self.max_decisions)),
                pooler_config=dict(task='token_classify',tok_pooling_type='ALL',use_activation=False),
                dtype='bfloat16',max_model_len=self.max_model_len,gpu_memory_utilization=.4,
                tensor_parallel_size=1,enforce_eager=True,max_num_seqs=1,seed=317,
                worker_extension_cls=self.worker_extension,
                enable_chunked_prefill=False,enable_prefix_caching=False,enable_mm_embeds=True,
                limit_mm_per_prompt=self.modality_limits,enable_lora=True,max_loras=1,max_cpu_loras=1,
                max_lora_rank=max(8,max(s['rank'] for s in specs.values())),
                lora_dtype=torch.bfloat16,lora_target_modules=self.packed_targets(specs))
            self.startup = self.engine.llm_engine.collective_rpc('finish_jev_context_startup',
                                                                  args=(bundle['policy_id'],))
            if (len(self.startup) != 1 or self.startup[0]['policy_id'] != bundle['policy_id'] or
                    self.startup[0]['head_dtype'] != 'torch.bfloat16' or self.startup[0]['profiling']):
                raise ValueError('worker context startup differs')
            self.runtime = self.runtime_receipt(profile)
        except BaseException:
            self.phase = 'failed'
            self.close()
            raise
        self.phase = 'ready'

    def score(self, inputs, positions, *, audio_token_id):
        from vllm import PoolingParams
        if self.phase != 'ready':
            raise ValueError('ready fixed context engine required')
        self.phase = 'evaluating'
        try:
            result = self._score(inputs,positions,audio_token_id,PoolingParams)
        except BaseException:
            self.phase = 'failed'
            raise
        self.phase = 'ready'
        return result

    def _score(self, inputs, positions, audio_token_id, params_class):
        if set(inputs) != {'input_ids','attention_mask','input_features','input_features_mask'}:
            raise ValueError('explicit label-free complete audio inputs required')
        ids = inputs['input_ids']
        if (ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.long or
                inputs['attention_mask'].shape != ids.shape or not (inputs['attention_mask'] == 1).all()):
            raise ValueError('one unpadded complete request required')
        ids = ids[0].tolist()
        if self.bundle['contract']['token']['id']==151706 and 151705 in ids:
            raise ValueError('timestamp reservation cannot appear in decision input')
        if (not ids or len(ids) > self.max_model_len or positions.dtype != torch.long or
                positions.ndim != 2 or positions.shape[0] != 1 or
                not 0 < positions.shape[1] <= self.max_decisions):
            raise ValueError('complete prompt and bounded marker count required')
        markers = [i for i,t in enumerate(ids) if t == self.bundle['contract']['token']['id']]
        if positions[0].tolist() != markers:
            raise ValueError('marker coverage differs')
        at = [i for i,t in enumerate(ids) if t == audio_token_id]
        if not at or not markers or at[-1] >= markers[0] or at != list(range(at[0],at[-1]+1)):
            raise ValueError('one contiguous expanded audio region required')
        features, mask = inputs['input_features'],inputs['input_features_mask']
        if (features.ndim != 3 or features.shape[0] != 1 or not torch.isfinite(features).all() or
                mask.shape != (1,features.shape[-1]) or not ((mask == 0) | (mask == 1)).all()):
            raise ValueError('finite features with aligned binary mask required')
        frames = int(mask.sum())
        if frames < 1 or not mask[0,:frames].bool().all() or mask[0,frames:].bool().any():
            raise ValueError('contiguous valid audio frames required')
        identity = dict(policy_id=self.bundle['policy_id'],positions=markers,
                        inputs={k:tensor_identity(v) for k,v in sorted(inputs.items())})
        rid = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        prompt = dict(prompt_token_ids=ids[:at[0]]+[audio_token_id]+ids[at[-1]+1:],
            multi_modal_data={'audio':dict(input_audio_features=features[0,:,:frames].contiguous(),
                audio_feature_lengths=torch.tensor([frames]),feature_attention_mask=mask[:,:frames].contiguous())},
            multi_modal_uuids={'audio':[rid]})
        params = params_class(use_activation=False,extra_kwargs=dict(policy_id=self.bundle['policy_id'],
            marker_positions=markers,expected_input_ids=ids))
        result = self.engine.encode([prompt],pooling_params=params,pooling_task='token_classify',
                                    use_tqdm=False,lora_request=self.lora)[0]
        values = result.outputs.data
        if (result.prompt_token_ids != ids or values.shape != (len(markers),1) or
                not torch.isfinite(values).all()
                or (not self.return_logits and ((values < 0) | (values > 1)).any())):
            raise ValueError('complete expanded prompt and finite readouts required')
        values = values[:,0].float()
        result = dict(request_id=rid,request_identity=identity, policy_id=self.bundle['policy_id'])
        if self.return_logits:
            result['logits'] = values.tolist()
            values = values.sigmoid()
        result['prob'] = values.tolist()
        return result

    def close(self):
        if self.engine is not None:
            self.engine.llm_engine.engine_core.shutdown(
                timeout=getattr(self, 'engine_shutdown_timeout_seconds', 30.))
            self.engine = None
        self.phase = 'closed'
