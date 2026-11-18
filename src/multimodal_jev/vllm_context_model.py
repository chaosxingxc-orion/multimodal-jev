"""Candidate fixed-policy Qwen3-ASR pooling model with a contextual head.

Explicit registration only. TP1, no cache/chunking/quantization. Startup worker
must finish profiling before serving. Native LoRA is supplied by the owner;
engine execution/parity and atomic policy switching are not yet admitted.
"""
import torch
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRForConditionalGeneration, Qwen3ASRMultiModalProcessor,
    Qwen3ASRProcessingInfo, Qwen3ASRDummyInputsBuilder,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from .context_decision_head import ContextDecisionHead
from .context_policy_artifacts import read_execution_bundle
from .vllm_context_pooler import ContextProbabilityPooler


def validate_context_runtime(config):
    if (config.parallel_config.tensor_parallel_size != 1 or
            config.parallel_config.pipeline_parallel_size != 1 or
            config.scheduler_config.enable_chunked_prefill or
            config.cache_config.enable_prefix_caching or config.quant_config is not None or
            config.model_config.dtype != torch.bfloat16 or not config.model_config.enforce_eager or
            config.lora_config is None):
        raise ValueError('context candidate requires BF16 eager TP1 PP1 native LoRA, no quant/cache/chunking')
    spec = getattr(config.model_config.hf_config,'jev_context_policy',None)
    if (type(spec) is not dict or set(spec) != {'snapshot','policy_id','contract','base_id','profiling_decisions'} or
            type(spec['profiling_decisions']) is not int or spec['profiling_decisions'] <= 0):
        raise ValueError('explicit authenticated joint policy specification required')
    return spec


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3ASRMultiModalProcessor, info=Qwen3ASRProcessingInfo,
    dummy_inputs=Qwen3ASRDummyInputsBuilder,
)
class Qwen3ASRForContextDecisions(Qwen3ASRForConditionalGeneration):
    is_pooling_model = True

    def __init__(self, *, vllm_config, prefix=''):
        spec = validate_context_runtime(vllm_config)
        bundle = read_execution_bundle(spec['snapshot'],expected_policy_id=spec['policy_id'],
            expected_contract=spec['contract'],expected_base_id=spec['base_id'])
        super().__init__(vllm_config=vllm_config,prefix=prefix)
        branch = bundle['contract']['branch']
        embedding = self.language_model.model.embed_tokens.weight
        if (embedding.dtype != torch.bfloat16 or embedding.shape[1] != branch['hidden_size'] or
                not 0 <= bundle['contract']['token']['id'] < embedding.shape[0]):
            raise ValueError('native BF16 embedding cannot hold the joint marker')
        head = ContextDecisionHead(**{k:branch[k] for k in
            ('hidden_size','num_heads','ffn_size','residual_scale')}).to(
                device=embedding.device,dtype=torch.bfloat16).eval()
        head.load_state_dict({k.removeprefix('decision_head.'):v for k,v in bundle['tensors'].items()
                              if k.startswith('decision_head.')},strict=True)
        self.pooler = ContextProbabilityPooler(head,bundle['contract']['token']['id'],
            bundle['policy_id'],spec['profiling_decisions'])
        self.pooler.memory_profiling = True
        # Plain CPU tensor, deliberately not a base checkpoint parameter.
        self._jev_marker_row = bundle['tensors']['decision_token_row']
        self._jev_base_weights_loaded = False
        self._jev_policy_id = bundle['policy_id']

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        # Base checkpoint loading can overwrite the spare embedding row.
        # Restore only after every native weight has been loaded.
        embedding = self.language_model.model.embed_tokens.weight
        with torch.no_grad():
            embedding[self.pooler.readout.decision_token_id].copy_(self._jev_marker_row)
        self._jev_base_weights_loaded = True
        # The head was authenticated/loaded separately from the base archive.
        loaded.update(n for n,_ in self.named_parameters() if n.startswith('pooler.'))
        return loaded


class ContextPolicyWorkerExtension:
    def replay_jev_context_hidden(self, expected_policy_id, source):
        model = self.get_model()
        if model._jev_policy_id != expected_policy_id or model.pooler.memory_profiling:
            raise ValueError('loaded serving policy required for hidden replay')
        value = torch.load(source, map_location='cpu', weights_only=True)
        if value['policy_id'] != expected_policy_id:
            raise ValueError('hidden capture policy differs')
        head = model.pooler.readout.head
        if any(not torch.equal(v.detach().cpu(),value['head'][k]) for k,v in head.state_dict().items()):
            raise ValueError('HF and vLLM head weights differ')
        device = next(head.parameters()).device
        with torch.no_grad():
            prob = head(value['hidden'].to(device),value['positions'].to(device),
                        value['inputs']['attention_mask'].to(device).bool()).cpu()
        return dict(probability=prob.tolist(),head_weights_exact=True,
                    hf_hidden_probability_error=float((prob-value['probability']).abs().max()))

    def capture_next_jev_context_readout(self, expected_policy_id, destination):
        from .context_hidden_capture import capture_next_readout
        model = self.get_model()
        if (model._jev_policy_id != expected_policy_id or
                model.pooler.memory_profiling or not model._jev_base_weights_loaded):
            raise ValueError('loaded serving policy required for diagnostic capture')
        capture_next_readout(model.pooler.readout, destination)
        return dict(policy_id=expected_policy_id, destination=destination)

    def finish_jev_context_startup(self, expected_policy_id):
        model = self.get_model()
        if (type(model).__name__ != 'Qwen3ASRForContextDecisions' or
                not model._jev_base_weights_loaded or model._jev_policy_id != expected_policy_id):
            raise ValueError('complete fixed context policy startup required')
        row = model.language_model.model.embed_tokens.weight[model.pooler.readout.decision_token_id]
        if not torch.equal(row.detach().cpu(),model._jev_marker_row):
            raise ValueError('decision marker overwritten during startup')
        model.pooler.memory_profiling = False
        return dict(policy_id=expected_policy_id,marker_exact=True,
                    head_dtype=str(next(model.pooler.readout.head.parameters()).dtype),
                    profiling=False,scope='Head/marker startup only; adapter activation needs request audit')


def register_context_model():
    from vllm import ModelRegistry
    ModelRegistry.register_model('Qwen3ASRForContextDecisions',
        'multimodal_jev.vllm_context_model:Qwen3ASRForContextDecisions')
