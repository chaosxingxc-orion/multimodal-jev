"""TP1 Q/V LoRA using the same BF16 operation boundaries as the actor."""
from types import MethodType

import torch

from multimodal_jev.span_fixed_rollout_projection import project
from tools.reports.context_bounded_rollout_worker import BoundedAcousticWorker


def actor_order_qkv(module, inputs, bias=None):
    weights = module.base_layer.weight.split(module.output_slices, dim=0)
    biases = (None,) * len(weights) if bias is None else bias.split(module.output_slices, dim=0)
    outputs = []
    for weight, part_bias, adapter in zip(weights, biases, module._jev_actor_order_lora, strict=True):
        value = project(inputs, weight, part_bias)
        if adapter is not None:
            a, b, scale = adapter
            value = value + project(project(inputs, a), b) * scale
        outputs.append(value)
    return torch.cat(outputs, dim=-1)


class ActorOrderAcousticWorker(BoundedAcousticWorker):
    def _bind_actor_order_lora(self):
        model = self.get_model()
        activation = self._jev_activation
        bundle = activation.current
        if (activation.phase != 'ready' or model._jev_policy_id != bundle['policy_id']
                or model.vllm_config.parallel_config.tensor_parallel_size != 1):
            raise ValueError('authenticated active TP1 policy required')
        entries = []
        names = set()
        for index, layer in enumerate(model.language_model.model.layers):
            wrapper = layer.self_attn.qkv_proj
            prefix = f'language_model.layers.{index}.self_attn.'
            if len(wrapper.output_slices) != 3:
                raise ValueError('complete packed Q/K/V geometry required')
            adapters = []
            for suffix in ('q_proj', 'k_proj', 'v_proj'):
                if suffix == 'k_proj':
                    adapters.append(None)
                    continue
                name = prefix + suffix
                names.add(name)
                spec = bundle['contract']['lora'][name]
                weights = tuple(bundle['tensors'][f'backbone.{name}.lora_{k}.default.weight'].to(
                    wrapper.base_layer.weight.device) for k in ('A', 'B'))
                adapters.append((*weights, spec['scaling']))
            entries.append((wrapper, tuple(adapters)))
        if names != set(bundle['contract']['lora']):
            raise ValueError('complete Q/V adapter coverage required')
        for wrapper, adapters in entries:
            wrapper._jev_actor_order_lora = adapters
            wrapper.apply = MethodType(actor_order_qkv, wrapper)
        return dict(arithmetic='actor_order_bf16_qv_v1', lora_layers=len(entries))

    def install_jev_dual_candidate(self, expected_policy_id):
        result = super().install_jev_dual_candidate(expected_policy_id)
        return dict(result, **self._bind_actor_order_lora())

    def switch_jev_context_policy(self, expected_policy_id, snapshot, policy_id, adapter, slot):
        result = super().switch_jev_context_policy(expected_policy_id, snapshot, policy_id, adapter, slot)
        try:
            return dict(result, **self._bind_actor_order_lora())
        except BaseException:
            model = self.get_model()
            self._jev_activation.phase = 'failed'
            model._jev_policy_id = model.pooler.readout.policy_id = 'activation-failed'
            raise
