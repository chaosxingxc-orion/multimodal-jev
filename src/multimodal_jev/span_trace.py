"""Read-only eager forward traces for locating native audio backend drift.

Trace tensors stay in local experiment storage and never enter model inputs.
HF and vLLM residual implementations differ: compare normalized layer inputs,
not the uncombined vLLM residual tuple against an HF layer output.
"""
import torch


class SpanTrace:
    def __init__(self):
        self.tensors = {}
        self.handles = []

    def capture(self, name, value):
        if isinstance(value, tuple):
            value = value[0]
        if not torch.is_tensor(value):
            raise ValueError('trace requires tensor: ' + name)
        if name in self.tensors:
            raise ValueError('unexpected repeated trace point: ' + name)
        # Normalize the singleton HF batch axis only, preserve all tokens.
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        self.tensors[name] = value.detach().cpu().clone()

    def watch(self, module, name):
        self.handles.append(module.register_forward_hook(
            lambda _module, _inputs, output: self.capture(name, output)))

    def watch_norm_input(self, module):
        def record(_module, args):
            self.capture('layer.0.pre_norm',args[0])
            self.capture('layer.0.norm_weight',_module.weight)
        self.handles.append(module.register_forward_pre_hook(record))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def attach_hf_trace(base):
    trace = SpanTrace()
    trace.watch(base.model.multi_modal_projector, 'audio_projected')
    language = base.model.language_model
    trace.watch_norm_input(language.layers[0].input_layernorm)
    for i, layer in enumerate(language.layers):
        trace.watch(layer.input_layernorm, f'layer.{i}.input_norm')
        trace.watch(layer.post_attention_layernorm, f'layer.{i}.post_attention_norm')
    for key in ('q_proj','k_proj','v_proj'):
        trace.watch(getattr(language.layers[0].self_attn,key), 'layer.0.'+key)
    trace.watch(language.norm, 'final_norm')
    return trace


def attach_vllm_trace(model):
    trace = SpanTrace()
    trace.watch(model.audio_tower, 'audio_projected')
    language = model.language_model.model
    trace.watch_norm_input(language.layers[0].input_layernorm)
    for i, layer in enumerate(language.layers):
        trace.watch(layer.input_layernorm, f'layer.{i}.input_norm')
        trace.watch(layer.post_attention_layernorm, f'layer.{i}.post_attention_norm')
    trace.watch(language.layers[0].self_attn.qkv_proj, 'layer.0.qkv_proj')
    trace.watch(language.norm, 'final_norm')
    return trace


def compare_traces(actor, rollout):
    rollout = dict(rollout)
    if 'layer.0.qkv_proj' in rollout:
        names = ['layer.0.'+key for key in ('q_proj','k_proj','v_proj')]
        parts = rollout.pop('layer.0.qkv_proj').split([actor[k].shape[-1] for k in names], -1)
        rollout.update(zip(names,parts))
    if set(actor) != set(rollout):
        raise ValueError('trace points differ')
    report = {}
    for name, x in actor.items():
        y = rollout[name]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError('trace shape/dtype differs: '+name)
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('nonfinite trace')
        delta = (x.float()-y.float()).abs()
        report[name] = dict(shape=list(x.shape),dtype=str(x.dtype),
                            maximum=float(delta.max()),mean=float(delta.mean()),
                            unequal_fraction=float((x!=y).float().mean()),
                            rms=float(delta.square().mean().sqrt()))
    return report


def run_hf_trace(config, batch, positions, mapping, output):
    from peft import PeftModel
    from .asr import load_runtime
    from .bf16_lora import cast_lora_bf16
    from .span_model import NativeAudioSpanModel
    import gc
    _,base = load_runtime(config['model'],'npu:0','bfloat16')
    adapted = PeftModel.from_pretrained(base,config['adapter'],is_trainable=False,
                                       autocast_adapter_dtype=False).eval()
    cast_lora_bf16(adapted)
    model = NativeAudioSpanModel(adapted.get_base_model()).eval()
    inputs = {k:v.to('npu:0',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
              for k,v in batch.items()}
    trace = attach_hf_trace(base)
    try:
        with torch.inference_mode():
            _, probabilities = model(inputs,positions.to('npu:0'),[m['token_ids'] for m in mapping])
        torch.save(trace.tensors,output/'hf-trace.pt')
        result = [p.cpu().tolist() for p in probabilities]
    finally:
        trace.close()
    del trace, model, adapted, base, inputs, probabilities, _
    gc.collect();torch.npu.empty_cache()
    return result
