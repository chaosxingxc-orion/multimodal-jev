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

    def watch_input(self, module, name):
        self.handles.append(module.register_forward_pre_hook(
            lambda _module, args: self.capture(name, args[0])))

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
    attach_audio_trace(trace,base.model.audio_tower)
    trace.watch(base.model.multi_modal_projector.linear_1,'audio.proj1')
    trace.watch(base.model.multi_modal_projector.linear_2,'audio.proj2')
    trace.watch(base.model.multi_modal_projector, 'audio_projected')
    language = base.model.language_model
    trace.watch_norm_input(language.layers[0].input_layernorm)
    for i, layer in enumerate(language.layers):
        trace.watch(layer.input_layernorm, f'layer.{i}.input_norm')
        trace.watch(layer.post_attention_layernorm, f'layer.{i}.post_attention_norm')
    for key in ('q_proj','k_proj','v_proj'):
        trace.watch(getattr(language.layers[0].self_attn,key), 'layer.0.'+key)
    attach_language_projection_trace(trace, language.layers[0])
    trace.watch(language.layers[0].mlp,'layer.0.mlp')
    trace.watch(language.norm, 'final_norm')
    return trace


def attach_vllm_trace(model):
    trace = SpanTrace()
    attach_audio_trace(trace,model.audio_tower)
    trace.watch(model.audio_tower.proj1,'audio.proj1')
    trace.watch(model.audio_tower.proj2,'audio.proj2')
    trace.watch(model.audio_tower, 'audio_projected')
    language = model.language_model.model
    trace.watch_norm_input(language.layers[0].input_layernorm)
    for i, layer in enumerate(language.layers):
        trace.watch(layer.input_layernorm, f'layer.{i}.input_norm')
        trace.watch(layer.post_attention_layernorm, f'layer.{i}.post_attention_norm')
    trace.watch(language.layers[0].self_attn.qkv_proj, 'layer.0.qkv_proj')
    attach_language_projection_trace(trace, language.layers[0])
    trace.watch(language.layers[0].mlp,'layer.0.mlp')
    trace.watch(language.norm, 'final_norm')
    return trace


def attach_language_projection_trace(trace, layer):
    for module, name in ((layer.self_attn.o_proj, 'o_proj'), (layer.mlp.down_proj, 'down_proj')):
        trace.watch_input(module, 'layer.0.'+name+'.input')
        trace.watch(module, 'layer.0.'+name+'.output')


def rank_trace_path(path, rank, world_size):
    from pathlib import Path
    if type(rank) is not int or type(world_size) is not int or not 0 <= rank < world_size:
        raise ValueError('valid rank and world size required')
    path = Path(path)
    return path.with_name(path.stem+f'.rank-{rank}-of-{world_size}'+path.suffix)


def compare_tp4_traces(actor, ranks):
    """Compare replicas separately and reassemble only known column shards.

    This ASR diagnostic requires four unique KV shards; replicated KV heads
    are intentionally rejected. All tensor manipulation is offline reporting.
    """
    if len(ranks)!=4 or any(set(r)!=set(ranks[0]) for r in ranks):
        raise ValueError('four aligned trace dictionaries required')
    split_names=('layer.0.o_proj.input','layer.0.down_proj.input')
    rebuilt=dict(ranks[0])
    for name in split_names:
        if any(r[name].shape[-1]*4!=actor[name].shape[-1] for r in ranks):
            raise ValueError('four column shards required: '+name)
        rebuilt[name]=torch.cat([r[name] for r in ranks],-1)
    widths=[actor['layer.0.'+key].shape[-1] for key in ('q_proj','k_proj','v_proj')]
    if any(w%4 for w in widths):
        raise ValueError('four unique QKV head shards required')
    pieces=[r['layer.0.qkv_proj'].split([w//4 for w in widths],-1) for r in ranks]
    rebuilt['layer.0.qkv_proj']=torch.cat([torch.cat([p[i] for p in pieces],-1) for i in range(3)],-1)
    replicas={}
    for rank,r in enumerate(ranks):
        replica_view=dict(rebuilt)
        for name in r:
            if name not in (*split_names,'layer.0.qkv_proj'):
                replica_view[name]=r[name]
        replicas[str(rank)]=compare_traces(actor,replica_view)
    return dict(reconstructed=compare_traces(actor,rebuilt),replicas=replicas,
                scope='Offline comparison only; QKV and input-column shards reconstructed; replicas checked separately')


def attach_audio_trace(trace,tower):
    def conv_input(module,args):
        value=args[0]
        trace.capture('audio.conv2d1.input',value)
        trace.capture('audio.conv2d1.input_stride',torch.tensor(value.stride()))
        trace.capture('audio.conv2d1.weight',module.weight)
        trace.capture('audio.conv2d1.weight_stride',torch.tensor(module.weight.stride()))
        trace.capture('audio.conv2d1.bias',module.bias)
    trace.handles.append(tower.conv2d1.register_forward_pre_hook(conv_input))
    for name in ('conv2d1','conv2d2','conv2d3','conv_out','ln_post'):
        trace.watch(getattr(tower,name),'audio.'+name)
    for i,layer in enumerate(tower.layers):
        trace.watch(layer,f'audio.layer.{i}.output')
    first=tower.layers[0]
    trace.watch(first.self_attn_layer_norm,'audio.layer.0.input_norm')
    trace.watch(first.self_attn,'audio.layer.0.attention')
    trace.watch(first.final_layer_norm,'audio.layer.0.post_attention_norm')
    for key in ('fc1','fc2'):
        trace.watch(getattr(first,key),'audio.layer.0.'+key)
    attn=first.self_attn
    keys=('qkv_proj',) if hasattr(attn,'qkv_proj') else ('q_proj','k_proj','v_proj')
    for key in keys:
        trace.watch(getattr(attn,key),'audio.layer.0.'+key)


def compare_traces(actor, rollout):
    rollout = dict(rollout)
    for prefix in ('layer.0.','audio.layer.0.'):
        if prefix+'qkv_proj' in rollout:
            names = [prefix+key for key in ('q_proj','k_proj','v_proj')]
            parts = rollout.pop(prefix+'qkv_proj').split([actor[k].shape[-1] for k in names], -1)
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
