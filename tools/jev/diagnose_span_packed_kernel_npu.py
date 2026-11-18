#!/usr/bin/env python3
"""Locate first-layer BF16 changes for two identical audio requests."""
import argparse
import json
from pathlib import Path


def main():
    import torch
    import torch_npu
    from peft import PeftModel
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.baseline import atomic_json, file_hash
    from multimodal_jev.simple_prepared_inputs import prepared
    from multimodal_jev.span_audio_collate import forward_span_audio_batch
    from multimodal_jev.span_model import NativeAudioSpanModel
    from multimodal_jev.span_device_scope import validate_tp1
    from multimodal_jev.span_runtime_profile import configure_actor_profile
    from multimodal_jev.span_parity import compare_span_policies
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();c=json.loads(Path(args.config).read_text());validate_tp1(c)
    for name,h in c['pins'].items():
        if file_hash(name)!=h:raise ValueError('changed dependency: '+name)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.npu.set_device(0);configure_actor_profile(c['runtime_profile'])
    processor,base=load_runtime(c['model'],'npu:0','bfloat16')
    adapted=PeftModel.from_pretrained(base,c['adapter'],is_trainable=False,autocast_adapter_dtype=False).eval()
    # Patch only exact Linear leaves: base projections, LoRA A/B, and head.
    # PEFT wrappers keep their original add/scaling/dropout semantics.
    import types
    from multimodal_jev.span_fixed_shape_linear import fixed_shape_linear
    patched = []
    def fixed_forward(module, inputs):
        if inputs.ndim not in (2, 3):
            raise ValueError('diagnostic requires 2D or 3D Linear inputs')
        values = inputs[None] if inputs.ndim == 2 else inputs
        result = fixed_shape_linear(values, module.weight, module.bias,
                                    tile_rows=128, tile_batch=8)
        return result[0] if inputs.ndim == 2 else result
    for name, module in adapted.named_modules():
        if type(module) is torch.nn.Linear:
            if module.weight.dtype != torch.bfloat16:
                raise ValueError('all patched Linear weights must remain BF16: ' + name)
            module.forward = types.MethodType(fixed_forward, module)
            patched.append(dict(name=name, shape=list(module.weight.shape)))
    if not patched or not any('lora_A' in p['name'] for p in patched):
        raise ValueError('expected base and adapter Linear leaves')
    atomic_json(root/'patched-modules.json',dict(modules=patched,
        tile_rows=128,tile_batch=8,production_profile_changed=False))
    # Candidate packed TND causal attention: one kernel call per layer.
    import transformers.models.qwen3.modeling_qwen3 as qwen3
    active_lengths = None
    attention_tensors = {}
    attention_calls = []
    def independent_attention(module, query, key, value, attention_mask,
                              scaling, dropout=0.0, **kwargs):
        if active_lengths is None or len(active_lengths) != query.shape[0]:
            raise ValueError('explicit full-prefill lengths required')
        if dropout != 0.0:
            raise ValueError('deterministic full prefill only')
        from multimodal_jev.span_packed_attention import packed_causal_attention
        attention_calls.append(dict(batch=query.shape[0],lengths=list(active_lengths)))
        output = packed_causal_attention(query, key, value, active_lengths, scaling=scaling)
        if module.layer_idx == 0:
            attention_tensors[phase[0]] = dict(
                q=query.detach().clone(), k=key.detach().clone(), v=value.detach().clone(),
                output=output.detach().clone(), lengths=list(active_lengths), scale=scaling)
        return output, None
    qwen3.eager_attention_forward = independent_attention
    model = NativeAudioSpanModel(adapted.get_base_model()).eval()
    def set_lengths(module, args, kwargs):
        nonlocal active_lengths
        active_lengths = [int(n) for n in kwargs['attention_mask'].sum(-1).cpu().tolist()]
    length_hook = model.backbone.register_forward_pre_hook(set_lengths, with_kwargs=True)

    artifact=json.loads(Path(c['inputs']).read_text())
    rows=sorted(list(prepared(c,artifact,processor=processor)),key=lambda r:(int(r[1]['input_features_mask'].sum()),r[0]['id']))
    row=rows[0];item=dict(inputs=row[1],positions=row[2],candidates=row[3]);other=dict(inputs=rows[-1][1],positions=rows[-1][2],candidates=rows[-1][3])
    layer=model.backbone.language_model.layers[0]
    modules={'input_norm':layer.input_layernorm,'q_proj':layer.self_attn.q_proj,
        'k_proj':layer.self_attn.k_proj,'q_norm':layer.self_attn.q_norm,'k_norm':layer.self_attn.k_norm,'v_proj':layer.self_attn.v_proj,
        'o_proj':layer.self_attn.o_proj,'post_norm':layer.post_attention_layernorm,
        'gate_proj':layer.mlp.gate_proj,'down_proj':layer.mlp.down_proj,
        'final_norm':model.backbone.language_model.norm,'head':model.lm_head}
    reference={};comparisons={};phase=['single'];handles=[]
    def capture(name):
        def hook(module,args,out):
            if phase[0]=='single':
                reference[name]=(args[0].detach().cpu(),out.detach().cpu())
            else:
                values=[]
                for index in range(0,8,2):
                    original_input,original_output=reference[name]
                    current_input=args[0][index:index+1,:original_input.shape[1]].detach().cpu()
                    current_output=out[index:index+1,:original_output.shape[1]].detach().cpu()
                    assert original_input.shape==current_input.shape and original_output.shape==current_output.shape
                    values.append(dict(input_max_abs=float((original_input.float()-current_input.float()).abs().max()),
                        output_max_abs=float((original_output.float()-current_output.float()).abs().max()),
                        output_changed_elements=int((original_output!=current_output).sum()),
                        output_elements=original_output.numel()))
                comparisons[name]=values
        return hook
    for name,module in modules.items():handles.append(module.register_forward_hook(capture(name)))
    options=dict(pad_token_id=processor.tokenizer.pad_token_id,audio_token_id=base.config.audio_token_id,
        audio_chunk_size=base.config.audio_config.n_window*2,audio_infer_window=base.config.audio_config.n_window_infer)
    with torch.inference_mode():
        inputs={k:v.to('npu:0',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in row[1].items()}
        ref=[v[0].cpu().tolist() for v in model(inputs,row[2].to('npu:0'),row[3])[1]]
        phase[0]='batch'
        out=forward_span_audio_batch(model,[item,other] * 4,device='npu:0',dtype=torch.bfloat16,**options)
        probs=[[v[0].cpu().tolist() for v in result] for result in out[::2]]
    for handle in handles:handle.remove()
    from multimodal_jev.span_packed_attention import packed_causal_attention
    single=attention_tensors['single'];batch=attention_tensors['batch']
    n=single['lengths'][0];kernel=[]
    with torch.inference_mode():
        for index in range(0,8,2):
            tensors=[batch[k][index:index+1,:,:n].contiguous() for k in ('q','k','v')]
            alone=packed_causal_attention(*tensors,[n],scaling=batch['scale'])
            kernel.append(dict(index=index,
                qkv_errors={k:float((single[k].float()-t.float()).abs().max())
                            for k,t in zip(('q','k','v'),tensors)},
                isolated_to_reference=float((alone.float()-single['output'].float()).abs().max()),
                isolated_to_packed=float((alone.float()-batch['output'][index:index+1,:n].float()).abs().max())))
    atomic_json(root/'result.json',dict(completed=True,request_id=row[4],request_identity=row[5],
        layer_comparisons=comparisons,isolated_kernel=kernel,
        reference=ref,batched=probs,parity=[compare_span_policies(ref,v) for v in probs],
        scope='First-layer post-rotary QKV and identical tensor isolated vs packed attention; no training or throughput admission.'))


if __name__=='__main__':main()
