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
    model=NativeAudioSpanModel(adapted.get_base_model()).eval()
    artifact=json.loads(Path(c['inputs']).read_text())
    rows=sorted(list(prepared(c,artifact,processor=processor)),key=lambda r:(int(r[1]['input_features_mask'].sum()),r[0]['id']))
    row=rows[len(rows)//2];item=dict(inputs=row[1],positions=row[2],candidates=row[3])
    layer=model.backbone.language_model.layers[0]
    modules={'input_norm':layer.input_layernorm,'q_proj':layer.self_attn.q_proj,
        'k_proj':layer.self_attn.k_proj,'v_proj':layer.self_attn.v_proj,
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
                for index in range(2):
                    original_input,original_output=reference[name]
                    current_input=args[0][index:index+1].detach().cpu()
                    current_output=out[index:index+1].detach().cpu()
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
        out=forward_span_audio_batch(model,[item,item],device='npu:0',dtype=torch.bfloat16,**options)
        probs=[[v[0].cpu().tolist() for v in result] for result in out]
    for handle in handles:handle.remove()
    # Isolated identical BF16 linear inputs: control shape alone, without audio drift.
    projection={}
    with torch.inference_mode():
        for name in ['q_proj','k_proj','v_proj','gate_proj','down_proj']:
            module=modules[name];x=reference[name][0].to('npu:0')
            single=module(x).cpu();batched=module(x.repeat(2,1,1)).cpu()
            projection[name]=[float((single.float()-batched[i:i+1].float()).abs().max()) for i in range(2)]
    atomic_json(root/'result.json',dict(completed=True,request_id=row[4],request_identity=row[5],
        layer_comparisons=comparisons,isolated_identical_projection_errors=projection,
        reference=ref,batched=probs,parity=[compare_span_policies(ref,v) for v in probs],
        scope='Identical audio input pair; first-layer and final readout BF16 hook diagnostics plus isolated projection shapes; no fix or training admission.'))


if __name__=='__main__':main()
