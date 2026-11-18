#!/usr/bin/env python3
"""Compare fixed-tile BF16 projections on captured real first-layer inputs.

Tests batched matrix multiply without changing model parameter/input precision.
No replacement is installed in the model; this is only a kernel-shape probe.
"""
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
    from multimodal_jev.span_model import NativeAudioSpanModel
    from multimodal_jev.span_fixed_shape_linear import fixed_shape_linear
    from multimodal_jev.span_device_scope import validate_tp1
    from multimodal_jev.span_runtime_profile import configure_actor_profile
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();c=json.loads(Path(a.config).read_text());validate_tp1(c)
    for name,h in c['pins'].items():
        if file_hash(name)!=h:raise ValueError('dependency changed: '+name)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.npu.set_device(0);configure_actor_profile(c['runtime_profile'])
    processor,base=load_runtime(c['model'],'npu:0','bfloat16')
    adapted=PeftModel.from_pretrained(base,c['adapter'],is_trainable=False,autocast_adapter_dtype=False).eval()
    model=NativeAudioSpanModel(adapted.get_base_model()).eval()
    rows=sorted(list(prepared(c,json.loads(Path(c['inputs']).read_text()),processor=processor)),
                key=lambda r:(int(r[1]['input_features_mask'].sum()),r[0]['id']))
    row=rows[len(rows)//2];layer=model.backbone.language_model.layers[0]
    modules=dict(q_proj=layer.self_attn.q_proj,k_proj=layer.self_attn.k_proj,
                 gate_proj=layer.mlp.gate_proj,down_proj=layer.mlp.down_proj)
    captured={};hooks=[]
    def capture(name):
        return lambda module,args:captured.__setitem__(name,args[0].detach())
    for name,module in modules.items():hooks.append(module.register_forward_pre_hook(capture(name)))
    with torch.inference_mode():
        inputs={k:v.to('npu:0',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in row[1].items()}
        model(inputs,row[2].to('npu:0'),row[3])
    for hook in hooks:hook.remove()
    results={}
    with torch.inference_mode():
        for name,module in modules.items():
            x=captured[name];linear=getattr(module,'base_layer',module)
            w=linear.weight;b=linear.bias
            assert x.dtype==w.dtype==torch.bfloat16
            def mm(z):return torch.nn.functional.linear(z,w,b)
            def bmm(z):
                out=torch.bmm(z,w.t().unsqueeze(0).expand(z.shape[0],-1,-1))
                return out if b is None else out+b
            reference=mm(x);cases=[]
            # Verify neighbor/slot/sequence-padding invariance, not just repeats.
            for tile_rows,tile_batch in [(128,8),(256,8)]:
                fixed=lambda z:fixed_shape_linear(z,w,b,tile_rows=tile_rows,tile_batch=tile_batch)
                ref=fixed(x)
                for batch in [1,2,4,8,16,32]:
                    z=x.expand(batch,-1,-1).contiguous().clone()
                    if batch>1:z[1:]*=.5
                    order=list(range(1,batch))+[0]
                    z=z[order];index=batch-1
                    for extra in [0,73]:
                        padded=torch.nn.functional.pad(z,(0,0,0,extra))
                        y=fixed(padded)[index:index+1,:x.shape[1]]
                        cases.append(dict(batch=batch,extra_tokens=extra,tile_rows=tile_rows,tile_batch=tile_batch,
                            fixed_to_single_max_abs=float((y.float()-ref.float()).abs().max()),
                            fixed_to_original_max_abs=float((y.float()-reference.float()).abs().max())))
                        del padded,y
                    del z
            results[name]=dict(input_shape=list(x.shape),weight_shape=list(w.shape),cases=cases)
            atomic_json(root/(name+'.json'),results[name])
    atomic_json(root/'result.json',dict(completed=True,request_id=row[4],results=results,
        training_promotion=False,scope='BF16 fixed-tile Linear on real captured activations, altered neighbors, relocated target and extra padding; excludes LoRA add, full model, backward, audio convolutions and timing.'))


if __name__=='__main__':main()
