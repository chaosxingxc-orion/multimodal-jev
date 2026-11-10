#!/usr/bin/env python3
"""Real HF conversion and MoE forward on synthetic expert weights; no training."""
import argparse
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path


def run():
    import torch
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.core_model_loading import dot_natural_key
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextExperts
    torch.set_num_threads(2);torch.manual_seed(17)
    count,hidden,intermediate=128,8,12
    # All 128 expert IDs, including decimal-width boundaries 9/10 and 99/100.
    old={f'model.layers.0.mlp.experts.{i}.{kind}_proj.weight':
         torch.randn((hidden,intermediate) if kind=='down' else (intermediate,hidden))*.1
         for i in reversed(range(count)) for kind in ('up','down','gate')}
    mapping=get_checkpoint_conversion_mapping('qwen3_omni_moe_thinker')
    assert len(mapping)==2
    converted={};conversions=[]
    for converter in mapping:
        target=None
        for name in sorted(old,key=dot_natural_key):
            renamed,pattern=converter.rename_source_key(name)
            if pattern is not None:
                if target is None:target=renamed
                assert target==renamed
                converter.add_tensor(renamed,name,pattern,old[name])
        assert target is not None
        converted.update(converter.convert(target))
        conversions.append(dict(sources=converter.source_patterns,targets=converter.target_patterns,
                                operations=[type(op).__name__ for op in converter.operations]))
    gate=converted['model.layers.0.mlp.experts.gate_up_proj']
    down=converted['model.layers.0.mlp.experts.down_proj']
    for i in range(count):
        prefix=f'model.layers.0.mlp.experts.{i}.'
        assert torch.equal(gate[i,:intermediate],old[prefix+'gate_proj.weight'])
        assert torch.equal(gate[i,intermediate:],old[prefix+'up_proj.weight'])
        assert torch.equal(down[i],old[prefix+'down_proj.weight'])
    config=Qwen3OmniMoeTextConfig(hidden_size=hidden,moe_intermediate_size=intermediate,
                                num_experts=count,num_experts_per_tok=2)
    experts=Qwen3OmniMoeThinkerTextExperts(config).eval()
    experts.load_state_dict({'gate_up_proj':gate,'down_proj':down},strict=True)
    inputs=torch.randn(count,hidden)
    routes=torch.stack((torch.arange(count),(torch.arange(count)+11)%count),dim=1)
    weights=torch.softmax(torch.randn(count,2),dim=-1)
    expected=torch.zeros_like(inputs)
    with torch.no_grad():
        for t in range(count):
            for k in range(2):
                prefix=f'model.layers.0.mlp.experts.{routes[t,k].item()}.'
                g=torch.nn.functional.linear(inputs[t],old[prefix+'gate_proj.weight'])
                u=torch.nn.functional.linear(inputs[t],old[prefix+'up_proj.weight'])
                expected[t]+=torch.nn.functional.linear(torch.nn.functional.silu(g)*u,
                    old[prefix+'down_proj.weight'])*weights[t,k]
        actual=experts(inputs,routes,weights)
        error=float((actual-expected).abs().max())
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        controls={}
        for name in ('expert_order','gate_up_order'):
            bad=gate.clone()
            if name=='expert_order':bad[[2,10]]=bad[[10,2]]
            else:bad=torch.cat((bad[:,intermediate:],bad[:,:intermediate]),dim=1)
            experts.load_state_dict({'gate_up_proj':bad,'down_proj':down},strict=True)
            diff=float((experts(inputs,routes,weights)-expected).abs().max())
            assert diff>1e-4
            controls[name]=diff
    paths={inspect.getfile(obj) for obj in (get_checkpoint_conversion_mapping,dot_natural_key,Qwen3OmniMoeThinkerTextExperts)}
    return dict(scope='Real fixed HF WeightConverter and CPU expert forward, synthetic weights with 128 '
        'experts and tiny hidden dimensions. No checkpoint payload, audio, from_pretrained, TP/EP, NPU or training.',
        seed=17,experts=count,source_tensors=len(old),converted_tensors=len(converted),
        exact_all_expert_values=True,forward_max_abs_error=error,atol=1e-6,rtol=1e-5,
        negative_control_max_abs_errors=controls,conversion_operations=conversions,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(paths)},
        weight_updates=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run()
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
