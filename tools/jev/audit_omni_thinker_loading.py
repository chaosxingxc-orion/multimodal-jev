#!/usr/bin/env python3
"""Inspect local tensor headers and actual HF meta model; never load weight payloads."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
import math
from pathlib import Path
import struct


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(model):
    index=json.loads((model/'model.safetensors.index.json').read_text())
    tensors={};headers={};groups={};layers=Counter()
    for shard in sorted(set(index['weight_map'].values())):
        path=model/shard
        if Path(shard).name!=shard:raise ValueError('local flat checkpoint shard required')
        with path.open('rb') as f:
            size=struct.unpack('<Q',f.read(8))[0]
            if size>64*1024*1024:raise ValueError('unexpected tensor header size')
            raw=f.read(size);header=json.loads(raw)
        headers[shard]=dict(header_sha256=hashlib.sha256(raw).hexdigest(),file_bytes=path.stat().st_size)
        end=0
        for name,value in sorted(((k,v) for k,v in header.items() if k!='__metadata__'),
                                 key=lambda item:item[1]['data_offsets'][0]):
            start,stop=value['data_offsets']
            if name in tensors or index['weight_map'].get(name)!=shard or start!=end or stop<start:
                raise ValueError('duplicate, unmapped or noncontiguous tensor')
            end=stop;count=math.prod(value['shape']);byte_count=stop-start
            item=dict(shape=value['shape'],dtype=value['dtype'],parameters=count,bytes=byte_count)
            tensors[name]=item
            group='.'.join(name.split('.')[:2]) if name.startswith('thinker.') else name.split('.')[0]
            agg=groups.setdefault(group,Counter());agg.update(parameters=count,bytes=byte_count,tensors=1)
            if name.startswith('thinker.model.layers.'):layers[int(name.split('.')[3])]+=byte_count
        if end+8+size!=path.stat().st_size:raise ValueError('shard extent differs from header')
    if set(tensors)!=set(index['weight_map']):raise ValueError('checkpoint index coverage differs')
    if sum(t['bytes'] for t in tensors.values())!=index['metadata']['total_size']:
        raise ValueError('checkpoint total bytes differ')
    return tensors,dict(index_sha256=sha(model/'model.safetensors.index.json'),
        config_sha256=sha(model/'config.json'),headers=headers,groups={k:dict(v) for k,v in groups.items()},
        layer_bytes=dict(sorted(layers.items())))


def run(model):
    tensors,report=inventory(model)
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration
    config=AutoConfig.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    with torch.device('meta'):
        thinker=Qwen3OmniMoeThinkerForConditionalGeneration(config.thinker_config)
    assert all(p.is_meta for p in thinker.parameters())
    actual={name:list(p.shape) for name,p in thinker.named_parameters(remove_duplicate=False)}
    expected={name.removeprefix('thinker.'):v['shape'] for name,v in tensors.items() if name.startswith('thinker.')}
    missing=sorted(actual.keys()-expected.keys());extra=sorted(expected.keys()-actual.keys())
    mismatches={k:dict(meta=actual[k],checkpoint=expected[k]) for k in actual.keys()&expected.keys()
                if actual[k]!=expected[k]}
    # Header-only reproduction of documented expert stack/concat shapes.
    # This does not execute HF conversion or validate tensor values/order.
    mapped=dict(expected);text=config.thinker_config.text_config
    for layer in range(text.num_hidden_layers):
        prefix=f'model.layers.{layer}.mlp.experts.'
        for suffix in ('gate_proj','up_proj','down_proj'):
            shapes=[mapped.pop(prefix+str(i)+'.'+suffix+'.weight') for i in range(text.num_experts)]
            expected_shape=([text.hidden_size,text.moe_intermediate_size] if suffix=='down_proj'
                            else [text.moe_intermediate_size,text.hidden_size])
            if any(s!=expected_shape for s in shapes):raise ValueError('expert shape differs')
        mapped[prefix+'gate_up_proj']=[text.num_experts,2*text.moe_intermediate_size,text.hidden_size]
        mapped[prefix+'down_proj']=[text.num_experts,text.hidden_size,text.moe_intermediate_size]
    mapped_equal=mapped==actual
    report.update(model=str(model),meta_parameter_count=sum(p.numel() for p in thinker.parameters()),
        meta_parameter_tensors=len(actual),checkpoint_thinker_tensors=len(expected),
        missing_checkpoint_keys=missing,extra_checkpoint_keys=extra,shape_mismatches=mismatches,
        exact_direct_key_shape_match=not(missing or extra or mismatches),
        header_expert_conversion_matches=mapped_equal,
        header_parameter_count=sum(math.prod(s) for s in expected.values()),
        hf_source_sha256=sha(inspect.getfile(Qwen3OmniMoeThinkerForConditionalGeneration)),
        torch_version=torch.__version__,weight_payload_read=False,weight_updates=0,
        scope='Actual local safetensors headers and fixed Transformers meta Thinker constructor. '
              'No weight payload integrity check, materialized model loading, audio forward, NPU or training.')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run(a.model)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('headers','missing_checkpoint_keys',
        'extra_checkpoint_keys','shape_mismatches','layer_bytes')},indent=2))
