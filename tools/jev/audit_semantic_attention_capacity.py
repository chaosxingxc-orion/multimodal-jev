#!/usr/bin/env python3
"""Account for explicit eager attention tensors, not measured NPU memory."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def tensor_budget(length, heads=16, batch=1, capacity_gib=64):
    if any(type(x) is not int or x<1 for x in (length, heads, batch, capacity_gib)):
        raise ValueError('positive integer dimensions and capacity required')
    elements=batch*heads*length*length
    # eager_causal_attention keeps `scores` alive across FP32 softmax and cast.
    # Count only the BF16 score and FP32 softmax output, excluding cast/mask/etc.
    score=2*elements;softmax=4*elements
    return dict(length=length,heads=heads,batch=batch,score_bf16_bytes=score,
        softmax_fp32_bytes=softmax,score_plus_softmax_bytes=score+softmax,
        score_plus_softmax_gib=(score+softmax)/2**30,
        exceeds_nominal_capacity=score+softmax>capacity_gib*2**30)


def audit(projection, model_config, output):
    if output.exists():raise ValueError('output exists')
    read=lambda p:json.loads(Path(p).read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    p=read(projection);m=read(model_config)
    heads=m['text_config']['num_attention_heads'];capacity=64
    source_paths=['src/multimodal_jev/span_attention_alignment.py',
        'src/multimodal_jev/span_runtime_profile.py','src/multimodal_jev/asr.py',
        'tools/jev/train_slurp_instruction_npu.py']
    source=Path(source_paths[0]).read_text()
    if 'probability=scores.softmax(-1,dtype=torch.float32).to(q.dtype)' not in source:
        raise ValueError('review changed attention implementation before accounting')
    stats={}
    for role,arms in p['populations'].items():
        stats[role]={}
        for arm,values in arms.items():
            histogram={int(k):v for k,v in values['length_histogram'].items()}
            if sum(histogram.values())!=values['requests']:raise ValueError('histogram population mismatch')
            exceed=sum(n for length,n in histogram.items() if tensor_budget(length,heads)['exceeds_nominal_capacity'])
            stats[role][arm]=dict(requests=values['requests'],requests_exceeding_64gib_pair=exceed,
                fraction_exceeding=exceed/values['requests'],maximum=tensor_budget(max(histogram),heads),
                median=tensor_budget(int(values['quantiles']['0.5']),heads))
    r=dict(schema='semantic-eager-attention-capacity-v1',
        pins={str(f):sha(f) for f in [projection,model_config,*source_paths]},
        implementation_sha256=sha(__file__),nominal_device_capacity_gib=capacity,
        capacity_basis='175910B3 reported65536MiB/card; nominal total, not available free capacity',
        maximum_length_fitting_only_two_tensors=math.isqrt(capacity*2**30//(6*heads)),
        populations=stats,
        dtype='BF16 scores plus FP32 softmax output, one request and one decoder layer',
        excluded=['BF16 probability cast','causal mask','Q/K/V and repeated KV','other activations',
            'weights and optimizer states','allocator/workspace','backward saved tensors','other layers'],
        DP4='Data parallelism/FSDP parameter sharding does not shard a single request attention matrix; same local16 heads, microbatch1 minimum.',
        G16='Sampling16 assignments from one categorical forward does not reduce the attention cost of that forward.',
        interpretation='The current explicit eager TP1 rollout cannot accommodate projected long inputs on a nominal64GiB card even before excluded costs. No actual OOM observed by this audit. Passing this arithmetic bound never proves fit.',
        decision='Do not inherit old eager runtime acceptance or automatically select32K. After task freeze, memory-efficient attention requires both actor/rollout implementation agreement and renewed probability/lifecycle acceptance. No change to DP4 or TP1 selection here.',
        training_dispatch_allowed=False,context_accepted=False,new_model_execution=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--projection',type=Path,required=True)
    p.add_argument('--model-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.projection,a.model_config,a.output)
    print(json.dumps({k:r[k] for k in ('maximum_length_fitting_only_two_tensors','populations')},indent=2))
