#!/usr/bin/env python3
"""Isolate the first observed MoE difference using saved layer-1 norm inputs."""
import argparse
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def run(args):
    import torch
    from safetensors import safe_open
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.core_model_loading import dot_natural_key
    from transformers.integrations import moe
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig

    torch.set_num_threads(16)
    start = time.monotonic()
    read = lambda p: json.loads(p.read_text())
    if file_hash(args.decoder/'result.json') != '1ca2828a5cdc39cd592f6c6bf983822f018f6b02212157fc7d7cf9a2fc707ece':
        raise ValueError('original first divergence diagnostic required')
    result = read(args.decoder/'result.json')
    if result['first_differing_layer'] != 1 or len(result['layers']) != 1:
        raise ValueError('original first divergent layer required')
    if file_hash(args.decoder/'layer-01.json') != result['layers'][0]['report_sha256']:
        raise ValueError('original layer report changed')
    report = read(args.decoder/'layer-01.json')
    if file_hash(args.first/'identity.json') != read(args.first/'result.json')['identity_sha256']:
        raise ValueError('original module identity changed')
    first_identity = read(args.first/'identity.json')
    for path, pin in first_identity['source_sha256'].items():
        if file_hash(Path(path)) != pin:
            raise ValueError('original HF modules required')
    if file_hash(args.prefix/'result.json') != 'fb10bfc904f23e4ebb408ce3a685c9eac66a55f3bfd48b50dfae95443c5e08e7':
        raise ValueError('original model identity required')
    if file_hash(args.prefix/'identity.json') != read(args.prefix/'result.json')['identity_sha256']:
        raise ValueError('original model identity changed')
    identity = read(args.prefix/'identity.json')
    order = identity['order']
    pins = {r['file']:r['sha256'] for r in identity['model_files']}
    verified = set()

    def verify(name):
        if name not in verified:
            if file_hash(args.model/name) != pins[name]:
                raise ValueError('original model file changed')
            verified.add(name)

    for name in ('config.json','model.safetensors.index.json'):
        verify(name)
    config = Qwen3OmniMoeTextConfig(**read(args.model/'config.json')['thinker_config']['text_config'])
    config._experts_implementation = first_identity['experts_implementation']
    if config._experts_implementation != 'grouped_mm':
        raise ValueError('original grouped experts dispatch required')
    with torch.device('meta'):
        block = hf.Qwen3OmniMoeThinkerTextSparseMoeBlock(config).to(torch.bfloat16)
    index = read(args.model/'model.safetensors.index.json')['weight_map']
    names = [k for k in index if k.startswith('thinker.model.layers.1.mlp.')]
    old = {}
    for filename in sorted({index[n] for n in names}):
        verify(filename)
        with safe_open(args.model/filename, framework='pt', device='cpu') as f:
            old.update({n.removeprefix('thinker.'):f.get_tensor(n) for n in names if index[n] == filename})
    converted = {}
    used = set()
    for converter in get_checkpoint_conversion_mapping('qwen3_omni_moe_thinker'):
        target = None
        for name in sorted(old, key=dot_natural_key):
            renamed, pattern = converter.rename_source_key(name)
            if pattern is not None:
                if target is None:
                    target = renamed
                if target != renamed:
                    raise ValueError('unexpected conversion target')
                converter.add_tensor(renamed,name,pattern,old[name])
                used.add(name)
        if target is None:
            raise ValueError('missing expert conversion')
        converted.update(converter.convert(target))
    prefix = 'model.layers.1.mlp.'
    for i in range(config.num_experts):
        gate_up = converted[prefix+'experts.gate_up_proj'][i]
        down = converted[prefix+'experts.down_proj'][i]
        for a,b in [(gate_up[:config.moe_intermediate_size],old[prefix+f'experts.{i}.gate_proj.weight']),
                    (gate_up[config.moe_intermediate_size:],old[prefix+f'experts.{i}.up_proj.weight']),
                    (down,old[prefix+f'experts.{i}.down_proj.weight'])]:
            if not torch.equal(a,b):
                raise ValueError('converted expert differs from checkpoint')
    state = {k.removeprefix(prefix):v for k,v in old.items() if k not in used}
    state.update({k.removeprefix(prefix):v for k,v in converted.items()})
    block.load_state_dict(state,strict=True,assign=True)
    block.eval().requires_grad_(False)
    del old,converted,state
    if any(p.is_meta or p.dtype != torch.bfloat16 for p in block.parameters()):
        raise ValueError('complete original BF16 block required')
    expected = [(s,o) for s in (1,2) for o in range(0,6,s)]
    if [(r['size'],r['offset']) for r in report['saved']] != expected:
        raise ValueError('complete original inputs required')
    calls = []
    paths = {}

    def difference(a,b):
        if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('incompatible or nonfinite tensors')
        return dict(equal=bool(torch.equal(a,b)),max_abs_error=float((a.float()-b.float()).abs().max()),
                    differing_elements=int((a != b).sum()))

    for record in report['saved']:
        size,offset = record['size'],record['offset']
        begin = time.monotonic()
        atomic_json(args.output/'progress.json',dict(phase='experts',size=size,offset=offset,heartbeat_unix=time.time()))
        p = args.decoder/record['file']
        if file_hash(p) != record['sha256']:
            raise ValueError('original layer capture changed')
        original = torch.load(p,map_location='cpu',weights_only=True)
        inputs = original['stages']['post_attention_norm']
        width = inputs.shape[1]
        values = dict(input=inputs.clone())
        routing = {}
        linear_calls = []

        def save(name,t):
            if name in values:
                raise ValueError('duplicate stage')
            values[name] = t.detach().clone()

        def router_hook(module,a,out):
            for name,t in zip(('logits','weights','indices'),out,strict=True):
                save('router.'+name,t.reshape(size,width,-1))
            _,perm = torch.sort(out[2].reshape(-1))
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(perm.numel())
            routing.update(perm=perm,inv=inv)

        def restore(t):
            return t[routing['inv']].reshape(size,width,config.num_experts_per_tok,-1)

        original_linear = moe._grouped_linear

        def grouped_linear(x,w,offs,*a,**kw):
            number = len(linear_calls)
            if number not in (0,1):
                raise ValueError('unexpected extra expert projection')
            expected_weight = block.experts.gate_up_proj if number == 0 else block.experts.down_proj
            if w is not expected_weight:
                raise ValueError('unexpected expert weight')
            if number == 0:
                if not torch.equal(x,inputs.reshape(-1,config.hidden_size)[routing['perm']//config.num_experts_per_tok]):
                    raise ValueError('unexpected grouped ordering')
            else:
                save('activation',restore(x))
            out = original_linear(x,w,offs,*a,**kw)
            save('gate_up' if number == 0 else 'down',restore(out))
            linear_calls.append(dict(stage='gate_up' if number == 0 else 'down',offsets=offs.tolist(),
                                     input_shape=list(x.shape),output_shape=list(out.shape)))
            return out

        hook = block.gate.register_forward_hook(router_hook)
        try:
            moe._grouped_linear = grouped_linear
            with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
                output = block(inputs)
                # Independent reconstruction in original token/top-k order.
                weighted = values['down']*values['router.weights'].unsqueeze(-1)
                save('weighted',weighted)
                reconstructed = weighted.sum(2)
                save('mlp',output)
                if not torch.equal(reconstructed,output):
                    raise ValueError('captured expert outputs do not reproduce actual aggregation')
        finally:
            moe._grouped_linear = original_linear
            hook.remove()
        if len(linear_calls) != 2:
            raise ValueError('missing grouped expert projections')
        exact = {name:difference(values[name],original['stages'][name]) for name in
                 ('router.logits','router.weights','router.indices','mlp')}
        if not all(c['equal'] for c in exact.values()):
            raise ValueError('isolated block failed to reproduce original complete-layer path')
        p = args.output/f'micro-{size}-offset-{offset}.pt'
        torch.save(dict(token_mask=original['token_mask'],stages=values),p)
        call = dict(size=size,offset=offset,tensor_file=p.name,tensor_sha256=file_hash(p),
                    exact_original=exact,linear_calls=linear_calls,seconds=time.monotonic()-begin)
        atomic_json(args.output/f'micro-{size}-offset-{offset}.json',call)
        calls.append(call)
        for j in range(size):
            paths[size,order[offset+j]] = p,j
        del values,original,inputs,output,weighted,reconstructed
    differences = []
    for row in order:
        p1,i1 = paths[1,row]
        p2,i2 = paths[2,row]
        a = torch.load(p1,map_location='cpu',weights_only=True)
        b = torch.load(p2,map_location='cpu',weights_only=True)
        mask = a['token_mask'][i1].bool()
        if not torch.equal(mask,b['token_mask'][i2].bool()):
            raise ValueError('original token masks differ')
        comparisons = []
        for name,v in a['stages'].items():
            left,right = v[i1,mask],b['stages'][name][i2,mask]
            diff = difference(left,right)
            coords = (left != right).nonzero()
            examples = []
            for coord in coords[:16].tolist():
                at = tuple(coord)
                example = dict(valid_coordinate=coord,micro1=float(left[at]),micro2=float(right[at]))
                if name in ('gate_up','activation','down','weighted'):
                    example['expert'] = int(a['stages']['router.indices'][i1,mask][coord[0],coord[1]])
                examples.append(example)
            comparisons.append(dict(stage=name,valid=diff,full_padded=difference(v[i1],b['stages'][name][i2]),examples=examples))
        differences.append(dict(row=row,stages=comparisons,first_valid_difference=next((c['stage'] for c in comparisons if not c['valid']['equal']),None)))
        del a,b
    for name in verified:
        if file_hash(args.model/name) != pins[name]:
            raise ValueError('model assets changed during diagnostic')
    atomic_json(args.output/'result.json',dict(status='diagnostic_complete',reports=calls,differences=differences,
        source_sha256=file_hash(__file__),decoder_result_sha256=file_hash(args.decoder/'result.json'),
        experts_implementation=config._experts_implementation,original_expert_tensors_verified=384,
        model_files_verified={n:pins[n] for n in sorted(verified)},elapsed_seconds=time.monotonic()-start,weight_updates=0,
        scope='Isolated original layer1 MoE from saved real norm inputs; grouped projection/activation/down/weighted outputs restored to token/top-k order. Exact reproduction of prior full-layer router and MoE outputs required. No attention, audio, later layers, actor probabilities, training or NPU.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('model','decoder','first','prefix','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)))
        raise
    atomic_json(args.output/'terminal.json',dict(exit_code=0))
