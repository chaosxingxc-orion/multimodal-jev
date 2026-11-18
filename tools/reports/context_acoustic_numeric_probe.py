"""Label-free, bounded NPU convolution diagnostic; no policy or training changes."""
import argparse
import json
from pathlib import Path


def difference(a, b):
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError('comparison shape/dtype mismatch')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('nonfinite diagnostic output')
    unequal = a != b
    coords = unequal.nonzero()[:8]
    return dict(exact=not bool(unequal.any()), count=int(unequal.sum()),
                max_abs=float((a.float()-b.float()).abs().max()),
                examples=[dict(index=c.tolist(), reference=float(a[tuple(c)]),
                               candidate=float(b[tuple(c)])) for c in coords])


def chunks(features):
    b, f, t = features.shape
    if b != 1 or t % 100:
        raise ValueError('one recording padded to 100-frame chunks required')
    return features.reshape(1, f, t//100, 100).permute(0, 2, 1, 3).reshape(t//100, 1, f, 100)


def run(package, output):
    import torch
    import torch_npu  # noqa: F401
    from safetensors import safe_open
    from torch.nn import functional as F
    from multimodal_jev.local_checkpoint import commit_json, commit_file
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.span_request_identity import tensor_identity
    from multimodal_jev.span_fixed_shape_linear import fixed_shape_linear
    from tools.reports.run_context_batch_probe_candidate import verify
    import hashlib
    package, output = Path(package), Path(output)
    require_training_storage(output)
    output.mkdir(exist_ok=False)
    try:
        manifest = verify(package)
        config = json.loads((package/'config.json').read_text())
        commit_json(output/'source.json', dict(files_sha256=manifest))
        torch.npu.set_device(0)
        convs, identities = [], {}
        with safe_open(str(Path(config['model'])/'model.safetensors'), framework='pt', device='cpu') as weights:
            for i in range(1, 4):
                suffix = f'audio_tower.conv2d{i}.weight'
                keys = [k for k in weights.keys() if k.endswith(suffix)]
                if len(keys) != 1:
                    raise ValueError(f'unique acoustic weight required: {keys}')
                key = keys[0]
                w, b = weights.get_tensor(key), weights.get_tensor(key[:-6]+'bias')
                identities[key] = tensor_identity(w)
                identities[key[:-6]+'bias'] = tensor_identity(b)
                conv = torch.nn.Conv2d(w.shape[1], w.shape[0], 3, 2, padding=1, dtype=torch.bfloat16)
                conv.load_state_dict({'weight': w, 'bias': b})
                convs.append(conv.eval().to('npu:0'))
            keys = [k for k in weights.keys() if k.endswith('audio_tower.conv_out.weight')]
            if len(keys) != 1:
                raise ValueError('unique conv_out weight required')
            key = keys[0]
            projection_weight = weights.get_tensor(key).to(dtype=torch.bfloat16, device='npu:0')
            bias_key = key[:-6]+'bias'
            projection_bias = weights.get_tensor(bias_key).to(dtype=torch.bfloat16, device='npu:0') if bias_key in weights.keys() else None
            identities[key] = tensor_identity(projection_weight)
            if projection_bias is not None:
                identities[bias_key] = tensor_identity(projection_bias)
        records, captures = [], {}
        with torch.inference_mode():
            for row in config['rows']:
                path = Path(config['cache_root'])/(row['cache_key']+'.pt')
                if hashlib.sha256(path.read_bytes()).hexdigest() != row['cache_sha256']:
                    raise ValueError('cache bytes changed')
                payload = torch.load(path, map_location='cpu', weights_only=True)
                values = dict(payload['inputs'], positions=payload['positions'])
                if {k: tensor_identity(v) for k, v in values.items()} != row['tensors']:
                    raise ValueError('cache tensors changed')
                features = values['input_features']
                mask = values['input_features_mask'].bool()
                if not torch.equal(mask, torch.arange(mask.shape[-1])[None] < mask.sum()):
                    raise ValueError('contiguous feature prefix required')
                count = features.shape[-1]//100
                if not 1 <= count <= 32:
                    raise ValueError('bounded <=32 chunk diagnostic required')
                zeroed = features.masked_fill(~mask[:, None, :], 0)
                native = chunks(features).to('npu:0')
                trimmed = chunks(zeroed).to('npu:0')
                padded = F.pad(native, (0, 0, 0, 0, 0, 0, 0, 32-count))
                result = dict(id=row['id'], cache_sha256=row['cache_sha256'], chunks=count,
                              valid_frames=int(mask.sum()), padding_input=difference(features, zeroed), stages=[])
                for i, conv in enumerate(convs, 1):
                    # Same previous-layer input isolates this layer's batch-shape sensitivity.
                    local_pad = F.pad(native, (0, 0, 0, 0, 0, 0, 0, 32-count))
                    ref = conv(native)
                    local = conv(local_pad)[:count]
                    accumulated = conv(padded)
                    cut = conv(trimmed)
                    stage = dict(layer=f'conv2d{i}', isolated_batch=difference(ref, local),
                                 cumulative_batch=difference(ref, accumulated[:count]),
                                 zeroed_padding=difference(ref, cut))
                    result['stages'].append(stage)
                    for mode, candidate in [('batch', accumulated[:count]), ('padding', cut)]:
                        key = f'{len(records)}-conv{i}-{mode}'
                        neq = (ref != candidate).nonzero()
                        if len(neq):
                            index = int(neq[0, 0])
                            captures[key] = dict(chunk=index, reference=ref[index].cpu(),
                                                 candidate=candidate[index].cpu())
                    native, padded, trimmed = F.gelu(ref), F.gelu(accumulated), F.gelu(cut)
                    stage['post_gelu_batch'] = difference(native, padded[:count])
                    stage['post_gelu_padding'] = difference(native, trimmed)
                def project(x):
                    b, c, f, t = x.shape
                    return fixed_shape_linear(x.permute(0, 3, 1, 2).contiguous().view(b, t, c*f),
                                              projection_weight, projection_bias)
                native_projected, padded_projected = project(native), project(padded)
                result['projection'] = difference(native_projected, padded_projected[:count])
                result['projection_shifted_slots'] = []
                for offset in (1, 7):
                    shifted = torch.cat([native[:1].expand(offset, -1, -1, -1), native], dim=0)
                    result['projection_shifted_slots'].append(dict(offset=offset,
                        difference=difference(native_projected, project(shifted)[offset:])))
                records.append(result)
                commit_json(output/f'row-{len(records):03d}.json', result)
        commit_file(output/'divergent-chunks.pt', lambda f: torch.save(captures, f))
        report = dict(status='finite_acoustic_convolution_diagnosed', rows=records, weights=identities,
                      scope='BF16 conv1/2/3, GELU and fixed conv_out; same real features; padding, batch shape and projection slots; no target labels',
                      runtime=dict(torch=torch.__version__, torch_npu=torch_npu.__version__),
                      full_model_parity_verified=False, training_dispatch_allowed=False)
        commit_json(output/'report.json', report)
        commit_json(output/'terminal.json', dict(exit_code=0))
    except BaseException as exc:
        commit_json(output/'terminal.json', dict(exit_code=1, error=repr(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.package, args.output)
