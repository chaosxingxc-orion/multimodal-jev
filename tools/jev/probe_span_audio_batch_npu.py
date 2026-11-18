#!/usr/bin/env python3
"""Pretrained BF16 serial/true-batch parity and inference capacity diagnostic.

No optimizer, promotion or throughput claim for FSDP training. Selection is
label-free, stratified by input frame count within the pinned learning fixture.
"""
import argparse
import json
import time
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
    from multimodal_jev.span_parity import compare_span_policies
    from multimodal_jev.span_runtime_profile import configure_actor_profile
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    from multimodal_jev.span_device_scope import validate_tp1
    validate_tp1(config)
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    for name, digest in config['pins'].items():
        if file_hash(name) != digest:
            raise ValueError('changed pinned input: ' + name)
    artifact = json.loads(Path(config['inputs']).read_text())
    assert artifact['passed'] and artifact['requests'] == 256 and len(artifact['records']) == 256
    # Frame count comes from the re-created, authenticated processor feature mask.
    torch.set_num_threads(4); torch.manual_seed(317); torch.npu.set_device(0)
    configure_actor_profile(config['runtime_profile'])
    processor, base = load_runtime(config['model'], 'npu:0', 'bfloat16')
    adapted = PeftModel.from_pretrained(base, config['adapter'], is_trainable=False,
                                      autocast_adapter_dtype=False).eval()
    assert all(p.dtype == torch.bfloat16 for p in adapted.parameters())
    model = NativeAudioSpanModel(adapted.get_base_model()).eval()
    all_inputs = list(prepared(config, artifact, processor=processor))
    ordered = sorted(all_inputs, key=lambda r: (int(r[1]['input_features_mask'].sum()), r[0]['id']))
    chosen = [ordered[round(i * (len(ordered) - 1) / 31)] for i in range(32)]
    items = [dict(inputs=r[1], positions=r[2], candidates=r[3]) for r in chosen]
    atomic_json(root/'inputs.json', dict(records=[dict(id=r[0]['id'], request_id=r[4],
        request_identity=r[5], frames=int(r[1]['input_features_mask'].sum()),
        tokens=r[1]['input_ids'].shape[-1]) for r in chosen],
        selection='32 deterministic frame-count quantiles of authenticated learning256'))
    options = dict(pad_token_id=processor.tokenizer.pad_token_id,
        audio_token_id=base.config.audio_token_id,
        audio_chunk_size=base.config.audio_config.n_window * 2,
        audio_infer_window=base.config.audio_config.n_window_infer)
    reference = []
    with torch.inference_mode():
        for item in items:
            inputs = {k: v.to('npu:0', dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                      for k, v in item['inputs'].items()}
            reference.append([v[0].cpu().tolist() for v in model(inputs, item['positions'].to('npu:0'), item['candidates'])[1]])
    del inputs
    atomic_json(root/'serial.json', dict(candidate_logprobs=reference))
    results = []
    for micro in (1, 2, 4, 8, 16, 32):
        torch.npu.empty_cache(); torch.npu.reset_peak_memory_stats()
        atomic_json(root/'progress.json', dict(phase='batch_forward', microbatch=micro, updated_unix=time.time()))
        calls = []
        hook = model.backbone.register_forward_hook(lambda *a: calls.append(1))
        values = []; torch.npu.synchronize(); start = time.monotonic()
        try:
            with torch.inference_mode():
                for offset in range(0, len(items), micro):
                    output = forward_span_audio_batch(model, items[offset:offset+micro],
                        device='npu:0', dtype=torch.bfloat16, **options)
                    values.extend([[v[0].cpu().tolist() for v in row] for row in output])
                    del output
            torch.npu.synchronize()
            elapsed = time.monotonic() - start
            assert len(calls) == 32 // micro
            parity = [compare_span_policies(a, b) for a, b in zip(reference, values, strict=True)]
            result = dict(microbatch=micro, backbone_calls=len(calls), seconds=elapsed,
                requests_per_second=32 / elapsed, maximum_allocated_bytes=torch.npu.max_memory_allocated(),
                maximum_reserved_bytes=torch.npu.max_memory_reserved(), candidate_logprobs=values,
                parity=parity, passed=all(v['passed'] for v in parity))
            atomic_json(root/f'micro-{micro:02d}.json', result)
            results.append({k: v for k, v in result.items() if k not in ('parity', 'candidate_logprobs')})
            # A failed numerical gate is evidence; do not promote or conceal it.
            if not result['passed']:
                break
        finally:
            hook.remove()
    atomic_json(root/'result.json', dict(results=results,
        passed=len(results)==6 and all(r['passed'] for r in results),
        training_promotion=False, scope='Pretrained BF16 LoRA inference, 32 length-stratified learning inputs; no backward/FSDP/optimizer or direct vLLM comparison. Single timing pass includes collation/transfers, not steady-state training throughput.'))


if __name__ == '__main__':
    main()
