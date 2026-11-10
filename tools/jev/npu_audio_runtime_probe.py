#!/usr/bin/env python3
"""Allocated-device checks and real Qwen audio forward/backward; no RL claim."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from multimodal_jev.baseline import atomic_json
    started = time.monotonic()

    def progress(phase):
        atomic_json(args.output / 'progress.json', {'phase': phase, 'observed_unix': time.time()})

    try:
        import torch
        import torch_npu
        from peft import LoraConfig, get_peft_model
        from multimodal_jev.asr import load_runtime, prepare
        from multimodal_jev.audio_policy import load_audio, teacher_forcing, response_log_probs
        from multimodal_jev.task_schema import DecisionSchema, JevExample
        config = json.loads(args.config.read_text())
        torch.set_num_threads(4)
        torch.manual_seed(config['seed'])
        progress('device_admission')
        devices = sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))
        expected = ['davinci' + str(i) for i in config['physical_devices']]
        if devices != expected or torch.npu.device_count() != len(expected):
            raise RuntimeError(f'device isolation mismatch: nodes={devices}, count={torch.npu.device_count()}')
        device_results = []
        for index in range(torch.npu.device_count()):
            device = f'npu:{index}'
            x = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
            y = x @ x.T
            reference = torch.arange(16, dtype=torch.float32).reshape(4, 4)
            torch.testing.assert_close(y.cpu(), reference @ reference.T, atol=0, rtol=0)
            device_results.append({'logical_device': device, 'name': torch.npu.get_device_name(index),
                                   'matmul_exact': True})
            del x, y
        torch.npu.set_device(0)
        torch.npu.manual_seed_all(config['seed'])
        progress('load_model')
        processor, model = load_runtime(config['model_path'], 'npu:0', config['dtype'])
        inputs = prepare(processor, config['audio'], 'npu:0')
        inputs = {k: v.to(model.dtype) if v.is_floating_point() else v for k, v in inputs.items()}
        progress('asr_generation')
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            response = processor.decode(generated[0, inputs['input_ids'].shape[-1]:].cpu(), skip_special_tokens=True)
        transcription = processor.extract_transcription(response)
        example = JevExample('npu-runtime-train', 'train', config['audio'],
                             DecisionSchema('language-change', 'noul',
                                            'Does the speaker request a language change?'), True)
        batch, boundary = teacher_forcing(processor, example, load_audio(config['audio']))
        batch = {k: v.to(device='npu:0', dtype=model.dtype if v.is_floating_point() else v.dtype)
                 for k, v in batch.items()}
        assert (batch['labels'][:, :boundary] == -100).all()
        names = [name for name, _ in model.named_modules()
                 if name.startswith('model.language_model.') and name.rsplit('.', 1)[-1] in {'q_proj', 'v_proj'}]
        if len(names) != 2 * model.config.text_config.num_hidden_layers:
            raise RuntimeError('unexpected adapter target inventory')
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                              target_modules=names, task_type='CAUSAL_LM', bias='none'))
        model.train()
        trainable = [p for p in model.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in trainable]
        optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0, foreach=False, fused=False)
        progress('forward')
        result = model(**batch, use_cache=False)
        logp = response_log_probs(result.logits, batch['input_ids'], boundary)
        if not torch.isfinite(logp).all() or not torch.isfinite(result.loss):
            raise RuntimeError('non-finite response probability/loss')
        torch.testing.assert_close(-logp.mean(), result.loss.float(), atol=1e-4, rtol=1e-4)
        progress('backward')
        result.loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
            raise RuntimeError('missing/non-finite trainable gradient')
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(norm) or norm <= 0:
            raise RuntimeError('non-finite or zero gradient norm')
        optimizer.step()
        torch.npu.synchronize()
        changed = sum(not torch.equal(old, p.detach()) for old, p in zip(before, trainable))
        if changed == 0 or any(not torch.isfinite(p).all() for p in trainable):
            raise RuntimeError('invalid optimizer update')
        states = list(optimizer.state.values())
        if len(states) != len(trainable) or any(float(s['step']) != 1 for s in states):
            raise RuntimeError('optimizer state step mismatch')
        receipt = {'status': 'passed', 'config': config,
                   'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                   'audio_sha256': hashlib.sha256(Path(config['audio']).read_bytes()).hexdigest(),
                   'device_nodes': devices, 'devices': device_results,
                   'model_device': str(next(model.parameters()).device),
                   'packages': {p: importlib.metadata.version(p) for p in
                                ['torch', 'torch-npu', 'transformers', 'peft', 'verl', 'vllm', 'vllm-ascend']},
                   'transcription': transcription, 'prompt_tokens': boundary,
                   'response_tokens': logp.shape[-1], 'loss': float(result.loss.detach()),
                   'gradient_norm_before_clip': float(norm), 'changed_tensors': changed,
                   'trainable_tensors': len(trainable), 'optimizer_states': len(states),
                   'trainable_parameters': sum(p.numel() for p in trainable),
                   'max_abs_ce_logprob_error': float(abs(-logp.mean() - result.loss).detach()),
                   'elapsed_seconds': time.monotonic() - started,
                   'scope': 'four-device arithmetic and single-device real audio BF16 LoRA step; not distributed training, verl update_policy, vLLM generation or academic task accuracy'}
        atomic_json(args.output / 'result.json', receipt)
        atomic_json(args.output / 'terminal.json', {'exit_code': 0, 'finished_unix': time.time()})
        print(json.dumps(receipt, indent=2), flush=True)
    except BaseException as exc:
        atomic_json(args.output / 'terminal.json', {'exit_code': 1, 'error': repr(exc), 'finished_unix': time.time()})
        raise


if __name__ == '__main__':
    main()
