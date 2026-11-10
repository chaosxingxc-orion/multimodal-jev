#!/usr/bin/env python3
"""Measure frozen TRAIN processor lengths without loading model weights."""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.budget_profile import schedule_profile
from multimodal_jev.budget_control import contract_hashes
from multimodal_jev.rl_preflight import validate_schedule
from multimodal_jev.sft import training_batches


def run(args):
    import torch
    import transformers
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.audio_policy import load_audio, teacher_forcing
    torch.set_num_threads(4)
    start = time.monotonic()
    paths = sorted(args.configs.glob('*.json'))
    configs = {p.name: json.loads(p.read_text()) for p in paths}
    primary = configs['17-pair-grpo-standard.json']
    preflight = validate_schedule(primary)
    initializer = json.loads(args.initializer_identity.read_text())
    if preflight['manifest_sha256'] != initializer['manifest_sha256']:
        raise ValueError('initializer manifest mismatch')
    wanted = set(initializer['example_ids'])
    batches = training_batches(configs['17-sft.json'])
    selected = {ex.example_id: ex for batch in batches for ex in batch}
    if set(selected) != wanted or len(selected) != 186:
        raise ValueError('registered initializer selection mismatch')
    model = Path(primary['actor']['model_path'])
    processor = Qwen3ASRProcessor.from_pretrained(model, local_files_only=True)
    counts = {}
    for uid, ex in sorted(selected.items()):
        inputs, boundary = teacher_forcing(processor, ex, load_audio(ex.audio))
        counts[uid] = {'prompt_tokens': boundary,
                       'gold_tokens': int((inputs['labels'] != -100).sum()),
                       'audio_frames': int(inputs['input_features_mask'].sum())}
        atomic_json(args.output / 'progress.json', {
            'phase': 'processor', 'completed': len(counts), 'total': len(selected),
            'heartbeat_unix': time.time()})
    profiles = {}
    for name, config in configs.items():
        seed = config['seed']
        sft = configs[f'{seed}-sft.json']
        if (config['batches'] != sft['batches'] or
                {uid for batch in config['batches'] for uid in batch} != wanted):
            raise ValueError('comparison data exposure mismatch')
        for key in ('initial_adapter', 'initial_checkpoint'):
            if config[key] != primary[key]:
                raise ValueError('comparison initializer changed')
        is_sft = name.endswith('-sft.json')
        actor = config if is_sft else config['actor']
        manifest = config['train_manifest'] if is_sft else config['manifest']
        if actor['model_path'] != str(model) or manifest != primary['manifest']:
            raise ValueError('comparison base or manifest changed')
        if is_sft:
            profiles[name] = schedule_profile(counts, config['batches'])
        else:
            cap = config['sampling']['max_tokens']
            if max(r['prompt_tokens'] for r in counts.values()) + cap > config['engine']['max_model_len']:
                raise ValueError('prompt plus response exceeds engine context')
            profiles[name] = schedule_profile(
                counts, config['batches'], samples=config['sampling']['n'],
                max_response_tokens=cap,
                no_grad_passes=1 + int(actor.get('reference') is not None))
    # Detect source audio changes during the audit as well as before it.
    for path, digest in preflight['audio_sha256'].items():
        if file_hash(path) != digest:
            raise ValueError('audio changed during profile')
    result = {'scope': 'TRAIN-only processor measurement; no forward, sampling or optimization; '
              'effective unpadded lengths, not FLOPs, padded execution costs or matched budgets',
              'initializer_identity_sha256': file_hash(args.initializer_identity),
              'contract_sha256': contract_hashes(),
              'config_sha256': {p.name: file_hash(p) for p in paths},
              'preflight': preflight, 'torch': torch.__version__,
              'transformers': transformers.__version__,
              'processor_files': {p.name: file_hash(p) for p in sorted(model.iterdir())
                                  if p.is_file() and p.suffix in {'.json', '.txt', '.jinja'}},
              'examples': counts, 'profiles': profiles,
              'elapsed_seconds': time.monotonic() - start}
    atomic_json(args.output / 'result.json', result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=Path, required=True)
    parser.add_argument('--initializer-identity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        atomic_json(args.output / 'terminal.json', {'exit_code': 1, 'error': repr(exc),
                                                   'finished_unix': time.time()})
        raise
    atomic_json(args.output / 'terminal.json', {'exit_code': 0, 'finished_unix': time.time()})


if __name__ == '__main__':
    main()
