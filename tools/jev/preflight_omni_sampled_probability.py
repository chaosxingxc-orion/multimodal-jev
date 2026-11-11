#!/usr/bin/env python3
"""Accepted real TRAIN78 K2 samples -> CPU processor/DataProto, no weights.

Run after the frozen sampling snapshot's independent verifier. The externally
supplied acceptance SHA is a trust boundary, not proof of Docker authenticity.
"""
import argparse
import importlib.metadata
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.omni_accepted_sampling import load_accepted_records
from multimodal_jev.omni_actor_batch import prepare
from multimodal_jev.omni_probability_batch import make_probability_batch
from multimodal_jev.omni_probability_capture import _aligned
from multimodal_jev.omni_probability_shards import shard_plan
from multimodal_jev.omni_sampling_acceptance import read_json
from multimodal_jev.sfc_reward_batch import digest
from sample_sfc_omni_learning import load_inputs
from probe_omni_fsdp_initialization import tensor_identity


def run(args):
    import torch
    from transformers import Qwen3OmniMoeProcessor

    begin = time.monotonic()
    torch.set_num_threads(2)
    config = read_json(args.root/'config.json')
    base, rows, preflight, checks = load_inputs(config)
    if {k: importlib.metadata.version(k) for k in base['versions']} != base['versions']:
        raise ValueError('fixed processor/runtime versions required')
    for name, pin in preflight['processor_assets'].items():
        if file_hash(Path(base['audio_model'])/name) != pin:
            raise ValueError('processor assets changed')
    processor = Qwen3OmniMoeProcessor.from_pretrained(base['audio_model'], local_files_only=True)
    records, identity = load_accepted_records(args.root, args.acceptance,
        args.acceptance_hash, rows, checks, processor.tokenizer)
    prepared = prepare(processor, rows, waveform_boundary='vllm_hop')
    batch = make_probability_batch(processor, rows, prepared, records,
        policy_sha256=identity['policy_sha256'],
        instruction_sha256=identity['instruction_sha256'], group_size=2, context_limit=4096)
    _aligned(batch, records)
    features = [dict(example_id=k[0], decision_id=k[1],
        original_samples=v['original_samples'], processed_samples=v['processed_samples'],
        audio_token_lengths=v['audio_token_lengths'],
        tensors={name: tensor_identity(v[name]) for name in
                 ('input_ids', 'input_features', 'feature_attention_mask')})
        for k, v in prepared.items()]
    plans = [shard_plan(records, world_size=2, micro_batch_size=size) for size in (1, 2)]
    atomic_json(args.output/'prepared-inputs.json', features)
    atomic_json(args.output/'shard-plans.json', plans)
    atomic_json(args.output/'result.json', dict(status='preflight_passed',
        acceptance_sha256=args.acceptance_hash, records_sha256=digest(records),
        sampling_identity_sha256=file_hash(args.root/'run/identity.json'),
        responses=len(records), response_tokens=sum(len(r['token_ids']) for r in records),
        tensors={k: tensor_identity(v) for k, v in batch.batch.items()},
        prepared_inputs_sha256=file_hash(args.output/'prepared-inputs.json'),
        shard_plans_sha256=file_hash(args.output/'shard-plans.json'),
        elapsed_seconds=time.monotonic()-begin, model_loaded=False, targets_read=False,
        weight_updates=0, scope='Accepted raw samples and real hop-aligned CPU processor/DataProto '
        'only. No vLLM feature comparison, NPU forward, numerical tolerance registration or RL efficacy.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('root', 'acceptance', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--acceptance-hash', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', dict(exit_code=1, error=repr(exc)))
        raise
    atomic_json(args.output/'terminal.json', dict(exit_code=0))
