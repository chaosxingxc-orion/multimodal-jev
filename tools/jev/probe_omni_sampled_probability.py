#!/usr/bin/env python3
"""Prepared-only entry point: frozen real-sample, two-rank NPU probabilities.

Dispatch needs independent sampling/initialization acceptance and a registered
numeric protocol. No scheduler, optimizer, automatic retry or CPU substitution.
"""
import argparse
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.omni_logprob_observer import PINNED_ACTOR_SOURCE
from multimodal_jev.omni_probability_audit import compare, validate_protocol
from multimodal_jev.omni_probability_shards import (
    merge_shards, recompute_observed_shard, shard_plan,
)
from multimodal_jev.omni_sampling_acceptance import read_json
from multimodal_jev.sfc_reward_batch import digest
from preflight_omni_sampled_probability import run as prepare_inputs
from probe_omni_fsdp_initialization import run as initialize

RUN_NAMES = ['micro1', 'micro1-same-logits-fp32', 'micro2', 'micro2-same-logits-fp32']


def prerequisites(args, base, records):
    """Rebind externally accepted artifacts; this is not a live Docker audit."""
    acceptance = args.initialization/'acceptance.json'
    if file_hash(acceptance) != args.initialization_hash:
        raise ValueError('externally pinned initialization acceptance required')
    accepted = read_json(acceptance)
    if (accepted['schema'] != 'omni-fsdp-initialization-audit-v1'
            or accepted['preflight_acceptance_sha256'] != args.preflight_hash
            or accepted['parameters'] != 31719205488
            or accepted['parameter_tensors'] != 1407 or accepted['units'] != 108
            or accepted['buffers'] != 4 or accepted['ranks'] != 2):
        raise ValueError('complete separately accepted two-rank initialization required')
    if file_hash(args.initialization/'run/original-parameters.json') != accepted['original_parameters_sha256']:
        raise ValueError('accepted initialization original parameters changed')
    if sorted(r['rank'] for r in accepted['rank_outputs']) != [0, 1]:
        raise ValueError('both accepted rank artifacts required')
    for row in accepted['rank_outputs']:
        root = args.initialization/'run'/f"rank-{row['rank']}"
        for name in ('result', 'parameters'):
            if file_hash(root/(name+'.json')) != row[name+'_sha256']:
                raise ValueError('accepted initialization rank artifact changed')
    if file_hash(args.preflight/'acceptance.json') != args.preflight_hash:
        raise ValueError('accepted CPU preflight changed')
    cpu_acceptance = read_json(args.preflight/'acceptance.json')
    if file_hash(args.preflight/'run/result.json') != cpu_acceptance['result_sha256']:
        raise ValueError('accepted CPU model manifest changed')
    cpu = read_json(args.preflight/'run/result.json')
    identity = read_json(args.root/'run/identity.json')
    assets = identity['model_files']
    pins = {row['file']: row['sha256'] for row in assets}
    policy = digest(dict(repository='Qwen/Qwen3-Omni-30B-A3B-Instruct',
        revision='26291f793822fb6be9555850f06dfe95f2d7e695', files=assets))
    if (len(pins) != len(assets) or pins != cpu['model_files']
            or identity['policy_sha256'] != policy
            or Path(base['audio_model']).resolve() != args.model.resolve()):
        raise ValueError('sampling and FSDP must use the same complete original model')
    protocol = read_json(args.protocol)
    validate_protocol(records, protocol=protocol, protocol_sha256=args.protocol_hash)
    if protocol['run_names'] != RUN_NAMES:
        raise ValueError('both native microbatches and same-logits references required')
    return protocol


def run(args, rank, local_rank, out):
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    if file_hash(inspect.getfile(DataParallelPPOActor)) != PINNED_ACTOR_SOURCE:
        raise ValueError('fixed native verl actor source required')
    prepared = out/'input-preflight'
    prepared.mkdir()
    processor, records, batch, base = prepare_inputs(SimpleNamespace(
        root=args.root, acceptance=args.acceptance, acceptance_hash=args.acceptance_hash,
        output=prepared))
    protocol = prerequisites(args, base, records)
    plans = [shard_plan(records, world_size=2, micro_batch_size=size) for size in (1, 2)]
    evidence = read_json(prepared/'result.json')
    binding = dict(protocol=protocol, protocol_sha256=args.protocol_hash, plans=plans,
        sampling_acceptance_sha256=args.acceptance_hash,
        initialization_acceptance_sha256=args.initialization_hash,
        cpu_preflight_acceptance_sha256=args.preflight_hash,
        tensors=evidence['tensors'], prepared_inputs_sha256=evidence['prepared_inputs_sha256'],
        processor_parity_sha256=evidence['processor_parity_sha256'],
        sampling_identity_sha256=evidence['sampling_identity_sha256'])
    # Written before loading/forward. The coordinator must separately establish
    # that the supplied protocol was frozen before dispatch, not selected later.
    atomic_json(out/'registered-inputs.json', dict(binding=binding, registered_unix=time.time()))
    config = OmegaConf.create(dict(use_remove_padding=False, use_fused_kernels=False,
        ulysses_sequence_parallel_size=1, use_dynamic_bsz=False, use_prefix_grouper=False,
        entropy_from_logits_with_chunking=False, use_torch_compile=False,
        fsdp_config={'dtype': 'bfloat16'}, calculate_sum_pi_squared=False,
        entropy_checkpointing=False))

    def after_verification(wrapped):
        bindings = [None, None]
        dist.all_gather_object(bindings, digest(binding))
        if bindings != [digest(binding)]*2:
            raise ValueError('all ranks must use identical registered inputs and protocol')
        if wrapped.training or any(p.requires_grad or p.grad is not None for p in wrapped.parameters()):
            raise ValueError('fully frozen verified actor required')
        actor = DataParallelPPOActor(config, wrapped)
        begin = time.monotonic()
        for plan in plans:
            name = f"micro{plan['micro_batch_size']}"
            atomic_json(out/'progress.json', dict(phase=name, heartbeat_unix=time.time()))
            observed = recompute_observed_shard(actor, batch, records, name=name, rank=rank,
                plan=plan, plan_sha256=digest(plan), pad_token_id=processor.tokenizer.pad_token_id)
            torch.npu.synchronize()
            atomic_json(out/(name+'.json'), observed)
            dist.barrier()
        atomic_json(out/'result.json', dict(status='probabilities_captured', rank=rank,
            binding_sha256=digest(binding), weight_updates=0, targets_read=False,
            forward_calls=sum(p['forwards_per_rank'] for p in plans),
            peak_allocated_bytes=torch.npu.max_memory_allocated(),
            peak_reserved_bytes=torch.npu.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-begin,
            scope='Native frozen NPU actor and same-forward FP32 normalization reference; '
                  'instrumented timing, no training or task-quality acceptance.'))
        dist.barrier()
        if rank == 0:
            runs = []
            for plan in plans:
                name = f"micro{plan['micro_batch_size']}"
                shards = [read_json(args.output/f'rank-{r}'/(name+'.json')) for r in (0, 1)]
                for kind, run_name in (('native', name), ('reference', name+'-same-logits-fp32')):
                    runs.append(merge_shards(records, [s[kind] for s in shards], name=run_name,
                        plan=plan, plan_sha256=digest(plan)))
            atomic_json(args.output/'merged-runs.json', runs)
            audit = compare(records, runs, protocol=protocol, protocol_sha256=args.protocol_hash)
            atomic_json(args.output/'probability-audit.json', audit)
        dist.barrier()
        if not read_json(args.output/'probability-audit.json')['numerical_gate_passed']:
            raise ValueError('registered probability gate failed; preserve evidence, no retry')

    init_root = args.output/'initialization'
    init_out = init_root/f'rank-{rank}'
    init_out.mkdir(parents=True, exist_ok=False)
    initialize(SimpleNamespace(model=args.model, preflight=args.preflight,
        preflight_hash=args.preflight_hash, output=init_root), rank, local_rank, init_out,
        after_verification=after_verification)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'acceptance', 'initialization', 'preflight', 'model', 'protocol', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('acceptance-hash', 'initialization-hash', 'preflight-hash', 'protocol-hash'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 2 or rank not in (0, 1) or rank != local_rank:
        raise ValueError('single-node two-rank torchrun topology required')
    out = args.output/f'rank-{rank}'
    out.mkdir(parents=True, exist_ok=False)
    try:
        run(args, rank, local_rank, out)
    except BaseException as exc:
        atomic_json(out/'terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time()))
        raise
    atomic_json(out/'terminal.json', dict(exit_code=0, finished_unix=time.time()))


if __name__ == '__main__':
    main()
