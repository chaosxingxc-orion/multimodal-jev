#!/usr/bin/env python3
"""Inspect whole-model dtype casts on real Omni buffers with meta parameters.

No pretrained tensor load, distributed initialization, NPU or model forward.
Calls only the actual text rotary module on explicit synthetic positions.
"""
import argparse
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash

HF_SOURCE = 'a55c7951f8ad316ab3abeadb09c51bc577ae3ffb673b40ebb5f17abac1348340'
WORKER_SOURCE = '1de59e6b6f48b26716c9c75649aaef842f9ace44f6da3a3ad64dd2a45f9535f7'


def difference(before, after):
    import torch
    if before.shape != after.shape or not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise ValueError('same-shape finite buffers required')
    error = (before.double()-after.double()).abs()
    return dict(elements=before.numel(), changed_elements=int(torch.count_nonzero(error)),
                max_abs_error=float(error.max()), mean_abs_error=float(error.mean()),
                exact=torch.equal(before, after))


def run(args):
    import torch
    import transformers
    import accelerate
    import verl
    from accelerate import init_empty_weights
    from transformers import initialization
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeThinkerConfig
    from torch.distributed.fsdp._runtime_utils import _cast_buffers_to_dtype_and_device

    begin = time.monotonic()
    torch.set_num_threads(2)
    worker = Path(verl.__file__).parent/'workers/fsdp_workers.py'
    if (file_hash(args.config) != args.config_hash or file_hash(inspect.getfile(hf)) != HF_SOURCE
            or file_hash(worker) != WORKER_SOURCE):
        raise ValueError('registered original config and fixed-runtime source required')
    raw = json.loads(args.config.read_text())
    config = Qwen3OmniMoeThinkerConfig(**raw['thinker_config'])
    config._attn_implementation = 'sdpa'
    # This is the explicit meta-parameter, real-buffer constructor context;
    # it is not a claim about from_pretrained or a completed worker load.
    with init_empty_weights(include_buffers=False), initialization.no_init_weights():
        model = hf.Qwen3OmniMoeThinkerForConditionalGeneration(config)
    count = sum(p.numel() for p in model.parameters())
    if count != 31719205488 or any(not p.is_meta for p in model.parameters()):
        raise ValueError('full original Thinker parameter structure must remain meta')
    before = {n:b.detach().clone() for n,b in model.named_buffers()}
    if not before or any(b.is_meta or b.device.type != 'cpu' or b.dtype != torch.float32 for b in before.values()):
        raise ValueError('complete real CPU FP32 constructor buffers required')
    persistent = set(model.state_dict())
    rope_before = {}
    positions = {}
    for length in (1230, 4096):
        positions[length] = torch.arange(length).unsqueeze(0)
        x = torch.zeros(1, length, 1, dtype=torch.bfloat16)  # Rotary reads dtype/device only.
        rope_before[length] = tuple(t.clone() for t in model.model.rotary_emb(x, positions[length]))

    # Match the worker's whole-model cast, then the actual FSDP buffer cast
    # helper; using its CPU device path does not establish FSDP/NPU support.
    model.to(torch.bfloat16)
    bf16 = {n:b.detach().clone() for n,b in model.named_buffers()}
    buffers = list(model.buffers())
    _cast_buffers_to_dtype_and_device(buffers, [torch.float32]*len(buffers), torch.device('cpu'))
    restored = {n:b.detach().clone() for n,b in model.named_buffers()}
    if set(before) != set(bf16) or set(before) != set(restored):
        raise ValueError('buffer roster changed during dtype casting')
    if any(not p.is_meta or p.dtype != torch.bfloat16 for p in model.parameters()):
        raise ValueError('parameters must stay BF16 meta without loading storage')
    reports = []
    for name, value in before.items():
        if bf16[name].dtype != torch.bfloat16 or restored[name].dtype != torch.float32:
            raise ValueError('expected FP32 -> BF16 -> FP32 buffer path')
        reports.append(dict(name=name, shape=list(value.shape), persistent=name in persistent,
                            before_dtype=str(value.dtype), intermediate_dtype=str(bf16[name].dtype),
                            final_dtype=str(restored[name].dtype), **difference(value, restored[name])))
    torch.save(dict(before=before, bf16=bf16, restored=restored), args.output/'buffers.pt')
    rope_after = {}; rotary_reports = []
    for length in (1230, 4096):
        x = torch.zeros(1, length, 1, dtype=torch.bfloat16)
        rope_after[length] = tuple(t.clone() for t in model.model.rotary_emb(x, positions[length]))
        rotary_reports.append(dict(sequence_length=length, output_dtype='torch.bfloat16',
            cos=difference(rope_before[length][0], rope_after[length][0]),
            sin=difference(rope_before[length][1], rope_after[length][1])))
    torch.save(dict(positions=positions, before=rope_before, after=rope_after), args.output/'rotary.pt')
    atomic_json(args.output/'result.json', dict(status='diagnostic_complete',
        config_sha256=args.config_hash, parameters=count, parameter_storage='meta',
        buffer_reports=reports, rotary_reports=rotary_reports,
        artifacts={n:file_hash(args.output/n) for n in ('buffers.pt', 'rotary.pt')},
        source_sha256={str(worker):file_hash(worker), inspect.getfile(hf):HF_SOURCE,
            inspect.getfile(inspect.unwrap(init_empty_weights)):file_hash(inspect.getfile(inspect.unwrap(init_empty_weights))),
            inspect.getfile(inspect.unwrap(_cast_buffers_to_dtype_and_device)):file_hash(inspect.getfile(inspect.unwrap(_cast_buffers_to_dtype_and_device))),
            'runner':file_hash(__file__)},
        versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                      accelerate=accelerate.__version__),
        buffer_cast_helper=inspect.getsource(_cast_buffers_to_dtype_and_device),
        elapsed_seconds=time.monotonic()-begin, weights_loaded=False, npu_used=False, weight_updates=0,
        scope='Full original config with meta parameters and real constructor buffers; actual whole-model dtype cast and FSDP buffer cast helper on CPU, plus rotary-only calls on synthetic positions. No pretrained load, FSDP collectives, NPU forward, sampled probabilities or task quality claim.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('config', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--config-hash', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', dict(exit_code=1, error=repr(exc)))
        raise
    atomic_json(args.output/'terminal.json', dict(exit_code=0))
