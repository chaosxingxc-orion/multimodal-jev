"""Finite Ascend attention replay on authenticated real layer-zero Q/K/V.

No model, labels, policy update or automatic runtime promotion. Compare input
layout and precision modes at batch1, batch8 and batch32 on identical values.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from tools.reports.context_acoustic_numeric_probe import difference

MODES = [(2, 'contiguous'), (2, 'strided_v'), (0, 'contiguous'), (1, 'contiguous')]


def layouts(q, k, v, mode):
    q, k, v = [t.contiguous() for t in (q, k, v)]
    if mode == 'strided_v':
        # Native fused QKV split has token stride q_width+2*kv_width.
        fused = torch.cat([q.flatten(1), k.flatten(1), v.flatten(1)], -1)
        v = fused[:, q.shape[1]*q.shape[2] + k.shape[1]*k.shape[2]:].view_as(v)
    elif mode != 'contiguous':
        raise ValueError('explicit input layout required')
    return q, k, v


def summarize(payload, outputs):
    rows = payload['rows']
    refs = torch.cat([r['expected'] for r in rows])
    recorded = payload['recorded_batch']
    results = {}
    for precision, layout in MODES:
        name = f'p{precision}_{layout}'
        singles = outputs[name + '_b1']
        results[name] = {str(b): dict(vs_original_hf=difference(refs, outputs[name+f'_b{b}']),
            vs_same_mode_single=difference(singles, outputs[name+f'_b{b}']),
            vs_original_batch=difference(recorded, outputs[name+f'_b{b}'])) for b in (1, 8, 32)}
    return results


def run(package, output):
    import torch_npu
    from multimodal_jev.local_checkpoint import commit_json, commit_file
    from multimodal_jev.local_runtime_storage import require_training_storage
    from tools.reports.run_context_batch_probe_candidate import verify
    package, output = Path(package), Path(output)
    require_training_storage(output); output.mkdir(exist_ok=False)
    try:
        identity = verify(package)
        commit_json(output/'source.json', dict(files_sha256=identity))
        config = json.loads((package/'config.json').read_text())
        payload = torch.load(package/'inputs.pt', map_location='cpu', weights_only=True)
        if len(payload['rows']) != 32:
            raise ValueError('complete ordered32 required')
        torch.set_num_threads(2); torch.npu.set_device(0)
        mask = torch.ones((2048, 2048), device='npu', dtype=torch.bool).triu(1)
        rows = [{k: v.to('npu') for k, v in row.items() if k in ('q', 'k', 'v')} for row in payload['rows']]
        outputs, costs = {}, {}
        with torch.no_grad():
            for precision, layout in MODES:
                for batch in (1, 8, 32):
                    name = f'p{precision}_{layout}_b{batch}'
                    parts, traces = [], []
                    torch.npu.synchronize(); start = time.perf_counter()
                    for offset in range(0, 32, batch):
                        group = rows[offset:offset+batch]
                        tensors = layouts(*(torch.cat([r[k] for r in group]) for k in ('q','k','v')), layout)
                        ends = []; total = 0
                        for row in group:
                            total += len(row['q']); ends.append(total)
                        traces.append(dict(lengths=[len(r['q']) for r in group],
                            strides=[list(t.stride()) for t in tensors]))
                        result = torch_npu.npu_fusion_attention(*tensors, 16, 'TND', atten_mask=mask,
                            scale=128**-.5, keep_prob=1., actual_seq_qlen=tuple(ends),
                            actual_seq_kvlen=tuple(ends), sparse_mode=3, inner_precise=precision,
                            next_tockens=0)[0]
                        parts.append(result.cpu())
                    torch.npu.synchronize()
                    outputs[name] = torch.cat(parts)
                    costs[name] = dict(seconds=time.perf_counter()-start, calls=len(parts), traces=traces)
        commit_file(output/'outputs.pt', lambda f: torch.save(outputs, f))
        report = dict(status='finite_attention_kernel_replay', rows=config['rows'],
            comparisons=summarize(payload, outputs), measurement=costs,
            runtime=dict(torch=torch.__version__, torch_npu=torch_npu.__version__),
            full_model_parity_verified=False, training_dispatch_allowed=False,
            scope='Layer0 real QKV kernel-only; includes transfers, no warmup; timings are not throughput evidence')
        commit_json(output/'report.json', report); commit_json(output/'terminal.json', dict(exit_code=0))
    except BaseException as exc:
        commit_json(output/'terminal.json', dict(exit_code=1, error=repr(exc))); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.package, a.output)
