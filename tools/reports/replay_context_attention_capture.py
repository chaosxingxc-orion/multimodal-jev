"""Recover the v0.0.28 diagnostic layout without rerunning or admitting NPU code.

Its HF o_proj prehook flattened [1,T,D] to [1,T*D]. Reshape only that
authenticated capture, using the independently captured Q width and row length.
Original files, failed terminal and all numerical values remain unchanged.
"""
import argparse
import json
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_language_stage_capture import compare


def normalize_hf(record, length):
    result = dict(record)
    q = record['layer_00.q_rope']
    values = record['layer_00.kernel_output']
    if len(q) != 1 or q[0].ndim != 2 or q[0].shape[0] != length or len(values) != 1:
        raise ValueError('one complete Q and kernel capture required')
    width = q[0].shape[1]
    if values[0].shape != (1, length * width):
        raise ValueError('exact known flattened HF kernel layout required')
    result['layer_00.kernel_output'] = [values[0].reshape(length, width)]
    return result


def replay(collection, output):
    collection, output = Path(collection), Path(output)
    transport_path = collection.with_suffix('.transport.json')
    transport = json.loads(transport_path.read_text())
    if (transport['status'] != 'transport_complete_independent_audit_required'
            or transport['state']['ExitCode'] != 1 or transport['state']['OOMKilled']):
        raise ValueError('complete original failed diagnostic transport required')
    tag = collection.name.removesuffix('-collected')
    launch = json.loads((collection.parent / (tag + '-launch.json')).read_text())
    if (transport['container_id'] != launch['container_id']
            or transport['image'] != launch['image']
            or transport['remote_root'] != launch['remote_root']
            or transport['files']['config.json'] != launch['config_sha256']
            or file_hash(collection.with_suffix('.tar')) != transport['archive_sha256']):
        raise ValueError('original launch and complete archive identity required')
    def read(name, tensor=False):
        path = collection / name
        if path.is_symlink() or file_hash(path) != transport['files'].get(name):
            raise ValueError('capture integrity differs: ' + name)
        return torch.load(path, map_location='cpu', weights_only=True) if tensor else json.loads(path.read_text())
    config = read('config.json')
    if read('evaluation/source.json')['files_sha256'] != launch['files_sha256']:
        raise ValueError('executed package identity differs')
    terminal = read('evaluation/terminal.json')
    if terminal.get('exit_code') != 1 or 'HF request length differs' not in terminal.get('error', ''):
        raise ValueError('known comparison layout failure required')
    lengths = [r['tensors']['input_ids']['shape'][1] for r in config['rows']]
    if len(lengths) != 32:
        raise ValueError('original complete rollout32 required')
    for mode in ('hf', 'baseline', 'acoustic'):
        if read(f'evaluation/{mode}/terminal.json') != {'exit_code': 0}:
            raise ValueError('all original forward phases must have completed')
    refs = [normalize_hf(read(f'evaluation/hf/row-{i:02d}.pt', True), n)
            for i, n in enumerate(lengths)]
    rows = []
    for i, row in enumerate(config['rows']):
        rows.append(dict(id=row['id'], comparisons={mode: compare([refs[i]],
            read(f'evaluation/{mode}/row-{i:02d}.pt', True), lengths=[lengths[i]])
            for mode in ('baseline', 'acoustic')}))
    batch = {mode: compare(refs, read(f'evaluation/{mode}/batch.pt', True), lengths=lengths)
             for mode in ('baseline', 'acoustic')}
    result = dict(status='failed_diagnostic_capture_layout_replayed_not_admission',
        container_id=transport['container_id'], original_exit_code=1,
        transport_sha256=file_hash(transport_path), rows=rows, batch=batch,
        full_model_parity_verified=False, training_dispatch_allowed=False,
        correction='HF kernel capture [1,T*D] reshaped to [T,D]; unchanged values')
    check_paths({'diagnostic_replay': output})
    if output.exists():
        raise FileExistsError(output)
    commit_json(output, result)
    return result


if __name__ == '__main__':
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = replay(a.collection, a.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'batch')}, indent=2))
