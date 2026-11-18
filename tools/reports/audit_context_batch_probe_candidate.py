"""Independently bind collected finite batch measurements to the sealed input.

No remote calls or runtime mutation. A successful audit covers fixed-policy
inference only, not actor updates, policy switching or training throughput.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from multimodal_jev.context_rollout_batch import bind_rollout
from tools.reports.launch_context_batch_probe_candidate import validate
from tools.reports.prepare_context_batch_probe_candidate import sha


def audit_measurements(serial, batched, rows, policy_id, input_sha256, max_queries, max_tokens):
    expected_kinds = ['warmup', 'timed', 'timed', 'reversed']
    lengths = [int(r['inputs']['input_ids'].shape[1]) for r in rows]
    reference = None
    timings, calls, peaks = {}, {}, {}
    for mode, report in [('serial', serial), ('batched', batched)]:
        if (report['mode'] != mode or report['policy_id'] != policy_id
                or report['input_sha256'] != input_sha256 or report['rows'] != len(rows)
                or [p['kind'] for p in report['passes']] != expected_kinds):
            raise ValueError('measurement source or pass population differs')
        timings[mode], calls[mode], peaks[mode] = [], [], []
        for index, p in enumerate(report['passes']):
            expected_order = list(range(len(rows)))
            if index == 3: expected_order.reverse()
            if p['order'] != expected_order or not math.isfinite(p['seconds']) or p['seconds'] <= 0:
                raise ValueError('measurement order or duration invalid')
            if len(p['records']) != len(rows) or len(p['sampling']) != len(rows):
                raise ValueError('incomplete per-input measurements')
            ids = []
            for slot, (row, record, sample) in enumerate(zip(rows, p['records'], p['sampling'], strict=True)):
                actual = bind_rollout(record, **row, policy_id=policy_id, group_size=16, seed=317+slot)['receipt']
                if actual != sample: raise ValueError('G16 receipt does not match actual input/probability')
                ids.append(record['request_id'])
            if len(set(ids)) != len(rows):
                raise ValueError('this fixed probe requires distinct requests for forward coverage')
            if reference is None: reference = p['records']
            if p['records'] != reference: raise ValueError('probability repeat/order/batch invariance failed')
            groups = p['telemetry']['counts']['language']
            if (not groups or any(not g or any(type(n) is not int or n <= 0 for n in g) for g in groups)
                    or Counter(n for g in groups for n in g) != Counter(lengths)):
                raise ValueError('actual language lengths do not cover the input population')
            if mode == 'serial' and groups != [[lengths[i]] for i in expected_order]:
                raise ValueError('serial model forward order differs')
            if mode == 'batched' and any(len(g) > max_queries or sum(g) > max_tokens for g in groups):
                raise ValueError('actual batch exceeds sealed capacity')
            memory = {k: p['telemetry'][k] for k in ('peak_allocated_bytes', 'peak_reserved_bytes')}
            if (any(type(v) is not int or v <= 0 for v in memory.values())
                    or memory['peak_allocated_bytes'] > memory['peak_reserved_bytes']):
                raise ValueError('invalid worker allocator peaks')
            if index in (1, 2):
                timings[mode].append(p['seconds']); calls[mode].append(len(groups)); peaks[mode].append(memory)
    means = {mode: sum(values)/len(values) for mode, values in timings.items()}
    return dict(status='finite_fixed_policy_inference_audited', rows=len(rows), policy_id=policy_id,
        exact_probabilities_and_G16=True, timed_seconds=timings, mean_seconds=means,
        serial_over_batched_time_ratio=means['serial']/means['batched'],
        actual_language_calls=calls,
        actual_call_reduction=all(b < a for a, b in zip(calls['serial'], calls['batched'])),
        worker_allocator_peaks=peaks,
        scope='fixed input/policy inference with hooks; no actor/switch/resume/whole-training speed claim',
        telemetry_limit='length coverage is not an independent tensor trace inside the device kernel')


def audit(prepared, collection, launch):
    import torch
    prepared, collection = Path(prepared), Path(collection)
    receipt, config = validate(prepared)
    launch = json.loads(Path(launch).read_text())
    transport_path = collection.with_suffix('.transport.json')
    transport = json.loads(transport_path.read_text())
    if (transport['status'] != 'transport_complete_independent_audit_required'
            or transport['state']['Status'] != 'exited' or transport['state']['ExitCode'] != 0
            or transport['state']['OOMKilled']):
        raise ValueError('successful terminal transport required')
    for key in ('container_id', 'image', 'remote_root'):
        if transport[key] != launch[key]: raise ValueError('transport/launch identity differs')
    if (launch['image'] != receipt['image'] or launch['remote_root'] != receipt['remote_root']
            or launch['files_sha256'] != receipt['files_sha256']
            or launch['config_sha256'] != sha(prepared/'package/config.json')):
        raise ValueError('launch differs from sealed source')
    if any(p.is_symlink() for p in collection.rglob('*')):
        raise ValueError('collection symlink forbidden')
    actual = {str(p.relative_to(collection)): sha(p) for p in collection.rglob('*') if p.is_file()}
    if actual != transport['files'] or actual['config.json'] != launch['config_sha256']:
        raise ValueError('collected files differ from transport/source')
    if sha(collection.with_suffix('.tar')) != transport['archive_sha256']:
        raise ValueError('transport archive changed')
    root = collection/'evaluation'
    def read(name): return json.loads((root/name).read_text())
    if read('source.json')['files_sha256'] != receipt['files_sha256']:
        raise ValueError('executed source differs')
    for name in ('terminal.json', 'serial-exit.json', 'batched-exit.json',
                 'serial/terminal.json', 'batched/terminal.json'):
        if read(name)['exit_code'] != 0: raise ValueError('measurement did not complete successfully')
    rows = torch.load(prepared/'package/inputs.pt', map_location='cpu', weights_only=True)
    if len(rows) != 128: raise ValueError('complete fixed TRAIN128 required')
    result = audit_measurements(read('serial/measurement.json'), read('batched/measurement.json'),
        rows, config['policy']['policy_id'], config['input_sha256'],
        config['policy']['max_batch_queries'], config['policy']['max_batch_tokens'])
    comparison = read('comparison.json')
    if (comparison['exact_probabilities_and_G16'] is not True or comparison['rows'] != result['rows']
            or comparison['actual_language_calls'] != [result['actual_language_calls'][m] for m in ('serial','batched')]
            or comparison['actual_call_reduction'] != result['actual_call_reduction']
            or comparison['serial_seconds'] != result['mean_seconds']['serial']
            or comparison['batched_seconds'] != result['mean_seconds']['batched']
            or comparison['time_ratio'] != result['serial_over_batched_time_ratio']):
        raise ValueError('runtime comparison disagrees with independent reconstruction')
    result.update(container_id=launch['container_id'], image=receipt['image'],
                  files_sha256=receipt['files_sha256'], transport_sha256=sha(transport_path))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('prepared', 'collection', 'launch', 'output'): parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    result = audit(args.prepared, args.collection, args.launch)
    with Path(args.output).open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))
