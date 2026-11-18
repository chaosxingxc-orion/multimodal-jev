"""Finite real-input macro128 probe using two already started isolated services.

No optimizer, labels, weight updates or service restarts. The authenticated
serial measurement is a parity reference, not an end-to-end speed baseline.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from multimodal_jev.context_rollout_batch import bind_rollout
from multimodal_jev.context_logits_wire import validate_logits_record
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_bulk_rpc_candidate import validate_rows
from tools.reports.context_macro_client import connect_pool


def check_reference(reference, rows, policy_id, input_sha256):
    if (reference['mode'] != 'serial' or reference['rows'] != 128 or len(rows) != 128
            or reference['policy_id'] != policy_id or reference['input_sha256'] != input_sha256
            or [p['kind'] for p in reference['passes']] != ['warmup','timed','timed','reversed']):
        raise ValueError('complete matching serial TRAIN128 measurement required')
    expected = reference['passes'][0]
    for p in reference['passes']:
        if len(p['records']) != len(rows) or len(p['sampling']) != len(rows):
            raise ValueError('complete serial records and sampling required')
        for i, (row, record) in enumerate(zip(rows, p['records'], strict=True)):
            validate_logits_record(record)
            sampling = bind_rollout(record, **row, policy_id=policy_id,
                                    group_size=16, seed=317+i)['receipt']
            if sampling != p['sampling'][i]: raise ValueError('serial sampling binding differs')
        if p['records'] != expected['records'] or p['sampling'] != expected['sampling']:
            raise ValueError('serial repeat/order invariance failed')
    return expected


def measure(pool, rows, expected, *, audio_token_id):
    policy = pool.bundle['policy_id']; passes = []
    orders = [list(range(128))]*3 + [list(reversed(range(128)))]
    for index, order in enumerate(orders):
        started = time.perf_counter()
        values = pool.score_many([rows[i] for i in order], audio_token_id=audio_token_id)
        seconds = time.perf_counter()-started
        if len(values) != len(order) or pool.bundle['policy_id'] != policy:
            raise ValueError('complete macro under unchanged policy required')
        canonical = [None]*128; sampling = [None]*128
        for i, record in zip(order, values, strict=True):
            validate_logits_record(record)
            canonical[i] = record
            sampling[i] = bind_rollout(record, **rows[i], policy_id=policy,
                                      group_size=16, seed=317+i)['receipt']
        if canonical != expected['records'] or sampling != expected['sampling']:
            raise ValueError('actual dual-replica logits/probability/G16 differs from serial reference')
        receipt = pool.last_receipt
        if (receipt['completed_indices'] != list(range(128)) or receipt['policy_id'] != policy
                or receipt['status'] != 'complete_macro_inference'):
            raise ValueError('complete macro receipt required')
        passes.append(dict(kind='warmup' if index==0 else 'reversed' if index==3 else 'timed',
            order=order, seconds=seconds, records=canonical, sampling=sampling,
            macro_receipt=dict(receipt)))
    seconds = sum(p['seconds'] for p in passes[1:3])/2
    return dict(policy_id=policy, rows=128, passes=passes, timed_mean_seconds=seconds,
        queries_per_second=128/seconds, exact_actual_logits=True,
        exact_probabilities_and_G16=True, runtime_identity=pool.runtime_identity,
        scope='Two isolated services, complete macro128 IPC/journal/validation; excludes startup, actor, export, activation and checkpoint. No training speedup claim.')


def run(config, provenance):
    root = Path(config['output']); check_paths({'measurement':root})
    if root.exists(): raise ValueError('existing finite measurement attempt')
    for path, expected_hash in config['pins'].items():
        if file_hash(path) != expected_hash: raise ValueError('pinned dependency changed: '+path)
    for key in ('inputs','serial_reference'):
        if config[key] not in config['pins']: raise ValueError('input/reference must be pinned')
    rows = validate_rows(torch.load(config['inputs'], map_location='cpu', weights_only=True))
    bundle = read_execution_bundle(config['policy']['snapshot'],
        expected_policy_id=config['policy']['policy_id'], expected_contract=config['policy']['contract'],
        expected_base_id=config['policy']['base_id'])
    expected = check_reference(json.loads(Path(config['serial_reference']).read_text()), rows,
        bundle['policy_id'], config['pins'][config['inputs']])
    root.mkdir(parents=True); pool = None
    try:
        commit_json(root/'provenance.json', provenance)
        pool = connect_pool(config['macro_client'], bundle=bundle)
        result = measure(pool, rows, expected, audio_token_id=config['audio_token_id'])
        commit_json(root/'measurement.json', result)
        pool.close(); pool = None
        commit_json(root/'terminal.json', dict(exit_code=0))
        return result
    except BaseException as exc:
        commit_json(root/'terminal.json', dict(exit_code=1, error=repr(exc))); raise
    finally:
        if pool is not None: pool.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--local',required=True)
    args=parser.parse_args(); config,provenance=load_config(args.config,args.local)
    run(config,provenance)
