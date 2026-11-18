"""Finite label-free transport measurement; no policy selection or training.

Use separate otherwise-identical services for each protocol. Include one warmup
then two timed passes over the same frozen input batch. Timing includes client
serialization, IPC, sequential engine forwards, and record validation.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from multimodal_jev.context_rollout_batch import bind_rollout
from tools.jev.context_rollout_rpc import ContextEngineClient
from tools.reports.context_bulk_rpc_candidate import BulkContextEngineClient, validate_rows
from tools.reports.serve_context_transport_candidate import PROTOCOLS


def measure(client, rows, *, protocol, audio_token_id, clock=time.perf_counter):
    if protocol not in PROTOCOLS:
        raise ValueError('Known explicit transport required')
    validate_rows(rows)
    passes = []
    for index in range(3):
        started = clock()
        if protocol == 'sequential_v1':
            records = [client.score(**row, audio_token_id=audio_token_id) for row in rows]
        else:
            records = client.score_many(rows, audio_token_id=audio_token_id)
        if len(records) != len(rows):
            raise ValueError('Incomplete pass')
        receipts = [bind_rollout(record, **row, policy_id=client.bundle['policy_id'],
                    group_size=16, seed=317+slot)['receipt']
                    for slot,(row,record) in enumerate(zip(rows,records,strict=True))]
        elapsed = clock()-started
        if not 0 < elapsed < float('inf'):
            raise ValueError('Finite positive elapsed time required')
        passes.append(dict(warmup=index==0,seconds=elapsed,records=records,sampling_receipts=receipts))
        if index and any(passes[index][k] != passes[0][k] for k in ('records','sampling_receipts')):
            raise ValueError('Same-policy repeated pass changed')
    return dict(protocol=protocol,policy_id=client.bundle['policy_id'],rows=len(rows),passes=passes,
                scope='Finite fixed-input transport timing; excludes startup and switching, not actor throughput')


def compare(sequential, bulk):
    if sequential['protocol']!='sequential_v1' or bulk['protocol']!='bulk_sequential_v1':
        raise ValueError('Explicit ordered protocol pair required')
    for key in ('policy_id','rows'):
        if sequential[key]!=bulk[key]:raise ValueError('Different comparison '+key)
    for report in (sequential,bulk):
        if len(report['passes'])!=3 or [p['warmup'] for p in report['passes']]!=[True,False,False]:
            raise ValueError('One warmup and two timed passes required')
    for a,b in zip(sequential['passes'],bulk['passes'],strict=True):
        for key in ('records','sampling_receipts'):
            if a[key]!=b[key]:raise ValueError('Input, probability or sampling mismatch')
    means=[sum(p['seconds'] for p in r['passes'][1:])/2 for r in (sequential,bulk)]
    if any(not 0<x<float('inf') for x in means):raise ValueError('Invalid timing')
    return dict(passed=True,rows=sequential['rows'],policy_id=sequential['policy_id'],
                exact_records_and_G16=True,sequential_mean_seconds=means[0],bulk_mean_seconds=means[1],
                observed_time_ratio=means[0]/means[1],
                scope='Two timed passes only; excludes startup, policy switching, actor and checkpoint costs')


def run(spec, output):
    output=Path(output);output.mkdir(exist_ok=False)
    def write(name,value):
        with (output/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    client=None
    try:
        protocol=spec['transport_protocol']
        if protocol not in PROTOCOLS:raise ValueError('Unknown transport')
        if str(Path(__file__).resolve()) not in spec['pins']:
            raise ValueError('Measurement entrypoint must be pinned')
        for name,digest in spec['pins'].items():
            if file_hash(name)!=digest:raise ValueError('Measurement dependency changed')
        if file_hash(spec['inputs'])!=spec['inputs_sha256']:raise ValueError('Input payload changed')
        rows=validate_rows(torch.load(spec['inputs'],map_location='cpu',weights_only=True))
        policy=spec['policy']
        bundle=read_execution_bundle(policy['snapshot'],expected_policy_id=policy['policy_id'],
            expected_contract=policy['contract'],expected_base_id=policy['base_id'])
        ready=json.loads(Path(spec['ready']).read_text())
        if (ready['transport_protocol']!=protocol or ready['policy_id']!=bundle['policy_id']
                or ready['contract']!=bundle['contract'] or ready['physical_devices']!=policy['physical_devices']
                or ready['model_forward_mode']!='sequential_single_row'
                or ready['entrypoint_sha256']!=spec['service_entrypoint_sha256']):
            raise ValueError('Actual service identity differs')
        write('input.json',spec);write('ready.json',ready)
        cls=ContextEngineClient if protocol=='sequential_v1' else BulkContextEngineClient
        client=cls(spec['socket_path'],spec['exchange'],bundle=bundle)
        report=measure(client,rows,protocol=protocol,audio_token_id=spec['audio_token_id'])
        client.close();client=None
        write('measurement.json',report);write('terminal.json',dict(passed=True))
        return report
    except BaseException as exc:
        write('terminal.json',dict(passed=False,error=repr(exc),retry=False))
        raise
    finally:
        if client is not None:client.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(json.loads(a.spec.read_text()),a.output)
