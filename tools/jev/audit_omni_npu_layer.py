#!/usr/bin/env python3
"""Audit a completed original NPU layer probe; no inference or dispatch."""
import argparse
import hashlib
from pathlib import Path
import subprocess

from multimodal_jev.audio2tool_inference import write_once
from multimodal_jev.baseline import file_hash
from multimodal_jev.omni_layer_output_audit import (
    paired_rows, validate_accounting, validate_output,
)
from watch_omni_npu_layer import check_container, check_pins, inspect, read, successful

PLAN_SHA = 'fe95d22789e7bb91a56866712b032856af644d342abe97acd6bf0d7ff0cf3df6'
PREFIX_SHA = 'fb10bfc904f23e4ebb408ce3a685c9eac66a55f3bfd48b50dfae95443c5e08e7'


def audit(root, prefix, output):
    if output.exists():
        raise FileExistsError('preserve prior diagnostic audit')
    # A running or failed job must not be reduced to a partial successful audit.
    for path in (root/'evaluation-terminal.json', root/'run/terminal.json'):
        if read(path)['exit_code'] != 0:
            raise ValueError('successful original supervisor and application required')
    _, prep = check_pins(root, PLAN_SHA)
    launch = read(root/'launch.json'); cid = launch['container_id']
    if (file_hash(root/'intent.json') != launch['intent_sha256'] or read(root/'intent.json') != prep
            or launch['source_commit'] != prep['source_commit'] or launch['physical_devices'] != [4]):
        raise ValueError('original dispatch identity changed')
    actual = inspect(cid); successful(actual,cid); check_container(actual,prep,cid)
    if actual['RestartCount'] != 0 or actual['State']['Error'] or actual['State']['Dead']:
        raise ValueError('probe runtime failure or restart')
    for name, digest in prep['source_files'].items():
        if hashlib.sha256(subprocess.check_output(['git','show',prep['source_commit']+':'+name])).hexdigest() != digest:
            raise ValueError('runner snapshot differs from Git')
    run = root/'run'; result = read(run/'result.json'); identity = read(run/'identity.json')
    complete = read(root/'execution-complete.json')
    if (complete['container_id'] != cid or complete['result_sha256'] != file_hash(run/'result.json')
            or result['identity_sha256'] != file_hash(run/'identity.json')
            or result['status'] != 'diagnostic_complete' or result['preflight_only'] is not False
            or result['calls'] != 9 or result['npu_forward_calls'] != 9 or result['weight_updates'] != 0):
        raise ValueError('complete nine-call original execution required')
    pre_run = Path(prep['preflight_root'])/'run'
    if file_hash(pre_run/'identity.json') != read(pre_run/'result.json')['identity_sha256']:
        raise ValueError('accepted CPU preflight identity changed')
    pre_identity = read(pre_run/'identity.json')
    expected_identity = dict(pre_identity, preflight_only=False)
    if identity != expected_identity:
        raise ValueError('runtime identity differs from accepted CPU load preflight')
    if file_hash(prefix/'result.json') != PREFIX_SHA or identity['prefix_result_sha256'] != PREFIX_SHA:
        raise ValueError('original TRAIN6 prefix required')
    prefix_result = read(prefix/'result.json')
    if file_hash(prefix/'identity.json') != identity['prefix_identity_sha256']:
        raise ValueError('original TRAIN6 identity changed')
    records = result['outputs']; capacity = validate_accounting(records)
    import torch
    torch.set_num_threads(2)
    tensors = {}; verified = []
    for record, call, original in zip(records,identity['calls'],prefix_result['records'],strict=True):
        size, offset = record['size'], record['offset']
        if (any(record[k] != v for k,v in call.items())
                or (size,offset) != (original['micro_batch_size'],original['offset'])
                or record['input_file'] != original['tensor_file']
                or record['input_sha256'] != original['tensor_sha256']
                or record['input_shape'] != [size,1230,2048]
                or record['output_file'] != f'micro-{size}-offset-{offset}.pt'):
            raise ValueError('original call/input/output binding differs')
        inp_path = prefix/record['input_file']; out_path = run/record['output_file']
        if (file_hash(inp_path) != record['input_sha256'] or file_hash(out_path) != record['output_sha256']
                or read(out_path.with_suffix('.json')) != record):
            raise ValueError('saved tensor or per-call report changed')
        inp = torch.load(inp_path,map_location='cpu',weights_only=True)
        saved = torch.load(out_path,map_location='cpu',weights_only=True)
        validate_output(saved,inp)
        if list(saved['output'].shape) != [size,1230,2048] or inp['token_mask'].sum(-1).tolist() != call['valid_lengths']:
            raise ValueError('original padded shape or lengths differ')
        tensors[size,offset] = saved
        verified.append(dict(size=size,offset=offset,input_sha256=record['input_sha256'],output_sha256=record['output_sha256']))
    pairs = [dict(row=identity['order'][offset],offset=offset,
        **paired_rows(tensors[1,offset],tensors[2,offset//2*2],offset%2)) for offset in range(6)]
    report = dict(schema='omni-npu-layer-diagnostic-audit-v1',container_id=cid,image=prep['image'],
        source_commit=prep['source_commit'],source_files_verified=len(prep['source_files']),
        result_sha256=file_hash(run/'result.json'),identity_sha256=file_hash(run/'identity.json'),
        queue_plan_sha256=PLAN_SHA,physical_devices=[4],calls=verified,capacity=capacity,
        micro1_micro2_comparisons=pairs,weight_updates=0,
        scope='All nine saved original first-layer NPU outputs and six paired microbatch comparisons. Differences are descriptive, with no numerical acceptance threshold. No CPU/NPU equivalence, full-actor probability, FSDP, gradient or training claim.')
    write_once(output,report)
    return report


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser()
    for name in ('root','prefix','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root,args.prefix,args.output),indent=2))
