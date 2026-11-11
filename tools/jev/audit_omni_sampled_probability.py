#!/usr/bin/env python3
"""Rebuild accepted sampled inputs and audit saved complete probability evidence.

CPU only; no model load or forward. Docker/source/protocol registration and
initialization authentication remain separate coordinator responsibilities.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.omni_probability_audit import compare, validate_protocol
from multimodal_jev.omni_probability_shards import audit_observed_shards, shard_plan
from multimodal_jev.omni_sampling_acceptance import read_json
from multimodal_jev.sfc_reward_batch import digest
from preflight_omni_sampled_probability import run as prepare_inputs


def run(args):
    prepared = args.output/'input-preflight'; prepared.mkdir()
    _, records, batch, _ = prepare_inputs(SimpleNamespace(root=args.sampling,
        acceptance=args.acceptance, acceptance_hash=args.acceptance_hash, output=prepared))
    protocol = read_json(args.protocol)
    validate_protocol(records, protocol=protocol, protocol_sha256=args.protocol_hash)
    plans = [shard_plan(records, world_size=2, micro_batch_size=size) for size in (1, 2)]
    inputs = read_json(prepared/'result.json')
    bindings = []
    for rank in (0, 1):
        root = args.root/f'rank-{rank}'
        if read_json(root/'terminal.json')['exit_code'] != 0:
            raise ValueError('both rank applications must complete successfully')
        binding = read_json(root/'registered-inputs.json')['binding']
        if (binding['protocol'] != protocol or binding['protocol_sha256'] != args.protocol_hash
                or binding['plans'] != plans
                or binding['sampling_acceptance_sha256'] != args.acceptance_hash
                or any(binding[k] != inputs[k] for k in ('tensors', 'prepared_inputs_sha256',
                    'processor_parity_sha256', 'sampling_identity_sha256'))):
            raise ValueError('fresh input preflight differs from registered rank inputs')
        result = read_json(root/'result.json')
        if (result['status'] != 'probabilities_captured' or result['rank'] != rank
                or result['binding_sha256'] != digest(binding)
                or result['weight_updates'] != 0 or result['targets_read'] is not False
                or result['forward_calls'] != sum(p['forwards_per_rank'] for p in plans)):
            raise ValueError('complete frozen rank result required')
        bindings.append(binding)
    if bindings[0] != bindings[1]:
        raise ValueError('same registered inputs required on both ranks')
    runs = []; observations = []
    for plan in plans:
        name = f"micro{plan['micro_batch_size']}"
        outputs = [read_json(args.root/f'rank-{rank}'/(name+'.json')) for rank in (0, 1)]
        checked = audit_observed_shards(batch, records, outputs, name=name,
            plan=plan, plan_sha256=digest(plan))
        runs.extend([checked['native'], checked['reference']])
        observations.append(dict(name=name, ranks=checked['observed_ranks'],
            normalization_calls=checked['normalization_calls']))
    if [r['name'] for r in runs] != protocol['run_names']:
        raise ValueError('all native and same-logits reference runs required')
    if digest(runs) != digest(read_json(args.root/'merged-runs.json')):
        raise ValueError('saved merged probabilities differ from raw observations')
    report = compare(records, runs, protocol=protocol, protocol_sha256=args.protocol_hash)
    if digest(report) != digest(read_json(args.root/'probability-audit.json')):
        raise ValueError('saved numerical audit does not reproduce')
    atomic_json(args.output/'probability-audit.json', report)
    if not report['numerical_gate_passed']:
        raise ValueError('registered numerical gate failed')
    atomic_json(args.output/'result.json', dict(status='saved_evidence_verified',
        protocol_sha256=args.protocol_hash, binding_sha256=digest(bindings[0]),
        observations=observations, responses=len(records),
        probability_audit_sha256=file_hash(args.output/'probability-audit.json'),
        scope='Fresh accepted CPU processor/DataProto and complete saved observation/merge/numeric '
              'consistency only. No independent NPU execution, Docker/source/initialization '
              'attestation or proof of pre-dispatch protocol registration.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'sampling', 'acceptance', 'protocol', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--acceptance-hash', required=True)
    parser.add_argument('--protocol-hash', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', dict(exit_code=1, error=repr(exc)))
        raise
    atomic_json(args.output/'terminal.json', dict(exit_code=0))


if __name__ == '__main__':
    main()
