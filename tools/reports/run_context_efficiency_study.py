"""Run SLURP efficiency arms after the audited matched100 baseline.

Tail admission and target parity remain requirements for later formal/transfer
work, not dependencies of these full-batch, source-only efficiency probes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.finalize_context_macro_training import wait_driver
from tools.reports.launch_context_batch_probe_candidate import write_new
from tools.reports import run_context_efficiency_successor as shared


def admit(c, baseline):
    required = {c['predecessor_driver'], c['predecessor_terminal'], c['comparison'],
                c['acoustic_operations'], c['bucket_operations']}
    if not required <= set(c['pins']):
        raise ValueError('exact baseline evidence and candidate configurations must be pinned')
    shared.pins(c)
    terminal = shared.read(c['predecessor_terminal'])
    comparison = shared.read(c['comparison'])
    if (terminal.get('exit_code') != 0
            or terminal.get('status') != 'service_batch32_100_steps_compared'
            or comparison.get('schema') != 'jev-service-batch-efficiency-comparison-v1'
            or comparison.get('matched_actual_queries') is not True
            or comparison.get('matched_reference_misses') is not True
            or comparison['candidate']['optimizer_steps'] != 100
            or comparison['candidate']['all_updates']['real_query_exposures'] != 12800
            or comparison['source_audits'][1]['status'] != 'finite_dp2_efficiency_audited'
            or comparison['source_audits'][1]['files_sha256'] != baseline['manifest_sha256']):
        raise ValueError('complete matched100 parent baseline required')
    # Recheck actual collected bytes; a previous success report alone is insufficient.
    audit = shared.audit_collected(baseline)
    if (audit['status'] != 'finite_dp2_efficiency_audited'
            or audit['files_sha256'] != baseline['manifest_sha256']):
        raise ValueError('independent baseline re-audit required')
    return audit


def run(c, provenance):
    shared.pins(c)
    acoustic = shared.read(c['acoustic_operations'])
    bucket = shared.read(c['bucket_operations'])
    shared.acoustic.validate(acoustic)
    shared.bucket.validate(bucket)
    root = Path(c['output']); check_paths({'efficiency_study': root})
    root.mkdir(parents=True, exist_ok=False)
    write_new(root/'config.json', c); write_new(root/'provenance.json', provenance)
    proc = Path('/proc')/str(os.getpid())
    write_new(root/'driver.json', dict(pid=os.getpid(),
        start_ticks=(proc/'stat').read_text().split(') ', 1)[1].split()[19],
        cmdline_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest()))
    try:
        predecessor = shared.read(c['predecessor_driver'])
        wait_driver(predecessor, Path(c['predecessor_terminal']))
        baseline = bucket['parent_operations']
        base_audit = admit(c, baseline)
        write_new(root/'baseline-audit.json', base_audit)
        result = shared.acoustic.namespace(acoustic)['launch'](acoustic, provenance)
        write_new(root/'acoustic-comparison.json', result)
        shared.pins(c)
        shared.bucket.run(bucket, provenance)
        result = shared.collect_and_audit(bucket, validator=shared.bucket.validate,
            command_builder=shared.bucket.command, output_auditor=shared.bucket.audit_outputs,
            result_dirs=('run', 'journal', 'service-0', 'service-1'))
        write_new(root/'bucket-audit.json', result)
        comparison = shared.compare_training(baseline, bucket, base_audit, result)
        write_new(root/'bucket-comparison.json', comparison)
        write_new(root/'terminal.json', dict(exit_code=0,
            status='acoustic_and_bucket_efficiency_compared', formal_dispatched=False,
            automatic_promotion=False, target_parity_verified=False, tail_admission_verified=False))
    except BaseException as exc:
        write_new(root/'terminal.json', dict(exit_code=1, error=repr(exc),
            restart=False, preserve_handles=True))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--local', required=True)
    args = p.parse_args(); c, provenance = load_config(args.config, args.local)
    import torch
    torch.set_num_threads(1)
    run(c, provenance)
