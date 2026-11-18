"""Run the sealed two-update epoch-tail diagnostic after the KV efficiency queue."""
import argparse
import hashlib
import os
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports import context_macro_tail_operations_v2 as tail
from tools.reports import context_macro_tail_operations as tail_original
from tools.reports import run_context_kv_training_study as shared

read = shared.read
write_new = shared.write_new
check_paths = shared.check_paths
wait_driver = shared.wait_driver
collect_and_audit = shared.collect_and_audit


def validate(c):
    required = {c[k] for k in ('operations', 'predecessor_driver', 'predecessor_config')}
    if not required <= set(c['pins']):
        raise ValueError('exact tail package and queue predecessor required')
    for path, digest in c['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('tail admission dependency changed: '+path)
    ops = read(c['operations']); receipt = tail.validate(ops)
    queue = read(c['predecessor_config']); kv = read(queue['operations'])
    if (c['predecessor_terminal'] != str(Path(queue['output'])/'terminal.json')
            or (ops['host'], ops['lease']) != (kv['host'], kv['lease'])):
        raise ValueError('exact predecessor on the same leased device set required')
    # This diagnoses original service32 tail handling, not the KV/acoustic path.
    parent = kv['parent_operations']['parent_operations']
    if receipt['parent_manifest_sha256'] != parent['manifest_sha256']:
        raise ValueError('tail must descend from the shared service32 baseline')
    if (ops.get('finite_training_dispatch_allowed') is not False
            or ops.get('tail_diagnostic_dispatch_allowed') is not False):
        raise ValueError('unreleased diagnostic-only package required')
    if (Path(ops['operations_root'])/'attempt.json').exists():
        raise ValueError('tail already attempted; reconcile exact handles')
    return ops


def run(c, provenance):
    ops = validate(c)
    root = Path(c['output']); check_paths({'tail_admission_queue': root})
    root.mkdir(parents=True, exist_ok=False)
    write_new(root/'config.json', c); write_new(root/'provenance.json', provenance)
    proc = Path('/proc')/str(os.getpid())
    write_new(root/'driver.json', dict(pid=os.getpid(),
        start_ticks=(proc/'stat').read_text().split(') ', 1)[1].split()[19],
        cmdline_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest()))
    try:
        driver = read(c['predecessor_driver'])
        write_new(root/'waiting.json', dict(predecessor=driver, mechanism='exact_pidfd', remote_polling=False))
        wait_driver(driver, Path(c['predecessor_terminal']))
        terminal = read(c['predecessor_terminal'])
        if terminal.get('exit_code') != 0 or terminal.get('status') != 'kv_training100_compared':
            raise ValueError('successful complete KV efficiency queue required')
        ops = validate(c)
        released = dict(ops, tail_diagnostic_dispatch_allowed=True)
        write_new(root/'released-operations.json', released)
        tail.run(released, provenance)
        result = collect_and_audit(released, validator=tail.validate, command_builder=tail.command,
            output_auditor=tail_original.audit_outputs, result_dirs=('run', 'journal', 'service-0', 'service-1'))
        if result['status'] != 'finite_dp2_tail_outputs_audited':
            raise ValueError('complete independently audited tail outputs required')
        write_new(root/'tail-audit.json', result)
        write_new(root/'terminal.json', dict(exit_code=0, status='tail_v2_collected_audited',
            formal_dispatched=False, acoustic_kv_tail_admitted=False,
            scope='original service32 epoch-tail diagnostic; no formal history or efficacy claim'))
    except BaseException as exc:
        write_new(root/'terminal.json', dict(exit_code=1, error=repr(exc), restart=False, preserve_handles=True))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--local', required=True)
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args(); c, provenance = load_config(args.config, args.local)
    import torch
    torch.set_num_threads(1)
    if args.validate_only:
        validate(c); print('tail v2 diagnostic queue validated; no dispatch')
    else: run(c, provenance)
