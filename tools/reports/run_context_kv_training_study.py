"""KV-only matched100 successor using the existing training and audit lifecycle."""
import argparse
import hashlib
import os
from pathlib import Path

from multimodal_jev.local_config import load_config
from tools.reports import context_kv_training_operations as training
from tools.reports import context_acoustic_training_operations as acoustic
from tools.reports import context_kv_probe_operations as probe_ops
from tools.reports import run_context_acoustic_training_study as shared
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.watch_context_acoustic_training import bind

read = shared.read
write_new = shared.write_new
check_paths = shared.check_paths
wait_driver = shared.wait_driver
collect_and_audit = shared.collect_and_audit
audit_collected = shared.audit_collected
_compare = bind(shared.compare, training=training)


def compare(baseline, candidate, audits):
    result = _compare(baseline, candidate, audits)
    result.update(schema='jev-kv-training-efficiency-v1',
                  scope='KV-only matched100 complete updates including activation; no efficacy/transfer claim')
    return result


def pins(c):
    required = {c[k] for k in ('operations', 'probe_operations', 'predecessor_driver', 'predecessor_config')}
    if not required <= set(c['pins']):
        raise ValueError('exact candidate, probe and predecessor required')
    for path, digest in c['pins'].items():
        if shared.file_hash(path) != digest:
            raise ValueError('KV training dependency changed: '+path)


def validate(c):
    pins(c)
    ops = read(c['operations']); probe = read(c['probe_operations'])
    training.validate(ops)
    _, config = probe_ops.validate(probe)
    baseline = ops['parent_operations']
    if read(read(c['predecessor_config'])['operations']) != baseline:
        raise ValueError('predecessor must train this exact acoustic parent')
    if probe['candidate_kv_cache_memory_bytes'] != ops['kv_cache_memory_bytes']:
        raise ValueError('training capacity must match measured probe')
    for i in range(2):
        policy = read(Path(baseline['prepared'])/f'package/service-{i}.json')['policy']
        expected = relocate(config['policy'], probe['remote_root'], baseline['remote_root'])
        expected['physical_devices'] = [6+i]
        if policy != expected:
            raise ValueError('probe and training must share initial policy and service settings')
    for name in acoustic.ADDITIONS[:3]:
        a = Path(probe['prepared'])/'package/tools/reports'/name
        b = Path(baseline['prepared'])/'package/tools/reports'/name
        if shared.file_hash(a) != shared.file_hash(b):
            raise ValueError('measured acoustic implementation differs')
    return ops, probe


def run(c, provenance):
    ops, probe = validate(c)
    root = Path(c['output']); check_paths({'kv_training_study': root})
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
        if terminal.get('exit_code') != 0 or terminal.get('status') != 'acoustic_training100_compared':
            raise ValueError('successful complete acoustic100 required; no implicit restart')
        validate(c)
        base = Path(probe['operations_root']); tag = probe['tag']
        admission = probe_ops.audit(probe, base/(tag+'-collected'), base/(tag+'-launch.json'))
        if not admission['exact_probabilities_and_logits'] or not admission['exact_sampling']:
            raise ValueError('exact KV numerical admission required')
        write_new(root/'kv-admission.json', admission)
        baseline = ops['parent_operations']
        old = audit_collected(baseline, validator=acoustic.validate,
                              command_builder=acoustic.command, output_auditor=acoustic.audit_outputs)
        write_new(root/'baseline-audit.json', old)
        pins(c)
        released = dict(ops, finite_training_dispatch_allowed=True)
        write_new(root/'released-operations.json', released)
        training.run(released, provenance)
        new = collect_and_audit(released, validator=training.validate, command_builder=training.command,
            output_auditor=training.audit_outputs, result_dirs=('run', 'journal', 'service-0', 'service-1'))
        write_new(root/'candidate-audit.json', new)
        write_new(root/'comparison.json', compare(baseline, released, [old, new]))
        write_new(root/'terminal.json', dict(exit_code=0, status='kv_training100_compared',
            automatic_promotion=False, formal_dispatched=False))
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
        validate(c); print('KV training dependency and measured-policy binding passed')
    else:
        run(c, provenance)
