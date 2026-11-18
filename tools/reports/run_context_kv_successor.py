"""Wait for the exact bucket-study process, then run one isolated KV probe."""
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
from tools.reports import context_kv_probe_operations as kv
from tools.reports import context_acoustic_probe_operations as acoustic


def pins(c):
    required = {c['predecessor_driver'], c['operations']}
    if not required <= set(c['pins']):
        raise ValueError('exact driver and prepared operations pins required')
    for path, digest in c['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('KV successor dependency changed: '+path)


def run(c, provenance):
    pins(c)
    ops = json.loads(Path(c['operations']).read_text())
    kv.validate(ops)
    root = Path(c['output']); check_paths({'kv_successor': root})
    root.mkdir(parents=True, exist_ok=False)
    write_new(root/'config.json', c); write_new(root/'provenance.json', provenance)
    proc = Path('/proc')/str(os.getpid())
    write_new(root/'driver.json', dict(pid=os.getpid(),
        start_ticks=(proc/'stat').read_text().split(') ',1)[1].split()[19],
        cmdline_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest()))
    try:
        driver = json.loads(Path(c['predecessor_driver']).read_text())
        write_new(root/'waiting.json', dict(predecessor=driver, mechanism='exact_pidfd', remote_polling=False))
        wait_driver(driver, Path(c['predecessor_terminal']))
        terminal = json.loads(Path(c['predecessor_terminal']).read_text())
        if terminal.get('exit_code') != 0 or terminal.get('status') != 'acoustic_and_bucket_efficiency_compared':
            raise ValueError('successful complete efficiency study required; preserve failed predecessor')
        pins(c); kv.validate(ops)
        parent = ops['parent_operations']; base = Path(parent['operations_root']); tag = parent['tag']
        audit = acoustic.audit(parent, base/(tag+'-collected'), base/(tag+'-launch.json'))
        if not audit['exact_probabilities_and_logits'] or not audit['exact_sampling']:
            raise ValueError('exact acoustic numerical admission required')
        write_new(root/'acoustic-admission.json', audit)
        released = dict(ops, finite_inference_dispatch_allowed=True)
        write_new(root/'released-operations.json', released)
        result = kv.namespace(released)['launch'](released, provenance)
        write_new(root/'comparison.json', result)
        write_new(root/'terminal.json', dict(exit_code=0, status='kv_capacity_independently_compared',
            automatic_promotion=False, training_dispatched=False))
    except BaseException as exc:
        write_new(root/'terminal.json', dict(exit_code=1, error=repr(exc), restart=False)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--local', required=True)
    args = p.parse_args(); c, provenance = load_config(args.config,args.local)
    import torch
    torch.set_num_threads(1)
    run(c, provenance)
