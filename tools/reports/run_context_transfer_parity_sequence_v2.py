"""Reuse the frozen three-target parity sequence with the repaired tail lifecycle."""
import argparse
import hashlib
import inspect
import os
from pathlib import Path

from multimodal_jev.local_config import load_config
from tools.reports import run_context_transfer_parity_sequence as original
from tools.reports import context_macro_tail_operations_v2 as tail
from tools.reports.watch_context_acoustic_training import bind


def sequence():
    validate = bind(original.validate, tail=tail)
    source = inspect.getsource(original.run)
    before = "terminal.get('status')!='tail_diagnostic_collected_audited'"
    if source.count(before) != 1:
        raise ValueError('unique predecessor terminal contract required')
    source = source.replace(before, "terminal.get('status')!='tail_v2_collected_audited'")
    ns = dict(vars(original), tail=tail, validate=validate)
    exec(compile(source, '<transfer-parity-tail-v2>', 'exec'), ns)
    return validate, ns['run']


validate, run = sequence()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--local', required=True)
    p.add_argument('--validate-only', action='store_true'); p.add_argument('--driver-root')
    args = p.parse_args(); c, provenance = load_config(args.config,args.local)
    import torch
    torch.set_num_threads(1)
    if args.validate_only:
        validate(c); print('three-target parity after tail v2 validated; no dispatch')
    else:
        if not args.driver_root:p.error('--driver-root required')
        validate(c)
        root = Path(args.driver_root); original.check_paths({'parity_driver':root})
        root.mkdir(parents=True,exist_ok=False)
        proc = Path('/proc')/str(os.getpid())
        original.write_new(root/'driver.json', dict(pid=os.getpid(),
            start_ticks=(proc/'stat').read_text().split(') ',1)[1].split()[19],
            cmdline_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest()))
        run(c, provenance)
