"""Wait for one exact driver, then collect and audit its terminal artifacts.

No model calls, polling loop, training restart or successor dispatch. A failed
attempt is recorded and preserved. Existing output is never overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select

from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.launch_context_batch_probe_candidate import write_new
from tools.reports.audit_context_macro_probe import collect_and_audit
from tools.reports.audit_context_macro_training import audit_outputs
from tools.reports.context_macro_training_operations import validate,command
from tools.reports.context_macro_efficiency_summary import summarize


def wait_driver(record, terminal):
    try:fd=os.pidfd_open(record['pid'])
    except ProcessLookupError:
        if not terminal.is_file():raise ValueError('driver missing without terminal; reconcile attempt')
        return
    try:
        proc=Path('/proc')/str(record['pid'])
        try:
            start=(proc/'stat').read_text().split(') ')[1].split()[19]
            digest=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest()
        except FileNotFoundError:
            if terminal.is_file():return
            raise ValueError('driver vanished without terminal')
        if start!=record['start_ticks'] or digest!=record['cmdline_sha256']:
            raise ValueError('driver PID reused or command changed')
        poll=select.poll();poll.register(fd,select.POLLIN);poll.poll()
    finally:os.close(fd)


def run(c, *, repair_legacy_access=False):
    base=Path(c['operations_root']);check_paths({'finalization':base})
    started=base/'finalization-started.json';terminal=base/'finalization-terminal.json'
    write_new(started,dict(driver=json.loads((base/'driver.json').read_text()),action='wait_collect_audit_only'))
    try:
        wait_driver(json.loads((base/'driver.json').read_text()),base/'terminal.json')
        state=json.loads((base/'terminal.json').read_text())
        if state['exit_code']!=0:
            write_new(terminal,dict(status='training_failed_preserved',driver_terminal=state,restart=False));return
        if repair_legacy_access:
            from tools.reports.repair_context_macro_artifact_access import run as repair
            repair(c)
        result=collect_and_audit(c,validator=validate,command_builder=command,
            output_auditor=audit_outputs,result_dirs=('run','journal','service-0','service-1'))
        training=json.loads((Path(c['prepared'])/'package/training.json').read_text())
        timing=summarize(base/'collected/run',start_update=training['start_update'],stop_update=training['stop_update'])
        write_new(base/'efficiency-summary.json',timing)
        write_new(terminal,dict(status='finite_training_collected_audited',audit_status=result['status'],
            fresh_process_replay_verified=result['fresh_process_replay_verified'],training_dispatch_allowed=False))
    except BaseException as exc:
        write_new(terminal,dict(status='finalization_failed_preserved',error=repr(exc),restart=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--repair-legacy-access',action='store_true',help='Repair terminal artifact modes with unchanged-byte receipt before collecting frozen legacy runtimes')
    a=p.parse_args();c,_=load_config(a.config,a.local)
    import torch
    torch.set_num_threads(1);run(c,repair_legacy_access=a.repair_legacy_access)
