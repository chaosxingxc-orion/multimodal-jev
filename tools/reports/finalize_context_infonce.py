"""Wait for an exact InfoNCE driver, audit, then bind its full source curve.

Deterministic postprocessing only: no model calls, retries, training dispatch or
inference dispatch. All full curves remain pending until actually evaluated.
"""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_infonce_formal import audit
from tools.reports.context_infonce_source_plan import build


def wait_exact(record,terminal):
    terminal=Path(terminal)
    try:fd=os.pidfd_open(record['pid'])
    except ProcessLookupError:
        if not terminal.is_file():raise ValueError('driver missing without terminal')
        return
    try:
        proc=Path('/proc')/str(record['pid'])
        try:
            stat=(proc/'stat').read_text().split(') ',1)[1].split()[19]
            argv=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        except FileNotFoundError:
            if terminal.is_file():return
            raise ValueError('driver disappeared without terminal')
        if stat!=record['process_start_ticks'] or argv!=record['argv']:
            raise ValueError('exact driver identity differs')
        poll=select.poll();poll.register(fd,select.POLLIN);poll.poll()
    finally:os.close(fd)
    if not terminal.is_file():raise ValueError('driver exited without terminal')


def run(c,provenance):
    root=Path(c['finalization_root']);check_paths({'finalization':root});root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        ops,_=load_config(c['operations_config'],c['operations_local'])
        if 'pipeline' in c:
            from multimodal_jev.baseline import file_hash
            dep=c['pipeline'];parent=Path(dep['root'])
            proof=Path(str(parent)+'-driver')/'started.json'
            if file_hash(proof)!=dep['driver_sha256']:
                raise ValueError('Stage1 pipeline driver identity changed')
            record=json.loads(proof.read_bytes());terminal_path=parent/'terminal.json'
        else:
            driver=Path(ops['operations_root'])/'driver'
            record=json.loads((driver/'started.json').read_text());terminal_path=driver/'terminal.json'
        commit_json(root/'waiting.json',record)
        wait_exact(record,terminal_path)
        terminal=json.loads(terminal_path.read_text())
        if terminal['exit_code']!=0:raise ValueError('training driver failed; preserve exact attempt')
        if 'pipeline' in c:
            launch=json.loads((Path(ops['operations_root'])/(ops['tag']+'-launch.json')).read_bytes())
            if (file_hash(parent/'formal-audit.json')!=terminal['audit_sha256']
                    or terminal['status']!='formal_infonce_artifacts_audited'
                    or terminal['result']['container_id']!=launch['container_id']
                    or launch['manifest_sha256']!=ops['manifest_sha256']):
                raise ValueError('exact pipeline and formal audit required')
        # build performs independent training audit before exposing any policy.
        plan=build(c,provenance)
        if 'pipeline' in c:
            from multimodal_jev.stage2_dataset_release import require_endpoint_layout
            require_endpoint_layout(plan,c['pipeline']['expected_stage1_layout'])
            if plan['source_audit']!=json.loads((parent/'formal-audit.json').read_bytes()):
                raise ValueError('independent formal audit changed')
        commit_json(root/'training-audit.json',plan['source_audit'])
        commit_json(root/'source-evaluation-plan.json',plan)
        result=dict(status='formal_infonce_audited_curve_bound',points=len(plan['checkpoints']),
            reference_endpoint=plan['reference_endpoint'],evaluation_complete=False)
        commit_json(root/'terminal.json',dict(exit_code=0,**result));return result
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False));raise


def detach(template,local):
    c,_=load_config(template,local);root=Path(c['finalization_root']).parent/(Path(c['finalization_root']).name+'-driver')
    check_paths({'finalization_driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.finalize_context_infonce','--config',str(Path(template).resolve()),
          '--local',str(Path(local).resolve()),'--action','run']
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,argv=argv,started_unix=time.time(),
        process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    commit_json(root/'started.json',record);return record


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['run','detach'],default='run');a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    print(json.dumps(detach(a.config,a.local) if a.action=='detach' else run(c,provenance),indent=2))
