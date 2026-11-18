"""Wait for the exact service study, run one tail diagnostic, collect and audit.

No periodic remote polling, retries, model analysis or formal training launch.
The existing study keeps sole ownership of its current resource window.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.finalize_context_macro_training import wait_driver
from tools.reports.launch_context_batch_probe_candidate import write_new
from tools.reports import context_macro_tail_operations as tail
from tools.reports.audit_context_macro_probe import collect_and_audit


def read(path):return json.loads(Path(path).read_text())


def validate(c):
    required={c['operations'],c['predecessor_driver'],c['study_config']}
    if not required<=set(c['pins']):raise ValueError('pinned operations and predecessor required')
    for path,digest in c['pins'].items():
        if file_hash(path)!=digest:raise ValueError('tail successor dependency changed: '+path)
    ops=read(c['operations']);receipt=tail.validate(ops)
    if (ops.get('tail_diagnostic_dispatch_allowed') is not True
            or ops.get('finite_training_dispatch_allowed') is not False):
        raise ValueError('explicit diagnostic-only dispatch required')
    if (Path(ops['operations_root'])/'attempt.json').exists():
        raise ValueError('tail already attempted; reconcile handles')
    study=read(c['study_config'])
    if (c['predecessor_terminal']!=str(Path(study['output'])/'terminal.json')
            or c['comparison']!=str(Path(study['output'])/'comparison.json')):
        raise ValueError('exact predecessor study output required')
    parent=read(study['training_operations'])
    if (receipt['parent_manifest_sha256']!=parent['manifest_sha256']
            or ops['host']!=parent['host'] or ops['lease']!=parent['lease']):
        raise ValueError('diagnostic must follow its actual parent study on same leased devices')
    return ops,parent


def run(c,provenance):
    ops,parent=validate(c)
    root=Path(c['output']);check_paths({'tail_successor':root});root.mkdir(parents=True,exist_ok=False)
    write_new(root/'config.json',c);write_new(root/'provenance.json',provenance)
    try:
        predecessor=read(c['predecessor_driver'])
        write_new(root/'waiting.json',dict(process=predecessor,terminal=c['predecessor_terminal']))
        wait_driver(predecessor,Path(c['predecessor_terminal']))
        ops,parent=validate(c)
        state=read(c['predecessor_terminal'])
        if state.get('exit_code')!=0 or state.get('status')!='service_batch32_100_steps_compared':
            raise ValueError('complete successful service study required')
        comparison=read(c['comparison'])
        if (comparison.get('schema')!='jev-service-batch-efficiency-comparison-v1'
                or comparison.get('matched_actual_queries') is not True
                or comparison.get('matched_reference_misses') is not True
                or comparison['candidate']['optimizer_steps']!=100
                or comparison['candidate']['all_updates']['real_query_exposures']!=12800
                or comparison['source_audits'][1]['status']!='finite_dp2_efficiency_audited'
                or comparison['source_audits'][1]['files_sha256']!=parent['manifest_sha256']):
            raise ValueError('audited complete parent comparison required')
        write_new(root/'predecessor-completed.json',dict(terminal_sha256=file_hash(c['predecessor_terminal']),
            comparison_sha256=file_hash(c['comparison']),parent_manifest_sha256=parent['manifest_sha256']))
        tail.run(ops,provenance)
        result=collect_and_audit(ops,validator=tail.validate,command_builder=tail.command,
            output_auditor=tail.audit_outputs,result_dirs=('run','journal','service-0','service-1'))
        if result['status']!='finite_dp2_tail_outputs_audited':raise ValueError('tail output audit did not pass')
        write_new(root/'tail-audit.json',result)
        write_new(root/'terminal.json',dict(exit_code=0,status='tail_diagnostic_collected_audited',
            formal_training_dispatched=False,formal_history=False))
        return result
    except BaseException as exc:
        write_new(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True))
        raise


def detach(template,local,driver_root):
    c,_=load_config(template,local);validate(c)
    root=Path(driver_root);check_paths({'driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-m','tools.reports.run_context_tail_successor',
        '--config',str(Path(template).resolve()),'--local',str(Path(local).resolve()),
        '--action','run','--driver-root',str(root.resolve())]
    write_new(root/'intent.json',dict(argv=argv,cwd=str(Path.cwd()),automatic_retry=False))
    with (root/'driver.log').open('x') as log:
        process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True)
    write_new(root/'spawn.json',dict(pid=process.pid,argv=argv))
    return dict(pid=process.pid,driver_root=str(root),status='spawned_once')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','run','detach'],default='validate')
    p.add_argument('--driver-root')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    import torch
    torch.set_num_threads(1)
    if a.action=='detach':
        if not a.driver_root:p.error('--driver-root required for detach')
        print(json.dumps(detach(a.config,a.local,a.driver_root)))
    elif a.action=='run':
        if a.driver_root:
            proc=Path('/proc')/str(os.getpid())
            write_new(Path(a.driver_root)/'driver.json',dict(pid=os.getpid(),
                start_ticks=(proc/'stat').read_text().split(') ')[1].split()[19],
                cmdline_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest()))
        run(c,provenance)
    else:validate(c);print('tail successor validated; no dispatch')
