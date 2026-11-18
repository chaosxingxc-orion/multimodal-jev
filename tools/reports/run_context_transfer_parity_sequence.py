"""Exact tail-successor wait, then serial FSC/MINDS/SLURP parity admission.

No model analysis, target scoring, automatic retries or formal training launch.
Each dataset reuses the same HF-first dual-service lifecycle and exclusive lease.
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
from tools.reports import context_transfer_parity_operations as parity
from tools.reports.audit_context_macro_probe import collect_and_audit

DATASETS=('FSC_native31','MINDS_native14','SLURP_clean_TEST60')


def read(path):return json.loads(Path(path).read_text())


def pins(c):
    required={c['predecessor_driver'],c['predecessor_config'],*c['operations']}
    if not required<=set(c['pins']):raise ValueError('explicit pinned predecessor and operation inputs required')
    for name,h in c['pins'].items():
        if file_hash(name)!=h:raise ValueError('parity sequence dependency changed: '+name)


def validate(c):
    pins(c);prior=read(c['predecessor_config'])
    if (c['predecessor_terminal']!=str(Path(prior['output'])/'terminal.json')
            or prior['operations'] not in c['pins'] or len(c['operations'])!=3):
        raise ValueError('exact tail predecessor and three parity datasets required')
    tail_ops=read(prior['operations']);tail_receipt=tail.validate(tail_ops)
    operations=[];common=None;paths=[]
    for expected,name in zip(DATASETS,c['operations'],strict=True):
        ops=read(name);receipt=parity.validate(ops)
        if (receipt['dataset']!=expected or ops.get('finite_inference_dispatch_allowed') is not True
                or ops['host']!=tail_ops['host'] or ops['lease']!=tail_ops['lease']
                or receipt['parent_manifest_sha256']!=tail_receipt['parent_manifest_sha256']
                or (Path(ops['operations_root'])/'attempt.json').exists()):
            raise ValueError('fresh serial parity-only successor on the same leased devices required')
        identity=(receipt['policy_id'],receipt['runtime_identity'],receipt['image'])
        if common is None:common=identity
        elif common!=identity:raise ValueError('same source policy/runtime required for all targets')
        paths.append((ops['remote_root'],ops['operations_root'],ops['tag']))
        operations.append(ops)
    for column in range(3):
        if len({p[column] for p in paths})!=3:raise ValueError('distinct dataset attempt paths and tags required')
    return tail_ops,operations


def run(c,provenance):
    validate(c)
    root=Path(c['output']);check_paths({'parity_sequence':root});root.mkdir(parents=True,exist_ok=False)
    write_new(root/'config.json',c);write_new(root/'provenance.json',provenance)
    try:
        prior=read(c['predecessor_driver'])
        write_new(root/'waiting.json',dict(process=prior,terminal=c['predecessor_terminal']))
        wait_driver(prior,Path(c['predecessor_terminal']))
        tail_ops,operations=validate(c)
        terminal=read(c['predecessor_terminal'])
        if terminal.get('exit_code')!=0 or terminal.get('status')!='tail_diagnostic_collected_audited':
            raise ValueError('successful tail completion required')
        admission=tail.audit_collected(tail_ops)
        if admission['status']!='finite_dp2_tail_outputs_audited':raise ValueError('independent tail audit required')
        write_new(root/'tail-audit.json',admission)
        completed=[]
        for dataset,ops in zip(DATASETS,operations,strict=True):
            pins(c);parity.validate(ops)
            write_new(root/(dataset+'-dispatch.json'),dict(operations=ops))
            parity.lifecycle(ops,provenance,validator=parity.validate,command_builder=parity.command)
            result=collect_and_audit(ops,validator=parity.validate,command_builder=parity.command,
                output_auditor=parity.audit_outputs,result_dirs=('hf','evaluation','journal','service-0','service-1'))
            if (result['status']!='target_width_macro_parity_outputs_audited'
                    or result['comparison']['dataset']!=dataset or result['comparison']['exact'] is not True):
                raise ValueError('independently collected exact target-width parity required')
            write_new(root/(dataset+'-audit.json'),result);completed.append(dataset)
        write_new(root/'terminal.json',dict(exit_code=0,status='three_target_width_parity_audited',
            completed=completed,formal_training_dispatched=False,target_scored=False))
        return completed
    except BaseException as exc:
        write_new(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True));raise


def detach(template,local,driver_root):
    c,_=load_config(template,local);validate(c)
    root=Path(driver_root);check_paths({'driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-m','tools.reports.run_context_transfer_parity_sequence','--config',str(Path(template).resolve()),
        '--local',str(Path(local).resolve()),'--action','run','--driver-root',str(root.resolve())]
    write_new(root/'intent.json',dict(argv=argv,cwd=str(Path.cwd()),automatic_retry=False))
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_new(root/'spawn.json',dict(pid=child.pid,argv=argv))
    return dict(pid=child.pid,driver_root=str(root),status='spawned_once')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','run','detach'],default='validate');p.add_argument('--driver-root')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    import torch
    torch.set_num_threads(1)
    if a.action=='detach':
        if not a.driver_root:p.error('--driver-root required')
        print(json.dumps(detach(a.config,a.local,a.driver_root)))
    elif a.action=='run':
        if a.driver_root:
            proc=Path('/proc')/str(os.getpid())
            write_new(Path(a.driver_root)/'driver.json',dict(pid=os.getpid(),
                start_ticks=(proc/'stat').read_text().split(') ')[1].split()[19],
                cmdline_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest()))
        run(c,provenance)
    else:validate(c);print('parity sequence validated; no dispatch')
