"""Endpoint TRAIN parity lifecycle using shared leases, exact wait and collect.

No training dispatch. Numeric failure is collected and retained for diagnosis;
successful transport still requires the independent parity result auditor.
"""
import argparse
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports import context_replica_training_operations as shared
from tools.reports.context_cached_storage_operations import command,storage_preflight
from tools.reports.prepare_context_train_parity_job import audit as audit_package


def validate(c):
    spec=c['parity_package'];audit_package(spec,{'source':'parity_lifecycle_validation'})
    if (c['package']!=spec['output'] or c['manifest_sha256']!=spec['manifest_sha256']
            or not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]*',c['host'])
            or not re.fullmatch('[a-z0-9][a-z0-9-]*',c['tag'])):
        raise ValueError('bound parity package and safe host/tag required')
    root=Path(c['package']);m=json.loads((root/'PREPARED.json').read_text())
    config=json.loads((root/'config.json').read_text())
    if m['image']!=shared.IMAGE or m['remote_root']!=spec['remote_root']:
        raise ValueError('fixed image and original parity destination required')
    for key in ('package','operations_root','lease'):shared.path(c[key])
    if (Path(c['operations_root']).is_relative_to(root)
            or root.is_relative_to(c['operations_root'])):
        raise ValueError('disjoint prepared package and lifecycle records required')
    command(c,m,config,producer=True)
    storage_preflight(config,m['remote_root'],producer=True)
    return m,config,{}


def launch(c,provenance):
    if c.get('finite_inference_dispatch_allowed') is not True:
        raise ValueError('explicit finite inference dispatch required')
    # Only validation, readonly preflight and the result label differ. Preserve
    # the common image/device check, nonblocking locks, one-attempt policy,
    # archive transfer, exact container handle, wait and collector implementation.
    source=inspect.getsource(shared.launch)
    old="    if c.get('finite_training_dispatch_allowed') is not True:\n        raise ValueError('finite training dispatch not enabled')"
    new="    if c.get('finite_inference_dispatch_allowed') is not True:\n        raise ValueError('finite inference dispatch not enabled')"
    if source.count(old)!=1:raise ValueError('shared launch authorization anchor changed')
    source=source.replace(old,new)
    begin="        check = storage + '\\ncheck_paths('"
    end="        checked(['python3', '-c', check])"
    if source.count(begin)!=1 or source.count(end)!=1:
        raise ValueError('shared storage preflight anchor changed')
    left=source.index(begin);right=source.index(end,left)
    source=source[:left]+"        check = storage + '\\n' + storage_preflight(config,root,producer=True)\n"+source[right:]
    source=source.replace("status='collected_independent_training_audit_required'",
        "status='collected_independent_train_parity_audit_required'")
    namespace=dict(shared.launch.__globals__,validate=validate,
        command=lambda c,m,config:command(c,m,config,producer=True),storage_preflight=storage_preflight)
    exec(compile(source,'<shared-train-parity-lifecycle>','exec'),namespace)
    return namespace['launch'](c,provenance)


def run_and_audit(c, provenance):
    """One exact launch/collect followed by independent output reconstruction."""
    root=Path(c['operations_root'])/'finalization'
    check_paths({'parity_finalization':root});root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        result=launch(c,provenance)
        if result['status']!='collected_independent_train_parity_audit_required':
            raise ValueError('complete parity collection required')
        base=Path(c['operations_root']);stem=base/c['tag']
        audit_config=dict(parity_package=c['parity_package'],deployment=c,
            **{k:str(stem)+suffix for k,suffix in (
                ('launch','-launch.json'),('inspect','-inspect.json'),('wait','-wait.json'),
                ('transport','-collected.transport.json'),('collection','-collected'),('archive','-collected.tar'))})
        commit_json(root/'audit-config.json',audit_config)
        from tools.reports.audit_context_train_parity import audit
        proof=audit(audit_config,provenance)
        commit_json(root/'audit.json',proof)
        commit_json(root/'terminal.json',dict(exit_code=0,status='train_parity_collected_and_audited',
            training_dispatched=False))
        return proof
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True))
        raise


def detach(template, local):
    c,_=load_config(template,local)
    if c.get('finite_inference_dispatch_allowed') is not True:
        raise ValueError('explicit finite inference dispatch required')
    root=Path(c['operations_root'])/'driver'
    check_paths({'parity_driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.context_train_parity_operations',
        '--config',str(Path(template).resolve()),'--local',str(Path(local).resolve()),'--action','run']
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,argv=argv,started_unix=time.time(),
        process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    commit_json(root/'started.json',record)
    return record


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','launch','run','detach'],default='validate')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    if a.action=='detach':result=detach(a.config,a.local)
    elif a.action=='run':result=run_and_audit(c,provenance)
    elif a.action=='launch':result=launch(c,provenance)
    else:
        m,config,_=validate(c)
        result=dict(status='train_parity_lifecycle_validated_not_executed',
            argv=command(c,m,config,producer=True),npu_executed=False,training_dispatch_allowed=False)
    print(json.dumps(result,indent=2))
