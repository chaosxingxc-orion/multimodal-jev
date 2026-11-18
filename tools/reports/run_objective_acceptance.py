"""One configured plugin prefix/recovery sequence; no retries or formal dispatch.

Call only after the current four-card owner yields the lease. Each launch uses
the existing nonblocking device lock; this entry never preempts another job.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.prepare_objective_training_candidate import read_bound
from tools.reports import context_objective_training_operations as prefix
from tools.reports import context_objective_recovery_operations as recovery
from tools.reports import prepare_objective_recovery as preparation


def validate(c):
    if c.get('finite_training_dispatch_allowed') is not True:
        raise ValueError('explicit finite sequence dispatch required')
    ops=c['prefix']
    if ops.get('finite_training_dispatch_allowed') is not True:
        raise ValueError('explicit prefix dispatch required')
    if c['monitor_interval_seconds']!=900:
        raise ValueError('900-second ordinary observation interval required')
    for binding in c['pins']:
        read_bound(binding)
    stage=c['recovery']
    if set(stage)!={'package','remote_root','operations_root','tag'}:
        raise ValueError('explicit recovery locations and tag required')
    local=[Path(c['output']),Path(ops['operations_root']),Path(ops['package']),
           Path(stage['package']),Path(stage['operations_root'])]
    for index,p in enumerate(local):
        if not p.is_absolute() or any(p.resolve().is_relative_to(q.resolve())
                or q.resolve().is_relative_to(p.resolve()) for q in local[:index]):
            raise ValueError('disjoint absolute sequence locations required')
    old=Path(ops['training_package']['remote_root']);new=Path(stage['remote_root'])
    if not new.is_absolute() or new.is_relative_to(old) or old.is_relative_to(new):
        raise ValueError('disjoint remote recovery required')
    if stage['tag']==ops['tag']:
        raise ValueError('distinct run identities required')
    return ops,stage


def run(c,provenance):
    ops,stage=validate(c);root=Path(c['output'])
    check_paths({'objective_sequence':root});root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        proof=prefix.run(ops,provenance)
        if proof['status']!='finite_objective_training_outputs_audited':
            raise ValueError('independent prefix audit required before recovery')
        commit_json(root/'prefix-result.json',proof)
        bound=prefix.read(Path(ops['operations_root'])/'finalization/audit-config.json')
        spec=dict(parent_audit=bound,output=stage['package'],remote_root=stage['remote_root'])
        commit_json(root/'recovery-preparation.json',spec)
        spec['manifest_sha256']=preparation.prepare(spec,provenance)['manifest_sha256']
        follow=dict(ops,training_package=spec,package=stage['package'],
            manifest_sha256=spec['manifest_sha256'],operations_root=stage['operations_root'],tag=stage['tag'])
        commit_json(root/'recovery-operations.json',follow)
        proof=recovery.run(follow,provenance)
        if (proof['status']!='finite_objective_recovery_audited'
                or proof['fresh_process_npu_resume_verified'] is not True
                or proof['replay']['exact_objective_traces'] is not True):
            raise ValueError('exact fresh-process recovery audit required')
        commit_json(root/'recovery-result.json',proof)
        result=dict(exit_code=0,status='objective_prefix_and_recovery_audited',
            formal_training_admitted=False,tail_and_efficiency_pending=True)
        commit_json(root/'terminal.json',result);return result
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True))
        raise


def detach(template,local):
    c,_=load_config(template,local);validate(c)
    root=Path(c['output']).with_name(Path(c['output']).name+'-driver')
    check_paths({'driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.run_objective_acceptance',
        '--config',str(Path(template).resolve()),'--local',str(Path(local).resolve()),'--action','run']
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,argv=argv,started_unix=time.time(),
        process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    commit_json(root/'started.json',record);return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--local',required=True);p.add_argument('--action',choices=['validate','run','detach'],default='validate')
    args=p.parse_args();c,provenance=load_config(args.config,args.local)
    if args.action=='detach':result=detach(args.config,args.local)
    elif args.action=='run':result=run(c,provenance)
    else:
        validate(c);result=dict(status='sequence_config_validated_only',npu_executed=False)
    print(json.dumps(result,indent=2))
