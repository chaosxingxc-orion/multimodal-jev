"""Evaluate the registered InfoNCE curve after the exact efficiency owner.

One immutable configuration, one attempt, serial four-card ownership. Failures
stop downstream work; no target tuning, retry, training or model polling.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.finalize_context_infonce import wait_exact
from tools.reports.context_infonce_selection import configured_steps,reconstruct_plan
from tools.reports.context_release_deployment import inherited
from tools.reports import context_cached_efficiency_successor as efficiency_sequence
from tools.reports import context_cached_efficiency as efficiency
from tools.reports import prepare_context_replica_selection as preparation
from tools.reports import context_replica_selection_operations as point
from tools.reports import collect_context_replica_infonce_curve as curve


def read(path):return json.loads(Path(path).read_text())


def validate(c):
    if c.get('selection_dispatch_allowed') is not True:raise ValueError('explicit complete selection dispatch required')
    entries=c['points']
    if [e['step'] for e in entries]!=configured_steps(c['point_template']) or any(type(e['step']) is not int for e in entries):
        raise ValueError('all registered ordered InfoNCE points required')
    prior=c['predecessor'];previous=read(prior['config']);base=Path(previous['output'])
    expected=dict(config=base/'config.json',terminal=base/'terminal.json',
        driver=base.with_name(base.name+'-driver')/'started.json')
    for k,p in expected.items():
        if prior[k]!=str(p):raise ValueError('exact efficiency predecessor path required: '+k)
    for k in ('config','driver'):
        if file_hash(prior[k])!=prior[k+'_sha256']:raise ValueError('efficiency predecessor changed: '+k)
    finite=efficiency_sequence.validate(previous)
    refops=read(finite['reference_predecessor']['config'])
    if c['deployment']!=inherited(refops):raise ValueError('same resource lease and release deployment required')
    template=c['point_template']
    if set(template)!={'parent','source_plan','source_plan_sha256','source_finalization','selection_inputs'}:
        raise ValueError('only shared point source/runtime/input template required')
    if template['parent']!=refops['reference_job']['parent']:raise ValueError('same audited Stage1 endpoint/runtime required')
    if file_hash(template['source_plan'])!=template['source_plan_sha256']:raise ValueError('source plan changed')
    root=Path(c['output']);paths=[root,root.with_name(root.name+'-driver')];remotes=[];tags=[]
    protected=[base,base.with_name(base.name+'-driver'),Path(refops['package']),Path(refops['operations_root']),
        Path(template['parent']['output']),Path(previous['experiment']['package']),Path(previous['experiment']['operations_root'])]
    protected.extend(Path(v[k]) for v in finite['stages'].values() for k in ('package','operations_root'))
    protected_remote=[Path(refops['reference_job']['remote_root']),Path(template['parent']['remote_root']),
        Path(previous['experiment']['remote_root']),*[Path(v['remote_root']) for v in finite['stages'].values()]]
    for e in entries:
        if set(e) not in ({'step','package','remote_root','operations_root','tag'},
                {'step','package','remote_root','operations_root','tag','manifest_sha256'}):
            raise ValueError('point only declares step, locations and optional preprepared identity')
        paths.extend([Path(e['package']),Path(e['operations_root'])]);remotes.append(Path(e['remote_root']));tags.append(e['tag'])
        if 'manifest_sha256' in e and file_hash(Path(e['package'])/'PREPARED.json')!=e['manifest_sha256']:
            raise ValueError('preprepared point identity changed')
    if len(set(tags))!=len(tags) or any(not re.fullmatch('[a-z0-9][a-z0-9-]*',s) for s in tags):
        raise ValueError('distinct safe selection tags required')
    for group,old in ((paths,protected),(remotes,protected_remote)):
        for path in group:point.lifecycle.shared.path(str(path))
        group=[p.resolve() for p in group];old=[p.resolve() for p in old]
        for i,p in enumerate(group):
            if any(p.is_relative_to(q) or q.is_relative_to(p) for q in group[:i]+old):
                raise ValueError('isolated selection paths required')
    check_paths({str(i):p for i,p in enumerate(paths)})
    return previous


def wait_efficiency(c):
    previous=validate(c);prior=c['predecessor']
    wait_exact(read(prior['driver']),prior['terminal'])
    if validate(c)!=previous:raise ValueError('efficiency predecessor changed during wait')
    terminal=read(prior['terminal'])
    if (terminal.get('exit_code')!=0 or terminal.get('status')!='cached_efficiency_collected_audited_and_measured'
            or terminal.get('updates')!=100 or terminal.get('formal_training_dispatched') is not False):
        raise ValueError('successful complete efficiency predecessor required')
    bound=read(Path(previous['output'])/'audit-config.json');stage=previous['experiment'];ops=bound['deployment']
    if (any(ops[k]!=stage[k] for k in ('package','operations_root','tag'))
            or bound['training_package']['remote_root']!=stage['remote_root']
            or inherited(ops)!=c['deployment']):
        raise ValueError('exact efficiency collection and deployment required')
    proof=efficiency.audit(bound,{'source':'source_curve_efficiency_predecessor'})
    if proof['status']!='cached_100step_efficiency_outputs_audited':raise ValueError('independent efficiency audit required')
    return proof


def run(c,provenance):
    validate(c);root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        commit_json(root/'waiting.json',c['predecessor'])
        commit_json(root/'efficiency-audit.json',wait_efficiency(c))
        plan=reconstruct_plan(c['point_template']);commit_json(root/'source-plan.json',plan)
        entries=[]
        for e in c['points']:
            spec=dict(c['point_template'],source_step=e['step'],output=e['package'],remote_root=e['remote_root'])
            if 'manifest_sha256' in e:
                spec['manifest_sha256']=e['manifest_sha256'];preparation.audit(spec,provenance)
            else:spec['manifest_sha256']=preparation.prepare(spec,provenance)['manifest_sha256']
            ops=dict(c['deployment'],selection_package=spec,package=e['package'],manifest_sha256=spec['manifest_sha256'],
                operations_root=e['operations_root'],tag=e['tag'],selection_dispatch_allowed=True)
            proof=point.run(ops,provenance)
            if (proof['status']!='complete_replica_infonce_selection_audited' or proof['source_step']!=e['step']
                    or proof['runtime_provenance_verified'] is not True):raise ValueError('independent full point audit required')
            path=Path(e['operations_root'])/'finalization/audit-config.json'
            entries.append(dict(step=e['step'],audit_config=str(path),audit_config_sha256=file_hash(path)))
            commit_json(root/f"completed-{e['step']:04d}.json",entries[-1])
        collected=dict(c['point_template'],evaluations=entries,output=str(root/'curve'))
        commit_json(root/'curve-config.json',collected)
        result=curve.collect(collected,provenance)
        if result['status']!='complete_infonce_source_curve_audited':raise ValueError('independent full curve required')
        commit_json(root/'terminal.json',dict(exit_code=0,status='replica_infonce_curve_collected_and_audited',
            points=len(entries),training_dispatched=False,target_inference_performed=False,restart=False))
        return result
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True));raise


def detach(template,local):
    c,_=load_config(template,local);validate(c)
    root=Path(c['output']).with_name(Path(c['output']).name+'-driver')
    check_paths({'selection_driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.context_replica_selection_sequence','--config',str(Path(template).resolve()),
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
    p.add_argument('--action',choices=['validate','run','detach'],default='validate');a=p.parse_args();c,provenance=load_config(a.config,a.local)
    if a.action=='detach':result=detach(a.config,a.local)
    elif a.action=='run':result=run(c,provenance)
    else:validate(c);result=dict(status='selection_sequence_validated_only',npu_executed=False)
    print(json.dumps(result,indent=2))
