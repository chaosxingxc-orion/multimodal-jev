"""Full GRPO source evaluation after exact formal training; no target dispatch."""
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
from tools.reports.context_release_deployment import inherited
from tools.reports import context_cached_formal_successor as predecessor
from tools.reports import context_cached_formal as formal
from tools.reports import context_cached_grpo_source_plan as source
from tools.reports import prepare_context_cached_grpo_selection as preparation
from tools.reports import context_cached_grpo_selection_operations as point
from tools.reports import collect_context_cached_grpo_curve as curve


def read(path):return json.loads(Path(path).read_text())


def validate(c):
    if c.get('selection_dispatch_allowed') is not True:raise ValueError('explicit full GRPO source evaluation required')
    prior=c['predecessor']
    for k in ('driver','config'):
        if file_hash(prior[k])!=prior[k+'_sha256']:raise ValueError('formal predecessor changed: '+k)
    previous=read(prior['config']);base=Path(previous['output'])
    for k,p in dict(config=base/'config.json',terminal=base/'terminal.json',
            driver=base.with_name(base.name+'-driver')/'started.json').items():
        if prior[k]!=str(p):raise ValueError('exact formal predecessor path required: '+k)
    selection,efficiency=predecessor.validate(previous)
    entries=c['points']
    if [e['step'] for e in entries]!=formal.STEPS or any(type(e['step']) is not int for e in entries):
        raise ValueError('all registered GRPO selection points required')
    root=Path(c['output']);paths=[root,root.with_name(root.name+'-driver')];remotes=[];tags=[]
    protected=[base,base.with_name(base.name+'-driver'),Path(previous['experiment']['package']),
        Path(previous['experiment']['operations_root']),Path(selection['output']),
        Path(selection['point_template']['parent']['output']),Path(efficiency['output']),
        Path(efficiency['experiment']['package']),Path(efficiency['experiment']['operations_root'])]
    protected.extend(Path(e[k]) for e in selection['points'] for k in ('package','operations_root'))
    old=[Path(previous['experiment']['remote_root']),Path(efficiency['experiment']['remote_root']),
        Path(selection['point_template']['parent']['remote_root']),*[Path(e['remote_root']) for e in selection['points']]]
    finite=predecessor.selection.efficiency_sequence.validate(efficiency)
    ref=read(finite['reference_predecessor']['config'])
    protected.extend([Path(finite['output']),Path(ref['package']),Path(ref['operations_root'])])
    protected.extend(Path(e[k]) for e in finite['stages'].values() for k in ('package','operations_root'))
    old.extend([Path(ref['reference_job']['remote_root']),*[Path(e['remote_root']) for e in finite['stages'].values()]])
    for e in entries:
        if set(e)!={'step','package','remote_root','operations_root','tag'}:raise ValueError('point only declares registered step and locations')
        paths.extend([Path(e['package']),Path(e['operations_root'])]);remotes.append(Path(e['remote_root']));tags.append(e['tag'])
    if len(set(tags))!=11 or any(not re.fullmatch('[a-z0-9][a-z0-9-]*',t) for t in tags):raise ValueError('unique safe point tags required')
    for group,existing in ((paths,protected),(remotes,old)):
        for p in group:predecessor.selection.point.lifecycle.shared.path(str(p))
        group=[p.resolve() for p in group];existing=[p.resolve() for p in existing]
        for i,p in enumerate(group):
            if any(p.is_relative_to(q) or q.is_relative_to(p) for q in group[:i]+existing):raise ValueError('isolated GRPO evaluation paths required')
    check_paths({str(i):p for i,p in enumerate(paths)})
    return previous,selection


def wait_formal(c):
    previous,selection=validate(c);prior=c['predecessor']
    wait_exact(read(prior['driver']),prior['terminal'])
    if validate(c)!=(previous,selection):raise ValueError('formal predecessor changed during wait')
    terminal=read(prior['terminal'])
    if (terminal.get('exit_code')!=0 or terminal.get('status')!='formal_cached_successor_collected_and_audited'
            or terminal.get('updates')!=634):raise ValueError('complete formal predecessor required')
    stage=previous['experiment'];root=Path(stage['operations_root'])/'finalization'
    path=root/'audit-config.json';bound=read(path);ops=bound['deployment']
    if (any(ops[k]!=stage[k] for k in ('package','operations_root','tag'))
            or ops['training_package']!=bound['training_package']
            or bound['training_package']['remote_root']!=stage['remote_root']
            or bound['training_package']['parent']!=selection['point_template']['parent']
            or inherited(ops)!=selection['deployment']
            or read(root/'terminal.json').get('status')!='formal_cached_training_collected_and_audited'):
        raise ValueError('actual formal collection, parent and deployment required')
    finalization=dict(training_audit_config=str(path),training_audit_config_sha256=file_hash(path))
    plan=source.build(finalization,{'source':'cached_grpo_evaluation_sequence'})
    return finalization,plan,ops,selection['point_template']['selection_inputs']


def run(c,provenance):
    validate(c);root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        commit_json(root/'waiting.json',c['predecessor'])
        finalization,plan,ops,inputs=wait_formal(c)
        plan_path=root/'source-plan.json';commit_json(plan_path,plan);commit_json(root/'source-finalization.json',finalization)
        template=dict(parent=ops['training_package']['parent'],source_plan=str(plan_path),source_plan_sha256=file_hash(plan_path),
            source_finalization=finalization,selection_inputs=inputs)
        entries=[]
        for e in c['points']:
            spec=dict(template,source_step=e['step'],output=e['package'],remote_root=e['remote_root'])
            spec['manifest_sha256']=preparation.prepare(spec,provenance)['manifest_sha256']
            deployment=dict(inherited(ops),selection_package=spec,package=e['package'],manifest_sha256=spec['manifest_sha256'],
                operations_root=e['operations_root'],tag=e['tag'],selection_dispatch_allowed=True)
            proof=point.run(deployment,provenance)
            if (proof['status']!='complete_replica_cached_grpo_selection_audited' or proof['source_step']!=e['step']
                    or proof['runtime_provenance_verified'] is not True):raise ValueError('independent complete GRPO selection point required')
            path=Path(e['operations_root'])/'finalization/audit-config.json'
            entry=dict(step=e['step'],audit_config=str(path),audit_config_sha256=file_hash(path));entries.append(entry)
            commit_json(root/f"completed-{e['step']:04d}.json",entry)
        cc=dict(template,evaluations=entries,output=str(root/'curve'));commit_json(root/'curve-config.json',cc)
        result=curve.collect(cc,provenance)
        if result['status']!='complete_source_curve_audited_and_roles_frozen':raise ValueError('complete independently audited GRPO curve required')
        commit_json(root/'terminal.json',dict(exit_code=0,status='cached_grpo_source_curve_collected_and_audited',
            points=11,target_inference_performed=False,training_dispatched=False,restart=False))
        return result
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True));raise


def detach(template,local):
    c,_=load_config(template,local);validate(c)
    root=Path(c['output']).with_name(Path(c['output']).name+'-driver')
    check_paths({'evaluation_driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.context_cached_grpo_evaluation_sequence','--config',str(Path(template).resolve()),
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
    else:validate(c);result=dict(status='grpo_evaluation_sequence_validated_only',npu_executed=False)
    print(json.dumps(result,indent=2))
