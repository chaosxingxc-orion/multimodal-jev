"""Run one audited 100-step efficiency experiment after finite acceptance.

No retry, formal dispatch, reference refresh, or model-driven scheduling.
The existing runtime, transport, numeric auditors and cost reporter are reused.
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
from tools.reports import context_cached_acceptance_sequence as sequence
from tools.reports import context_cached_efficiency as efficiency
from tools.reports import context_cached_efficiency_cost as cost


def read(path):return json.loads(Path(path).read_text())


def validate(c):
    if c.get('efficiency_dispatch_allowed') is not True:
        raise ValueError('explicit 100-step efficiency authorization required')
    prior=c['predecessor'];previous=read(prior['config']);base=Path(previous['output'])
    expected=dict(config=base/'config.json',terminal=base/'terminal.json',
        driver=base.with_name(base.name+'-driver')/'started.json')
    for key,path in expected.items():
        if prior[key]!=str(path):raise ValueError('exact finite predecessor path required: '+key)
    for key in ('driver','config'):
        if file_hash(prior[key])!=prior[key+'_sha256']:
            raise ValueError('pinned finite predecessor changed: '+key)
    reference_ops=sequence.validate(previous)
    stage=c['experiment']
    if set(stage)!={'package','remote_root','operations_root','tag'}:
        raise ValueError('efficiency experiment only declares locations and tag')
    if not re.fullmatch('[a-z0-9][a-z0-9-]*',stage['tag']):raise ValueError('safe experiment tag required')
    if stage['tag'] in [v['tag'] for v in previous['stages'].values()]:
        raise ValueError('new efficiency tag required')
    local=[Path(c['output']),Path(c['output']).with_name(Path(c['output']).name+'-driver'),
        Path(stage['package']),Path(stage['operations_root'])]
    protected=[base,base.with_name(base.name+'-driver'),Path(reference_ops['operations_root']),
        Path(reference_ops['package']),Path(reference_ops['reference_job']['parent']['output'])]
    for v in previous['stages'].values():protected.extend([Path(v['package']),Path(v['operations_root'])])
    remote=[Path(stage['remote_root'])]
    protected_remote=[Path(v['remote_root']) for v in previous['stages'].values()]
    protected_remote.extend(Path(reference_ops['reference_job'][k]['remote_root'])
        for k in ('parent',))
    protected_remote.append(Path(reference_ops['reference_job']['remote_root']))
    for paths,existing in ((local,protected),(remote,protected_remote)):
        for p in paths:sequence.training.lifecycle.shared.path(str(p))
        paths=[p.resolve() for p in paths];existing=[p.resolve() for p in existing]
        for i,p in enumerate(paths):
            if any(p.is_relative_to(q) or q.is_relative_to(p) for q in paths[:i]+existing):
                raise ValueError('isolated efficiency paths required')
    check_paths({str(i):p for i,p in enumerate(local)})
    return previous


def wait_acceptance(c):
    previous=validate(c);prior=c['predecessor']
    wait_exact(read(prior['driver']),prior['terminal'])
    # Recheck bindings after waiting; a status string alone is never admission.
    if validate(c)!=previous:raise ValueError('finite predecessor changed during wait')
    terminal=read(prior['terminal'])
    if (terminal.get('exit_code')!=0 or terminal.get('status')!='cached_finite_sequence_audited'
            or terminal.get('formal_training_dispatched') is not False):
        raise ValueError('successful independently audited finite sequence required')
    evidence={}
    for name in sequence.STAGES:
        stage=previous['stages'][name];root=Path(stage['operations_root'])/'finalization'
        final=read(root/'terminal.json');ops=read(root/'config.json');bound=read(root/'audit-config.json')
        if (final.get('exit_code')!=0 or final.get('status')!=sequence.STATUSES[name]
                or bound!=sequence.audit_config(ops,bound['reference_audit'])
                or any(ops[k]!=stage[k] for k in ('package','operations_root','tag'))
                or ops['training_package']['output']!=stage['package']
                or ops['training_package']['remote_root']!=stage['remote_root']):
            raise ValueError('exact finite stage evidence required: '+name)
        evidence[name]=bound
    prefix=evidence['prefix'];parent=prefix['training_package']
    refops=read(previous['reference_predecessor']['config'])
    for bound in evidence.values():
        if (any(bound['deployment'][k]!=refops[k] for k in ('host','lease'))
                or bound['reference_audit']!=read(previous['reference_predecessor']['audit_config'])):
            raise ValueError('same predecessor reference, host and lease required')
    stage=c['experiment']
    spec=dict(parent=parent['parent'],reference_job=parent['reference_job'],acceptance=evidence,
        acceptance_kind='efficiency',output=stage['package'],remote_root=stage['remote_root'])
    return spec,prefix['deployment'],prefix['reference_audit']


def cost_config(bound,output):
    ref=bound['reference_audit'];train_root=Path(bound['collection']);ref_root=Path(ref['collection'])
    paths=dict(training_config=train_root/'config.json',reference_build_config=ref_root/'reference-build-config.json',
        reference_result=ref_root/'reference-build/result.json',training_inspect=bound['inspect'],
        training_handle=bound['launch'],reference_inspect=ref['inspect'],reference_handle=ref['launch'],
        training_transport=bound['transport'],reference_transport=ref['transport'])
    return dict(run_root=str(train_root/'run'),output=str(output),
        **{k:dict(path=str(p),sha256=file_hash(p)) for k,p in paths.items()})


def run(c,provenance):
    validate(c);root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        commit_json(root/'waiting.json',c['predecessor'])
        spec,prior,ref=wait_acceptance(c)
        # package_action independently replays all three gates before writing.
        commit_json(root/'preparation-config.json',spec)
        prepared=efficiency.package_action(spec,provenance)
        spec=dict(spec,manifest_sha256=prepared['manifest_sha256'])
        from tools.reports.context_release_deployment import inherited
        ops=inherited(prior)
        stage=c['experiment']
        ops.update(training_package=spec,reference_audit=ref,package=stage['package'],
            manifest_sha256=spec['manifest_sha256'],operations_root=stage['operations_root'],
            tag=stage['tag'],efficiency_dispatch_allowed=True)
        commit_json(root/'deployment.json',ops)
        collected=efficiency.launch(ops,provenance)
        if collected.get('status')!='collected_independent_cached_training_audit_required':
            raise ValueError('complete efficiency collection required')
        bound=sequence.audit_config(ops,ref);commit_json(root/'audit-config.json',bound)
        proof=efficiency.audit(bound,provenance)
        if (proof.get('status')!='cached_100step_efficiency_outputs_audited'
                or proof.get('formal_training_admitted') is not False):
            raise ValueError('independent 100-step output audit required')
        commit_json(root/'audit.json',proof)
        measure=cost_config(bound,root/'cost.json');commit_json(root/'cost-config.json',measure)
        cost.run(measure,provenance)
        terminal=dict(exit_code=0,status='cached_efficiency_collected_audited_and_measured',
            updates=100,formal_training_dispatched=False,efficacy_evaluated=False,restart=False)
        commit_json(root/'terminal.json',terminal);return terminal
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True))
        raise


def detach(template,local):
    c,_=load_config(template,local);validate(c)
    root=Path(c['output']).with_name(Path(c['output']).name+'-driver')
    check_paths({'efficiency_driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.context_cached_efficiency_successor',
        '--config',str(Path(template).resolve()),'--local',str(Path(local).resolve()),'--action','run']
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,argv=argv,started_unix=time.time(),
        process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    commit_json(root/'started.json',record);return record


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','run','detach'],default='validate');a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    if a.action=='detach':result=detach(a.config,a.local)
    elif a.action=='run':result=run(c,provenance)
    else:validate(c);result=dict(status='efficiency_successor_validated_only',npu_executed=False)
    print(json.dumps(result,indent=2))
