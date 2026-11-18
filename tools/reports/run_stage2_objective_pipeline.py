"""Configured editable-objective tail, 100-step efficiency, then full GRPO.

Wait exact predecessor processes, retain independent audits, and use the shared
four-card lease. No retries, model monitoring, or efficiency checkpoint resume.
"""
import argparse
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from types import FunctionType, SimpleNamespace

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.prepare_objective_training_candidate import read_bound
from tools.reports.finalize_context_infonce import wait_exact
from tools.reports.context_isolated_source_audit import IsolatedSourceAuditEngine
from tools.reports.objective_parallel_admission import MODULES,clone_graph
from tools.reports.context_cached_formal_successor import monitored_training
from tools.reports.context_release_deployment import inherited

STAGES=('tail','efficiency','formal')


def read(p):return json.loads(Path(p).read_text())


def implementation(engine):
    names=(*MODULES,'audit_context_cached_training','context_cached_efficiency','context_cached_formal',
        'prepare_stage2_objective_run','stage2_objective_run_operations')
    modules=[importlib.import_module('tools.reports.'+n) for n in dict.fromkeys(names)]
    views=clone_graph(modules,{('tools.reports.context_infonce_selection','reconstruct_plan'):engine.reconstruct})
    return views['tools.reports.prepare_stage2_objective_run'],views['tools.reports.stage2_objective_run_operations']


def validate(c):
    if c.get('stage2_training_dispatch_allowed') is not True or c.get('monitor_interval_seconds')!=900:
        raise ValueError('explicit dispatch and 900-second monitoring required')
    if set(c['stages'])!=set(STAGES):raise ValueError('tail, efficiency, formal stages required')
    for pin in c['pins']:read_bound(pin)
    prior=c['predecessor'];previous=json.loads(read_bound(prior['config']))
    driver=json.loads(read_bound(prior['driver']))
    if 'tools.reports.run_stage2_parallel_acceptance' not in driver['argv']:
        raise ValueError('exact editable acceptance predecessor required')
    base=Path(previous['output'])
    if (prior['config']['path']!=str(base/'config.json')
            or prior['driver']['path']!=str(base.with_name(base.name+'-driver')/'started.json')
            or prior['terminal']!=str(base/'terminal.json')):
        raise ValueError('exact predecessor locations required')
    prefix=previous['sequence']['prefix']
    local=[Path(c['output']),Path(c['output']+'-driver')];remote=[];tags=[]
    for stage in c['stages'].values():
        if set(stage)!={'package','remote_root','operations_root','tag'}:raise ValueError('explicit stage locations only')
        if not re.fullmatch('[a-z0-9][a-z0-9-]*',stage['tag']):raise ValueError('safe tag required')
        local.extend(Path(stage[k]) for k in ('package','operations_root'));remote.append(Path(stage['remote_root']));tags.append(stage['tag'])
    protected=[base,Path(prefix['package']),Path(prefix['operations_root']),Path(previous['sequence']['output'])]
    protected.extend(Path(previous['sequence']['recovery'][k]) for k in ('package','operations_root'))
    oldremote=[Path(prefix['training_package']['remote_root']),Path(previous['sequence']['recovery']['remote_root'])]
    for paths,old in ((local,protected),(remote,oldremote)):
        for i,p in enumerate(paths):
            if not p.is_absolute() or any(p.is_relative_to(q) or q.is_relative_to(p) for q in paths[:i]+old):
                raise ValueError('isolated absolute stage locations required')
    if len(set(tags))!=3:raise ValueError('distinct stage tags required')
    check_paths({str(i):p for i,p in enumerate(local)})
    return previous


def cpu_evidence(c):
    report=json.loads(read_bound(c['cpu_report']));snapshot=Path(c['cpu_source'])
    files=read(snapshot/'FILES.json')
    if (file_hash(snapshot/'FILES.json')!=report['source_manifest_sha256']
            or report['image']!='sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
            or report['state']['ExitCode'] or report['state']['OOMKilled'] or report['terminal']['exit_code']
            or any(report['counts'][k] for k in ('failures','errors','skipped'))
            or report['counts']['tests']!=c['expected_cpu_tests']):
        raise ValueError('complete fixed-image CPU proof required')
    for pin in c['pins']:
        relative=str(Path(pin['path']).relative_to(c['project']))
        if files.get(relative)!=pin['sha256'] or file_hash(snapshot/relative)!=pin['sha256']:
            raise ValueError('pipeline source differs from fixed-image tests: '+relative)
    return report


def wait_acceptance(c):
    previous=validate(c);prior=c['predecessor']
    wait_exact(json.loads(read_bound(prior['driver'])),prior['terminal'])
    if validate(c)!=previous:raise ValueError('predecessor changed during wait')
    if read(prior['terminal']).get('status')!='objective_prefix_and_recovery_audited' or read(prior['terminal']).get('exit_code')!=0:
        raise ValueError('editable prefix/recovery predecessor failed')
    stages=previous['sequence']
    evidence={k:read(Path(stages[k]['operations_root'])/'finalization/audit-config.json') for k in ('prefix','recovery')}
    if evidence['prefix']['deployment']!=stages['prefix']:
        raise ValueError('prefix evidence differs from pinned predecessor')
    return evidence


def wait_curve(c,source_policy):
    entry=c['source_curve'];config=json.loads(read_bound(entry['config']))
    root=Path(config['output'])
    wait_exact(json.loads(read_bound(entry['driver'])),root/'terminal.json')
    terminal=read(root/'terminal.json')
    if terminal.get('exit_code')!=0 or terminal.get('status')!='complete_infonce_source_curve_audited':
        raise ValueError('complete independently audited Stage1 curve required')
    result=read(root/'curve/result.json');source=json.loads(read_bound(result['source_evaluation']))
    for pin in [source['curve'],source['source_audits'],*[p['result'] for p in source['points']]]:read_bound(pin)
    if (result['points']!=11 or source['roles']['stage2_initial_and_reference']['policy_id']!=source_policy
            or source['target_tuning_allowed'] is not False):
        raise ValueError('full source curve and same fixed reference endpoint required')
    return dict(source_evaluation=result['source_evaluation'],roles=source['roles'],points=11)


def run(c,provenance):
    validate(c);cpu_evidence(c);root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        commit_json(root/'waiting.json',c['predecessor']);evidence=wait_acceptance(c)
        prefix=evidence['prefix'];reference=prefix['deployment']['reference_audit']
        with IsolatedSourceAuditEngine(cpu_workers=c['cpu_workers'],io_workers=c['io_workers']) as engine:
            prepare,operations=implementation(engine)
            for kind in STAGES:
                if kind=='formal':
                    policy=read(Path(prefix['training_package']['output'])/'config.json')['source_policy_id']
                    commit_json(root/'source-curve-admission.json',wait_curve(c,policy))
                stage=c['stages'][kind]
                spec=dict(prefix_package=prefix['training_package'],kind=kind,output=stage['package'],remote_root=stage['remote_root'])
                commit_json(root/(kind+'-preparation.json'),spec)
                spec['manifest_sha256']=prepare.prepare(spec,provenance)['manifest_sha256']
                ops=dict(inherited(prefix['deployment']),training_package=spec,reference_audit=reference,
                    acceptance=dict(evidence),package=stage['package'],manifest_sha256=spec['manifest_sha256'],
                    operations_root=stage['operations_root'],tag=stage['tag'],stage2_training_dispatch_allowed=True)
                commit_json(root/(kind+'-deployment.json'),ops)
                monitor_root=root/kind;monitor_root.mkdir()
                runner=FunctionType(monitored_training.__code__,dict(monitored_training.__globals__,formal=operations))
                proof=runner(ops,provenance,monitor_root)
                commit_json(root/(kind+'-result.json'),proof)
                evidence[kind]=read(Path(ops['operations_root'])/'finalization/audit-config.json')
        result=dict(exit_code=0,status='stage2_objective_formal_collected_and_audited',updates=634,
            selection_evaluation_pending=True,transfer_evaluation_pending=True,restart=False)
        commit_json(root/'terminal.json',result);return result
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True));raise


def detach(template,local):
    c,_=load_config(template,local);validate(c);cpu_evidence(c)
    root=Path(c['output']+'-driver');root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.run_stage2_objective_pipeline','--config',str(Path(template).resolve()),
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
    p.add_argument('--action',choices=['validate','run','detach'],default='validate');a=p.parse_args();c,prov=load_config(a.config,a.local)
    if a.action=='detach':result=detach(a.config,a.local)
    elif a.action=='run':result=run(c,prov)
    else:validate(c);cpu_evidence(c);result=dict(status='pipeline_validated_only',npu_executed=False)
    print(json.dumps(result,indent=2))
