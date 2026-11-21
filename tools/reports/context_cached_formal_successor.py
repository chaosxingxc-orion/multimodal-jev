"""Run full cached GRPO after the exact source-curve owner releases four cards.

The efficiency gates, complete source curve and actual Stage1 endpoint remain
independent requirements. No selection-best initialization or efficiency resume.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import threading

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.finalize_context_infonce import wait_exact
from tools.reports.context_release_deployment import inherited
from tools.reports import context_replica_selection_sequence as selection
from tools.reports import collect_context_replica_infonce_curve as curve
from tools.reports import context_cached_formal as formal
from tools.reports.observe_context_replica_training import observe,INTERVAL


def read(path):return json.loads(Path(path).read_text())


def validate(c):
    if c.get('formal_training_dispatch_allowed') is not True:raise ValueError('explicit full GRPO dispatch required')
    prior=c['predecessor']
    for k in ('config','driver'):
        if file_hash(prior[k])!=prior[k+'_sha256']:raise ValueError('source-curve predecessor changed: '+k)
    previous=read(prior['config']);base=Path(previous['output'])
    expected=dict(config=base/'config.json',driver=base.with_name(base.name+'-driver')/'started.json',terminal=base/'terminal.json')
    for k,p in expected.items():
        if prior[k]!=str(p):raise ValueError('exact source-curve predecessor path required: '+k)
    efficiency=selection.validate(previous)
    stage=c['experiment']
    if set(stage)!={'package','remote_root','operations_root','tag'}:raise ValueError('formal experiment only declares locations and tag')
    if not re.fullmatch('[a-z0-9][a-z0-9-]*',stage['tag']):raise ValueError('safe formal tag required')
    root=Path(c['output']);local=[root,root.with_name(root.name+'-driver'),Path(stage['package']),Path(stage['operations_root'])]
    protected=[base,base.with_name(base.name+'-driver'),Path(efficiency['output']),
        Path(efficiency['experiment']['package']),Path(efficiency['experiment']['operations_root']),
        Path(previous['point_template']['parent']['output'])]
    protected.extend(Path(p[k]) for p in previous['points'] for k in ('package','operations_root'))
    remote=Path(stage['remote_root']);old=[Path(efficiency['experiment']['remote_root']),
        Path(previous['point_template']['parent']['remote_root']),*[Path(p['remote_root']) for p in previous['points']]]
    # Include every upstream reference and finite job, not just its controller.
    finite=selection.efficiency_sequence.validate(efficiency)
    ref=read(finite['reference_predecessor']['config'])
    protected.extend([Path(finite['output']),Path(ref['operations_root']),Path(ref['package'])])
    protected.extend(Path(p[k]) for p in finite['stages'].values() for k in ('package','operations_root'))
    old.extend([Path(ref['reference_job']['remote_root']),*[Path(p['remote_root']) for p in finite['stages'].values()]])
    for group,existing in ((local,protected),([remote],old)):
        for p in group:selection.point.lifecycle.shared.path(str(p))
        group=[p.resolve() for p in group];existing=[p.resolve() for p in existing]
        for i,p in enumerate(group):
            if any(p.is_relative_to(q) or q.is_relative_to(p) for q in group[:i]+existing):
                raise ValueError('isolated formal paths required')
    check_paths({str(i):p for i,p in enumerate(local)})
    return previous,efficiency


def wait_source(c):
    previous,efficiency=validate(c);prior=c['predecessor']
    wait_exact(read(prior['driver']),prior['terminal'])
    if validate(c)!=(previous,efficiency):raise ValueError('predecessor changed during wait')
    terminal=read(prior['terminal'])
    if (terminal.get('exit_code')!=0 or terminal.get('status')!='replica_infonce_curve_collected_and_audited'
            or terminal.get('points')!=11 or terminal.get('training_dispatched') is not False):
        raise ValueError('complete independently audited source curve required')
    source_root=Path(previous['output']);cc=read(source_root/'curve-config.json')
    expected=[]
    for p in previous['points']:
        path=Path(p['operations_root'])/'finalization/audit-config.json'
        expected.append(dict(step=p['step'],audit_config=str(path),audit_config_sha256=file_hash(path)))
    if cc!=dict(previous['point_template'],evaluations=expected,output=str(source_root/'curve')):
        raise ValueError('curve configuration differs from exact predecessor')
    checked=curve.reconstruct(cc)
    # Bind the published curve to reconstruction, including all regressions.
    if read(source_root/'curve/curve.json')!=checked['curve']:raise ValueError('published source curve differs')
    source=read(source_root/'curve/source-evaluation.json')
    endpoint=checked['plan']['reference_endpoint']
    if source['roles']['stage2_initial_and_reference']!=endpoint:raise ValueError('fixed Stage1 endpoint required')
    selection.wait_efficiency(previous)
    bound=read(Path(efficiency['output'])/'audit-config.json');prior_spec=bound['training_package']
    if prior_spec['parent']!=previous['point_template']['parent']:raise ValueError('same Stage1 endpoint runtime required')
    stage=c['experiment']
    spec=dict(parent=prior_spec['parent'],reference_job=prior_spec['reference_job'],efficiency_audit=bound,
        acceptance_kind='formal_source',evaluation_steps=list(formal.STEPS),
        output=stage['package'],remote_root=stage['remote_root'])
    return spec,bound,dict(reference_endpoint=endpoint,selected_checkpoint=checked['curve']['selected_checkpoint'],
        source_curve_sha256=file_hash(source_root/'curve/curve.json'),source_selection_used_for_initialization=False)


def monitored_training(ops,provenance,root,*,run_stage=None):
    """Reuse the exact-handle observer at 900s; never affect training decisions."""
    config={k:ops[k] for k in ('host','tag','operations_root','manifest_sha256')}
    config['interval_seconds']=INTERVAL
    commit_json(root/'monitor-config.json',config)
    stop=threading.Event()
    def monitor():
        index=0
        while not stop.wait(INTERVAL):
            try:result=observe(config)
            except Exception as exc:result=dict(status='observation_error',error=repr(exc),restart=False)
            commit_json(root/f'observation-{index:04d}.json',result);index+=1
            state=result.get('state') or {}
            if not state.get('Running',True) and state.get('Status') in ('exited','dead'):return
    worker=threading.Thread(target=monitor,name='formal-900s-observer',daemon=True);worker.start()
    try:return (formal.run if run_stage is None else run_stage)(ops,provenance)
    finally:
        stop.set();worker.join(timeout=60)
        commit_json(root/'monitor-terminal.json',dict(owner_finished=True,observer_stopped=not worker.is_alive(),
            interval_seconds=INTERVAL,training_success_verified=False))


def run(c,provenance):
    validate(c);root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        commit_json(root/'waiting.json',c['predecessor'])
        spec,bound,source=wait_source(c);commit_json(root/'source-admission.json',source)
        commit_json(root/'preparation-config.json',spec)
        prepared=formal.package_action(spec,provenance)
        spec=dict(spec,manifest_sha256=prepared['manifest_sha256'])
        stage=c['experiment'];ops=dict(inherited(bound['deployment']),training_package=spec,
            reference_audit=bound['reference_audit'],package=stage['package'],manifest_sha256=spec['manifest_sha256'],
            operations_root=stage['operations_root'],tag=stage['tag'],formal_training_dispatch_allowed=True)
        commit_json(root/'deployment.json',ops)
        proof=monitored_training(ops,provenance,root)
        if (proof['status']!='formal_cached_reference_training_outputs_audited'
                or proof['real_query_exposures']!=81026 or len(proof['updates'])!=634):
            raise ValueError('complete independently audited formal GRPO required')
        commit_json(root/'audit.json',proof)
        terminal=dict(exit_code=0,status='formal_cached_successor_collected_and_audited',updates=634,
            selection_evaluation_pending=True,transfer_evaluation_pending=True,restart=False)
        commit_json(root/'terminal.json',terminal);return terminal
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True));raise


def detach(template,local):
    c,_=load_config(template,local);validate(c)
    root=Path(c['output']).with_name(Path(c['output']).name+'-driver')
    check_paths({'formal_successor_driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.context_cached_formal_successor','--config',str(Path(template).resolve()),
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
    else:validate(c);result=dict(status='formal_successor_validated_only',npu_executed=False)
    print(json.dumps(result,indent=2))
