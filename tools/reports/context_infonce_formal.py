"""Freeze full TRAIN InfoNCE configuration on an independently admitted runtime.

No dispatch until prefix, fresh replay and actual tail artifacts all pass.
"""
import argparse
import json
from pathlib import Path
import shutil
import re
from types import FunctionType, SimpleNamespace

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_topology_batch_plan import derive
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_infonce_recovery import parent
from tools.reports import context_instruction_operations as shared


def formal_config(original,old,new,design):
    config=json.loads(json.dumps(original));stage=design['stage1'];plan=derive(config['batch_config'])
    shared.topology(config,plan['dp_replicas'],['davinci'+str(d) for d in config['physical_devices']])
    from tools.reports.context_replica_formal_candidate import window
    steps=window(plan,stage['evaluation_steps'])
    interval=steps[1]
    if (steps!=[0,*range(interval,plan['total_updates'],interval),plan['total_updates']] or config['instruction_objective']!=stage['instruction_objective']
            or config['lr_schedule']!=stage['lr_schedule'] or config['batch_config']!=stage['batch_config']
            or stage['reference_endpoint']!=plan['total_updates']):
        raise ValueError('preregistered interval/end-point curve and admitted Stage1 settings required')
    config.update(start_update=0,stop_update=plan['total_updates'],resume_root=None,
        engineering_prefix=False,training_dispatch_allowed=True,checkpoint_interval=interval,
        experiment_kind='formal_infonce_source',evaluation_steps=steps,reference_endpoint=stage['reference_endpoint'],
        selection_rule=design['selection_rule'],monitor_seconds=design['monitor_seconds'])
    if 'lora' in stage or 'lora' in design['stage2']:
        from multimodal_jev.context_stage_lora import bind_design
        config['stage_lora_transition']=bind_design(design,config['source_contract'])
    for key in ('source_root','schedule','parent_schedule'):
        if not config[key].startswith(old+'/'):raise ValueError('internal path differs')
        config[key]=new+config[key][len(old):]
    config['pins']={new+k[len(old):] if k.startswith(old+'/') else k:v for k,v in config['pins'].items()}
    return config


def admission(c,source_manifest):
    reports={}
    plan=source_manifest['batch_plan']
    for kind,expected in [('replay','finite_infonce_replay_audited'),('tail','finite_infonce_tail_audited')]:
        path=Path(c[kind+'_audit']);report=json.loads(path.read_text())
        if file_hash(path)!=c[kind+'_audit_sha256'] or report['status']!=expected or report['world_size']!=plan['dp_replicas']:
            raise ValueError('independent '+kind+' admission required')
        ops,_=load_config(c[kind+'_operations_template'],c[kind+'_operations_local'])
        if kind=='replay':
            from tools.reports.context_infonce_recovery import validate
        else:
            from tools.reports.context_infonce_tail_operations import validate
        m,_,_=validate(ops)
        transport=Path(ops['operations_root'])/(ops['tag']+'-collected.transport.json')
        if (report['manifest_sha256']!=ops['manifest_sha256']
                or file_hash(transport)!=report['transport_sha256']
                or any(m['files'].get(n)!=h for n,h in source_manifest['files'].items() if n.startswith('source/'))):
            raise ValueError('admission does not cover this runtime')
        receipt=json.loads(transport.read_text());collection=Path(ops['operations_root'])/(ops['tag']+'-collected')
        if any(file_hash(collection/n)!=h for n,h in receipt['files'].items()):raise ValueError('admission evidence changed')
        reports[kind]=report
    if (reports['replay']['fresh_process_npu_resume_verified'] is not True
            or [u['real_queries'] for u in reports['tail']['updates']]!=[plan['tail_real_queries'],plan['global_batch']]):
        raise ValueError('exact replay and true zero-weight tail required')
    return {k:dict(container_id=v['container_id'],manifest_sha256=v['manifest_sha256']) for k,v in reports.items()}


def prepare(c,provenance):
    ops,m,original,_,_=parent(c)
    evidence=admission(c,m)
    design=json.loads(Path(c['design']).read_text())
    dest=Path(c['package']);remote=shared.lifecycle.path(c['remote_root'])
    if Path(remote).is_relative_to(m['remote_root']) or Path(m['remote_root']).is_relative_to(remote):
        raise ValueError('fresh disjoint formal root required')
    config=formal_config(original,m['remote_root'],remote,design)
    check_paths({'formal_package':dest});dest.mkdir(parents=True,exist_ok=False)
    shutil.copytree(Path(ops['package'])/'source',dest/'source')
    shutil.copy2(Path(ops['package'])/'run.py',dest/'run.py')
    commit_json(dest/'config.json',config)
    manifest=dict(m,status='formal_infonce_prepared_only',remote_root=remote,provenance=provenance,
        parent_manifest_sha256=ops['manifest_sha256'],parent_audit_sha256=c['parent_audit_sha256'],
        admission=evidence,design_sha256=file_hash(c['design']),
        files={str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()})
    commit_json(dest/'PREPARED.json',manifest)
    return dict(status=manifest['status'],manifest_sha256=file_hash(dest/'PREPARED.json'),
                total_updates=config['stop_update'],evaluation_steps=config['evaluation_steps'])


def validate(c):
    if (not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]*',c['host'])
            or not re.fullmatch('[a-z0-9][a-z0-9-]*',c['tag'])):
        raise ValueError('safe host alias and tag required')
    ops,pm,original,_,_=parent(c);evidence=admission(c,pm)
    root=Path(c['package']);m=json.loads((root/'PREPARED.json').read_text())
    t=json.loads((root/'config.json').read_text());remote=shared.lifecycle.path(m['remote_root'])
    if (remote!=c['remote_root'] or Path(remote).is_relative_to(pm['remote_root'])
            or Path(pm['remote_root']).is_relative_to(remote)):
        raise ValueError('fresh disjoint formal root required')
    actual={str(p.relative_to(root)):file_hash(p) for p in root.rglob('*')
            if p.is_file() and p!=root/'PREPARED.json'}
    if (file_hash(root/'PREPARED.json')!=c['manifest_sha256'] or actual!=m['files']
            or any(p.is_symlink() for p in root.rglob('*'))
            or m['status']!='formal_infonce_prepared_only'
            or m['parent_manifest_sha256']!=ops['manifest_sha256']
            or m['parent_audit_sha256']!=c['parent_audit_sha256'] or m['admission']!=evidence
            or m['design_sha256']!=file_hash(c['design'])
            or any(m[k]!=pm[k] for k in ('image','training_devices','inference_services','batch_plan','layout'))
            or (root/'run.py').read_text()!=shared.RUNNER
            or file_hash(root/'run.py')!=pm['files']['run.py']
            or {k:h for k,h in actual.items() if k.startswith('source/')}!=
               {k:h for k,h in pm['files'].items() if k.startswith('source/')}):
        raise ValueError('formal package must retain exactly admitted runtime and evidence')
    design=json.loads(Path(c['design']).read_text())
    if t!=formal_config(original,pm['remote_root'],remote,design):
        raise ValueError('formal configuration differs from admitted design')
    from tools.reports.audit_context_infonce_training import checkpoint_steps
    checkpoint_steps(t)
    shared.generate(root/'source');shared.lifecycle.release_evidence(c,t)
    return m,t,{}


def launch(c,provenance):
    if c.get('formal_training_dispatch_allowed') is not True:
        raise ValueError('explicit formal training dispatch required')
    training=json.loads((Path(c['package'])/'config.json').read_text())
    if training['batch_config']['training_devices']!=4 or training['batch_config']['inference_instances']!=0:
        raise ValueError('current formal Stage1 budget is four training devices')
    lifecycle_launch=FunctionType(shared.lifecycle.launch.__code__,
        dict(shared.lifecycle.launch.__globals__,validate=validate,command=shared.command),'formal_launch')
    proxy=SimpleNamespace(IMAGE=shared.lifecycle.IMAGE,launch=lifecycle_launch)
    runner=FunctionType(shared.launch.__code__,dict(shared.launch.__globals__,
        lifecycle=proxy,validate=validate,command=shared.command),'launch')
    return runner(c,provenance)


def audit(c):
    from tools.reports.audit_context_infonce_training import audit_collected
    result=audit_collected(c,validator=validate,command_builder=shared.command)
    operations,_=load_config(c['operations_config'],c['operations_local'])
    config=json.loads((Path(operations['package'])/'config.json').read_text());plan=derive(config['batch_config'])
    if (len(result['updates'])!=plan['total_updates']
            or result['real_exposures']!=config['batch_config']['source_queries']*config['batch_config']['epochs']):
        raise ValueError('full source exposure and update budget required')
    result['status']='formal_infonce_artifacts_audited'
    result['efficacy_evaluated']=False
    return result


def detach(template,local):
    import subprocess,sys,time
    c,_=load_config(template,local);validate(c)
    if c.get('formal_training_dispatch_allowed') is not True:
        raise ValueError('explicit formal training dispatch required')
    root=Path(c['operations_root'])/'driver';check_paths({'driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.context_infonce_formal','--config',str(Path(template).resolve()),
          '--local',str(Path(local).resolve()),'--action','launch']
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,argv=argv,started_unix=time.time(),
        process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    commit_json(root/'started.json',record);return record


if __name__=='__main__':
    import time
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','validate','launch','detach','audit'],default='prepare')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    if a.action=='prepare':result=prepare(c,provenance)
    elif a.action=='detach':result=detach(a.config,a.local)
    elif a.action=='audit':
        result=audit(c);commit_json(c['output'],dict(result,provenance=provenance))
    elif a.action=='validate':result=dict(status='validated_only',argv=shared.command(c,*validate(c)[:2]))
    else:
        terminal=Path(c['operations_root'])/'driver/terminal.json'
        try:result=launch(c,provenance)
        except BaseException as exc:
            if terminal.parent.is_dir():commit_json(terminal,dict(exit_code=1,error=repr(exc),finished_unix=time.time()))
            raise
        else:
            if terminal.parent.is_dir():commit_json(terminal,dict(exit_code=0,result=result,finished_unix=time.time()))
    print(json.dumps(result,indent=2))
