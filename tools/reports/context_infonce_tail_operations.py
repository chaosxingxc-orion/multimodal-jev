"""Finite real-audio tail acceptance on the admitted InfoNCE DP8 lifecycle."""
import argparse
import json
from pathlib import Path
import shutil
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports import context_instruction_operations as shared
from tools.reports.context_infonce_recovery import parent
from tools.reports.context_infonce_tail import contract,generate

ENTRY='tools/reports/context_infonce_tail.py'
RUNNER=shared.RUNNER.replace('tools/reports/context_infonce_trainer.py',ENTRY)


def replay(c):
    report=json.loads(Path(c['replay_audit']).read_text())
    if (file_hash(c['replay_audit'])!=c['replay_audit_sha256']
            or report['status']!='finite_infonce_replay_audited'
            or report['fresh_process_npu_resume_verified'] is not True
            or report['replay']!=dict(exact_masters=True,exact_adam=True,exact_batch=True,
                                     exact_bf16_execution=True,rank_shards=report['world_size'],step=2)):
        raise ValueError('exact declared-DP fresh-process replay acceptance required')
    return report


def prepare(c,provenance):
    ops,m,original,_,report=parent(c);r=replay(c)
    dest=Path(c['package']);remote=shared.lifecycle.path(c['remote_root'])
    check_paths({'tail_package':dest});dest.mkdir(parents=True,exist_ok=False)
    shutil.copytree(Path(ops['package'])/'source',dest/'source')
    shutil.copy2(ENTRY,dest/'source'/ENTRY)
    # Schedule validation is reused by the new tail entry.
    for name in ('tools/reports/context_replica_formal_candidate.py','tools/reports/context_replica_trainer_candidate.py',
                 'tools/reports/context_macro_trainer_candidate.py','tools/reports/context_dual_trainer_candidate.py'):
        shutil.copy2(name,dest/'source'/name)
    config=json.loads(json.dumps(original));old=m['remote_root']
    for key in ('source_root','schedule','parent_schedule'):config[key]=remote+config[key][len(old):]
    config['pins']={k:v for k,v in config['pins'].items() if not k.startswith(old+'/')}
    config['pins'].update({remote+'/source/'+str(p.relative_to(dest/'source')):file_hash(p)
                          for p in (dest/'source').rglob('*') if p.is_file()})
    config['experiment_kind']='infonce_tail_admission';contract(config);generate(dest/'source')
    commit_json(dest/'config.json',config);(dest/'run.py').write_text(RUNNER)
    manifest=dict(m,status='infonce_tail_prepared_only',remote_root=remote,provenance=provenance,
        parent_manifest_sha256=ops['manifest_sha256'],parent_container_id=report['container_id'],
        replay_container_id=r['container_id'],tail_admission=contract(config),
        files={str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()})
    commit_json(dest/'PREPARED.json',manifest)
    return dict(status=manifest['status'],manifest_sha256=file_hash(dest/'PREPARED.json'))


def validate(c):
    import re
    if (not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]*',c['host'])
            or not re.fullmatch('[a-z0-9][a-z0-9-]*',c['tag'])):raise ValueError('safe host/tag required')
    ops,pm,original,_,report=parent(c);r=replay(c)
    root=Path(c['package']);m=json.loads((root/'PREPARED.json').read_text());t=json.loads((root/'config.json').read_text())
    remote=shared.lifecycle.path(m['remote_root'])
    if Path(remote).is_relative_to(pm['remote_root']) or Path(pm['remote_root']).is_relative_to(remote):
        raise ValueError('fresh disjoint tail root required')
    actual={str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file() and p!=root/'PREPARED.json'}
    if (file_hash(root/'PREPARED.json')!=c['manifest_sha256'] or actual!=m['files']
            or any(p.is_symlink() for p in root.rglob('*')) or (root/'run.py').read_text()!=RUNNER
            or m['status']!='infonce_tail_prepared_only' or m['tail_admission']!=contract(t)
            or m['parent_manifest_sha256']!=ops['manifest_sha256']
            or m['parent_container_id']!=report['container_id'] or m['replay_container_id']!=r['container_id']
            or any(m[k]!=pm[k] for k in ('image','training_devices','inference_services','batch_plan','layout'))):
        raise ValueError('complete frozen bounded tail package required')
    exclude={'pins','source_root','schedule','parent_schedule','experiment_kind'}
    if {k:v for k,v in t.items() if k not in exclude}!={k:v for k,v in original.items() if k not in exclude}:
        raise ValueError('tail may not change training settings')
    for key in ('source_root','schedule','parent_schedule'):
        if t[key]!=remote+original[key][len(pm['remote_root']):]:raise ValueError('internal path differs')
    for name,h in pm['files'].items():
        if name.startswith('source/') and actual.get(name)!=h:raise ValueError('parent runtime changed')
    for name,h in t['pins'].items():
        shared.lifecycle.path(name)
        if name.startswith(remote+'/') and actual.get(name[len(remote)+1:])!=h:raise ValueError('internal pin changed')
    generate(root/'source');shared.lifecycle.release_evidence(c,t)
    return m,t,{}


def launch(c,provenance):
    report=json.loads(Path(c['tail_cpu_report']).read_text())
    if (file_hash(c['tail_cpu_report'])!=c['tail_cpu_report_sha256'] or report['image']!=shared.lifecycle.IMAGE
            or report['state']['ExitCode']!=0 or report['terminal']['exit_code']!=0
            or report['counts']!=dict(tests=c.get('tail_cpu_expected_tests',6),failures=0,errors=0,skipped=0)):
        raise ValueError('fixed image tail validation required')
    import xml.etree.ElementTree as ET
    junit=Path(report['collection'])/'run/junit.xml'
    if file_hash(junit)!=report['files'].get('run/junit.xml'):
        raise ValueError('tail CPU test evidence changed')
    cases=list(ET.parse(junit).getroot().iter('testcase'))
    if not any(node.attrib.get('name')=='test_actual_tail_selected_without_changing_formal_schedule'
               and not list(node) for node in cases):
        raise ValueError('actual tail selector CPU test required')
    manifest=Path(report['collection'])/'package/FILES.json'
    if file_hash(manifest)!=report['source_manifest_sha256']:raise ValueError('CPU source manifest changed')
    files=json.loads(manifest.read_text());package=Path(c['package'])/'source'
    if any(file_hash(package/n)!=h for n,h in files.items() if (package/n).is_file()):
        raise ValueError('tail candidate differs from CPU snapshot')
    runner=FunctionType(shared.lifecycle.launch.__code__,dict(shared.lifecycle.launch.__globals__,
        validate=validate,command=shared.command),'tail_launch')
    return runner(c,provenance)


def detach(template,local):
    import subprocess,sys,time
    c,_=load_config(template,local);validate(c)
    root=Path(c['operations_root'])/'driver';check_paths({'driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m',__name__ if __name__!='__main__' else 'tools.reports.context_infonce_tail_operations',
          '--config',str(Path(template).resolve()),'--local',str(Path(local).resolve()),'--action','launch']
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,argv=argv,started_unix=time.time(),
        process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    commit_json(root/'started.json',record);return record


def audit(c):
    from tools.reports.audit_context_infonce_training import audit_collected
    result=audit_collected(c,validator=validate,command_builder=shared.command)
    if not result['tail_admission'] or [u['real_queries'] for u in result['updates']]!=result['tail_admission']['real_queries']:
        raise ValueError('actual tail and next epoch updates required')
    result['status']='finite_infonce_tail_audited'
    return result


if __name__=='__main__':
    import time
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','validate','launch','detach','audit'],required=True)
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
