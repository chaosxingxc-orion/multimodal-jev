"""Prepare a fresh-process second-update replay from an audited DP8 prefix.

Copies the accepted runtime unchanged. Only isolated output paths and the
authenticated resume window change; never edits the parent training package.
"""
import argparse
import json
from pathlib import Path
import shutil
import re
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports import context_instruction_operations as shared


def parent(c):
    ops,_=load_config(c['parent_operations_template'],c['parent_operations_local'])
    m,t,_=shared.validate(ops)
    report=json.loads(Path(c['parent_audit']).read_text())
    if (file_hash(c['parent_audit'])!=c['parent_audit_sha256']
            or report['status']!='finite_infonce_artifacts_audited'
            or report['manifest_sha256']!=ops['manifest_sha256']
            or [u['step'] for u in report['updates']]!=[1,2]):
        raise ValueError('audited two-update parent required')
    collected=Path(ops['operations_root'])/(ops['tag']+'-collected')
    transport=Path(ops['operations_root'])/(ops['tag']+'-collected.transport.json')
    if file_hash(transport)!=report['transport_sha256']:
        raise ValueError('parent transport changed')
    files=json.loads(transport.read_text())['files']
    for name,h in files.items():
        if file_hash(collected/name)!=h:raise ValueError('audited parent bytes changed')
    return ops,m,t,collected,report


def derive_config(original,old,new,commit):
    """Keep optimizer/schedule contract equal; bind only committed replay inputs."""
    c=json.loads(json.dumps(original))
    for key in ('source_root','schedule','parent_schedule'):
        if not c[key].startswith(old+'/'):raise ValueError('parent internal path differs')
        c[key]=new+c[key][len(old):]
    c['pins']={new+k[len(old):] if k.startswith(old+'/') else k:v for k,v in c['pins'].items()}
    c.update(start_update=1,stop_update=2,resume_root=old+'/run')
    from multimodal_jev.context_topology_batch_plan import derive
    if commit['updates']!=1 or len(commit['rank_shards'])!=derive(c['batch_config'])['dp_replicas']:
        raise ValueError('complete step1 declared-DP commit required')
    for name,h in commit['rank_shards'].items():
        c['pins'][old+'/run/'+name]=h
    return c


def prepare(c,provenance):
    ops,m,original,collected,report=parent(c)
    dest=Path(c['package']);remote=shared.lifecycle.path(c['remote_root'])
    if Path(remote).is_relative_to(m['remote_root']) or Path(m['remote_root']).is_relative_to(remote):
        raise ValueError('fresh replay root must be disjoint')
    check_paths({'replay_package':dest});dest.mkdir(parents=True,exist_ok=False)
    shutil.copytree(Path(ops['package'])/'source',dest/'source')
    commit_path=collected/'run/checkpoint-0001-committed.json'
    config=derive_config(original,m['remote_root'],remote,json.loads(commit_path.read_text()))
    config['pins'][m['remote_root']+'/run/checkpoint-0001-committed.json']=file_hash(commit_path)
    commit_json(dest/'config.json',config);shutil.copy2(Path(ops['package'])/'run.py',dest/'run.py')
    manifest=dict(m,status='infonce_replay_prepared_only',remote_root=remote,provenance=provenance,
        parent_container_id=report['container_id'],parent_manifest_sha256=ops['manifest_sha256'],
        parent_audit_sha256=c['parent_audit_sha256'],
        files={str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()})
    commit_json(dest/'PREPARED.json',manifest)
    return dict(status=manifest['status'],manifest_sha256=file_hash(dest/'PREPARED.json'),
                formal_training_allowed=False)


def validate(c):
    if (not re.fullmatch(r'(?:[a-zA-Z0-9][a-zA-Z0-9_.-]*@)?[a-zA-Z0-9][a-zA-Z0-9_.-]*',c['host'])
            or not re.fullmatch('[a-z0-9][a-z0-9-]*',c['tag'])):
        raise ValueError('safe host alias and tag required')
    ops,pm,original,collected,report=parent(c)
    root=Path(c['package']);m=json.loads((root/'PREPARED.json').read_text());t=json.loads((root/'config.json').read_text())
    remote=shared.lifecycle.path(m['remote_root'])
    if Path(remote).is_relative_to(pm['remote_root']) or Path(pm['remote_root']).is_relative_to(remote):
        raise ValueError('fresh replay root must be disjoint')
    if file_hash(root/'PREPARED.json')!=c['manifest_sha256']:raise ValueError('replay manifest changed')
    actual={str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file() and p!=root/'PREPARED.json'}
    if actual!=m['files'] or any(p.is_symlink() for p in root.rglob('*')):raise ValueError('replay files changed')
    commit=collected/'run/checkpoint-0001-committed.json'
    expected=derive_config(original,pm['remote_root'],m['remote_root'],json.loads(commit.read_text()))
    expected['pins'][pm['remote_root']+'/run/checkpoint-0001-committed.json']=file_hash(commit)
    if (t!=expected or m['status']!='infonce_replay_prepared_only'
            or m['parent_container_id']!=report['container_id']
            or m['parent_audit_sha256']!=c['parent_audit_sha256']
            or m['parent_manifest_sha256']!=ops['manifest_sha256']
            or any(m[k]!=pm[k] for k in ('image','training_devices','inference_services','batch_plan','layout'))
            or file_hash(root/'run.py')!=pm['files']['run.py']
            or {k:h for k,h in actual.items() if k.startswith('source/')} !=
               {k:h for k,h in pm['files'].items() if k.startswith('source/')}):
        raise ValueError('replay must reuse exact parent runtime and bound optimizer state')
    shared.generate(root/'source')
    shared.lifecycle.release_evidence(c,t)
    return m,t,{}


def command(c,m,t):
    # Shared command requires all pins under explicit mounts; resume tensors
    # live outside the new root and are added as a read-only input.
    resume=t['resume_root']
    base=dict(t,pins={k:v for k,v in t['pins'].items() if not k.startswith(resume+'/')})
    argv=shared.command(c,m,base)
    at=argv.index('--entrypoint');argv[at:at]=['-v',resume+':'+resume+':ro']
    return argv


def launch(c,provenance):
    # Keep CPU source checks, resource lease and exact wait/collect from the
    # parent lifecycle, swapping only the replay validator and mount builder.
    lifecycle_launch=FunctionType(shared.lifecycle.launch.__code__,
        dict(shared.lifecycle.launch.__globals__,validate=validate,command=command),'replay_launch')
    # Shared launch dynamically composes lifecycle.launch; provide its module
    # globals explicitly rather than mutating a module used by other tasks.
    from types import SimpleNamespace
    proxy=SimpleNamespace(IMAGE=shared.lifecycle.IMAGE,launch=lifecycle_launch)
    runner=FunctionType(shared.launch.__code__,dict(shared.launch.__globals__,
        lifecycle=proxy,validate=validate,command=command),'launch')
    return runner(c,provenance)


def compare(continuous,replayed,*,world_size=8):
    """Only call after both runs pass independent runtime/artifact audits."""
    import torch
    from tools.jev.context_replay_audit import require_exact
    continuous,replayed=Path(continuous),Path(replayed)
    def read(root,name):return json.loads((root/name).read_text())
    def tensor(root,name):return torch.load(root/name,map_location='cpu',weights_only=True)
    require_exact(read(continuous,'batch-0001.json'),read(replayed,'batch-0001.json'),'batch')
    a=read(continuous,'checkpoint-0002-committed.json');b=read(replayed,'checkpoint-0002-committed.json')
    require_exact(a['contract'],b['contract'],'contract')
    if a['updates']!=2 or b['updates']!=2:raise ValueError('step2 replay required')
    if type(world_size) is not int or not 1<=world_size<=8:raise ValueError('declared DP width required')
    for rank in range(world_size):
        name=f'rank-{rank}/checkpoint-0002.pt'
        require_exact(tensor(continuous,name),tensor(replayed,name),f'rank-{rank}')
        if read(replayed,f'rank-{rank}/restored.json')!=dict(exact_masters=True,exact_adam=True,completed_updates=1):
            raise ValueError('exact restored step1 required')
    require_exact(tensor(continuous,'execution-0002.pt'),tensor(replayed,'execution-0002.pt'),'execution')
    require_exact(tensor(continuous,'policy-0002/execution.pt'),tensor(replayed,'policy-0002/execution.pt'),'policy')
    return dict(exact_masters=True,exact_adam=True,exact_batch=True,exact_bf16_execution=True,rank_shards=world_size,step=2)


def audit(c):
    from tools.reports.audit_context_infonce_training import audit_collected
    ops,_=load_config(c['operations_config'],c['operations_local'])
    _,_,_,previous,report=parent(ops)
    result=audit_collected(c,validator=validate,command_builder=command)
    if result['container_id']==report['container_id']:raise ValueError('fresh container required')
    current=Path(ops['operations_root'])/(ops['tag']+'-collected')
    result['replay']=compare(previous/'run',current/'run',world_size=result['world_size'])
    result['fresh_process_npu_resume_verified']=True
    result['status']='finite_infonce_replay_audited'
    return result


def detach(template,local):
    import subprocess,sys,time
    c,_=load_config(template,local);validate(c)
    root=Path(c['operations_root'])/'driver';check_paths({'driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.context_infonce_recovery','--config',str(Path(template).resolve()),
          '--local',str(Path(local).resolve()),'--action','launch']
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,argv=argv,started_unix=time.time(),
                process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    commit_json(root/'started.json',record);return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','validate','launch','audit','detach'],required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    if a.action=='prepare':result=prepare(c,provenance)
    elif a.action=='launch':
        import time
        terminal=Path(c['operations_root'])/'driver/terminal.json'
        try:result=launch(c,provenance)
        except BaseException as exc:
            if terminal.parent.is_dir():commit_json(terminal,dict(exit_code=1,error=repr(exc),finished_unix=time.time()))
            raise
        else:
            if terminal.parent.is_dir():commit_json(terminal,dict(exit_code=0,result=result,finished_unix=time.time()))
    elif a.action=='detach':result=detach(a.config,a.local)
    elif a.action=='audit':
        result=audit(c);commit_json(c['output'],dict(result,provenance=provenance))
    else:result=dict(status='validated_only',argv=command(c,*validate(c)[:2]))
    print(json.dumps(result,indent=2))
