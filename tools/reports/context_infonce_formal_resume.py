"""Resume an admitted Stage1 runtime unchanged from a committed checkpoint.

Only paths, start cursor, read-only resume pins and resume experiment kind
change. Parameter/Adam validation is independent and repeated by every NPU rank.
"""
import argparse,json,shutil,subprocess,sys,time
from pathlib import Path
from types import FunctionType,SimpleNamespace
from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports import context_infonce_formal as formal
from tools.reports import context_instruction_operations as shared
from tools.reports.context_infonce_recovery import command


def derive(original,old,new,commit,commit_sha):
    from tools.reports.audit_context_infonce_training import expected_contract
    if (commit['contract']!=expected_contract(original) or not 0<commit['updates']<original['stop_update']
            or len(commit['rank_shards'])!=len(original['physical_devices'])):
        raise ValueError('complete original-contract intermediate checkpoint required')
    c=json.loads(json.dumps(original))
    for key in ('source_root','schedule','parent_schedule'):
        if not c[key].startswith(old+'/'):raise ValueError('internal path mismatch')
        c[key]=new+c[key][len(old):]
    c['pins']={new+k[len(old):] if k.startswith(old+'/') else k:v for k,v in c['pins'].items()}
    c.update(start_update=commit['updates'],resume_root=old+'/run',experiment_kind='formal_infonce_resume')
    for name,h in commit['rank_shards'].items():
        if name!=f'rank-{int(name.split("/")[0].split("-")[1])}/checkpoint-{commit["updates"]:04d}.pt':raise ValueError('unexpected shard path')
        c['pins'][old+'/run/'+name]=h
    c['pins'][old+f'/run/checkpoint-{commit["updates"]:04d}-committed.json']=commit_sha
    return c


def parent(c):
    ops,_=load_config(c['parent_template'],c['parent_local']);m,t,_=formal.validate(ops)
    collection=Path(ops['operations_root'])/(ops['tag']+'-collected')
    transport=json.loads(collection.with_suffix('.transport.json').read_text())
    step=c['resume_step'];name=f'run/checkpoint-{step:04d}-committed.json';p=collection/name
    if file_hash(p)!=transport['files'].get(name):raise ValueError('checkpoint commit transport mismatch')
    commit=json.loads(p.read_text())
    if commit['updates']!=step:raise ValueError('checkpoint cursor mismatch')
    from tools.reports.audit_context_infonce_training import restore_shard
    from tools.reports.context_master_export_audit import audit
    import torch
    states=[]
    for rank in range(len(t['physical_devices'])):
        name=f'rank-{rank}/checkpoint-{step:04d}.pt';p=collection/'run'/name
        if file_hash(p)!=commit['rank_shards'].get(name) or file_hash(p)!=transport['files'].get('run/'+name):raise ValueError('checkpoint shard changed')
        state=torch.load(p,map_location='cpu',weights_only=True);restore_shard(state,t,rank=rank,step=step);states.append(state)
    p=collection/'run'/commit['execution_file']
    if file_hash(p)!=commit['execution_sha256']:raise ValueError('execution changed')
    proof=audit(states,torch.load(p,map_location='cpu',weights_only=True),world_size=len(states))
    return ops,m,t,commit,file_hash(collection/f'run/checkpoint-{step:04d}-committed.json'),proof


def prepare(c,provenance):
    ops,m,t,commit,h,proof=parent(c)
    root=Path(c['package']);root.mkdir(parents=True,exist_ok=False)
    shutil.copytree(Path(ops['package'])/'source',root/'source');shutil.copy2(Path(ops['package'])/'run.py',root/'run.py')
    config=derive(t,m['remote_root'],c['remote_root'],commit,h);commit_json(root/'config.json',config)
    manifest=dict(m,status='formal_infonce_resume_prepared_only',remote_root=c['remote_root'],
        parent_manifest_sha256=ops['manifest_sha256'],resume_commit_sha256=h,provenance=provenance,
        files={str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file()})
    commit_json(root/'PREPARED.json',manifest)
    return dict(manifest_sha256=file_hash(root/'PREPARED.json'),resume_step=commit['updates'],checkpoint_audit=proof)


def validate(c):
    ops,m,t,commit,h,_=parent(c)
    root=Path(c['package']);manifest=json.loads((root/'PREPARED.json').read_text());actual={str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file() and p!=root/'PREPARED.json'}
    if (file_hash(root/'PREPARED.json')!=c['manifest_sha256'] or actual!=manifest['files']
            or manifest['parent_manifest_sha256']!=ops['manifest_sha256'] or manifest['resume_commit_sha256']!=h
            or manifest['status']!='formal_infonce_resume_prepared_only' or manifest['remote_root']!=c['remote_root']
            or any(p.is_symlink() for p in root.rglob('*'))
            or {n:v for n,v in actual.items() if n.startswith('source/') or n=='run.py'}!={n:v for n,v in m['files'].items() if n.startswith('source/') or n=='run.py'}
            or any(manifest[k]!=m[k] for k in ('image','layout','training_devices','inference_services','batch_plan'))):raise ValueError('unchanged admitted runtime required')
    config=json.loads((root/'config.json').read_text())
    if config!=derive(t,m['remote_root'],c['remote_root'],commit,h):raise ValueError('resume changed configuration')
    if Path(c['remote_root']).is_relative_to(m['remote_root']) or Path(m['remote_root']).is_relative_to(c['remote_root']):raise ValueError('disjoint resume root required')
    return manifest,config,{}


def launch(c,p):
    if c.get('formal_training_dispatch_allowed') is not True:raise ValueError('formal dispatch required')
    lifecycle=FunctionType(shared.lifecycle.launch.__code__,dict(shared.lifecycle.launch.__globals__,validate=validate,command=command),'resume_launch')
    proxy=SimpleNamespace(IMAGE=shared.lifecycle.IMAGE,launch=lifecycle)
    return FunctionType(shared.launch.__code__,dict(shared.launch.__globals__,lifecycle=proxy,validate=validate,command=command),'launch')(c,p)


def run(c,p):
    root=Path(c['operations_root']);root.mkdir(parents=True,exist_ok=True)
    try:
        result=launch(c,p);commit_json(root/'launch-result.json',result)
        from tools.reports.audit_context_infonce_training import audit_collected
        proof=audit_collected(dict(operations_config=c['self_template'],operations_local=c['self_local'],initial_policy=c['initial_policy']),validator=validate,command_builder=command)
        commit_json(root/'resume-audit.json',proof);commit_json(root/'terminal.json',dict(exit_code=0,status='formal_resume_collected_and_audited'))
    except BaseException as exc:commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc)));raise

if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--local',required=True);a.add_argument('--action',choices=['prepare','launch','detach'],required=True);x=a.parse_args();c,p=load_config(x.config,x.local)
    if x.action=='prepare':print(json.dumps(prepare(c,p)))
    elif x.action=='launch':run(c,p)
    else:
        validate(c);d=Path(c['operations_root'])/'driver';d.mkdir(parents=True,exist_ok=False)
        argv=[sys.executable,'-B','-m','tools.reports.context_infonce_formal_resume','--config',str(Path(x.config).resolve()),'--local',str(Path(x.local).resolve()),'--action','launch']
        with (d/'driver.log').open('x') as f:child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        commit_json(d/'started.json',dict(pid=child.pid,argv=argv,process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19],unix=time.time()));print(child.pid)
