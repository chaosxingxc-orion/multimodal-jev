"""Collect exact terminal probe outputs and independently bind every rollout.

No training admission or optimizer/speedup claim. Original containers and files
are preserved, including failed outputs. Collections are never overwritten.
"""
import argparse
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_macro_rollout import MacroRolloutPool, digest, plan_rollouts
from multimodal_jev.context_logits_wire import validate_logits_record
from multimodal_jev.context_rollout_batch import bind_rollout
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_macro_probe_operations import validate, inspect_runtime, command
from tools.reports.context_macro_measurement import check_reference
from tools.reports.launch_context_batch_probe_candidate import write_new


def audit_outputs(package, collection):
    package,collection=Path(package),Path(collection)
    def read(path):return json.loads(path.read_text())
    config=read(package/'measurement.json');client=config['macro_client']
    rows=torch.load(package/'inputs.pt',map_location='cpu',weights_only=True)
    policy=config['policy']['policy_id']
    expected=check_reference(read(package/'serial-reference.json'),rows,policy,file_hash(package/'inputs.pt'))
    for role in ('service-0','service-1','measurement'):
        if read(collection/role/'terminal.json')['exit_code']!=0:raise ValueError('successful terminal outputs required')
    for i in range(2):
        service=read(package/f'service-{i}.json');ready=read(collection/f'service-{i}/ready.json')
        if (ready['physical_devices']!=[6+i] or ready['policy_id']!=policy
                or ready['contract']!=config['policy']['contract']
                or ready['max_batch_queries']!=service['policy']['max_batch_queries']
                or ready['model_forward_mode']!=client['runtime_identity']['profile']):
            raise ValueError('service startup does not match sealed configuration')
    result=read(collection/'measurement/measurement.json')
    if (result['policy_id']!=policy or result['rows']!=128
            or result['runtime_identity']!=client['runtime_identity'] or len(result['passes'])!=4):
        raise ValueError('complete four-pass matching runtime required')
    journal=collection/'journal'/client['attempt_id'];times=[];owners=[]
    for index,p in enumerate(result['passes']):
        order=list(range(128));kind='warmup' if index==0 else 'reversed' if index==3 else 'timed'
        if index==3:order.reverse()
        if (p['order']!=order or p['kind']!=kind or not math.isfinite(p['seconds']) or p['seconds']<=0
                or p['records']!=expected['records'] or p['sampling']!=expected['sampling']):
            raise ValueError('full pass order/results or timing differs')
        directory=journal/f'batch-{index:08d}'
        manifest=read(directory/'INPUT.json');complete=read(directory/'COMPLETE.json')
        identities,costs=MacroRolloutPool._inputs([rows[i] for i in order])
        jobs=plan_rollouts(costs,client['rollout_queries'])
        if (manifest['inputs']!=identities or manifest['jobs']!=jobs
                or manifest['expected_indices']!=list(range(128)) or manifest['policy_id']!=policy
                or manifest['runtime_identity']!=client['runtime_identity']
                or manifest['batch_id']!=directory.name):
            raise ValueError('actual input/rollout ownership manifest differs')
        ordered=[None]*128;hashes={}
        for job in jobs:
            recorded=read(directory/(job['rollout_id']+'.json'))
            if (any(recorded[k]!=v for k,v in job.items()) or recorded['policy_id']!=policy
                    or len(recorded['records'])!=len(job['indices'])
                    or not math.isfinite(recorded['seconds']) or recorded['seconds']<=0):
                raise ValueError('rollout ownership/population/timing differs')
            hashes[job['rollout_id']]=digest(recorded)
            for slot,r in zip(job['indices'],recorded['records'],strict=True):
                validate_logits_record(r)
                bind_rollout(r,**rows[order[slot]],policy_id=policy,group_size=16,seed=317+order[slot])
                if ordered[slot] is not None or r!=expected['records'][order[slot]]:
                    raise ValueError('duplicate or incorrect actual rollout result')
                ordered[slot]=r
        if (any(r is None for r in ordered) or complete!=p['macro_receipt']
                or complete['input_sha256']!=digest(manifest) or complete['rollout_hashes']!=hashes
                or complete['result_sha256']!=digest(ordered) or complete['completed_indices']!=list(range(128))
                or complete['actor_update_performed'] is not False
                or complete['status']!='complete_macro_inference' or complete['policy_id']!=policy
                or complete['batch_id']!=directory.name):
            raise ValueError('macro completion does not match actual rollout files')
        owners.append([j['replica_id'] for j in jobs])
        if kind=='timed':times.append(p['seconds'])
    mean=sum(times)/len(times)
    if (result['timed_mean_seconds']!=mean or result['queries_per_second']!=128/mean
            or result['exact_actual_logits'] is not True or result['exact_probabilities_and_G16'] is not True):
        raise ValueError('reported aggregate differs')
    return dict(status='finite_two_replica_macro_audited',rows=128,passes=4,timed_seconds=times,
        mean_seconds=mean,queries_per_second=128/mean,rollout_replica_owners=owners,
        exact_actual_logits=True,exact_probabilities_and_G16=True,policy_id=policy,
        scope='Complete dual-instance fixed-policy inference; no actor, policy-update, recovery or training-speed claim')


def collect_and_audit(c, *, validator=validate, command_builder=command,
                      output_auditor=audit_outputs, result_dirs=('measurement','journal','service-0','service-1')):
    receipt=validator(c);base=Path(c['operations_root']);check_paths({'collection':base})
    terminal=json.loads((base/'terminal.json').read_text())
    if terminal['exit_code']!=0:raise ValueError('successful driver terminal required; preserve failure files')
    transfer_timeout=c.get('collection_transfer_timeout_seconds',1800)
    if type(transfer_timeout) is not int or not 55<=transfer_timeout<=7200:
        raise ValueError('bounded collection transfer timeout required')
    def ssh(args, *, timeout=55):
        result=subprocess.run(['ssh','-o','ConnectTimeout=10',c['host'],shlex.join(args)],
            capture_output=True,text=True,timeout=timeout)
        if result.returncode:
            raise RuntimeError('remote collection command failed: '+result.stderr[-8000:])
        return result.stdout
    inspections={}
    for role in ('service-0','service-1','client','preflight'):
        h=json.loads((base/(role+'-launch.json')).read_text())
        if h['argv']!=command_builder(c,role):raise ValueError('launch command differs from configuration')
        obs=json.loads(ssh(['docker','inspect',h['container_id']]))[0]
        inspect_runtime(obs,h,terminal=True);inspections[role]=obs
    remote=c['remote_root']
    if (not result_dirs or any(not isinstance(n,str) or not n or '/' in n or n in ('.','..')
            for n in result_dirs)):raise ValueError('explicit result directory names required')
    verify_code="""import hashlib,json,sys
from pathlib import Path
root=Path(ROOT);sys.path.insert(0,str(root/'package'))
from tools.reports.run_context_batch_probe_candidate import verify
actual=verify(root/'package')
if actual!=EXPECTED:raise ValueError('remote sealed package changed')
p=root/'package/training.json'
if p.exists():
 c=json.loads(p.read_text());mapped={m['target']:m['source'] for m in c.get('publication_mounts',[])}
 for name,digest in c['pins'].items():
  h=hashlib.sha256()
  with open(mapped.get(name,name),'rb') as stream:
   for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
  if h.hexdigest()!=digest:raise ValueError('remote dependency changed: '+name)
print(actual)
""".replace('ROOT',repr(remote)).replace('EXPECTED',repr(receipt['files_sha256']))
    if ssh(['python3','-B','-c',verify_code]).strip()!=receipt['files_sha256']:
        raise ValueError('remote package verification differs')
    # Enumerate only result directories; never collect models/caches/base weights.
    code="""from pathlib import Path
import hashlib,json,tarfile,os
r=Path(ROOT);names=RESULT_DIRS
files={}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def fail(error):raise error
for name in names:
 base=r/name
 if not base.is_dir() or base.is_symlink():raise ValueError('missing or linked result directory')
 for directory,dirs,leaves in os.walk(base,onerror=fail,followlinks=False):
  for child in dirs+leaves:
   p=Path(directory)/child
   if p.is_symlink():raise ValueError('symlink in result')
  for child in leaves:
   p=Path(directory)/child
   if not p.is_file():raise ValueError('unexpected result file type')
   files[str(p.relative_to(r))]=sha(p)
archive=r/'results.tar'
with tarfile.open(archive,'x') as t:
 for name in sorted(files):t.add(r/name,arcname=name,recursive=False)
print(json.dumps(dict(files=files,archive_sha256=sha(archive))))
""".replace('ROOT',repr(remote)).replace('RESULT_DIRS',repr(tuple(result_dirs)))
    collection=base/'collected';collection.mkdir(exist_ok=False)
    transport=json.loads(ssh(['python3','-B','-c',code],timeout=transfer_timeout))
    write_new(base/'transport.json',transport)
    archive=base/'results.tar'
    subprocess.run(['scp','-q',c['host']+':'+remote+'/results.tar',str(archive)],check=True,capture_output=True,timeout=transfer_timeout)
    if file_hash(archive)!=transport['archive_sha256']:raise ValueError('transport archive changed')
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        if (len(members)!=len(transport['files']) or {m.name for m in members}!=set(transport['files'])
                or any(not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in members)):
            raise ValueError('unexpected archive members')
        for member in members:
            path=collection/member.name;path.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source,path.open('xb') as dest:
                shutil.copyfileobj(source,dest,length=4*1024*1024)
    if {str(p.relative_to(collection)):file_hash(p) for p in collection.rglob('*') if p.is_file()}!=transport['files']:
        raise ValueError('collected bytes differ')
    write_new(base/'collected-inspect.json',inspections)
    result=output_auditor(Path(c['prepared'])/'package',collection)
    result.update(files_sha256=receipt['files_sha256'],transport_sha256=file_hash(base/'transport.json'),
        containers={r:obs['Id'] for r,obs in inspections.items()},image=receipt['image'])
    write_new(base/'audit.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,_=load_config(a.config,a.local)
    print(json.dumps(collect_and_audit(c),indent=2))
