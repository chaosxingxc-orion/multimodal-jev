"""Read-only snapshot of one immutable commit while training may continue.

Never collects mutable logs or claims terminal training success. The original
terminal collector remains unchanged. Only LoRA/head recovery artifacts and
configuration are transported; no base model or reference cache is copied.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.jev.collect_context_rl_training import sha, unpack
from tools.reports.committed_signal_report import write


def inventory(root, step, launch, following_steps=0):
    root = Path(root)
    if type(step) is not int or step < 1:
        raise ValueError('explicit committed step required')
    if type(following_steps) is not int or following_steps not in (0,2):
        raise ValueError('zero or two completed comparison updates required')
    files = {}
    def include(name, expected=None):
        relative = Path(name); p = root/relative
        if relative.is_absolute() or '..' in relative.parts or p.resolve() != p or not p.is_file():
            raise ValueError('regular self-contained checkpoint artifact required')
        digest = sha(p)
        if expected is not None and digest != expected:
            raise ValueError('committed artifact changed: '+name)
        files[name] = digest
        return p
    for name, key in (('config.json','config_sha256'),('service-config.json','service_config_sha256'),
                      ('PREPARED.json','manifest_sha256')):
        include(name, launch[key])
    name = f'run/checkpoint-{step:04d}-committed.json'
    commit = json.loads(include(name).read_text())
    shards = {f'rank-{r}/checkpoint-{step:04d}.pt' for r in range(4)}
    if (commit['updates'] != step or set(commit['rank_shards']) != shards or
            commit['execution_file'] != f'execution-{step:04d}.pt' or
            commit['joint_policy_directory'] != f'policy-{step:04d}'):
        raise ValueError('exact DP4 recovery commit required')
    for name, digest in commit['rank_shards'].items():
        include('run/'+name, digest)
    include('run/'+commit['execution_file'], commit['execution_sha256'])
    policy = f'run/policy-{step:04d}'
    manifest = json.loads(include(policy+'/MANIFEST.json',commit['joint_policy_manifest_sha256']).read_text())
    if manifest['policy_id'] != commit['joint_policy']['policy_id']:
        raise ValueError('committed policy identity differs')
    for name, digest in manifest['files'].items():
        include(policy+'/'+name, digest)
    adapter = f'run/adapter-{step:04d}'
    provenance = json.loads(include(adapter+'/CONTEXT_POLICY.json').read_text())
    if (provenance['policy_id'] != manifest['policy_id'] or
            set(provenance['files']) != {'adapter_config.json','adapter_model.safetensors'}):
        raise ValueError('complete native adapter provenance required')
    for name, digest in provenance['files'].items():
        include(adapter+'/'+name, digest)
    policy_id=manifest['policy_id'];chain=[]
    for next_step in range(step+1,step+following_steps+1):
        exported=json.loads(include(f'run/policy-export-{next_step:04d}.json').read_text())
        directory=f'run/policy-{next_step:04d}'
        following=json.loads(include(directory+'/MANIFEST.json',exported['policy_manifest_sha256']).read_text())
        if exported['update']!=next_step or exported['policy_id']!=following['policy_id']:
            raise ValueError('following policy publication differs')
        for name,digest in following['files'].items():include(directory+'/'+name,digest)
        adapter=f'run/adapter-{next_step:04d}'
        native=json.loads(include(adapter+'/CONTEXT_POLICY.json').read_text())
        if native['policy_id']!=following['policy_id']:raise ValueError('following adapter differs')
        for name,digest in native['files'].items():include(adapter+'/'+name,digest)
        activation=json.loads(include(f'run/activation-{next_step:04d}.json').read_text())
        if activation['error'] is not None or activation['policy_id']!=following['policy_id']:
            raise ValueError('following policy activation failed')
        receipt=json.loads(include(f'run/rollout-{next_step-1:04d}.json').read_text())
        if receipt['update']!=next_step-1 or receipt['behavior_policy_id']!=policy_id:
            raise ValueError('following behavior chain differs')
        include(f'run/batch-{next_step-1:04d}.json');include(f'run/groups-{next_step-1:04d}.pt')
        for rank in range(4):
            record=json.loads(include(f'run/rank-{rank}/update-{next_step:04d}.json').read_text())
            if (record['update']!=next_step or record['next_policy_id']!=following['policy_id'] or
                    record['rollout']!=receipt):
                raise ValueError('complete following all-rank update required')
        policy_id=following['policy_id'];chain.append(policy_id)
    if following_steps:
        return dict(step=step,policy_id=manifest['policy_id'],following_policy_ids=chain,
            comparison_stop_update=step+following_steps,files=files,
            following_optimizer_states_collected=False)
    return dict(step=step, policy_id=manifest['policy_id'], files=files)


def remote_script(root, step, launch, expected=None, following_steps=0):
    # Pure stdlib reader. Recheck the identical inventory before archiving;
    # local unpack checks every transported byte, including concurrent changes.
    import inspect
    source = 'import json,hashlib,tarfile,sys\nfrom pathlib import Path\n'
    source += inspect.getsource(sha)+'\n'+inspect.getsource(inventory)
    source += '\nr=Path('+repr(root)+')\n'
    source += 'proof=inventory(r,'+repr(step)+','+repr(launch)+','+repr(following_steps)+')\n'
    if expected is None:
        source += 'print(json.dumps(proof))\n'
    else:
        source += 'assert proof=='+repr(expected)+", 'checkpoint changed between reads'\n"
        source += "with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:\n"
        source += " for name in sorted(proof['files']):tar.add(r/name,arcname=name,recursive=False)\n"
    return source


def reader(launch):
    root=launch['remote_root']
    if (not re.fullmatch(r'/[A-Za-z0-9_./-]+',root) or '..' in Path(root).parts or
            str(Path(root))!=root or root=='/' or
            not re.fullmatch(r'sha256:[0-9a-f]{64}',launch['image'])):
        raise ValueError('explicit canonical read-only mount and image required')
    # Snapshot files are root-owned by the training container. An isolated
    # CPU stdlib reader needs only this read-only mount, no inherited devices.
    return ['docker','run','--rm','--network=none','--read-only','--cpus=1','--memory=1g',
        '--mount','type=bind,src='+root+',dst='+root+',readonly',
        '--entrypoint','python',launch['image']]


def collect(c, provenance):
    launch = json.loads(Path(c['launch']).read_text())
    if sha(c['launch']) != c['launch_sha256']:
        raise ValueError('original launch identity changed')
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',c['host']) or
            not re.fullmatch(r'[0-9a-f]{64}',launch['container_id'])):
        raise ValueError('explicit host and container identity required')
    root = Path(c['output']).resolve()
    check_paths({'snapshot':root})
    root.mkdir(parents=True,exist_ok=False)
    write(root/'attempt.json',dict(config=c,provenance=provenance,implementation_sha256=sha(__file__)))
    def ssh(argv):
        return ['ssh','-o','ConnectTimeout=10',c['host'],shlex.join(argv)]
    inspected = json.loads(subprocess.check_output(ssh(['docker','inspect',launch['container_id']]),text=True,timeout=55))[0]
    if inspected['Id'] != launch['container_id'] or inspected['Image'] != launch['image']:
        raise ValueError('actual source container differs')
    write(root/'source-inspect.json',inspected)
    read_command=reader(launch)
    following=c.get('following_steps',0)
    proof = json.loads(subprocess.check_output(ssh(read_command+['-c',remote_script(launch['remote_root'],c['step'],launch,following_steps=following)]),text=True,timeout=55))
    write(root/'inventory.json',proof)
    archive = root/'checkpoint.tar'
    with archive.open('xb') as stream:
        subprocess.run(ssh(read_command+['-c',remote_script(launch['remote_root'],c['step'],launch,proof,following_steps=following)]),stdout=stream,check=True,timeout=55)
    output = root/'collection';output.mkdir()
    unpack(archive,output,proof['files'])
    result=dict(schema='context-committed-checkpoint-snapshot-v1',container_id=launch['container_id'],
        image=launch['image'],step=c['step'],policy_id=proof['policy_id'],files=proof['files'],
        archive_sha256=sha(archive),collection=str(output),training_success_certified=False,
        fresh_process_resume_verified=False,scope='Immutable committed LoRA/head snapshot; tensor and recovery audit required')
    if following:
        result.update(schema='context-committed-following-snapshot-v1',
            comparison_stop_update=proof['comparison_stop_update'],following_policy_ids=proof['following_policy_ids'],
            following_optimizer_states_collected=False,
            scope='Committed state plus two completed policy/sample updates; no following Adam states or replay proof')
    write(root/'COMPLETE.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    result=collect(c,provenance)
    print(json.dumps({k:result[k] for k in ('step','container_id','collection','fresh_process_resume_verified')}))
