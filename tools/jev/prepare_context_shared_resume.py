#!/usr/bin/env python3
"""Bind fresh-process replay to independently collected shared update one.

Prepare only. The original Stage1 reference remains fixed while the behavior
policy and FP32 optimizer state resume from the first committed RL update.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(*, package, collection, proof_path, launch_path, output, remote_root):
    package,collection,output=map(Path,(package,collection,output))
    proof=json.loads(Path(proof_path).read_text())
    launch=json.loads(Path(launch_path).read_text())
    if (proof.get('passed') is not True or proof['state']['Status']!='exited'
            or proof['state']['ExitCode']!=0 or proof['state']['OOMKilled']
            or proof['actor_container']!=launch['container_id']
            or proof['service_container']!=launch['container_id']
            or proof['image']!=launch['image']):
        raise ValueError('successful independently collected shared prefix required')
    for name,digest in proof['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('relative collected path required')
        if sha(collection/relative)!=digest:raise ValueError('collected artifact changed: '+name)
    def bound(name):
        if name not in proof['files']:raise ValueError('artifact absent from collection proof: '+name)
        return collection/name
    if sha(bound('PREPARED.json'))!=launch['manifest_sha256']:
        raise ValueError('executed package identity differs')
    manifest=json.loads(bound('PREPARED.json').read_text())
    if sha(package/'PREPARED.json')!=launch['manifest_sha256']:
        raise ValueError('local source snapshot differs')
    for name,digest in manifest['files'].items():
        if sha(package/name)!=digest:raise ValueError('candidate snapshot changed: '+name)
    config=json.loads(bound('config.json').read_text())
    service=json.loads(bound('service-config.json').read_text())
    if (config['start_update']!=0 or config['stop_update']!=2 or not config['engineering_prefix']
            or config['training_dispatch_allowed'] or config['algorithm']!='grpo_v1'):
        raise ValueError('original bounded GRPO prefix required')
    previous=launch['remote_root']
    if (not remote_root.startswith('/data/xingchao/multimodal-jev/simple-final-v1/')
            or remote_root==previous):
        raise ValueError('distinct local-storage replay root required')
    commit=json.loads(bound('run/checkpoint-0001-committed.json').read_text())
    if commit['updates']!=1 or len(commit['rank_shards'])!=4:
        raise ValueError('complete first-update commit required')
    policy_dir=commit['joint_policy_directory']
    if policy_dir!='policy-0001':raise ValueError('exact step-one policy directory required')
    policy=json.loads(bound('run/'+policy_dir+'/MANIFEST.json').read_text())
    if (sha(bound('run/'+policy_dir+'/MANIFEST.json'))!=commit['joint_policy_manifest_sha256']
            or policy['policy_id']!=commit['joint_policy']['policy_id']
            or policy['contract']!=config['source_contract']):
        raise ValueError('committed joint-policy identity differs')
    required=['checkpoint-0001-committed.json',*commit['rank_shards']]
    for name,digest in commit['rank_shards'].items():
        if sha(bound('run/'+name))!=digest:raise ValueError('rank shard changed')
    for directory in (policy_dir,'adapter-0001'):
        assets=[p for p in (collection/'run'/directory).rglob('*') if p.is_file()]
        if not assets:raise ValueError('missing policy/adapter assets')
        for asset in assets:
            if asset.is_symlink():raise ValueError('policy symlink forbidden')
            name=str(asset.relative_to(collection/'run'))
            bound('run/'+name);required.append(name)
    # Keep fixed reference/source paths. Relocate only paths inside the old run.
    def relocate(value):
        if isinstance(value,str):
            return remote_root+value[len(previous):] if value==previous or value.startswith(previous+'/') else value
        if isinstance(value,list):return [relocate(v) for v in value]
        if isinstance(value,dict):return {relocate(k):relocate(v) for k,v in value.items()}
        return value
    resumed=relocate(config);service=relocate(service)
    resumed.update(start_update=1,resume_root=previous+'/run',
        initial_adapter=previous+'/run/adapter-0001',
        diagnostic_scope='Fresh process replay of shared-container update2; not research Stage2')
    for name in required:
        resumed['pins'][previous+'/run/'+name]=sha(bound('run/'+name))
    service['pins']=resumed['pins']
    service['policy'].update(snapshot=previous+'/run/'+policy_dir,policy_id=policy['policy_id'],
                            adapter=resumed['initial_adapter'])
    if any(resumed[k]!=config[k] for k in ('source_policy','source_policy_id','reference_adapter','source_contract')):
        raise ValueError('fixed reference changed')
    runner=(package/'run.py').read_text()
    old="if c['start_update']!=0 or c['stop_update']!=2:"
    if runner.count(old)!=1:raise ValueError('unexpected runner cursor guard')
    runner=runner.replace(old,"if c['start_update']!=1 or c['stop_update']!=2:")
    runner=runner.replace('initial bounded candidate only; resume requires separate binding',
                          'bound first-update replay only')
    output.mkdir(exist_ok=False)
    shutil.copytree(package/'source',output/'source')
    for name,value in [('config.json',resumed),('service-config.json',service)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    (output/'run.py').write_text(runner)
    result=dict(status='prepared_only_no_dispatch',remote_root=remote_root,
        source_container=launch['container_id'],source_remote_root=previous,
        source_proof_sha256=sha(proof_path),source_manifest_sha256=sha(package/'PREPARED.json'),
        fixed_reference_policy_id=config['source_policy_id'],initial_behavior_policy_id=policy['policy_id'],
        required_readonly_mount=previous+':'+previous+':ro',
        files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    (output/'PREPARED.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('package','collection','proof-path','launch-path','output','remote-root'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(**vars(args)),indent=2))
