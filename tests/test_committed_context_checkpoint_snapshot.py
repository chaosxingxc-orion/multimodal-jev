import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.reports.collect_committed_context_checkpoint import inventory,remote_script,sha,unpack,reader


def fixture(root):
    def put(name,value):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return sha(p)
    launch={key:put(name,{}) for name,key in [('config.json','config_sha256'),
        ('service-config.json','service_config_sha256'),('PREPARED.json','manifest_sha256')]}
    shards={f'rank-{r}/checkpoint-0064.pt':put(f'run/rank-{r}/checkpoint-0064.pt',{'rank':r}) for r in range(4)}
    execution=put('run/execution-0064.pt',{})
    files={'execution.pt':put('run/policy-0064/execution.pt',{})}
    policy=put('run/policy-0064/MANIFEST.json',dict(policy_id='policy',files=files))
    files={name:put('run/adapter-0064/'+name,{}) for name in ('adapter_config.json','adapter_model.safetensors')}
    put('run/adapter-0064/CONTEXT_POLICY.json',dict(policy_id='policy',files=files))
    put('run/checkpoint-0064-committed.json',dict(updates=64,rank_shards=shards,execution_file='execution-0064.pt',
        execution_sha256=execution,joint_policy_directory='policy-0064',joint_policy=dict(policy_id='policy'),
        joint_policy_manifest_sha256=policy))
    put('run/mutable-progress.json',{'update':97});put('reference-cache/large.json',{})
    return launch


def test_remote_reader_and_archive_transport_exclude_mutable_state(tmp_path):
    root=tmp_path/'remote';root.mkdir();launch=fixture(root)
    proof=json.loads(subprocess.check_output([sys.executable,'-c',remote_script(str(root),64,launch)],text=True))
    assert proof==inventory(root,64,launch)
    assert 'run/mutable-progress.json' not in proof['files']
    assert not any(n.startswith('reference-cache/') for n in proof['files'])
    archive=tmp_path/'checkpoint.tar'
    with archive.open('wb') as stream:
        subprocess.run([sys.executable,'-c',remote_script(str(root),64,launch,proof)],stdout=stream,check=True)
    unpack(archive,tmp_path/'out',proof['files'])
    assert sha(tmp_path/'out/config.json')==launch['config_sha256']


def test_reader_mounts_only_snapshot_readonly_without_devices():
    argv=reader(dict(remote_root='/data/run',image='sha256:'+'a'*64))
    assert '--device' not in argv and '--volumes-from' not in argv
    assert 'type=bind,src=/data/run,dst=/data/run,readonly' in argv
    assert '--read-only' in argv and '--network=none' in argv


@pytest.mark.parametrize('fault',['shard','cursor','config','symlink','traversal','adapter'])
def test_inconsistent_commit_rejected(tmp_path,fault):
    launch=fixture(tmp_path);p=tmp_path/'run/checkpoint-0064-committed.json';c=json.loads(p.read_text())
    if fault=='shard':(tmp_path/'run/rank-0/checkpoint-0064.pt').write_text('changed')
    if fault=='cursor':c['updates']=65
    if fault=='config':launch['config_sha256']='wrong'
    if fault=='symlink':
        p0=tmp_path/'run/rank-0/checkpoint-0064.pt';p0.rename(tmp_path/'other');p0.symlink_to(tmp_path/'other')
    if fault=='traversal':c['rank_shards']['../outside']='x'
    if fault=='adapter':(tmp_path/'run/adapter-0064/adapter_model.safetensors').write_text('changed')
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError):inventory(tmp_path,64,launch)


@pytest.mark.parametrize('fault',[None,'rank','behavior','activation','export'])
def test_completed_following_updates_are_bound(tmp_path,fault):
    launch=fixture(tmp_path);current='policy'
    def put(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value));return sha(p)
    for step in (65,66):
        policy=f'policy{step}'
        execution=put(f'run/policy-{step:04d}/execution.pt',{})
        manifest=put(f'run/policy-{step:04d}/MANIFEST.json',dict(policy_id=policy,files={'execution.pt':execution}))
        put(f'run/policy-export-{step:04d}.json',dict(update=step,policy_id=policy,policy_manifest_sha256=manifest))
        put(f'run/adapter-{step:04d}/CONTEXT_POLICY.json',dict(policy_id=policy,files={}))
        put(f'run/activation-{step:04d}.json',dict(error=None,policy_id=policy))
        receipt=dict(update=step-1,behavior_policy_id=current)
        put(f'run/rollout-{step-1:04d}.json',receipt)
        for name in (f'run/batch-{step-1:04d}.json',f'run/groups-{step-1:04d}.pt'):put(name,{})
        for rank in range(4):
            put(f'run/rank-{rank}/update-{step:04d}.json',dict(update=step,next_policy_id=policy,rollout=receipt))
        current=policy
    if fault=='rank':(tmp_path/'run/rank-3/update-0066.json').unlink()
    if fault=='behavior':put('run/rollout-0064.json',dict(update=64,behavior_policy_id='wrong'))
    if fault=='activation':put('run/activation-0065.json',dict(error='failed',policy_id='policy65'))
    if fault=='export':put('run/policy-0065/execution.pt',{'changed':True})
    if fault:
        with pytest.raises(ValueError):inventory(tmp_path,64,launch,2)
    else:
        result=inventory(tmp_path,64,launch,2)
        actual=json.loads(subprocess.check_output([sys.executable,'-c',remote_script(str(tmp_path),64,launch,following_steps=2)],text=True))
        assert result==actual and result['following_policy_ids']==['policy65','policy66']
        assert not result['following_optimizer_states_collected']
