import json
from pathlib import Path
import pytest
from tools.jev.prepare_context_shared_resume import prepare,sha


def setup(tmp):
    package=tmp/'package';col=tmp/'collection';package.mkdir();col.mkdir()
    remote='/data/xingchao/multimodal-jev/simple-final-v1/continuous'
    def write(root,name,value):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value));return p
    contract={'runtime':'test'}
    c=dict(start_update=0,stop_update=2,engineering_prefix=True,training_dispatch_allowed=False,
           algorithm='grpo_v1',source_policy='/fixed/policy',source_policy_id='reference',
           reference_adapter='/fixed/adapter',source_contract=contract,
           initial_adapter='/fixed/adapter',reference_cache=remote+'/reference-cache',
           schedule=remote+'/source/schedule',pins={remote+'/source/module.py':'unused'})
    write(package,'config.json',c)
    write(package,'service-config.json',dict(policy=dict(snapshot='/fixed/policy',policy_id='reference'),pins=c['pins']))
    (package/'run.py').write_text("if c['start_update']!=0 or c['stop_update']!=2:\n raise ValueError('initial bounded candidate only; resume requires separate binding')\n")
    write(package,'source/module.py',{'example':'fixture only'})
    manifest=dict(files={str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file()})
    write(package,'PREPARED.json',manifest)
    for name in ['config.json','service-config.json','PREPARED.json']:
        (col/name).write_bytes((package/name).read_bytes())
    policy=write(col,'run/policy-0001/MANIFEST.json',dict(policy_id='updated',contract=contract))
    write(col,'run/policy-0001/execution.pt',{'fixture':'weights'})
    write(col,'run/adapter-0001/adapter_model.safetensors',{'fixture':'adapter'})
    shards={}
    for i in range(4):
        name=f'rank-{i}/checkpoint-0001.pt';shards[name]=sha(write(col,'run/'+name,{'rank':i}))
    write(col,'run/checkpoint-0001-committed.json',dict(updates=1,rank_shards=shards,
        joint_policy_directory='policy-0001',joint_policy={'policy_id':'updated'},joint_policy_manifest_sha256=sha(policy)))
    launch=write(tmp,'launch.json',dict(container_id='continuous',image='fixed',remote_root=remote,
                                      manifest_sha256=sha(package/'PREPARED.json')))
    proof=write(tmp,'proof.json',dict(passed=True,state=dict(Status='exited',ExitCode=0,OOMKilled=False),
        actor_container='continuous',service_container='continuous',image='fixed',
        files={str(p.relative_to(col)):sha(p) for p in col.rglob('*') if p.is_file()}))
    return dict(package=package,collection=col,proof_path=proof,launch_path=launch,output=tmp/'resume',
                remote_root='/data/xingchao/multimodal-jev/simple-final-v1/replay')


def test_resume_changes_behavior_not_reference(tmp_path):
    args=setup(tmp_path);result=prepare(**args)
    c=json.loads((args['output']/'config.json').read_text())
    s=json.loads((args['output']/'service-config.json').read_text())
    assert c['start_update']==1 and c['stop_update']==2
    assert c['source_policy_id']=='reference' and c['reference_adapter']=='/fixed/adapter'
    assert s['policy']['policy_id']=='updated'
    assert c['reference_cache']==args['remote_root']+'/reference-cache'
    assert c['resume_root'].endswith('/continuous/run')
    assert c['pins']==s['pins'] and len(c['pins'])==9
    assert '!=1' in (args['output']/'run.py').read_text()
    assert result['fixed_reference_policy_id']!=result['initial_behavior_policy_id']


@pytest.mark.parametrize('fault',['failed','container','missing_binding','mutated_shard','snapshot','same_root'])
def test_reject_before_creating_replay(tmp_path,fault):
    args=setup(tmp_path);proof=json.loads(args['proof_path'].read_text())
    if fault=='failed':proof['passed']=False
    if fault=='container':proof['actor_container']='wrong'
    if fault=='missing_binding':del proof['files']['run/checkpoint-0001-committed.json']
    if fault=='mutated_shard':(args['collection']/'run/rank-0/checkpoint-0001.pt').write_text('changed')
    if fault=='snapshot':(args['package']/'source/module.py').write_text('changed')
    if fault=='same_root':args['remote_root']=json.loads(args['launch_path'].read_text())['remote_root']
    args['proof_path'].write_text(json.dumps(proof))
    with pytest.raises(ValueError):prepare(**args)
    assert not args['output'].exists()
