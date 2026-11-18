import copy
import json
from pathlib import Path

import pytest
import torch

from tools.reports import prepare_context_cached_recovery as module


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))


def fixture(tmp_path,monkeypatch):
    from tools.reports import context_release_deployment
    monkeypatch.setattr(context_release_deployment,'audit',lambda *a:{'fixture':'external release mount admission'})
    from test_prepare_context_cached_training import setup
    from test_audit_context_replica_training import candidate
    from test_context_stage_lora import config as lora_config,contract as lora_contract
    from multimodal_jev.context_stage_lora import transition
    from tools.reports import prepare_context_cached_training as preparation
    from tools.reports.context_replica_training_operations import IMAGE
    spec,_,_,_=setup(tmp_path,monkeypatch)
    source=Path(spec['parent']['output']);original=module.read(source/'config.json')
    c=candidate(2);c.update(original)
    c['macro_client']['runtime_identity']=copy.deepcopy(c['dual_runtime_identity'])
    c['batch_config']['inference_batch_queries']=32
    c['pins']={'/shared/model':'b'*64}
    lora=lora_config();c['source_contract']=lora_contract(lora)
    c['stage_lora_transition']=transition(lora,lora,reuse_stage1_adapter=True)
    c.update(physical_devices=[4,5],source_root='/data/parent/source',
        encoded_audio_cache='/data/inputs/cache',audio_mirror_root='/data/audio/run',
        initial_adapter='/data/parent/initial/adapter',reference_adapter='/data/parent/initial/adapter')
    # Full model/reference production is an explicit fixture. The package,
    # evidence bytes, rebinding, pins and independent reconstruction are real.
    monkeypatch.setattr(preparation,'reference_binding',lambda *a,**kw:dict(
        root='/data/producer/dataset',manifest_sha256=module.file_hash(
            Path(spec['reference_job']['collection'])/'reference-build/dataset/MANIFEST.json'),
        identity={'policy_id':c['source_policy_id']},population={}))
    monkeypatch.setattr(preparation,'ReferenceDataset',lambda *a,**kw:None)
    write(source/'config.json',c)
    m=module.read(source/'PREPARED.json');m['image']=IMAGE;write(source/'PREPARED.json',m)
    for n in ('a','b'):write(source/f'service-{n}.json',dict(policy={},pins={}))
    storage=source/'source/src/multimodal_jev/local_runtime_storage.py';storage.parent.mkdir(parents=True)
    storage.write_text('from pathlib import Path\ndef check_paths(value):pass\n')
    write(source/'RUNTIME-SOURCES.json',{str(p.relative_to(source/'source')):module.file_hash(p)
        for p in (source/'source').rglob('*.py')})
    spec['manifest_sha256']=preparation.prepare(spec,{})['manifest_sha256']
    parent=Path(spec['output']);m=module.read(parent/'PREPARED.json');c=module.read(parent/'config.json')
    run=tmp_path/'parent-result/run';run.mkdir(parents=True)
    for rank in range(2):
        p=run/f'rank-{rank}/checkpoint-0001.pt';p.parent.mkdir()
        torch.save({'fixture':'trainable shard'},p)
    write(run/'policy-0001/MANIFEST.json',{'policy_id':'behavior1'})
    torch.save({'lora':torch.ones(2)},run/'policy-0001/execution.pt')
    write(run/'adapter-0001/adapter_config.json',{'r':4})
    (run/'adapter-0001/adapter_model.safetensors').write_bytes(b'fixture adapter')
    commit=dict(updates=1,contract=module.expected_contract(c,module.mounted_names(m)),
        rank_shards={f'rank-{r}/checkpoint-0001.pt':module.file_hash(run/f'rank-{r}/checkpoint-0001.pt') for r in range(2)},
        joint_policy_directory='policy-0001',joint_policy_manifest_sha256=module.file_hash(run/'policy-0001/MANIFEST.json'),
        joint_policy={'policy_id':'behavior1'})
    write(run/'checkpoint-0001-committed.json',commit)
    transport=tmp_path/'transport.json';write(transport,{'files':{'run/'+str(p.relative_to(run)):module.file_hash(p)
        for p in run.rglob('*') if p.is_file()}})
    proof=dict(status='finite_cached_reference_training_outputs_audited',world_size=2,
        manifest_sha256=spec['manifest_sha256'],container_id='a'*64)
    monkeypatch.setattr(module,'audit_parent',lambda *a:copy.deepcopy(proof))
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    config=dict(parent_audit=dict(training_package=spec,collection=str(run.parent),transport=str(transport)),
        output=str(tmp_path/'recovery'),remote_root='/data/recovery')
    return config,run


def test_recovery_keeps_reference_and_runtime_but_restores_behavior(tmp_path,monkeypatch):
    c,run=fixture(tmp_path,monkeypatch);parent=Path(c['parent_audit']['training_package']['output'])
    before={str(p):module.file_hash(p) for root in (parent,run) for p in root.rglob('*') if p.is_file()}
    result=module.prepare(c,{});c['manifest_sha256']=result['manifest_sha256']
    report=module.audit(c,{})
    assert report['actor_updates']==1 and report['initial_behavior_policy_id']=='behavior1'
    root=Path(c['output']);current=module.read(root/'config.json');original=module.read(parent/'config.json')
    assert current['source_policy_id']==original['source_policy_id']!='behavior1'
    assert current['reference_dataset']==original['reference_dataset']
    assert current['reference_adapter']=='/data/recovery/initial/adapter'
    assert current['initial_adapter']=='/data/recovery/resume/adapter-0001'
    assert current['start_update']==1 and current['stop_update']==2
    assert (root/'RUNTIME-SOURCES.json').read_bytes()==(parent/'RUNTIME-SOURCES.json').read_bytes()
    assert before=={str(p):module.file_hash(p) for base in (parent,run) for p in base.rglob('*') if p.is_file()}
    from tools.reports import context_cached_recovery_operations as operations
    ops=dict(training_package=c,package=c['output'],manifest_sha256=c['manifest_sha256'],
        host='host',tag='recovery-test',shared_root='/shared',driver_root='/driver',
        operations_root=str(tmp_path/'ops'),lease=str(tmp_path/'lease'))
    m,config,_=operations.validate(ops)
    argv=operations.shared.command(ops,m,config)
    assert '/data/recovery/resume:/data/recovery/resume:ro' in argv
    assert '/data/producer/dataset:/data/producer/dataset:ro' in argv
    with pytest.raises(ValueError,match='explicit'):operations.launch(ops,{})


@pytest.mark.parametrize('fault',['reference','behavior','runtime','shard','extra','cursor'])
def test_rehashed_recovery_tampering_rejected(tmp_path,monkeypatch,fault):
    c,_=fixture(tmp_path,monkeypatch);module.prepare(c,{})
    root=Path(c['output'])
    if fault in ('reference','cursor'):
        v=module.read(root/'config.json')
        if fault=='reference':v['source_policy_id']='behavior1'
        else:v['start_update']=0
        write(root/'config.json',v)
    elif fault=='behavior':
        v=module.read(root/'service-a.json');v['policy']['policy_id']='reference';write(root/'service-a.json',v)
    elif fault=='runtime':(root/'source/test.py').write_text('changed')
    elif fault=='shard':(root/'resume/rank-0/checkpoint-0001.pt').write_bytes(b'changed')
    else:(root/'resume/base-model.bin').write_bytes(b'unwanted')
    m=module.read(root/'PREPARED.json');m['files']={str(p.relative_to(root)):module.file_hash(p)
        for p in root.rglob('*') if p.is_file() and p.name!='PREPARED.json'}
    write(root/'PREPARED.json',m);c['manifest_sha256']=module.file_hash(root/'PREPARED.json')
    with pytest.raises(ValueError):module.audit(c,{})


def test_source_step1_corruption_rejected_before_package(tmp_path,monkeypatch):
    c,run=fixture(tmp_path,monkeypatch);(run/'rank-1/checkpoint-0001.pt').write_bytes(b'bad')
    with pytest.raises(ValueError):module.prepare(c,{})
    assert not Path(c['output']).exists()


def test_replay_auditor_keeps_reference_separate_from_behavior():
    from tools.reports import audit_context_cached_recovery as audit
    namespace=dict(audit.cached.shared.audit_outputs.__globals__)
    prior={'policy_id':'step1','tensors':{}}
    auditor=audit.replay_auditor(namespace,prior)
    assert auditor.__globals__['replay_prior'] is prior
    assert auditor.__globals__['expected_contract'] is audit.cached.shared.expected_contract


@pytest.mark.parametrize('exit_code',[0,9])
def test_actual_shared_launch_path_preserves_resume_mount_and_exact_wait(tmp_path,monkeypatch,exit_code):
    import shlex
    import subprocess
    from tools.reports import context_cached_recovery_operations as operations
    c,_=fixture(tmp_path,monkeypatch);c['manifest_sha256']=module.prepare(c,{})['manifest_sha256']
    lease=tmp_path/'lease';lease.touch()
    ops=dict(training_package=c,package=c['output'],manifest_sha256=c['manifest_sha256'],
        host='host',tag='recovery-test',shared_root='/shared',driver_root='/driver',
        operations_root=str(tmp_path/'ops'),lease=str(lease),finite_training_dispatch_allowed=True)
    shared=operations.shared.lifecycle.shared;cid='c'*64;calls=[];waits=[];collected=[]
    monkeypatch.setattr(shared,'check_paths',lambda _:None)
    def remote(argv,**kwargs):
        if argv[0]=='scp':return subprocess.CompletedProcess(argv,0,'','')
        args=shlex.split(argv[-1]);calls.append(args)
        if args[:3]==['docker','image','inspect']:out=json.dumps([{'Id':shared.IMAGE}])
        elif args[:2]==['npu-smi','info']:out='\n'.join(f'No running processes found in NPU {d}' for d in (4,5,6,7))
        elif args[:2]==['docker','create']:out=cid
        elif args[:2]==['docker','inspect']:out=json.dumps([{'Id':cid,'State':{'ExitCode':exit_code}}])
        else:out=''
        return subprocess.CompletedProcess(argv,0,out,'')
    monkeypatch.setattr(shared.subprocess,'run',remote)
    monkeypatch.setattr(shared,'wait_exact',lambda remote,handle:(waits.append(handle) or exit_code))
    monkeypatch.setattr(shared,'collect',lambda handle,*a,**kw:(collected.append(handle) or {}))
    import tools.reports.context_dual_probe_operations as runtime
    def check(*args):
        if exit_code:raise ValueError('failed replay collected')
    monkeypatch.setattr(runtime,'check_runtime',check)
    if exit_code:
        with pytest.raises(ValueError,match='failed replay'):operations.launch(ops,{})
    else:operations.launch(ops,{})
    assert waits==[cid] and [r['container_id'] for r in collected]==[cid]
    create=[v for v in calls if v[:2]==['docker','create']]
    assert len(create)==1 and '/data/recovery/resume:/data/recovery/resume:ro' in create[0]


def replay_fixture(tmp_path):
    from test_context_replay_audit import state
    roots=[tmp_path/'continuous',tmp_path/'replay']
    for root in roots:
        write(root/'batch-0001.json',{'policy_id':'behavior1','records':['audio']})
        write(root/'checkpoint-0001-committed.json',{'joint_policy':{'policy_id':'behavior1'}})
        write(root/'checkpoint-0002-committed.json',{'updates':2,'contract':{'reference':'stage1'}})
        write(root/'rollout-0001.json',dict(queries=[{'reference_record':{'prob':[.3]},
            'behavior_record':{'prob':[.4]}}],behavior_policy_id='behavior1',
            fixed_reference_policy_id='stage1',runtime_identity={'fixed':True}))
        for rank in range(2):
            write(root/f'rank-{rank}/restored.json',dict(exact_masters=True,exact_adam=True,completed_updates=1))
            torch.save(state(),root/f'rank-{rank}/checkpoint-0002.pt')
        torch.save({'w':torch.ones(2,dtype=torch.bfloat16)},root/'execution-0002.pt')
        (root/'policy-0002').mkdir()
        torch.save({'w':torch.ones(2,dtype=torch.bfloat16)},root/'policy-0002/execution.pt')
        torch.save([{'samples':torch.tensor([0,1])}],root/'groups-0001.pt')
    return roots


@pytest.mark.parametrize('fault',[None,'adam','samples','reference','behavior','restore'])
def test_exact_dp2_replay_comparison(tmp_path,fault):
    from tools.reports.audit_context_cached_recovery import compare
    left,right=replay_fixture(tmp_path)
    if fault=='adam':
        p=right/'rank-1/checkpoint-0002.pt';v=torch.load(p,weights_only=True)
        v['optimizer']['state'][0]['exp_avg'][0]+=.1;torch.save(v,p)
    elif fault=='samples':torch.save([{'samples':torch.tensor([1,1])}],right/'groups-0001.pt')
    elif fault=='restore':write(right/'rank-0/restored.json',{})
    elif fault:
        p=right/'rollout-0001.json';v=module.read(p)
        if fault=='reference':v['fixed_reference_policy_id']='behavior1'
        else:v['behavior_policy_id']='stage1'
        write(p,v)
    if fault:
        with pytest.raises(ValueError):compare(left,right)
    else:
        result=compare(left,right)
        assert result['rank_shards']==2 and result['exact_sampled_groups']
