"""Run the real launcher until prerequisite failure; remote mutations forbidden."""
import json
import runpy
import sys
from pathlib import Path
import pytest
from multimodal_jev.resident_service_admission import IMAGE


@pytest.fixture
def launcher(tmp_path,monkeypatch):
    source=Path(__file__).parents[1]/'tools/jev/launch_simple_resident_train.py'
    target=tmp_path/'launch_resident_train.py';target.write_text(source.read_text())
    monkeypatch.setattr(sys,'argv',[str(target),'--phase','service','--state-dir',str(tmp_path)])
    payload=json.dumps(dict(passed=True))+'\n';calls=[]
    for suffix,filename in [('switch-finalize-v2','verified.json'),('restore-audit','state-verified.json'),('resident-batch-finalize','verified.json')]:
        tag='sampled-npu-grpo_v1-'+suffix
        for name,value in [('launch',dict(container_id=tag,remote_root='/data/'+tag)),
            ('observed',dict(container_id=tag,image=IMAGE,state=dict(Status='exited',ExitCode=0)))]:
            (tmp_path/(tag+'-'+name+'.json')).write_text(json.dumps(value))
        (tmp_path/(tag+'-verified.json')).write_text(payload)
    mode={'occupied':True,'root_exists':False}
    import subprocess
    def remote(argv,**kwargs):
        assert argv[:2]==['ssh','npu']
        import shlex
        command=shlex.split(argv[2]);calls.append(command)
        if command[:2]==['python3','-c']:return str(mode['root_exists'])+'\n'
        if command[0]=='cat':return payload
        if command==['npu-smi','info']:
            return '\n'.join('No running processes found in NPU '+str(i) for i in range(4 if mode['occupied'] else 5))
        raise AssertionError('Unexpected operation before admission: '+str(command))
    monkeypatch.setattr(subprocess,'check_output',remote)
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('No copy/process allowed before admission'))
    return target,mode,calls,tmp_path


def test_busy_fifth_card_never_creates_service_or_training(launcher):
    path,_,calls,root=launcher
    with pytest.raises(ValueError,match='occupied'):runpy.run_path(str(path),run_name='__main__')
    assert calls[-1]==['npu-smi','info']
    assert not list(root.glob('*-created.json')) and not list(root.glob('*-source.tar.gz'))


def test_existing_exact_handle_never_restarts(launcher):
    path,_,calls,root=launcher
    (root/'sampled-npu-grpo_v1-resident-service-created.json').write_text('{}')
    with pytest.raises(RuntimeError,match='exact handle'):runpy.run_path(str(path),run_name='__main__')
    assert calls==[]


def test_existing_remote_root_is_not_overwritten(launcher):
    path,mode,calls,_=launcher;mode['root_exists']=True
    with pytest.raises(RuntimeError,match='no overwrite'):runpy.run_path(str(path),run_name='__main__')
    assert len(calls)==1


def test_exit_zero_without_independent_acceptance_cannot_dispatch(launcher):
    path,_,calls,root=launcher
    p=root/'sampled-npu-grpo_v1-resident-batch-finalize-verified.json';p.write_text(json.dumps(dict(passed=False)))
    with pytest.raises(ValueError,match='prerequisite'):runpy.run_path(str(path),run_name='__main__')
    assert not any(c[0]=='npu-smi' for c in calls)


def test_formal_candidate_refuses_before_service_or_host_mutation(launcher,monkeypatch):
    path,_,calls,root=launcher
    # Copied launcher resolves its repository relative to this temporary path.
    repository=path.resolve().parents[2]
    # Substitute only the candidate read to avoid creating files outside fixture.
    original=Path.read_text
    def read(target,*args,**kwargs):
        if target==repository/'configs/experiments/slurp-simple-sampled-rl-v1.json':
            return json.dumps(dict(schema='slurp-simple-sampled-rl-v1',training_dispatch_allowed=False))
        return original(target,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setattr(sys,'argv',[str(path),'--phase','service','--formal','--state-dir',str(root)])
    with pytest.raises(ValueError,match='not been released'):runpy.run_path(str(path),run_name='__main__')
    assert len(calls)==1 and calls[0][:2]==['python3','-c']


@pytest.mark.parametrize('formal,resuming,resume_fault',[(False,False,'fixture'),(False,False,'fixture_live'),(False,False,'fixture_graph'),(False,False,None),(True,False,None),(True,True,None),
    (True,True,'live'),(True,True,'partial'),(True,True,'changed_graph'),(True,True,'changed_experiment'),
    (True,True,'service_lineage')])
def test_both_launch_phases_bind_service_actor_and_immutable_sources(tmp_path,monkeypatch,formal,resuming,resume_fault):
    """Execute both actual launcher branches with a simulated SSH/Docker host."""
    import hashlib,shlex,tarfile
    fixture_resume=bool(resume_fault and resume_fault.startswith('fixture'))
    import subprocess
    repository=tmp_path/'repo';target=repository/'tools/jev/launch_simple_resident_train.py'
    target.parent.mkdir(parents=True)
    source=Path(__file__).parents[1]/'tools/jev/launch_simple_resident_train.py'
    target.write_text(source.read_text())
    (repository/'src').mkdir();(repository/'src/example.py').write_text('VALUE=1\n')
    local=tmp_path/'state';local.mkdir();remote_files={};containers={};creates=[]
    base='/data/xingchao/multimodal-jev/simple-final-v1';stem='sampled-npu-grpo_v1'
    def put(path,value):path.write_text(json.dumps(value)+'\n')
    payload=json.dumps(dict(passed=True))+'\n'
    for suffix,filename in [('switch-finalize-v2','verified.json'),('restore-audit','state-verified.json'),('resident-batch-finalize','verified.json')]:
        tag=stem+'-'+suffix;root=base+'/'+tag+'-v1'
        put(local/(tag+'-launch.json'),dict(container_id=tag,remote_root=root))
        put(local/(tag+'-observed.json'),dict(container_id=tag,image=IMAGE,state=dict(Status='exited',ExitCode=0)))
        (local/(tag+'-verified.json')).write_text(payload);remote_files[root+'/run/'+filename]=payload.encode()
    policy=dict(model='model',base_id='base',adapter='/data/initial',adapter_id='initial',runtime_profile='fixed',max_model_len=4096)
    remote_files[base+'/'+stem+'-parity-v2/config.json']=json.dumps(policy).encode()
    for name in ['adapter_config.json','adapter_model.safetensors']:remote_files['/data/initial/'+name]=name.encode()
    put(local/'prefix-prepared.json',dict(argv=['docker','create','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro']))
    put(local/(stem+'-continue-config.json'),dict(pins={},physical_devices=[0,1,2,3],global_batch=128))
    if formal:
        from test_simple_stage2_release import fixture as release_fixture
        release,_,experiment=release_fixture(tmp_path/'release-evidence')
        p=repository/'configs/experiments/slurp-simple-sampled-rl-v1.json';p.parent.mkdir(parents=True);put(p,experiment)
        for role,suffix in [('continuous','resident-train-audit'),('restore','resident-restore-audit'),('resume_update','resume-resident-train-audit')]:
            t=stem+'-'+suffix;root=base+'/'+t+'-v1'
            entry=release['stage2_initialization']['lifecycle'][role]
            payload=Path(entry['report']).read_bytes()
            (local/(t+'-verified.json')).write_bytes(payload)
            put(local/(t+'-launch.json'),dict(container_id=t,image=IMAGE,remote_root=root))
            put(local/(t+'-observed.json'),dict(container_id=t,image=IMAGE,state=dict(Status='exited',ExitCode=0)))
            remote_files[root+'/run/state-verified.json']=payload
            training=base+'/'+stem+('-resident-train-v1' if role=='continuous' else '-resident-restore-v1')
            remote_files[root+'/config.json']=json.dumps(dict(training_root=training)).encode()
            remote_files[training+'/config.json']=Path(entry['training_config']).read_bytes()
        final=base+'/stage1-selection-finalize-v1';gate=base+'/stage1-selection-v1'
        put(local/'stage1-selection-finalize-launch.json',dict(container_id='finalizer',image=IMAGE,remote_root=final))
        put(local/'stage1-selection-finalize-observed.json',dict(container_id='finalizer',image=IMAGE,state=dict(Status='exited',ExitCode=0)))
        remote_files[final+'/run/result.json']=json.dumps(dict(passed=True,updates=634,recordings=9226,parity=dict(full_candidate_parity=True))).encode()
        put(local/'stage1-selection-launch.json',dict(remote_root=gate))
        policy['inputs']=base+'/selection-input.json'
        remote_files[policy['inputs']]=b'{}'
        remote_files[gate+'/config.json']=json.dumps(policy).encode()
        remote_files[gate+'/launch.json']=b'{}';remote_files[gate+'/observed-final.json']=b'{}'
        put(local/'continuation-config.json',dict(pins={},physical_devices=[4,5,6,7],global_batch=128))
        stem='stage2-grpo_v1'
        if resuming:
            t='stage2-grpo_v1-prefix-audit';audit_root=base+'/'+t+'-v1';source_root=base+'/stage2-grpo_v1-resident-train-v1'
            prior=dict(pins={},physical_devices=[0,1,2,3],global_batch=128,
                stage2_initialization=dict(algorithm='grpo_v1',mode='final_stage1_fresh_optimizer',
                start_update=0,stop_update=634,total_updates=634,reset_optimizer=True,
                gate=gate,selection_inputs=policy['inputs'],selection_updates=list(range(11)),
                experiment=source_root+'/experiment.json',lifecycle=release['stage2_initialization']['lifecycle']))
            remote_files[prior['stage2_initialization']['experiment']]=json.dumps(experiment).encode()
            prior['pins'][prior['stage2_initialization']['experiment']]=hashlib.sha256(json.dumps(experiment).encode()).hexdigest()
            remote_files[source_root+'/config.json']=json.dumps(prior).encode()
            remote_files[source_root+'/launch.json']=json.dumps(dict(container_id='stopped')).encode()
            containers['stopped']=dict(Id='stopped',Image=IMAGE,State=dict(Status='running' if resume_fault=='live' else 'exited',ExitCode=137))
            receipt=dict(passed=True,stage='simple_stage2_rl',audit_mode='stopped_prefix',complete_training=False,
                updates=64,new_updates=64,ranks=4,policies={str(i):'base:p'+str(i) for i in range(65)},
                checks=[dict(passed=True,updates=i,requests=128,policy_id='base:p'+str(i)) for i in range(1,65)],
                exact_master_export=True,exact_peft_reload=True,exact_pre_update_recovery=True,
                selection=[dict(passed=True,updates=s) for s in (0,64)],
                initial_state_checks=[dict(passed=True)]*4,files={source_root+'/config.json':'hash'})
            if resume_fault=='partial':receipt['checks'].pop()
            if resume_fault=='changed_experiment':remote_files[prior['stage2_initialization']['experiment']]+=b' '
            payload=json.dumps(receipt)+'\n';(local/(t+'-verified.json')).write_text(payload)
            remote_files[audit_root+'/run/state-verified.json']=payload.encode()
            remote_files[audit_root+'/config.json']=json.dumps(dict(training_root=source_root,
                training_observed=audit_root+'/training-observed.json')).encode()
            remote_files[audit_root+'/training-observed.json']=b'{}'
            put(local/(t+'-launch.json'),dict(container_id='audit',image=IMAGE,remote_root=audit_root))
            put(local/(t+'-observed.json'),dict(container_id='audit',image=IMAGE,state=dict(Status='exited',ExitCode=0)))
            stem+='-resume1'
    if fixture_resume:
        source_root=base+'/sampled-npu-grpo_v1-resident-train-v1'
        t='sampled-npu-grpo_v1-resident-train-audit';audit_root=base+'/'+t+'-v1'
        accepted=dict(passed=True,updates=3,new_updates=2,ranks=4,checks=[dict(passed=True)]*2,
            exact_master_export=True,exact_peft_reload=True,exact_pre_update_recovery=True,
            files={source_root+'/config.json':'hash'})
        payload=json.dumps(accepted)+chr(10);(local/(t+'-verified.json')).write_text(payload)
        remote_files[audit_root+'/run/state-verified.json']=payload.encode()
        remote_files[audit_root+'/config.json']=json.dumps(dict(training_root=source_root,
            training_observed=audit_root+'/training-observed.json')).encode()
        remote_files[audit_root+'/training-observed.json']=b'{}'
        put(local/(t+'-launch.json'),dict(container_id='audit',image=IMAGE,remote_root=audit_root))
        put(local/(t+'-observed.json'),dict(container_id='audit',image=IMAGE,state=dict(Status='exited',ExitCode=0)))
        remote_files[source_root+'/launch.json']=json.dumps(dict(container_id='closed-fixture')).encode()
        containers['closed-fixture']=dict(Id='closed-fixture',Image=IMAGE,
            State=dict(Status='running' if resume_fault=='fixture_live' else 'exited',ExitCode=0))
        remote_files[source_root+'/config.json']=json.dumps(dict(pins={},physical_devices=[0,1,2,3],global_batch=128,
            sampled_probe=dict(algorithm='grpo_v1',fixture_initialization='restore_updated_rl_then_resident_loop'))).encode()
        remote_files[source_root+'/run/checkpoint-0003-committed.json']=json.dumps(dict(
            reference_policy_id='base:fixed',cursor=dict(policy_id='base:third'))).encode()
        for name in ['adapter_config.json','adapter_model.safetensors']:
            remote_files[source_root+'/run/loop-adapter-0003/'+name]=name.encode()
        stem='sampled-npu-grpo_v1-resume'
    def remote(argv,**kwargs):
        assert argv[:2]==['ssh','npu'];cmd=shlex.split(argv[2])
        if cmd[:2]==['python3','-c']:
            if cmd[2].startswith('# fixture_resume_evidence_graph'):
                if resume_fault=='fixture_graph':raise subprocess.CalledProcessError(1,argv)
                return json.dumps(dict(passed=True,verified_files=1))
            if cmd[2].startswith('# formal_resume_evidence_graph'):
                if resume_fault=='changed_graph':raise subprocess.CalledProcessError(1,argv)
                return json.dumps(dict(passed=True,verified_files=1))
            return 'False\n'
        if cmd==['npu-smi','info']:return '\n'.join('No running processes found in NPU '+str(i) for i in range(5))
        if cmd[0]=='cat':return remote_files[cmd[1]].decode()
        if cmd[0]=='sha256sum':return hashlib.sha256(remote_files[cmd[1]]).hexdigest()+' '+cmd[1]+'\n'
        if cmd[0] in ('mkdir','tar'):return ''
        if cmd[:2]==['docker','create']:
            cid='container'+str(len(creates));creates.append(cmd)
            mounts=[];devices=[]
            for i,arg in enumerate(cmd):
                if arg=='-v':
                    parts=cmd[i+1].split(':');mounts.append(dict(Source=parts[0],Destination=parts[1],RW=len(parts)<3 or parts[2]!='ro'))
                if arg=='--device':devices.append(dict(PathOnHost=cmd[i+1].split(':')[0]))
            containers[cid]=dict(Id=cid,Image=IMAGE,State=dict(Status='running',Running=True,Restarting=False,OOMKilled=False),Mounts=mounts,HostConfig=dict(Devices=devices))
            return cid+'\n'
        if cmd[:2]==['docker','start']:
            if cmd[2]=='container0':
                root=base+'/'+stem+'-resident-service-v1'
                c=json.loads(remote_files[root+'/config.json'])
                remote_files[root+'/run/ready.json']=json.dumps(dict(socket_path=c['socket_path'],base_id='base',runtime_profile='fixed',physical_devices=[4])).encode()
            return cmd[2]+'\n'
        if cmd[:2]==['docker','inspect']:return json.dumps([containers[cmd[2]]])
        raise AssertionError(cmd)
    def copy(argv,**kwargs):
        assert argv[0]=='scp' and argv[2].startswith('npu:')
        remote_files[argv[2][4:]]=Path(argv[1]).read_bytes()
    monkeypatch.setattr(subprocess,'check_output',remote);monkeypatch.setattr(subprocess,'run',copy)
    # Call from an unrelated cwd; archived source must still come from repository.
    monkeypatch.chdir(local)
    for phase in ('service','trainer'):
        monkeypatch.setattr(sys,'argv',[str(target),'--phase',phase,'--state-dir',str(local)]+(['--formal'] if formal else ['--fixture-resume'] if fixture_resume else [])+
            (['--resume-audit-tag','stage2-grpo_v1-prefix-audit','--run-id','resume1'] if resuming else []))
        if resume_fault=='service_lineage' and phase=='trainer':
            path=local/(stem+'-resident-service-launch.json');value=json.loads(path.read_text())
            value['resume_update']=63;put(path,value)
            with pytest.raises(ValueError,match='same recovery audit'):
                runpy.run_path(str(target),run_name='__main__')
            assert len(creates)==1
            return
        if resume_fault and resume_fault not in ('service_lineage','fixture'):
            with pytest.raises((ValueError,subprocess.CalledProcessError)):
                runpy.run_path(str(target),run_name='__main__')
            assert not creates and not any(path.startswith(base+'/'+stem+'-resident-') for path in remote_files)
            return
        runpy.run_path(str(target),run_name='__main__')
    assert len(creates)==2
    service=containers['container0'];trainer=containers['container1']
    assert service['HostConfig']['Devices'][0]['PathOnHost']=='/dev/davinci4'
    assert [d['PathOnHost'] for d in trainer['HostConfig']['Devices'][:4]]==['/dev/davinci'+str(i) for i in range(4)]
    for process in (service,trainer):
        mounts={m['Destination']:m for m in process['Mounts']}
        assert mounts[base]['RW'] is False and mounts['/mnt/cephfs/xingchao']['RW'] is False
        assert mounts['/jev-ipc']['Source']==base+'/'+stem+'-resident-service-v1/ipc'
    c=json.loads(remote_files[base+'/'+stem+'-resident-train-v1/config.json'])
    plan=c['stage2_initialization' if formal else 'sampled_probe']
    assert plan['stop_update']==(634 if formal else 4 if fixture_resume else 3) and plan['rollout_physical_devices']==[4]
    for key in ['service_launch','service_inspect','service_ready','accepted_switch','accepted_restore','accepted_resident_batch']:
        path=plan[key];assert c['pins'][path]==hashlib.sha256(remote_files[path]).hexdigest()
    if fixture_resume:
        assert c['continuous_resume']['completed_updates']==3
        assert c['continuous_resume']['training_root']==source_root
        assert plan['fixture_initialization']=='restore_continuous_then_one_update'
        runner=remote_files[base+'/'+stem+'-resident-train-v1/run.py'].decode()
        assert 'probe_simple_continuous_resume_update.py' in runner
    if formal:
        assert 'sampled_probe' not in c and c['physical_devices']==[0,1,2,3]
        assert plan['start_update']==0 and plan['reset_optimizer'] is True and len(plan['selection_updates'])==11
        assert plan['gate']==base+'/stage1-selection-v1'
        if not resuming:
            for entry in plan['lifecycle'].values():
                for path in entry.values():assert c['pins'][path]==hashlib.sha256(remote_files[path]).hexdigest()
        else:
            assert plan['experiment']==prior['stage2_initialization']['experiment']
            assert plan['lifecycle']==prior['stage2_initialization']['lifecycle']
            assert c['continuous_resume']['completed_updates']==64
            assert c['continuous_resume']['training_root']==source_root
        assert c['pins'][plan['experiment']]==hashlib.sha256(remote_files[plan['experiment']]).hexdigest()
        runner=remote_files[base+'/'+stem+'-resident-train-v1/run.py'].decode()
        assert 'train_simple_stage2_npu.py' in runner and 'simple_stage2_release import authenticate' in runner
    for role in ('service','train'):
        with tarfile.open(local/(stem+'-resident-'+role+'-source.tar.gz')) as archive:
            assert 'src/example.py' in archive.getnames()
            assert 'tools/jev/launch_simple_resident_train.py' in archive.getnames()
        launch=json.loads((local/(stem+'-resident-'+role+'-launch.json')).read_text())
        assert launch['next_regular_observation_not_before']-launch['created_unix']>=600
        if resuming:
            assert launch['resume_audit_tag']=='stage2-grpo_v1-prefix-audit' and launch['resume_update']==64
