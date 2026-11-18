"""Source admission uses real shards/export and authenticated audit file graphs."""
import json
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.baseline import file_hash
from multimodal_jev.simple_continuous_resume import authenticate
from multimodal_jev.resident_service_admission import IMAGE
from multimodal_jev.slurp_instruction_resume import FIELDS
from test_simple_resident_train_audit import fixture as source_fixture


def fixture(tmp_path,monkeypatch):
    module,_,root,put=source_fixture(tmp_path,monkeypatch)
    train=json.loads((root/'config.json').read_text())
    for key in FIELDS:train.setdefault(key,'fixed')
    train.update(microbatch_per_rank=1,task_mode='direct_final',loss_reduction='real_request_mean',physical_devices=[0,1,2,3])
    put(root/'config.json',train)
    put(root/'launch.json',dict(container_id='training',image=IMAGE,config_sha256=file_hash(root/'config.json')))
    put(root/'terminal.json',dict(exit_code=0))
    observed=tmp_path/'source-observed.json'
    put(observed,dict(container_id='training',image=IMAGE,state=dict(Status='exited',ExitCode=0)))
    # The independent model/audio audit itself has separate tests. Build its
    # authenticated file graph, including all real tensor and model assets.
    files={str(p):file_hash(p) for p in root.rglob('*') if p.is_file()}
    files[str(observed)]=file_hash(observed)
    audit_root=tmp_path/'audit';audit=audit_root/'run/state-verified.json'
    put(audit,dict(passed=True,exact_master_export=True,exact_peft_reload=True,ranks=4,
        exact_pre_update_recovery=True,updates=3,files=files))
    audit_launch=tmp_path/'audit-launch.json';audit_observed=tmp_path/'audit-observed.json'
    put(audit_launch,dict(container_id='audit',image=IMAGE,remote_root=str(audit_root)))
    put(audit_observed,dict(container_id='audit',image=IMAGE,state=dict(Status='exited',ExitCode=0)))
    c=dict(train,continuous_resume=dict(training_root=str(root),completed_updates=3,algorithm='grpo_v1',
        reference_policy_id='base:reference',audit=str(audit),audit_launch=str(audit_launch),
        audit_observed=str(audit_observed),training_observed=str(observed)),
        pins={str(p):file_hash(p) for p in [audit,audit_launch,audit_observed,observed]})
    return c,root,put


def test_four_shards_authenticated_before_any_model_loading(tmp_path,monkeypatch):
    c,root,_=fixture(tmp_path,monkeypatch);r=authenticate(c)
    assert r['completed_updates']==3 and len(r['states'])==4
    assert r['reference_policy_id']=='base:reference' and r['adapter']==str(root/'run/loop-adapter-0003')


@pytest.mark.parametrize('fault',['audit_pin','unaccepted','audit_running','wrong_container','wrong_image',
    'missing_asset','missing_shard','missing_source','corrupt_shard','reference','algorithm','topology',
    'schedule','lr','cursor','bool_cursor','commit','source_running','missing_inputs','missing_rollout'])
def test_reject_unaccepted_or_changed_resume_sources(tmp_path,monkeypatch,fault):
    c,root,put=fixture(tmp_path,monkeypatch);plan=c['continuous_resume']
    if fault in ('reference','algorithm'):
        plan['reference_policy_id' if fault=='reference' else 'algorithm']='wrong'
    elif fault=='topology':c['physical_devices']=[4,5,6,7]
    elif fault=='schedule':c['schedule_sha256']='other'
    elif fault=='lr':c['lr']=.1
    elif fault in ('cursor','bool_cursor'):plan['completed_updates']=2 if fault=='cursor' else True
    elif fault=='corrupt_shard':
        with (root/'run/loop-rank-1/checkpoint-0003.pt').open('ab') as f:f.write(b'changed')
    elif fault=='commit':
        p=root/'run/checkpoint-0003-committed.json';v=json.loads(p.read_text());v['passed']=False;put(p,v)
    else:
        key='audit_observed' if fault in ('audit_running','wrong_container','wrong_image') else 'training_observed' if fault=='source_running' else 'audit'
        p=Path(plan[key]);v=json.loads(p.read_text())
        if fault=='unaccepted':v['exact_peft_reload']=False
        elif fault.endswith('running'):v['state']['Status']='running'
        elif fault=='wrong_container':v['container_id']='other'
        elif fault=='wrong_image':v['image']='other'
        elif fault=='missing_asset':v['files'].pop(str(root/'run/loop-adapter-0003/adapter_model.safetensors'))
        elif fault=='missing_shard':v['files'].pop(str(root/'run/loop-rank-0/checkpoint-0003.pt'))
        elif fault=='missing_source':v['files'].pop(str(root/'config.json'))
        elif fault=='missing_inputs':v['files'].pop(str(root/'run/inputs-0002.json'))
        elif fault=='missing_rollout':v['files'].pop(str(root/'run/rollout-after-0003.json'))
        else:v['extra']='changed'
        put(p,v)
        if fault!='audit_pin':c['pins'][str(p)]=file_hash(p)
    with pytest.raises(ValueError):authenticate(c)


def formal_fixture(tmp_path,monkeypatch):
    c,root,put=fixture(tmp_path,monkeypatch)
    train=json.loads((root/'config.json').read_text());train.pop('sampled_probe')
    train['stage2_initialization']=dict(mode='final_stage1_fresh_optimizer',algorithm='grpo_v1',
        start_update=0,stop_update=634,total_updates=634,reset_optimizer=True,gate='fixed-final-stage1',
        selection_inputs='fixed-inputs',selection_updates=[0,64,128,192,256,317,384,448,512,576,634],
        experiment='released-experiment',lifecycle={'continuous':'accepted','restore':'accepted'})
    put(root/'config.json',train);c.pop('sampled_probe');c['stage2_initialization']=dict(train['stage2_initialization'])
    launch=json.loads((root/'launch.json').read_text());launch['config_sha256']=file_hash(root/'config.json')
    put(root/'launch.json',launch);(root/'terminal.json').unlink()
    observed=Path(c['continuous_resume']['training_observed'])
    record=json.loads(observed.read_text());record['state']['ExitCode']=137;put(observed,record)
    for rank in range(4):
        path=root/'run'/f'loop-rank-{rank}/checkpoint-0003.pt'
        state=torch.load(path,weights_only=True);state['stage']='simple_stage2_rl';torch.save(state,path)
    path=root/'run/checkpoint-0003-committed.json';commit=json.loads(path.read_text())
    commit['files']={p:file_hash(p) for p in commit['files']}
    from multimodal_jev.slurp_training import content_hash
    commit['cursor']['checkpoint_id']=content_hash(commit['files']);put(path,commit)
    audit=Path(c['continuous_resume']['audit']);value=json.loads(audit.read_text())
    value.update(stage='simple_stage2_rl',audit_mode='stopped_prefix',complete_training=False,new_updates=3,
        checks=[dict(passed=True)]*3,selection=[dict(updates=0)],initial_state_checks=[dict(passed=True)]*4)
    value['policies']={'0':'base:reference','1':'base:initial'}
    for step in (2,3):
        v=json.loads((root/'run'/f'checkpoint-{step:04d}-committed.json').read_text())
        value['policies'][str(step)]=v['cursor']['policy_id']
    value['checks']=[dict(passed=True,updates=i,policy_id=value['policies'][str(i)]) for i in range(1,4)]
    value['files']={str(p):file_hash(p) for p in root.rglob('*') if p.is_file()}
    value['files'][str(observed)]=file_hash(observed);put(audit,value)
    c['pins'].update({str(p):file_hash(p) for p in (audit,observed)})
    return c,root,put


def test_formal_stopped_source_keeps_adam_cursor_without_success_terminal(tmp_path,monkeypatch):
    c,_,_=formal_fixture(tmp_path,monkeypatch)
    result=authenticate(c,formal=True)
    assert result['stage']=='simple_stage2_rl' and result['completed_updates']==3
    assert all(s['optimizer']['state'][0]['step'].item()==3 for s in result['states'])


@pytest.mark.parametrize('fault',['mode','completed','partial_updates','selection','initial_state',
    'reference_gate','reward_config','selection_schedule','algorithm','running'])
def test_formal_recovery_rejects_incomplete_audit_or_changed_definition(tmp_path,monkeypatch,fault):
    c,_,put=formal_fixture(tmp_path,monkeypatch)
    if fault in ('reference_gate','reward_config','selection_schedule','algorithm'):
        key={'reference_gate':'gate','reward_config':'experiment','selection_schedule':'selection_updates',
             'algorithm':'algorithm'}[fault]
        c['stage2_initialization'][key]='different'
    elif fault=='running':
        p=Path(c['continuous_resume']['training_observed']);v=json.loads(p.read_text())
        v['state']['Status']='running';put(p,v);c['pins'][str(p)]=file_hash(p)
    else:
        p=Path(c['continuous_resume']['audit']);v=json.loads(p.read_text())
        if fault=='mode':v['audit_mode']='terminal'
        elif fault=='completed':v['complete_training']=True
        elif fault=='partial_updates':v['checks'].pop()
        elif fault=='selection':v['selection']=[]
        else:v['initial_state_checks']=[]
        put(p,v);c['pins'][str(p)]=file_hash(p)
    with pytest.raises(ValueError):authenticate(c,formal=True)


def test_formal_mode_never_implicitly_admits_engineering_checkpoint(tmp_path,monkeypatch):
    c,_,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='formal stopped prefix'):authenticate(c,formal=True)
