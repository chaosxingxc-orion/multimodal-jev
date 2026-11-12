import pytest
import json
from pathlib import Path
from multimodal_jev.baseline import file_hash
from multimodal_jev.slurp_instruction_resume import authenticate,load_full_schedule
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.slurp_instruction_resume import FIELDS, validate_lineage, schedule_cursor


def configs():
    previous = {k: 'fixed-'+k for k in FIELDS}
    previous['updates'] = 2
    current = dict(previous, previous_root='/runs/prefix', initial_adapter='/runs/prefix/run/adapter-step0002',
                   start_update=2, total_updates=634, updates=315)
    return previous, current


def test_full_budget_continues_from_saved_update():
    old, new = configs()
    validate_lineage(old, new, 2, 634)


@pytest.mark.parametrize('key', FIELDS)
def test_no_silent_training_factor_change(key):
    old, new = configs(); new[key] = 'changed'
    with pytest.raises(ValueError, match='changed'):
        validate_lineage(old, new, 2, 634)


@pytest.mark.parametrize('change', [dict(start_update=0), dict(total_updates=635), dict(updates=633),
                                   dict(initial_adapter='/unrelated/adapter'), dict(updates=0)])
def test_cursor_budget_and_adapter_cannot_reset(change):
    old, new = configs(); new.update(change)
    with pytest.raises(ValueError):
        validate_lineage(old, new, 2, 634)


@pytest.mark.parametrize('step,expected', [(0,(0,0)),(2,(0,2)),(316,(0,316)),
                                         (317,(1,0)),(633,(1,316)),(634,(2,0))])
def test_epoch_boundary_does_not_repeat_last_batch(step, expected):
    assert schedule_cursor(step,[317,317]) == expected


@pytest.mark.parametrize('step', [-1,635,2.5,True])
def test_reject_invalid_schedule_cursor(step):
    with pytest.raises(ValueError):
        schedule_cursor(step,[317,317])


def acceptance_fixture(tmp_path):
    root=tmp_path/'training';gate=tmp_path/'gate';adapter=tmp_path/'repair/adapter-step0002'
    old,new=configs();new.update(previous_root=str(root),previous_gate=str(gate),
        initial_adapter=str(adapter),accepted_adapter=str(adapter),pins={})
    def put(path, value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));new['pins'][str(path)]=file_hash(path)
    put(root/'config.json',old)
    put(root/'launch.json',dict(container_id='container',image='image'))
    put(gate/'training-observed.json',dict(container_id='container',image='image',state=dict(Status='exited',ExitCode=0)))
    files={}
    for path in [root/'run'/f'rank-{i}'/'checkpoint-0002.pt' for i in range(4)]+[adapter/'adapter_config.json',adapter/'adapter_model.safetensors']:
        put(path,{'fixture':str(path)});files[str(path)]=file_hash(path)
    state=dict(passed=True,training_root=str(root),updates=2,adapter_id='repaired',adapter=str(adapter),
        exact_master_export=True,exact_peft_reload=True,files=files)
    put(gate/'state-verified.json',state)
    verified=dict(passed=True,training_root=str(root),updates=2,adapter_id='repaired',
        state_sha256=file_hash(gate/'state-verified.json'),full_candidate_parity=True,exact_master_export=True,files={})
    put(gate/'verified.json',verified)
    return new,gate,put


def test_accept_lossless_repair_only_through_pinned_gate(tmp_path):
    config,_,_=acceptance_fixture(tmp_path)
    assert authenticate(config)['adapter_id']=='repaired'


@pytest.mark.parametrize('damage', ['running','failed','wrong_container','reload','master','adapter_id',
                                   'missing_rank','missing_weights','changed_bytes','parity','wrong_state_hash'])
def test_authentication_rejects_incomplete_or_changed_acceptance(tmp_path,damage):
    config,gate,put=acceptance_fixture(tmp_path)
    if damage in ('running','failed','wrong_container'):
        p=gate/'training-observed.json';v=json.loads(p.read_text())
        if damage=='running':v['state']['Status']='running'
        elif damage=='failed':v['state']['ExitCode']=1
        else:v['container_id']='other'
        put(p,v)
    elif damage in ('reload','master','adapter_id','missing_rank','missing_weights'):
        p=gate/'state-verified.json';v=json.loads(p.read_text())
        if damage=='reload':v['exact_peft_reload']=False
        elif damage=='master':v['exact_master_export']=False
        elif damage=='adapter_id':v['adapter_id']='other'
        else:
            suffix='rank-3/checkpoint-0002.pt' if damage=='missing_rank' else 'adapter_model.safetensors'
            v['files']={k:h for k,h in v['files'].items() if not k.endswith(suffix)}
        put(p,v)
        q=gate/'verified.json';w=json.loads(q.read_text());w['state_sha256']=file_hash(p);put(q,w)
    elif damage=='changed_bytes':
        (Path(config['initial_adapter'])/'adapter_model.safetensors').write_text('changed')
    else:
        p=gate/'verified.json';v=json.loads(p.read_text())
        if damage=='parity':v['full_candidate_parity']=False
        else:v['state_sha256']='wrong'
        put(p,v)
    with pytest.raises(ValueError):authenticate(config)


def full_schedule_fixture(tmp_path):
    root=tmp_path/'schedule';root.mkdir();processor=tmp_path/'processor';processor.mkdir()
    config=dict(schedule=str(root),release='/dataset',release_sha256='release',global_batch=128,
        total_updates=4,context_limit=4096,model_path='/model',pins={})
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        config['pins'][str(path)]=file_hash(path)
    epochs=[]
    for i in range(2):
        batches=[[{'id':f'{i}:0'}],[{'id':f'{i}:1'}]]
        p=root/f'epoch-{i}.json';put(p,dict(batches=batches))
        epochs.append(dict(epoch=i,updates=2,file_sha256=file_hash(p),batches_sha256=content_hash(batches)))
    put(root/'MANIFEST.json',dict(release='/dataset',manifest_sha256='release',global_batch=128,
        world_size=4,epochs=epochs,total_nonpadding_recording_uses=4))
    config['schedule_sha256']=file_hash(root/'MANIFEST.json')
    put(processor/'config.json',dict(schedule_sha256=config['schedule_sha256'],model='/model',context_limit=4096))
    put(processor/'launch.json',dict(container_id='processor',image='image',config_sha256=file_hash(processor/'config.json')))
    put(processor/'observed-final.json',dict(container_id='processor',image='image',state=dict(Status='exited',ExitCode=0)))
    put(processor/'run/terminal.json',dict(exit_code=0))
    p=processor/'run/result.json';put(p,dict(passed=True,schedule_sha256=config['schedule_sha256'],
        release_sha256='release',requests=4,context_limit=4096,context_overflow=[]))
    config['processor_acceptance']=str(p)
    return config,processor,put


def test_full_schedule_keeps_epoch_and_batch_identity(tmp_path):
    config,_,_=full_schedule_fixture(tmp_path)
    batches,updates,_=load_full_schedule(config)
    assert [(e,b) for e,b,_ in batches]==[(0,0),(0,1),(1,0),(1,1)]
    assert updates==[2,2]


@pytest.mark.parametrize('damage',['live','failure','partial','overflow','model','budget','epoch_bytes'])
def test_full_schedule_requires_actual_completed_matching_processor(tmp_path,damage):
    config,p,put=full_schedule_fixture(tmp_path)
    if damage=='budget':config['total_updates']=3
    elif damage=='epoch_bytes':(Path(config['schedule'])/'epoch-1.json').write_text('{}')
    elif damage in ('live','failure'):
        path=p/'observed-final.json';v=json.loads(path.read_text())
        if damage=='live':v['state']['Status']='running'
        else:v['state']['ExitCode']=1
        put(path,v)
    elif damage=='model':
        config['model_path']='/other'
    else:
        path=p/'run/result.json';v=json.loads(path.read_text())
        if damage=='partial':v['requests']=3
        else:v['context_overflow']=[{'id':'too-long'}]
        put(path,v)
    with pytest.raises(ValueError):load_full_schedule(config)
