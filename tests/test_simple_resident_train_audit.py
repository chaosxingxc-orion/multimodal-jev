"""Real four-shard tensors, sampling and parity; external audio/model I/O mocked."""
import copy
import importlib.util
import json
import sys
import types
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.baseline import file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.simple_audio_batch import compile_batch,renderer_for_policy
from multimodal_jev.simple_sampled_rollout import actor_batch
from multimodal_jev.simple_resident_acceptance import compare_updated_batch
from test_simple_resident_batch import setup
from test_simple_sampled_checkpoint import fixture as tensor_fixture


def fixture(tmp_path,monkeypatch,*,formal_prefix=False,formal_start=0,formal_stop=3,fixture_resume=False):
    tools=Path(__file__).parents[1]/'tools/jev';monkeypatch.syspath_prepend(str(tools))
    spec=importlib.util.spec_from_file_location('resident_train_audit',tools/'verify_simple_resident_train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    collector,population,slots,policy,_,_=setup(monkeypatch)
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    root=tmp_path/'train';source=tmp_path/'initial';old=tmp_path/'reference';current=tmp_path/'behavior'
    train=dict(base_id='base',model_path='base',seed=317,lr=1e-4,epoch_file=str(tmp_path/'schedule/epoch.json'),
        schedule_sha256='schedule',release_sha256='release',release='data',
        sampled_probe=dict(algorithm='grpo_v1',stop_update=3,behavior_gate=str(current),reference_gate=str(old)))
    if fixture_resume:
        train['sampled_probe'].update(fixture_initialization='restore_continuous_then_one_update',stop_update=4)
        train['continuous_resume']=dict(algorithm='grpo_v1')
    if formal_prefix:
        train.pop('sampled_probe')
        train['stage2_initialization']=dict(algorithm='grpo_v1',stop_update=634)
    put(root/'config.json',train)
    states,exported,_=tensor_fixture()
    def scheduled_slots(step):
        result=copy.deepcopy(slots)
        if formal_prefix and step%317==316:
            result[65:]=[dict(result[0],padding=True,weight=0.) for _ in range(63)]
        return result
    if formal_prefix:
        import multimodal_jev.simple_stage2_audit as stage2
        # Source/initial-state and full selection are exercised independently.
        # This fixture executes every intervening audit operation with real
        # tensors, complete G16 sampling and resident probability checks.
        source_result=dict(files={},names=states[0]['names'],checks=[dict(passed=True)]*4,
            initialization=dict(reference_policy_id='base:reference',policy_id='base:initial'))
        if formal_start:
            # Earlier prefix authentication is deliberately substituted. The
            # following segment's actual tensors, G16 sampling, padding and
            # terminal behavior are all checked by the real audit loop.
            source_result['recovery']=dict(completed_updates=formal_start,policy_id='base:initial',
                selection_history=[],prefix_audit=dict(checks=[dict(passed=True,updates=i)
                    for i in range(1,formal_start+1)],policies={str(i):'base:initial'
                    for i in range(formal_start+1)},initial_state_checks=[dict(passed=True)]*4))
        monkeypatch.setattr(stage2,'source',lambda *a,**k:source_result)
        monkeypatch.setattr(stage2,'selections',lambda *a,through_update,**kw: [dict(updates=0,through_update=through_update)])
    if fixture_resume:
        import verify_simple_continuous_restore as continuous_restore
        monkeypatch.setattr(continuous_restore,'run',lambda *a,**k:dict(files={},requests=128,parity={'passed':True}))
        import multimodal_jev.simple_continuous_resume as continuous_source
        monkeypatch.setattr(continuous_source,'authenticate',lambda *a,**k:dict(files={},completed_updates=3,
            reference_policy_id='base:reference',policy_id='base:initial',states=states))
    initial=source/'run/rl-rank-0/checkpoint-0001.pt';initial.parent.mkdir(parents=True);torch.save(states[0],initial)
    import verify_simple_sampled_restore as recovery
    def recover(*args,**kwargs):
        assert kwargs['expected_mode']=='restore_updated_rl_then_resident_loop'
        return dict(files={})
    monkeypatch.setattr(recovery,'run',recover)
    import verify_slurp_simple_parity as parity
    monkeypatch.setattr(parity,'verify',lambda p:dict(files={},adapter_id='reference' if Path(p)==old else 'initial',training_root=str(source)))
    import multimodal_jev.simple_rl_schedule as schedules
    class Schedule:
        files={};sha256='schedule'
        def __init__(self,*args,**kwargs):pass
        def validate_population(self,rows):return dict(passed=True)
        def at(self,step):
            assert step in (range(formal_start,formal_stop) if formal_prefix else (3,) if fixture_resume else (1,2))
            return scheduled_slots(step),dict(coefficient=max(.1,1.-step*.1),
                real_requests=65 if formal_prefix and step%317==316 else 128)
    monkeypatch.setattr(schedules,'SimpleRLSchedule',Schedule)
    import multimodal_jev.slurp_official_training as release
    monkeypatch.setattr(release,'load',lambda *a,**k:(list(population.values()),{},{}))
    transformer=types.ModuleType('transformers')
    transformer.Qwen3ASRProcessor=types.SimpleNamespace(from_pretrained=lambda *a,**k:collector.processor)
    monkeypatch.setitem(sys.modules,'transformers',transformer)
    # Substitute only the safetensors loader; reconstruction compares real tensors.
    exports={};safe=types.ModuleType('safetensors.torch');safe.load_file=lambda p:exports[p]
    monkeypatch.setitem(sys.modules,'safetensors.torch',safe)
    import multimodal_jev.asr as asr
    monkeypatch.setattr(asr,'load_runtime',lambda *a:('processor','base'))
    peft=types.ModuleType('peft')
    class Loaded:
        def eval(self):return self
        def named_parameters(self):return list(exported.items())
    peft.PeftModel=types.SimpleNamespace(from_pretrained=lambda *a,**k:Loaded())
    monkeypatch.setitem(sys.modules,'peft',peft)
    behavior='base:initial';fixed='base:reference'
    for updates in (range(formal_start+1,formal_stop+1) if formal_prefix else (4,) if fixture_resume else (2,3)):
        step=updates-1
        entries,renderer,compiled=compile_batch(population,{},scheduled_slots(step),collector.processor,policy_id=behavior,global_batch=128)
        b=collector.engine.rollout(dict(policy,adapter_id=behavior.split(':')[1],population={'requests':len(renderer)}),renderer)
        reference=collector.engine.rollout(dict(policy,population={'requests':len(renderer)}),renderer_for_policy(renderer,fixed))
        _,sampling=actor_batch(entries,b,reference,policy_id=behavior,reference_policy_id=fixed,seed=317,update=step)
        for name,value in [('behavior',b),('reference',reference),('inputs',dict(compiled=compiled,sampling=sampling,next_optimizer_update=updates))]:
            put(root/'run'/f'{name}-{step:04d}.json',value)
        adapter=root/'run'/f'loop-adapter-{updates:04d}';adapter.mkdir()
        (adapter/'adapter_model.safetensors').write_bytes(f'adapter{updates}'.encode())
        put(adapter/'adapter_config.json',dict(r=4,target_modules=['q_proj','v_proj']))
        aid=checkpoint_identity(adapter)[0];new='base:'+aid
        exports[str(adapter/'adapter_model.safetensors')]=exported
        hashes={}
        for rank,state in enumerate(copy.deepcopy(states)):
            state.update(updates=updates,stage='simple_stage2_rl' if formal_prefix else 'continuous_rl_engineering',adapter_id=aid,
                         reference_policy_id=fixed,schedule_sha256='schedule',next_update=updates)
            for v in state['optimizer']['state'].values():v['step']=torch.tensor(float(updates))
            path=root/'run'/f'loop-rank-{rank}'/f'checkpoint-{updates:04d}.pt';path.parent.mkdir(exist_ok=True)
            torch.save(state,path);hashes[str(path)]=file_hash(path)
        committed=dict(passed=True,files=hashes,reference_policy_id=fixed,cursor=dict(
            schema='jev-span-training-session-v1',completed_updates=updates,policy_id=new,
            schedule_sha256='schedule',checkpoint_id=content_hash(hashes)))
        put(root/'run'/f'checkpoint-{updates:04d}-committed.json',committed)
        result=collector.engine.rollout(dict(policy,adapter_id=aid,population={'requests':len(renderer)}),renderer_for_policy(renderer,new))
        result['reference_compared']=False
        # Backend rows are unique; zero-weight duplicate slots still require
        # distinct actor identities. Match the old input identity to the new
        # policy's record rather than zip 128 actor slots with 65 backend rows.
        by_input={}
        import hashlib
        for record in result['records']:
            identity=dict(record['request_identity'],policy_id=behavior)
            uid=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            by_input[uid]=record
        rows=[dict(request_id=s['actor_request_id'],candidate_logprobs=by_input[s['request_id']]['candidate_logprobs'])
              for s in sampling['sampling']]
        ranks=[rows[i:i+32] for i in range(0,128,32)]
        check=compare_updated_batch(ranks,result,sampling,policy_id=new,updates=updates)
        put(root/'run'/f'rollout-after-{updates:04d}.json',result)
        for rank in range(4):
            out=root/'run'/f'loop-rank-{rank}'
            put(out/f'after-{updates:04d}.json',ranks[rank])
            put(out/f'update-{updates:04d}.json',dict(completed_updates=updates,next_update=updates,
                behavior_policy_id=behavior,policy_id=new,coefficient=max(.1,1.-step*.1),
                real_requests=65 if formal_prefix and step%317==316 else 128,
                schedule_sha256='schedule',checkpoint=committed,acceptance=check,metrics={'actor/grad_norm':[.5]}))
        behavior=new
    for rank in range(4):
        put(root/'run'/f'loop-rank-{rank}'/'result.json',dict(passed=True,completed_updates=formal_stop if formal_prefix else 4 if fixture_resume else 3,
            phase='accepted',policy_id=behavior,schedule_sha256='schedule',reference_policy_id=fixed))
    config=dict(training_root=str(root))
    if formal_prefix and formal_stop<634:config.update(audit_mode='stopped_prefix',completed_updates=formal_stop)
    return module,config,root,put


def test_two_updates_real_sampling_shards_and_probabilities(tmp_path,monkeypatch):
    m,c,root,_=fixture(tmp_path,monkeypatch);r=m.run(c,tmp_path/'out')
    assert r['passed'] and r['new_updates']==2 and len(r['checks'])==2 and r['exact_peft_reload']
    assert all(x['requests']==128 and x['maximum_logprob_error']==0 for x in r['checks'])
    assert any(k.endswith('rl-rank-0/checkpoint-0001.pt') for k in r['files'])


@pytest.mark.parametrize('fault',['adam','master','reference','state_schedule','commit','sampling',
    'input_cursor','report_cursor','report_schedule','coefficient','gradient_nan','gradient_zero',
    'duplicate_slot','probability','final_reference','final_schedule','final_policy'])
def test_reject_corruption_even_with_refreshed_commit_hashes(tmp_path,monkeypatch,fault):
    m,c,root,put=fixture(tmp_path,monkeypatch);out=root/'run/loop-rank-2'
    if fault in ('adam','master','reference','state_schedule'):
        path=out/'checkpoint-0003.pt';state=torch.load(path,weights_only=True)
        if fault=='adam':state['optimizer']['state'][0]['step']=torch.tensor(2.)
        elif fault=='master':state['parameters'][state['names'][0]]+=1
        elif fault=='reference':state['reference_policy_id']='moving'
        else:state['schedule_sha256']='other'
        torch.save(state,path)
        p=root/'run/checkpoint-0003-committed.json';v=json.loads(p.read_text());v['files'][str(path)]=file_hash(path)
        v['cursor']['checkpoint_id']=content_hash(v['files']);put(p,v)
        for rank in range(4):
            q=root/'run'/f'loop-rank-{rank}'/'update-0003.json';report=json.loads(q.read_text());report['checkpoint']=v;put(q,report)
    else:
        if fault=='commit':p=root/'run/checkpoint-0003-committed.json'
        elif fault in ('sampling','input_cursor'):p=root/'run/inputs-0002.json'
        elif fault in ('duplicate_slot','probability'):p=out/'after-0003.json'
        elif fault.startswith('final_'):p=out/'result.json'
        else:p=out/'update-0003.json'
        v=json.loads(p.read_text())
        if fault=='commit':v['passed']=False
        elif fault=='sampling':v['sampling']['sampling'][0]['slot']=4
        elif fault=='input_cursor':v['next_optimizer_update']=2
        elif fault=='report_cursor':v['next_update']=2
        elif fault=='report_schedule':v['schedule_sha256']='other'
        elif fault=='coefficient':v['coefficient']=.01
        elif fault.startswith('gradient_'):v['metrics']['actor/grad_norm']=[float('nan') if fault=='gradient_nan' else 0.]
        elif fault=='duplicate_slot':v[-1]=copy.deepcopy(v[0])
        elif fault=='probability':v[0]['candidate_logprobs'][0][0]+=.2
        elif fault=='final_reference':v['reference_policy_id']='moving'
        elif fault=='final_schedule':v['schedule_sha256']='other'
        else:v['policy_id']='stale'
        put(p,v)
    with pytest.raises(ValueError):m.run(c,tmp_path/'out')


def test_valid_export_with_incorrect_peft_loader_is_rejected(tmp_path,monkeypatch):
    m,c,_,_=fixture(tmp_path,monkeypatch)
    class Wrong:
        def eval(self):return self
        def named_parameters(self):return []
    monkeypatch.setattr(sys.modules['peft'].PeftModel,'from_pretrained',lambda *a,**k:Wrong())
    with pytest.raises(ValueError,match='PEFT reload differs'):m.run(c,tmp_path/'out')


def test_formal_stopped_prefix_checks_every_update_without_claiming_terminal(tmp_path,monkeypatch):
    m,c,root,_=fixture(tmp_path,monkeypatch,formal_prefix=True)
    # No final success record is needed for recovery from an exited process.
    for rank in range(4):(root/'run'/f'loop-rank-{rank}'/'result.json').unlink()
    r=m.run(c,tmp_path/'out')
    assert r['passed'] and r['new_updates']==3 and len(r['checks'])==3
    assert r['audit_mode']=='stopped_prefix' and r['complete_training'] is False
    assert r['exact_peft_reload'] and r['selection']==[dict(updates=0,through_update=3)]
    assert all(x['requests']==128 for x in r['checks'])


def test_formal_prefix_rejects_corrupt_middle_update(tmp_path,monkeypatch):
    m,c,root,put=fixture(tmp_path,monkeypatch,formal_prefix=True)
    path=root/'run/inputs-0001.json';value=json.loads(path.read_text())
    value['sampling']['sampling'][0]['slot']=4;put(path,value)
    with pytest.raises(ValueError,match='sampling replay'):
        m.run(c,tmp_path/'out')


def test_formal_prefix_must_load_its_last_adapter_even_before_budget_end(tmp_path,monkeypatch):
    m,c,_,_=fixture(tmp_path,monkeypatch,formal_prefix=True)
    class Wrong:
        def eval(self):return self
        def named_parameters(self):return []
    monkeypatch.setattr(sys.modules['peft'].PeftModel,'from_pretrained',lambda *a,**k:Wrong())
    with pytest.raises(ValueError,match='PEFT reload differs'):m.run(c,tmp_path/'out')


def test_engineering_fixture_cannot_be_promoted_by_prefix_flag(tmp_path,monkeypatch):
    m,c,_,_=fixture(tmp_path,monkeypatch)
    c.update(audit_mode='stopped_prefix',completed_updates=3)
    with pytest.raises(ValueError,match='only defined for formal'):
        m.run(c,tmp_path/'out')


def test_resumed_audit_covers_new_segment_and_retains_certified_policy_prefix(tmp_path,monkeypatch):
    m,c,root,_=fixture(tmp_path,monkeypatch,formal_prefix=True)
    import multimodal_jev.simple_stage2_audit as audit
    source=audit.source(c,verify_parity=None)
    first=json.loads((root/'run/loop-rank-0/update-0001.json').read_text())
    prefix=dict(checks=[first['acceptance']],policies={'0':'base:initial','1':first['policy_id']},
        initial_state_checks=source['checks'])
    source['recovery']=dict(completed_updates=1,policy_id=first['policy_id'],prefix_audit=prefix,
        selection_history=[dict(updates=0)])
    monkeypatch.setattr(audit,'source',lambda *a,**k:source)
    seen=[]
    def selections(*args,through_update,history):
        seen.extend(history);return [dict(updates=0,through_update=through_update)]
    monkeypatch.setattr(audit,'selections',selections)
    # Only the independently accepted prefix report supplies the old update.
    # Replaying this file would reveal an incorrect reset to update0.
    (root/'run/inputs-0000.json').unlink()
    result=m.run(c,tmp_path/'out')
    assert result['segment_start_update']==1 and result['segment_new_updates']==2
    assert result['new_updates']==3 and len(result['checks'])==3
    assert set(result['policies'])=={0,1,2,3} and seen==[dict(updates=0)]


@pytest.mark.parametrize('start,stop',[(315,318),(632,634)])
def test_formal_epoch_padding_and_terminal_cursor_in_real_audit_loop(tmp_path,monkeypatch,start,stop):
    m,c,root,_=fixture(tmp_path,monkeypatch,formal_prefix=True,formal_start=start,formal_stop=stop)
    result=m.run(c,tmp_path/'out')
    assert result['updates']==stop and result['segment_new_updates']==stop-start
    boundary=317 if start==315 else 634
    inputs=json.loads((root/'run'/f'inputs-{boundary-1:04d}.json').read_text())
    assert inputs['sampling']['global_real_requests']==65 and inputs['sampling']['global_rows']==128
    check=result['checks'][boundary-1]
    assert check['requests']==128 and check['unique_inputs']==65
    assert result['complete_training'] is (stop==634)
    assert result['policies'][stop]==json.loads((root/'run/loop-rank-0/result.json').read_text())['policy_id']


@pytest.mark.parametrize('fault',['padding_denominator','adam_cursor','terminal_cursor'])
def test_terminal_segment_rejects_wrong_padding_or_final_state(tmp_path,monkeypatch,fault):
    m,c,root,put=fixture(tmp_path,monkeypatch,formal_prefix=True,formal_start=633,formal_stop=634)
    if fault=='adam_cursor':
        path=root/'run/loop-rank-0/checkpoint-0634.pt';state=torch.load(path,weights_only=True)
        state['optimizer']['state'][0]['step'].fill_(633);torch.save(state,path)
        p=root/'run/checkpoint-0634-committed.json';value=json.loads(p.read_text())
        value['files'][str(path)]=file_hash(path);value['cursor']['checkpoint_id']=content_hash(value['files']);put(p,value)
    else:
        path=root/'run/loop-rank-0'/('update-0634.json' if fault=='padding_denominator' else 'result.json')
        value=json.loads(path.read_text());value['real_requests' if fault=='padding_denominator' else 'completed_updates']=128 if fault=='padding_denominator' else 633;put(path,value)
    with pytest.raises(ValueError):m.run(c,tmp_path/'out')


@pytest.mark.parametrize('fault',[None,'adam','reference','sampling','probability','terminal_cursor'])
def test_continuous_checkpoint_next_update_audit(tmp_path,monkeypatch,fault):
    m,c,root,put=fixture(tmp_path,monkeypatch,fixture_resume=True)
    if fault=='adam':
        path=root/'run/loop-rank-0/checkpoint-0004.pt';state=torch.load(path,weights_only=True)
        state['optimizer']['state'][0]['step']=torch.tensor(3.);torch.save(state,path)
        cp=root/'run/checkpoint-0004-committed.json';commit=json.loads(cp.read_text())
        commit['files'][str(path)]=file_hash(path);commit['cursor']['checkpoint_id']=content_hash(commit['files']);put(cp,commit)
    elif fault=='reference':
        path=root/'config.json';value=json.loads(path.read_text());value['continuous_resume']['algorithm']='rloo_v1';put(path,value)
    elif fault=='sampling':
        path=root/'run/inputs-0003.json';value=json.loads(path.read_text());value['sampling']['update']=2;put(path,value)
    elif fault=='probability':
        path=root/'run/loop-rank-0/after-0004.json';value=json.loads(path.read_text());value[0]['candidate_logprobs'][0][0]+=.2;put(path,value)
    elif fault=='terminal_cursor':
        path=root/'run/loop-rank-0/result.json';value=json.loads(path.read_text());value['completed_updates']=3;put(path,value)
    if fault:
        with pytest.raises(ValueError):m.run(c,tmp_path/'out')
    else:
        result=m.run(c,tmp_path/'out')
        assert result['updates']==4 and result['new_updates']==1 and result['resumed_from_update']==3
        assert result['recovery_probability_requests']==128 and len(result['checks'])==1
        assert result['checks'][0]['requests']==128
