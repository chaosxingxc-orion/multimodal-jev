"""Orchestration regression with real tensors/files/sampling; model/audio loaders mocked.

Actual PEFT and audio execution is still mandatory in the independent image audit.
"""
import copy,importlib.util,json,math,sys,types
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.baseline import file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.simple_sampled_rollout import actor_batch
from test_simple_sampled_checkpoint import fixture as tensor_fixture
from test_simple_sampled_rollout import actor_fixture


def fixture(tmp_path,monkeypatch):
    tools=Path(__file__).parents[1]/'tools/jev';monkeypatch.syspath_prepend(str(tools))
    spec=importlib.util.spec_from_file_location('sampled_audit',tools/'verify_simple_sampled_checkpoint.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
    root=tmp_path/'training';reference=tmp_path/'reference';gate=tmp_path/'gate';base=tmp_path/'base'
    base.mkdir();(base/'model.safetensors').write_bytes(b'base')
    base_id=checkpoint_identity(base)[0]
    for parent in (root,reference):
        a=parent/'run/adapter-step0002';a.mkdir(parents=True)
        (a/'adapter_model.safetensors').write_bytes(b'original adapter')
        put(a/'adapter_config.json',dict(r=4,target_modules=['q','v']))
    old_id=checkpoint_identity(reference/'run/adapter-step0002')[0];policy=base_id+':'+old_id
    adapter=root/'run/rl-adapter-step0001';adapter.mkdir()
    (adapter/'adapter_model.safetensors').write_bytes(b'updated adapter')
    put(adapter/'adapter_config.json',dict(r=4,target_modules=['q','v']))
    adapter_id=checkpoint_identity(adapter)[0]
    from multimodal_jev.slurp_instruction_resume import FIELDS
    c={k:'fixed' for k in FIELDS};c.update(model_path=str(base),base_id=base_id,task_mode='direct_final',
        updates=2,global_batch=128,microbatch_per_rank=1,lr=1e-4,seed=317,pins={},epoch_file=str(tmp_path/'epoch.json'),
        sampled_probe=dict(algorithm='grpo_v1',behavior_gate=str(gate),fixture_initialization='exact_checkpoint_restore'))
    put(root/'config.json',c);put(reference/'config.json',c)
    put(root/'launch.json',dict(container_id='probe',image='fixed',config_sha256=file_hash(root/'config.json')))
    put(root/'terminal.json',dict(exit_code=0));observed=tmp_path/'observed.json'
    put(observed,dict(container_id='probe',image='fixed',state=dict(Status='exited',ExitCode=0)))
    batches=[list(range(128))];put(Path(c['epoch_file']),dict(batches=batches,report={'batches_sha256':content_hash(batches)}))
    entries=[];records=[]
    for i in range(128):
        item,(b,_)=actor_fixture();item['policy_id']=policy;item['request_weight']=.5 if i%2==0 else 1.5
        identity=b['request_identity'];identity['policy_id']=policy;identity['request']['audio']=str(i)
        import hashlib
        b['request_id']=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        entries.append(dict(identity=identity,item=item));records.append(b)
    behavior=dict(passed=True,backend='vllm',base_id=base_id,adapter_id=old_id,records=records)
    put(gate/'run/vllm/result.json',behavior)
    items,sampling=actor_batch(entries,behavior,behavior,policy_id=policy,reference_policy_id=policy,seed=317,update=0)
    compiled={'complete_inputs':128};put(root/'run/rl-inputs.json',dict(compiled=compiled,sampling=sampling))
    renderer=[(None,None,None,None,r['request_id']) for r in records]
    states,exported,_=tensor_fixture();committed={}
    for rank,state in enumerate(states):
        state.update(adapter_id=adapter_id,reference_policy_id=policy)
        prefix=copy.deepcopy(state);prefix['updates']=2
        for v in prefix['optimizer']['state'].values():v['step']=torch.tensor(2.)
        for parent in (root,reference):
            p=parent/'run'/f'rank-{rank}/checkpoint-0002.pt';p.parent.mkdir(parents=True);torch.save(prefix,p)
        out=root/'run'/f'rl-rank-{rank}';out.mkdir();torch.save(state,out/'checkpoint-0001.pt')
        committed[f'rl-rank-{rank}/checkpoint-0001.pt']=file_hash(out/'checkpoint-0001.pt')
        rows=[dict(request_id=item['request_id'],candidate_logprobs=[[-math.log(60)]*60]) for item in items[rank*32:(rank+1)*32]]
        put(out/'after.json',dict(records=rows))
        put(out/'restored.json',dict(rank=rank,world_size=4,previous_updates=1,exact_masters=True,exact_adam=True,
            exact_resident_probabilities=True,requests=32,records=rows))
        put(out/'result.json',dict(passed=True,rank=rank,updates=1,algorithm='grpo_v1',base_id=base_id,
            adapter_id=adapter_id,reference_policy_id=policy,phase='awaiting_acceptance',metrics={'actor/grad_norm':[.5]},
            admission=dict(global_rows=128,global_real_requests=128,stage='task',coefficient=1.,reduction='real_request_mean',algorithm='grpo_v1',group_size=16),
            execution_dtypes={str(i):['torch.bfloat16','torch.bfloat16'] for i in range(112)}))
        put(root/'run'/f'rank-{rank}/terminal.json',dict(exit_code=0))
    put(root/'run/rl-checkpoint-0001-committed.json',dict(adapter_id=adapter_id,world_size=4,files=committed))
    # Mock only expensive external dependencies; file hashes, state/Adam
    # reconstruction, distribution checks and G16 sampling all execute for real.
    st=types.ModuleType('safetensors.torch');st.load_file=lambda path:exported
    monkeypatch.setitem(sys.modules,'safetensors.torch',st)
    peft=types.ModuleType('peft')
    class Loaded:
        def eval(self):return self
        def named_parameters(self):return list(exported.items())
    peft.PeftModel=types.SimpleNamespace(from_pretrained=lambda *a,**k:Loaded());monkeypatch.setitem(sys.modules,'peft',peft)
    import multimodal_jev.asr as asr
    import multimodal_jev.slurp_official_training as release
    import multimodal_jev.simple_audio_batch as audio
    import verify_slurp_simple_parity as parity
    monkeypatch.setattr(asr,'load_runtime',lambda *a:('processor','base'))
    monkeypatch.setattr(release,'load',lambda *a,**kw:([{'id':'placeholder'}],{},{}))
    monkeypatch.setattr(audio,'compile_batch',lambda *a,**kw:(entries,renderer,compiled))
    monkeypatch.setattr(parity,'verify',lambda p:dict(training_root=str(reference),updates=2,requests=256,adapter_id=old_id,files={}))
    return m,dict(training_root=str(root),training_observed=str(observed),pins={}),root,put


def test_complete_real_state_and_sampling_audit_with_mocked_external_loaders(tmp_path,monkeypatch):
    m,c,root,_=fixture(tmp_path,monkeypatch);r=m.run(c,tmp_path/'out')
    assert r['passed'] and r['tensors']==112 and r['sampling_signal']['samples']==2048
    signal=r['sampling_signal'];assert signal['all_wrong_groups']+signal['all_correct_groups']+signal['mixed_reward_groups']==128
    assert signal['real_request_weight']==128


@pytest.mark.parametrize('fault',['commit','master','sampling','coverage','dtype','partial','admission','cursor','live'])
def test_reject_false_successful_update_artifacts(tmp_path,monkeypatch,fault):
    m,c,root,put=fixture(tmp_path,monkeypatch);out=root/'run/rl-rank-0'
    if fault=='commit':
        p=root/'run/rl-checkpoint-0001-committed.json';v=json.loads(p.read_text());v['files'].pop(next(iter(v['files'])));put(p,v)
    elif fault=='master':
        p=out/'checkpoint-0001.pt';v=torch.load(p,weights_only=True);v['parameters'][v['names'][0]]+=1;torch.save(v,p)
        p=root/'run/rl-checkpoint-0001-committed.json';v=json.loads(p.read_text());v['files']['rl-rank-0/checkpoint-0001.pt']=file_hash(out/'checkpoint-0001.pt');put(p,v)
    elif fault=='sampling':
        p=root/'run/rl-inputs.json';v=json.loads(p.read_text());v['sampling']['sampling'][0]['actions'][0][0][0]=59;put(p,v)
    elif fault in ('coverage','partial'):
        p=out/'restored.json';v=json.loads(p.read_text())
        if fault=='coverage':v['records'][0]['request_id']='foreign'
        else:v['records'].pop()
        put(out/'after.json',dict(records=v['records']));put(p,v)
    elif fault=='live':
        p=Path(c['training_observed']);v=json.loads(p.read_text());v['state']['Status']='running';put(p,v)
    else:
        p=out/'result.json';v=json.loads(p.read_text())
        if fault=='dtype':v['execution_dtypes']['0']=['torch.float32']*2
        elif fault=='admission':v['admission']['global_real_requests']=127
        else:v['updates']=2
        put(p,v)
    with pytest.raises(ValueError):m.run(c,tmp_path/'out')
