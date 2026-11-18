"""Second-update audit with real tensors/sampling; external runtime mocked."""
import importlib.util,json
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.baseline import file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.span_hybrid_objective import sft_coefficient
from test_simple_sampled_checkpoint_audit import fixture as initial_fixture


def fixture(tmp_path,monkeypatch):
    _,config,root,put=initial_fixture(tmp_path,monkeypatch)
    import verify_simple_sampled_restore as recovery
    import verify_slurp_simple_parity as parity
    import multimodal_jev.simple_next_behavior as transport
    import multimodal_jev.simple_rl_schedule as schedule
    class Schedule:
        files={}
        def __init__(self,*a,**k):pass
        def validate_population(self,rows):return dict(passed=True,recordings=40513,uses=81026)
        def at(self,step):
            assert step==1
            return [],dict(coefficient=sft_coefficient(1,634))
    monkeypatch.setattr(schedule,'SimpleRLSchedule',Schedule)
    # First-step recovery itself is exercised by separate exact-state tests.
    monkeypatch.setattr(recovery,'run',lambda *a,**k:dict(files={}))
    train=json.loads((root/'config.json').read_text());old=Path(train['sampled_probe']['behavior_gate'])
    original=json.loads((old/'run/vllm/result.json').read_text())
    fixed=original['base_id']+':'+original['adapter_id']
    updated=tmp_path/'updated';source=tmp_path/'previous'
    source_adapter=root/'run/rl-adapter-step0001'
    behavior_id=checkpoint_identity(source_adapter)[0];behavior=train['base_id']+':'+behavior_id
    train['sampled_probe'].update(behavior_gate=str(updated),reference_gate=str(old),total_rl_updates=634,
        fixture_initialization='restore_updated_rl_then_second_update')
    put(root/'config.json',train);put(source/'config.json',dict(sampled_probe=dict(behavior_gate=str(old))))
    def verify(path):
        is_old=Path(path)==old
        return dict(updates=2 if is_old else 1,requests=256,files={},
            adapter_id=original['adapter_id'] if is_old else behavior_id,training_root=str(source))
    monkeypatch.setattr(parity,'verify',verify)
    adapter=root/'run/rl-adapter-step0002';adapter.mkdir()
    (adapter/'adapter_model.safetensors').write_bytes(b'second adapter')
    put(adapter/'adapter_config.json',dict(r=4,target_modules=['q','v']))
    adapter_id=checkpoint_identity(adapter)[0];committed={};items=[]
    for rank in range(4):
        out=root/'run'/f'rl-rank-{rank}'
        state=torch.load(out/'checkpoint-0001.pt',weights_only=True)
        path=source/'run'/f'rl-rank-{rank}/checkpoint-0001.pt';path.parent.mkdir(parents=True);torch.save(state,path)
        state.update(updates=2,adapter_id=adapter_id,reference_policy_id=fixed,behavior_policy_id=behavior)
        for v in state['optimizer']['state'].values():v['step']=torch.tensor(2.)
        torch.save(state,out/'checkpoint-0002.pt')
        committed[f'rl-rank-{rank}/checkpoint-0002.pt']=file_hash(out/'checkpoint-0002.pt')
        r=json.loads((out/'result.json').read_text());r.update(updates=2,adapter_id=adapter_id,
            behavior_policy_id=behavior,coefficient=sft_coefficient(1,634))
        r['admission']['coefficient']=r['coefficient'];put(out/'result.json',r)
        restored=json.loads((out/'restored.json').read_text());restored['previous_updates']=2
        for row in restored['records']:
            row['request_id']=row['request_id'].replace('::update0-', '::update1-')
            items.append(dict(request_id=row['request_id']))
        put(out/'restored.json',restored);put(out/'after.json',dict(records=restored['records']))
    put(root/'run/rl-checkpoint-0002-committed.json',dict(adapter_id=adapter_id,world_size=4,files=committed))
    epoch=json.loads(Path(train['epoch_file']).read_text());epoch['batches']*=2
    from multimodal_jev.slurp_training import content_hash
    epoch['report']['batches_sha256']=content_hash(epoch['batches']);put(Path(train['epoch_file']),epoch)
    receipt=dict(next_optimizer_update=2,compiled={'rows':128},sampling={'update':1})
    put(root/'run/rl-inputs.json',receipt);put(updated/'run/vllm/result.json',{})
    monkeypatch.setattr(transport,'next_batch',lambda *a,**k:(items,receipt))
    path=Path(__file__).parents[1]/'tools/jev/verify_simple_sampled_continue.py'
    spec=importlib.util.spec_from_file_location('continue_audit',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m,config,root,put


def test_second_state_export_and_actual_cursor(tmp_path,monkeypatch):
    m,c,root,_=fixture(tmp_path,monkeypatch);report=m.run(c,tmp_path/'out')
    assert report['updates']==2 and report['exact_pre_update_recovery'] and report['tensors']==112


@pytest.mark.parametrize('fault',['reset_adam','reference','behavior','commit','input','coefficient','probability','dtype'])
def test_second_update_rejects_reset_or_wrong_provenance(tmp_path,monkeypatch,fault):
    m,c,root,put=fixture(tmp_path,monkeypatch);out=root/'run/rl-rank-1'
    if fault in ('reset_adam','reference','behavior'):
        p=out/'checkpoint-0002.pt';s=torch.load(p,weights_only=True)
        if fault=='reset_adam':s['optimizer']['state'][0]['step']=torch.tensor(1.)
        elif fault=='reference':s['reference_policy_id']='moving'
        else:s['behavior_policy_id']='old'
        torch.save(s,p);q=root/'run/rl-checkpoint-0002-committed.json';v=json.loads(q.read_text())
        v['files']['rl-rank-1/checkpoint-0002.pt']=file_hash(p);put(q,v)
    elif fault=='commit':
        p=root/'run/rl-checkpoint-0002-committed.json';v=json.loads(p.read_text());v['files'].pop(next(iter(v['files'])));put(p,v)
    elif fault=='input':put(root/'run/rl-inputs.json',{'wrong':True})
    elif fault=='probability':
        p=out/'restored.json';v=json.loads(p.read_text());v['records'][0]['candidate_logprobs']=[[0.]*60]
        put(p,v);put(out/'after.json',dict(records=v['records']))
    else:
        p=out/'result.json';v=json.loads(p.read_text())
        if fault=='coefficient':v['coefficient']=1.
        else:v['execution_dtypes']['0']=['torch.float32']*2
        put(p,v)
    with pytest.raises(ValueError):m.run(c,tmp_path/'out')
