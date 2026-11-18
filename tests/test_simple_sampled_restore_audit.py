import importlib.util,json,math
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.baseline import file_hash
from test_simple_sampled_checkpoint import fixture as states_fixture


def fixture(tmp_path,monkeypatch):
    tools=Path(__file__).parents[1]/'tools/jev';monkeypatch.syspath_prepend(str(tools))
    import verify_slurp_simple_parity
    spec=importlib.util.spec_from_file_location('restore_audit',tools/'verify_simple_sampled_restore.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
    source=tmp_path/'source';root=tmp_path/'restore';gate=tmp_path/'gate';old=tmp_path/'old'
    from multimodal_jev.slurp_instruction_resume import FIELDS
    common={k:'fixed' for k in FIELDS};common['base_id']='base'
    source_config=dict(common,sampled_probe=dict(algorithm='grpo_v1',behavior_gate=str(old)))
    put(source/'config.json',source_config);put(old/'config.json',dict(adapter_id='old'))
    c=dict(common,pins={},sampled_probe=dict(algorithm='grpo_v1',behavior_gate=str(gate),fixture_initialization='restore_updated_rl_only'))
    put(root/'config.json',c);put(root/'launch.json',dict(container_id='restore',image='fixed',config_sha256=file_hash(root/'config.json')))
    put(root/'terminal.json',dict(exit_code=0));obs=tmp_path/'observed.json';put(obs,dict(container_id='restore',image='fixed',state=dict(Status='exited',ExitCode=0)))
    put(gate/'config.json',dict(inputs=str(gate/'inputs.json')))
    put(gate/'inputs.json',dict(records=[dict(id=str(i)) for i in range(256)]))
    records=[dict(id=str(i),candidate_logprobs=[[-math.log(60)]*60]) for i in range(256)]
    for b in ('hf','vllm'):put(gate/'run'/b/'result.json',dict(records=records))
    states,_,_=states_fixture()
    for rank,state in enumerate(states):
        state['reference_policy_id']='base:old'
        p=source/'run'/f'rl-rank-{rank}/checkpoint-0001.pt';p.parent.mkdir(parents=True,exist_ok=True);torch.save(state,p)
        out=root/'run'/f'rank-{rank}';out.mkdir(parents=True)
        torch.save(state,out/'checkpoint-0001.pt')
        put(out/'result.json',dict(passed=True,rank=rank,updates=1,adapter_id='new',base_id='base',
            checkpoint_source=str(p),execution_dtypes={str(i):['torch.bfloat16','torch.bfloat16'] for i in range(112)}))
        put(out/'terminal.json',dict(exit_code=0));put(out/'restored-policy.json',dict(records=records[rank::4]))
    monkeypatch.setattr(verify_slurp_simple_parity,'verify',lambda p:dict(updates=1,requests=256,sampled_resident_parity={'hf':{}},files={},training_root=str(source),adapter_id='new'))
    return m,dict(training_root=str(root),training_observed=str(obs),pins={}),root,put


def test_independently_recomputes_full_restored_state_and_probabilities(tmp_path,monkeypatch):
    m,c,root,_=fixture(tmp_path,monkeypatch);r=m.run(c,tmp_path/'out')
    assert r['exact_master_adam'] and r['requests']==256 and r['maximum_logprob_error']==0


@pytest.mark.parametrize('fault',['master','adam','partial','probability','source','live'])
def test_reject_false_fresh_process_recovery(tmp_path,monkeypatch,fault):
    m,c,root,put=fixture(tmp_path,monkeypatch);out=root/'run/rank-1'
    if fault in ('master','adam'):
        p=out/'checkpoint-0001.pt';state=torch.load(p,weights_only=True)
        value=state['parameters'][state['names'][0]] if fault=='master' else state['optimizer']['state'][0]['exp_avg']
        value[0]=torch.nextafter(value[0],torch.tensor(float('inf')));torch.save(state,p)
    elif fault in ('partial','probability'):
        p=out/'restored-policy.json';r=json.loads(p.read_text())
        if fault=='partial':r['records'].pop()
        else:r['records'][0]['candidate_logprobs']=[[math.log(.5)]+[math.log(.5/59)]*59]
        put(p,r)
    elif fault=='source':
        p=out/'result.json';r=json.loads(p.read_text());r['checkpoint_source']='other';put(p,r)
    else:
        p=Path(c['training_observed']);r=json.loads(p.read_text());r['state']['Status']='running';put(p,r)
    with pytest.raises(ValueError):m.run(c,tmp_path/'out')
