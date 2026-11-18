import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pytest
pytest.importorskip('torch')
from multimodal_jev.baseline import file_hash

spec=importlib.util.spec_from_file_location('simple_parity',Path(__file__).parents[1]/'tools/jev/verify_slurp_simple_parity.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture(root,monkeypatch,count=256):
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    train=root/'train';gate=root/'gate';adapter=root/'adapter'
    put(train/'config.json',{'task_mode':'direct_final'})
    put(train/'launch.json',dict(container_id='train',image='fixed',config_sha256=file_hash(train/'config.json')))
    put(gate/'training-observed.json',dict(container_id='train',image='fixed',state=dict(Status='exited',ExitCode=0)))
    put(gate/'state-verified.json',dict(passed=True,training_root=str(train),updates=2,
        exact_master_export=True,exact_peft_reload=True,durable_commit_verified=True,
        adapter_id='a',adapter=str(adapter),files={}))
    rows=[];outputs=[]
    for i in range(count):
        row=dict(id=str(i),request={'audio':str(i)},audio_processing={'wave':str(i)},positions=[[2]],
            candidate_mapping=[dict(token_ids=list(range(60)))],input_ids={'hash':str(i)},
            input_features={'hash':str(i)},input_features_mask={'hash':'mask'})
        identity=dict(request=row['request'],policy_id='b:a',audio_processing=row['audio_processing'],
            positions=row['positions'],candidate_ids=[list(range(60))],inputs={k:row[k] for k in
            ('input_ids','input_features','input_features_mask')})
        digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        rows.append(row);outputs.append(dict(id=str(i),request_id=digest,request_identity=identity,
            candidate_logprobs=[[-math.log(60)]*60],parity={'passed':True}))
    inputs=gate/'inputs.json';put(inputs,dict(passed=True,population_mode='prefix',requests=count,recordings=count,records=rows))
    put(gate/'config.json',dict(base_id='b',adapter_id='a',adapter=str(adapter),inputs=str(inputs),pins={str(inputs):file_hash(inputs)}))
    put(gate/'launch.json',dict(container_id='parity',image='fixed',config_sha256=file_hash(gate/'config.json')))
    put(gate/'observed-final.json',dict(container_id='parity',image='fixed',state=dict(Status='exited',ExitCode=0)))
    for backend in ('hf','vllm'):
        put(gate/'run'/backend/'terminal.json',dict(exit_code=0))
        put(gate/'run'/backend/'result.json',dict(passed=True,backend=backend,base_id='b',adapter_id='a',records=outputs))
    put(gate/'run/terminal.json',dict(exit_code=0))
    monkeypatch.setattr(m,'checkpoint_identity',lambda path:('a',{}))
    return gate,put


def test_recompute_entire_prefix(tmp_path,monkeypatch):
    gate,_=fixture(tmp_path,monkeypatch)
    report=m.verify(gate)
    assert report['requests']==256 and report['maximum_logprob_error']==0


@pytest.mark.parametrize('fault',['forged_pass','missing','request','durability','live'])
def test_false_pass_or_missing_evidence_is_rejected(tmp_path,monkeypatch,fault):
    gate,put=fixture(tmp_path,monkeypatch)
    if fault in ('forged_pass','missing','request'):
        p=gate/'run/vllm/result.json';data=json.loads(p.read_text())
        if fault=='missing':data['records'].pop()
        if fault=='request':data['records'][0]['request_id']='foreign'
        if fault=='forged_pass':data['records'][0]['candidate_logprobs']=[[math.log(.5)]+[math.log(.5/59)]*59]
    elif fault=='durability':
        p=gate/'state-verified.json';data=json.loads(p.read_text());data['durable_commit_verified']=False
    else:
        p=gate/'observed-final.json';data=json.loads(p.read_text());data['state']['Status']='running'
    put(p,data)
    with pytest.raises(ValueError):m.verify(gate)


def selection_fixture(root,monkeypatch):
    gate,put=fixture(root,monkeypatch,count=9226)
    train=root/'train'
    c=dict(task_mode='direct_final',total_updates=634,base_id='b',runtime_profile='fixed',release='/release',release_sha256='release-sha')
    put(train/'config.json',c)
    launch=json.loads((train/'launch.json').read_text());launch['config_sha256']=file_hash(train/'config.json');put(train/'launch.json',launch)
    p=gate/'state-verified.json';state=json.loads(p.read_text());state.update(updates=634,epoch=2,next_batch=0);put(p,state)
    p=gate/'inputs.json';artifact=json.loads(p.read_text());artifact['population_mode']='full_selection';put(p,artifact)
    p=gate/'config.json';c=json.loads(p.read_text());c.update(runtime_profile='fixed',checkpoint_evidence=str(gate/'state-verified.json'),training_observed=str(gate/'training-observed.json'))
    c['pins'].update({str(q):file_hash(q) for q in [gate/'inputs.json',gate/'state-verified.json',gate/'training-observed.json']});put(p,c)
    launch=json.loads((gate/'launch.json').read_text());launch['config_sha256']=file_hash(p);put(gate/'launch.json',launch)
    import multimodal_jev.slurp_official_training as released
    import multimodal_jev.simple_selection_inputs as selection
    def load(path,digest,role,*,purpose):
        assert (path,digest,role,purpose)==('/release','release-sha','selection','evaluate')
        return list(range(9226)),{},{}
    def verify_inputs(artifact,rows,task,digest):
        # Full real compiler/input binding is separately covered in
        # test_simple_selection_inputs; this fixture tests terminal provenance.
        assert len(rows)==9226 and digest=='release-sha'
        assert artifact['population_mode']=='full_selection'
    monkeypatch.setattr(released,'load',load);monkeypatch.setattr(selection,'verify',verify_inputs)
    return gate,put


def test_complete_selection_requires_terminal_state_and_recomputes_every_request(tmp_path,monkeypatch):
    gate,_=selection_fixture(tmp_path,monkeypatch)
    report=m.verify(gate)
    assert report['requests']==9226 and report['population_mode']=='full_selection'
    assert report['updates']==634 and report['maximum_logprob_error']==0


@pytest.mark.parametrize('fault',['nonterminal_state','wrong_profile','partial','unpinned_state'])
def test_reject_selection_without_exact_terminal_and_complete_probabilities(tmp_path,monkeypatch,fault):
    gate,put=selection_fixture(tmp_path,monkeypatch)
    c=json.loads((gate/'config.json').read_text())
    if fault=='partial':
        p=gate/'run/vllm/result.json';data=json.loads(p.read_text());data['records'].pop();put(p,data)
    elif fault=='wrong_profile':c['runtime_profile']='other'
    elif fault=='unpinned_state':c['pins'].pop(c['checkpoint_evidence'])
    else:
        p=gate/'state-verified.json';s=json.loads(p.read_text());s['updates']=633;put(p,s);c['pins'][str(p)]=file_hash(p)
    put(gate/'config.json',c)
    launch=json.loads((gate/'launch.json').read_text());launch['config_sha256']=file_hash(gate/'config.json');put(gate/'launch.json',launch)
    with pytest.raises(ValueError):m.verify(gate)


def sampled_fixture(root,monkeypatch):
    gate,put=fixture(root,monkeypatch);train=root/'train'
    c=dict(task_mode='direct_final',base_id='b',runtime_profile='fixed',sampled_probe={'algorithm':'grpo_v1'})
    put(train/'config.json',c)
    launch=json.loads((train/'launch.json').read_text());launch['config_sha256']=file_hash(train/'config.json');put(train/'launch.json',launch)
    outputs=json.loads((gate/'run/hf/result.json').read_text())['records'];sampling=[];resident=[]
    for i,row in enumerate(outputs[:128]):
        prior=dict(row['request_identity'],policy_id='b:old')
        rid=hashlib.sha256(json.dumps(prior,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        uid=f'{rid}::update0-slot{i}'
        sampling.append(dict(request_id=rid,actor_request_id=uid,policy_id='b:old',update=0,slot=i))
        resident.append(dict(request_id=uid,candidate_logprobs=row['candidate_logprobs']))
    path=train/'run/rl-inputs.json';put(path,dict(sampling={'sampling':sampling,'policy_id':'b:old','reference_policy_id':'b:old','update':0}));files={str(path):file_hash(path)}
    for rank in range(4):
        path=train/'run'/f'rl-rank-{rank}/after.json';put(path,dict(records=resident[rank*32:(rank+1)*32]));files[str(path)]=file_hash(path)
    p=gate/'state-verified.json';state=json.loads(p.read_text());state.update(updates=1,algorithm='grpo_v1',reference_policy_id='b:old',supervision_records_verified=128,files=files);put(p,state)
    p=gate/'config.json';c=json.loads(p.read_text());c.update(runtime_profile='fixed',checkpoint_evidence=str(gate/'state-verified.json'),training_observed=str(gate/'training-observed.json'))
    c['pins'].update({str(q):file_hash(q) for q in [gate/'state-verified.json',gate/'training-observed.json']});put(p,c)
    launch=json.loads((gate/'launch.json').read_text());launch['config_sha256']=file_hash(p);put(gate/'launch.json',launch)
    return gate,put


def test_sampled_update_requires_resident_and_fresh_backends(tmp_path,monkeypatch):
    gate,_=sampled_fixture(tmp_path,monkeypatch)
    report=m.verify(gate)
    assert report['updates']==1
    assert all(r['requests']==128 and r['passed'] for r in report['sampled_resident_parity'].values())


def test_second_update_uses_behavior_identity_not_fixed_reference(tmp_path,monkeypatch):
    gate,put=sampled_fixture(tmp_path,monkeypatch);train=tmp_path/'train'
    p=train/'config.json';c=json.loads(p.read_text())
    c['sampled_probe']['fixture_initialization']='restore_updated_rl_then_second_update';put(p,c)
    p=train/'launch.json';v=json.loads(p.read_text());v['config_sha256']=file_hash(train/'config.json');put(p,v)
    p=train/'run/rl-inputs.json';v=json.loads(p.read_text());v['sampling'].update(update=1,reference_policy_id='b:fixed')
    for r in v['sampling']['sampling']:
        r['update']=1;r['actor_request_id']=r['actor_request_id'].replace('::update0-', '::update1-')
    put(p,v)
    p=gate/'state-verified.json';state=json.loads(p.read_text())
    state.update(updates=2,behavior_policy_id='b:old',reference_policy_id='b:fixed')
    for rank in range(4):
        q=train/'run'/f'rl-rank-{rank}/after.json';v=json.loads(q.read_text())
        for r in v['records']:r['request_id']=r['request_id'].replace('::update0-', '::update1-')
        put(q,v)
    state['files']={k:file_hash(k) for k in state['files']};put(p,state)
    c=json.loads((gate/'config.json').read_text());c['pins'][str(p)]=file_hash(p);put(gate/'config.json',c)
    p=gate/'launch.json';v=json.loads(p.read_text());v['config_sha256']=file_hash(gate/'config.json');put(p,v)
    report=m.verify(gate)
    assert report['updates']==2 and report['sampled_resident_parity']['vllm']['passed']


@pytest.mark.parametrize('fault',['missing_pin','resident_drift','wrong_algorithm','wrong_cursor'])
def test_reject_false_sampled_backend_acceptance(tmp_path,monkeypatch,fault):
    gate,put=sampled_fixture(tmp_path,monkeypatch)
    p=gate/'state-verified.json';state=json.loads(p.read_text())
    if fault=='missing_pin':state['files'].pop(next(iter(state['files'])))
    elif fault=='resident_drift':
        q=tmp_path/'train/run/rl-rank-0/after.json';data=json.loads(q.read_text())
        data['records'][0]['candidate_logprobs']=[[math.log(.5)]+[math.log(.5/59)]*59]
        put(q,data);state['files'][str(q)]=file_hash(q)
    elif fault=='wrong_algorithm':state['algorithm']='rloo_v1'
    else:state['updates']=2
    put(p,state)
    c=json.loads((gate/'config.json').read_text());c['pins'][str(p)]=file_hash(p);put(gate/'config.json',c)
    launch=json.loads((gate/'launch.json').read_text());launch['config_sha256']=file_hash(gate/'config.json');put(gate/'launch.json',launch)
    with pytest.raises(ValueError):m.verify(gate)
