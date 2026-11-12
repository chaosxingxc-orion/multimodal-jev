import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import verify_span_resident_policy_suite as module
from test_intent_gate_identity import fixture,rehash
from multimodal_jev.baseline import atomic_json


def case():
    artifact,expected,rows=fixture()
    cfg=dict(base_id='base',adapter_id='adapter')
    hf=dict(passed=True,backend='hf',base_id='base',adapter_id='adapter',records=rows)
    vl=copy.deepcopy(hf);vl.update(backend='vllm',slot=1,worker=[dict(registered_adapter_ids=[1],
        lora_dtype='torch.bfloat16',head_projection_dtype='torch.bfloat16',actual_lora_buffer_dtypes={'layer':['torch.bfloat16']})])
    return cfg,artifact,expected,hf,vl


def test_full_task_and_joint_candidate_recomputation():
    c,a,e,h,v=case();r=module.verify_policy(c,a,e,h,v,1)
    assert r['passed'] and r['requests']==2 and r['joint_ratio_range']==[1.,1.]


@pytest.mark.parametrize('bad',['both_wrong_task','both_wrong_audio','both_stale_policy','nan','missing','slot','dtype','joint_ratio'])
def test_agreement_alone_is_not_acceptance(bad):
    c,a,e,h,v=case()
    if bad.startswith('both_'):
        for report in (h,v):
            row=report['records'][1];i=row['request_identity']
            if bad=='both_wrong_task':i['request']['questions']=['wrong','final']
            elif bad=='both_wrong_audio':i['audio_processing']={'waveform':'wrong'}
            else:i['policy_id']='base:stale'
            rehash(row)
    elif bad=='nan':v['records'][0]['candidate_logprobs'][0][0]=float('nan')
    elif bad=='missing':v['records'].pop()
    elif bad=='slot':v['worker'][0]['registered_adapter_ids']=[2]
    elif bad=='dtype':v['worker'][0]['actual_lora_buffer_dtypes']['layer']=['torch.float32']
    else:
        import math
        # Every marginal log-probability error <.02, joint ratio exceeds1.02.
        v['records'][1]['candidate_logprobs']=[[math.log(.506),math.log(.494)]]*2
    with pytest.raises(ValueError):module.verify_policy(c,a,e,h,v,1)


def test_running_container_cannot_be_accepted_from_ready_files(tmp_path,monkeypatch):
    receipt=dict(root=str(tmp_path),image='fixed',physical_devices=[4],argv=['create'],launch_exit_code=0,container_id='cid')
    atomic_json(tmp_path/'launch.json',receipt)
    atomic_json(tmp_path/'prepared.json',dict(image='fixed',physical_devices=[4],create_argv=['create']))
    monkeypatch.setattr(module,'completed',lambda *a:False)
    with pytest.raises(ValueError,match='successful resident'):module.verify(tmp_path,'pilot','manifest','fixed')


def full_fixture(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace as NS
    from multimodal_jev.baseline import file_hash
    root=tmp_path/'suite';root.mkdir();(root/'run').mkdir();(root/'source').mkdir()
    atomic_json(root/'source-manifest.json',{})
    artifacts=[];records=[];by_arm={a:[] for a in ['D','Q']}
    c,a,e,h,v=case()
    for arm,index in [('D',0),('Q',1)]:
        for n in range(89):
            uid=f'audio{n}::{arm}';source=copy.deepcopy(a['records'][index]);source['id']=uid
            row=copy.deepcopy(h['records'][index]);row['id']=uid;rehash(row)
            artifacts.append(source);records.append(row)
            by_arm[arm].append({'id':f'audio{n}','request':NS(model_input=lambda r=source['request']:r)})
    artifact=tmp_path/'inputs.json';atomic_json(artifact,dict(passed=True,requests=178,recordings=89,records=artifacts))
    monkeypatch.setattr(module,'load_pilot',lambda p,m,a:(by_arm[a],{}))
    monkeypatch.setattr(module,'checkpoint_identity',lambda p:(p,{}))
    monkeypatch.setattr(module,'completed',lambda *a:True)
    policies=[];summary=[];files={};pins={}
    for key in ['first','second']:
        cfg=dict(model='base',base_id='base',adapter='adapter',adapter_id='adapter',runtime_profile='profile',max_model_len=4096,
            physical_devices=[4],population=dict(recordings=89,requests=178),inputs=str(artifact),pins={str(artifact):file_hash(artifact)})
        original=tmp_path/f'{key}.json';atomic_json(original,cfg);pins[str(original)]=file_hash(original)
        policies.append(dict(key=key,config=str(original)));sub=root/'run'/key
        for d in [sub,sub/'run',sub/'run/hf',sub/'run/vllm']:d.mkdir()
        atomic_json(sub/'config.json',cfg)
        hf=dict(h,records=copy.deepcopy(records));vl=dict(v,records=copy.deepcopy(records))
        atomic_json(sub/'run/hf/result.json',hf);atomic_json(sub/'run/vllm/result.json',vl)
        atomic_json(sub/'run/vllm/worker.json',vl['worker'])
        for name in ['terminal.json','hf/terminal.json','vllm/terminal.json']:atomic_json(sub/'run'/name,{'exit_code':0})
        for name in ['hf/runtime-profile.json','vllm/runtime-profile.json']:atomic_json(sub/'run'/name,{'profile':'profile'})
        atomic_json(sub/'run/READY.json',dict(passed=True,requests=178,base_id='base',adapter_id='adapter',result_sha256=file_hash(sub/'run/vllm/result.json')))
        for name in ['config.json','run/READY.json','run/terminal.json','run/hf/result.json','run/hf/terminal.json','run/vllm/result.json','run/vllm/terminal.json']:
            files[str(sub/name)]=file_hash(sub/name)
        summary.append(dict(key=key,adapter_id='adapter',slot=1,requests=178,maximum_logprob_error=0.))
    atomic_json(root/'config.json',dict(schema='jev-resident-policy-suite-v1',policies=policies,pins=pins))
    atomic_json(root/'run/result.json',dict(passed=True,policies=summary,files=files))
    atomic_json(root/'run/terminal.json',{'exit_code':0});atomic_json(root/'run/runtime-profile.json',{'profile':'profile'})
    identity=dict(image='fixed',physical_devices=[4],config_sha256=file_hash(root/'config.json'),source_manifest_sha256=file_hash(root/'source-manifest.json'))
    atomic_json(root/'prepared.json',dict(identity,create_argv=['create']))
    atomic_json(root/'launch.json',dict(identity,root=str(root),argv=['create'],launch_exit_code=0,container_id='cid'))
    return root


def test_complete_host_directory_verification(tmp_path,monkeypatch):
    root=full_fixture(tmp_path,monkeypatch)
    result=module.verify(root,'pilot','manifest','fixed')
    assert result['passed'] and [p['requests'] for p in result['policies']]==[178,178]


@pytest.mark.parametrize('bad',['missing_pin','summary','runtime','worker','terminal'])
def test_host_rejects_incomplete_or_forged_artifacts(tmp_path,monkeypatch,bad):
    root=full_fixture(tmp_path,monkeypatch);sub=root/'run/first'
    if bad in ['missing_pin','summary']:
        p=root/'run/result.json';r=module.read(p)
        if bad=='missing_pin':r['files'].pop(str(sub/'run/hf/result.json'))
        else:r['policies'][0]['maximum_logprob_error']=.001
        atomic_json(p,r)
    elif bad=='runtime':atomic_json(sub/'run/hf/runtime-profile.json',{'profile':'wrong'})
    elif bad=='worker':atomic_json(sub/'run/vllm/worker.json',[])
    else:atomic_json(root/'run/terminal.json',{'exit_code':1})
    with pytest.raises(ValueError):module.verify(root,'pilot','manifest','fixed')
