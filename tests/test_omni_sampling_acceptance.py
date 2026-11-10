import json
from pathlib import Path
import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.omni_rollout_records import sampling_kwargs
from multimodal_jev.omni_sampling_acceptance import audit_records, read_json
from multimodal_jev.sfc_reward_batch import digest


class Tokenizer:
    def __len__(self):return 100
    def decode(self,ids,skip_special_tokens):return 'false' if ids==[3,4] else 'INVALID'


def fixture(tmp_path):
    run=tmp_path/'run';(run/'requests').mkdir(parents=True)
    rows=[dict(id=str(i),source_group=str(i%4),split='train_id',
        input={'decisions':[{'id':str(d)} for d in range(13)]}) for i in range(6)]
    checks={(r['id'],d['id']):dict(input_ids_sha256=digest([1,2]),prompt_sha256=digest([r['id'],d['id']]))
        for r in rows for d in r['input']['decisions']}
    sampling=sampling_kwargs(2,17);sampling.update(stop=[],stop_token_ids=[])
    write=lambda p,v:p.write_text(json.dumps(v))
    records=[]
    for row in rows:
        for d in row['input']['decisions']:
            key=row['id'],d['id'];name=digest(key);out=run/'requests';rr=[]
            for rep in (0,1):
                rr.append(dict(example_id=row['id'],decision_id=d['id'],request_sha256=digest(row),
                    replicate=rep,policy_sha256='a'*64,instruction_sha256='b'*64,**checks[key],
                    raw_text='false',token_ids=[3,4],log_probs=[-.3,-.4],finish_reason='stop'))
            write(out/(name+'.intent.json'),dict(example_id=key[0],decision_id=key[1],policy_sha256='a'*64,sampling=sampling,started_unix=1.))
            write(out/(name+'.raw.json'),dict(outputs=[dict(finished=True,prompt_token_ids=[1,2],outputs=[
                dict(index=rep,text='false',token_ids=[3,4],finish_reason='stop',logprobs=[{'3':-.3},{'4':-.4}]) for rep in (1,0)])]))
            write(out/(name+'.result.json'),dict(records=rr,input_ids=[1,2],seconds=1.))
            records.extend(rr)
    (run/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    write(run/'identity.json',{'synthetic':True});write(run/'terminal.json',{'exit_code':0})
    write(run/'result.json',dict(status='passed',model_loaded=True,weight_updates=0,bundles=6,decisions=78,responses=156,
        records_sha256=file_hash(run/'records.jsonl'),identity_sha256=file_hash(run/'identity.json')))
    args=dict(policy_sha256='a'*64,instruction_sha256='b'*64,sampling=sampling)
    return run,rows,checks,args


def test_independent_audit_accepts_full_raw_roster_including_reverse_completion(tmp_path):
    run,rows,checks,args=fixture(tmp_path)
    report=audit_records(run,rows,checks,Tokenizer(),**args)
    assert report['responses']==156 and report['response_tokens']==312 and report['request_seconds']==78


@pytest.mark.parametrize('change',['missing_file','extra_file','prompt','probability','missing_logprob','nan',
    'duplicate_rep','finish','text','tokens','converted','merged','cost','policy','sampling','input',
    'summary','terminal','roster','duplicate_json'])
def test_artifact_corruption_is_rejected_even_with_passed_summary(tmp_path,change):
    run,rows,checks,args=fixture(tmp_path);name=digest(('0','0'));folder=run/'requests'
    kind='raw';path=folder/(name+'.raw.json');obj=read_json(path);answer=obj['outputs'][0]['outputs'][0]
    if change=='missing_file':path.unlink()
    elif change=='extra_file':(folder/'unknown.json').write_text('{}')
    elif change=='prompt':obj['outputs'][0]['prompt_token_ids']=[2,1]
    elif change=='probability':answer['logprobs'][0]['3']=-.8
    elif change=='missing_logprob':answer['logprobs'][0]={}
    elif change=='nan':answer['logprobs'][0]['3']=float('nan')
    elif change=='duplicate_rep':answer['index']=0
    elif change=='finish':answer['finish_reason']='abort'
    elif change=='text':answer['text']='repaired'
    elif change=='tokens':answer['token_ids']=[3]
    elif change in ('converted','cost','input'):
        kind='result';path=folder/(name+'.result.json');obj=read_json(path)
        if change=='converted':obj['records'][0]['log_probs'][0]=-.9
        if change=='cost':obj['seconds']=-1
        if change=='input':obj['input_ids']=[2,1]
    elif change in ('policy','sampling'):
        kind='intent';path=folder/(name+'.intent.json');obj=read_json(path)
        if change=='policy':obj['policy_sha256']='c'*64
        if change=='sampling':obj['sampling']['temperature']=.7
    elif change=='merged':
        path=run/'records.jsonl';lines=path.read_text().splitlines();path.write_text('\n'.join(lines[:-1])+'\n')
        summary=read_json(run/'result.json');summary['records_sha256']=file_hash(path)
        (run/'result.json').write_text(json.dumps(summary))
    elif change=='summary':path=run/'result.json';obj=read_json(path);obj['responses']=155
    elif change=='terminal':path=run/'terminal.json';obj={'exit_code':1}
    elif change=='roster':rows.pop()
    elif change=='duplicate_json':path.write_text('{"outputs":[],"outputs":[]}')
    if change not in ('missing_file','extra_file','merged','roster','duplicate_json'):
        path.write_text(json.dumps(obj))
    with pytest.raises(ValueError):audit_records(run,rows,checks,Tokenizer(),**args)


@pytest.mark.parametrize('change',[None,'running','exit','image','oom','cid','config','devices','privileged'])
def test_container_gate_checks_live_docker_identity_and_device_mapping(tmp_path,monkeypatch,change):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'tools/jev'))
    from test_sfc_jev import load_tool
    tool=load_tool('verify_sfc_omni_sampling')
    (tmp_path/'config.json').write_text('{}')
    plan=dict(image='image',source_commit='commit',config_sha256=file_hash(tmp_path/'config.json'))
    launch=dict(container_id='a'*64,source_commit='commit',physical_devices=[6,7])
    state=dict(Id='a'*64,Image='image',State=dict(Status='exited',ExitCode=0,OOMKilled=False),
               HostConfig=dict(Privileged=False,Devices=[{'PathOnHost':'/dev/davinci6'},{'PathOnHost':'/dev/davinci7'}]))
    if change=='running':state['State']['Status']='running'
    if change=='exit':state['State']['ExitCode']=1
    if change=='image':state['Image']='other'
    if change=='oom':state['State']['OOMKilled']=True
    if change=='cid':state['Id']='b'*64
    if change=='config':(tmp_path/'config.json').write_text('{"changed":true}')
    if change=='devices':state['HostConfig']['Devices'].append({'PathOnHost':'/dev/davinci0'})
    if change=='privileged':state['HostConfig']['Privileged']=True
    monkeypatch.setattr(tool,'inspect',lambda cid:state)
    if change is None:assert tool.verify_container(tmp_path,plan,launch)==state
    else:
        with pytest.raises(ValueError):tool.verify_container(tmp_path,plan,launch)
