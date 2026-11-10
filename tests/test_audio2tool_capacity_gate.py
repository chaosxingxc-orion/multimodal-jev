"""Synthetic envelope controls; never evidence of a real successful NPU run."""
import copy
import json

import pytest

from multimodal_jev import audio2tool_engine as engine
from multimodal_jev import audio2tool_inference as api


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'capacity';run=root/'run';run.mkdir(parents=True)
    source=tmp_path/'source';source.mkdir();(source/'runner.py').write_text('fixture')
    sha=api.file_hash(source/'runner.py');cid='a'*64;image='sha256:'+'b'*64
    argv=['docker','create','--device','/dev/davinci6:/dev/davinci6','--device','/dev/davinci7:/dev/davinci7',
          '-e','OFFLINE=1','-v','/source:/source:ro',image,'timeout','3600s','python','/source/runner.py']
    plan=dict(argv=argv,image=image,source_root=str(source),source_files={'runner.py':sha})
    item=dict(Id=cid,Image=image,Name='/audio2tool-engine-capacity-v1',
        State=dict(Status='exited',ExitCode=0,OOMKilled=False,Running=False,Restarting=False,Dead=False,Error=''),
        HostConfig=dict(NanoCpus=16*10**9,Memory=128*1024**3,ShmSize=8*1024**3,NetworkMode='host',Privileged=False,
            Devices=[{'PathOnHost':'/dev/davinci6'},{'PathOnHost':'/dev/davinci7'}],Binds=['/source:/source:ro']),
        Config=dict(Cmd=argv[argv.index(image)+1:],Env=['OFFLINE=1']))
    prior={k:dict(id=k,input_tokens=10,input_ids_sha256='ids',audio_receipts=[{}]) for k in ('a','b','c','d')}
    records=[dict(selection=k,**{a:v for a,v in p.items() if a!='audio_receipts'},audio_count=1,
                  input_matches_processor=True,token_ids=[3],finish_reason='length',engine_seconds=.1) for k,p in prior.items()]
    identity=dict(preflight_only=False,versions=engine.VERSIONS,engine=engine.ENGINE,
        sampling=dict(temperature=0.,max_tokens=1,n=1,seed=17),source_files={'runner.py':sha})
    def write(p,r):p.write_text(json.dumps(r))
    write(root/'preparation.json',plan);write(root/'inspection.json',item);write(run/'identity.json',identity)
    write(run/'terminal.json',{'exit_code':0})
    result=dict(status='passed',model_loaded=True,weight_updates=0,annotations_read=False,
        identity_sha256=api.file_hash(run/'identity.json'),records=records)
    write(run/'result.json',result)
    for r in records:write(run/(r['selection']+'.json'),r)
    # Only fixed plan identity and actual processor dependency are replaced.
    # Container/result/source checks execute normally on explicit fake data.
    original=api.file_hash
    monkeypatch.setattr(api,'file_hash',lambda p:api.CAPACITY_PREPARATION_SHA if p==root/'preparation.json' else original(p))
    monkeypatch.setattr(engine,'prior_records',lambda _: (prior,{}))
    return root,item,identity,result,cid


@pytest.mark.parametrize('mutation',[None,'running','failed','oom','wrong_id','wrong_image','command','device','cpu_preflight','no_model','missing_case','source'])
def test_capacity_requires_real_success_envelope(tmp_path,monkeypatch,mutation):
    root,item,identity,result,cid=fixture(tmp_path,monkeypatch);run=root/'run'
    if mutation=='running':item['State']['Status']='running';item['State']['Running']=True
    if mutation=='failed':item['State']['ExitCode']=1
    if mutation=='oom':item['State']['OOMKilled']=True
    if mutation=='wrong_id':item['Id']='c'*64
    if mutation=='wrong_image':item['Image']='sha256:'+'c'*64
    if mutation=='command':item['Config']['Cmd']+=['--preflight-only']
    if mutation=='device':item['HostConfig']['Devices'][0]['PathOnHost']='/dev/davinci0'
    if mutation=='cpu_preflight':identity['preflight_only']=True
    if mutation=='no_model':result['model_loaded']=False
    if mutation=='missing_case':result['records'].pop()
    if mutation=='source':(tmp_path/'source/runner.py').write_text('modified')
    (root/'inspection.json').write_text(json.dumps(item));(run/'identity.json').write_text(json.dumps(identity))
    result['identity_sha256']=api.file_hash(run/'identity.json');(run/'result.json').write_text(json.dumps(result))
    args=(root,root/'inspection.json',api.file_hash(root/'inspection.json'),cid,tmp_path/'processor')
    if mutation is None:assert api.require_capacity(*args)['container_id']==cid
    else:
        with pytest.raises(ValueError):api.require_capacity(*args)
