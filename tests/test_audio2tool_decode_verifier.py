import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
import verify_audio2tool_decode as verifier


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.mark.parametrize('fault',[None,'running','failed','oom','devices','memory','command','mount','environment','identity'])
def test_exact_preflight_container(fault):
    argv=['docker','create','-v','source:/source:ro','-e','FIXED=true','image','python','profile','--preflight-only']
    item=dict(Id=verifier.PREFLIGHT_ID,Image='image',State=dict(Status='exited',ExitCode=0,OOMKilled=False),
              Config=dict(Cmd=argv[argv.index('image')+1:],Env=['FIXED=true']),
              HostConfig=dict(NanoCpus=2*10**9,Memory=20*1024**3,NetworkMode='none',Devices=[],Privileged=False,Binds=['source:/source:ro']))
    if fault=='running':item['State']['Status']='running'
    if fault=='failed':item['State']['ExitCode']=1
    if fault=='oom':item['State']['OOMKilled']=True
    if fault=='devices':item['HostConfig']['Devices']=['npu']
    if fault=='memory':item['HostConfig']['Memory']=0
    if fault=='command':item['Config']['Cmd']=[]
    if fault=='mount':item['HostConfig']['Binds']=[]
    if fault=='environment':item['Config']['Env']=[]
    if fault=='identity':item['Id']='other'
    if fault:
        with pytest.raises(ValueError):verifier.verify_preflight(item,dict(argv=argv,image='image'))
    else:verifier.verify_preflight(item,dict(argv=argv,image='image'))


@pytest.mark.parametrize('fault',[None,'command','intent_hash','devices','capacity_hash','source'])
def test_intent_only_appends_authenticated_capacity(tmp_path,fault):
    save(tmp_path/'capacity-evidence.json',{'fixture':True})
    prep=dict(argv=['image','profile'],image='image',source_commit='commit')
    intent=dict(prep,argv=prep['argv']+['--capacity-evidence',str(tmp_path/'capacity-evidence.json'),
                                     '--capacity-evidence-hash',verifier.file_hash(tmp_path/'capacity-evidence.json')])
    if fault=='command':intent['argv'].append('--preflight-only')
    save(tmp_path/'intent.json',intent)
    launch=dict(intent_sha256=verifier.file_hash(tmp_path/'intent.json'),image='image',source_commit='commit',physical_devices=[6,7])
    if fault=='intent_hash':launch['intent_sha256']='wrong'
    if fault=='devices':launch['physical_devices']=[4,5]
    if fault=='capacity_hash':save(tmp_path/'capacity-evidence.json',{'changed':True})
    if fault=='source':launch['source_commit']='other'
    if fault:
        with pytest.raises(ValueError):verifier.check_intent(tmp_path,prep,intent,launch)
    else:verifier.check_intent(tmp_path,prep,intent,launch)


@pytest.mark.parametrize('fault',[None,'native','jev','saved_identity','changed_result','wrong_root','inspection'])
def test_both_capacity_admissions_and_runtime_binding(tmp_path,monkeypatch,fault):
    root=tmp_path/'profile';root.mkdir();proof={};snapshots={};saved={};calls=[]
    for family in ('native','jev'):
        prior=tmp_path/('audio2tool-engine-capacity-v1' if family=='native' else 'audio2tool-jev-engine-capacity-v1')
        cid=('a' if family=='native' else 'b')*64;save(prior/'launch.json',dict(container_id=cid))
        path=root/(family+'-capacity-inspection.json');save(path,dict(Id=cid))
        snapshots[family]=dict(Id=cid,fresh=True)
        proof[family]=dict(root=str(prior),container_id=cid,inspection=str(path),inspection_sha256=verifier.file_hash(path),intent_sha256='intent')
        saved[family]=dict(container_id=cid,result_sha256='result',inspection_sha256=verifier.file_hash(path))
    save(root/'run/identity.json',dict(capacity=saved))
    if fault=='native':snapshots['native']['Id']='other'
    if fault=='jev':snapshots['jev']['Id']='other'
    if fault=='wrong_root':proof['native']['root']='other'
    if fault=='inspection':save(root/'native-capacity-inspection.json',{'changed':True})
    if fault=='saved_identity':save(root/'run/identity.json',dict(capacity={}))
    save(root/'capacity-evidence.json',proof)
    registry=tmp_path/'registry';registry.write_bytes(b'fixture')
    def admit(prior,path,pin,cid):
        item=json.loads(Path(path).read_text());calls.append((cid,item.get('fresh',False)))
        assert verifier.file_hash(path)==pin
        if item['Id']!=cid:raise ValueError('actual capacity container differs')
        return dict(container_id=cid,inspection_sha256=pin,result_sha256='changed' if fault=='changed_result' and item.get('fresh') else 'result')
    monkeypatch.setattr(verifier,'admit_native',lambda root,path,pin,cid,*a:admit(root,path,pin,cid))
    monkeypatch.setattr(verifier,'admit_jev',lambda root,intent,path,pin,cid,*a:admit(root,path,pin,cid))
    prep=dict(argv=['--registry',str(registry),'--release','release'])
    if fault:
        with pytest.raises(ValueError):verifier.capacities(root,snapshots,prep)
    else:
        result=verifier.capacities(root,snapshots,prep)
        assert set(result)=={'native','jev'} and len(calls)==4


def test_evidence_and_preparation_pins_precede_artifact_access(tmp_path):
    evidence=tmp_path/'evidence.json';save(evidence,{})
    with pytest.raises(ValueError,match='externally pinned'):
        verifier.envelope(tmp_path,evidence,'wrong')
