import json
from pathlib import Path

import pytest

from multimodal_jev import audio2tool_jev_capacity as gate
from multimodal_jev.audio2tool_acceptance import IMAGE
from multimodal_jev.baseline import file_hash


def fixture(tmp_path, monkeypatch):
    root=tmp_path/'capacity';root.mkdir();run=root/'run';run.mkdir();source=root/'source';source.mkdir()
    put=lambda p,v:p.write_text(json.dumps(v))
    script=source/gate.RUNNER;script.parent.mkdir(parents=True);script.write_text('# synthetic envelope fixture')
    pins={gate.RUNNER:file_hash(script)}
    preflight=tmp_path/'preflight';preflight.mkdir();put(preflight/'intent.json',dict(source_files=pins))
    monkeypatch.setattr(gate,'PREFLIGHT',preflight)
    monkeypatch.setattr(gate,'file_hash',lambda p:'f5a3c9d312dea0c6a893af70c9074727d2f7ebcfb01c9bb2ab7a1b6b36022727'
                        if Path(p)==preflight/'intent.json' else file_hash(p))
    release=tmp_path/'release';context=tmp_path/'context';cid='a'*64
    cmd=['timeout','--signal=TERM','--kill-after=30s','3600s','python3','-B','/source/'+gate.RUNNER,
         '--root','/mnt/cephfs/xingchao/datasets/audio2tool','--model',gate.MODEL,'--release',str(release),
         '--registry','/mnt/cephfs/xingchao/datasets/audio2tool/tools_registry.csv','--asset-audit',
         '/mnt/cephfs/xingchao/experiments/multimodal-jev/sfc-qwen3-omni-train6-v1/asset-audit.json',
         '--context-run',str(context),'--native-processor-run',
         '/mnt/cephfs/xingchao/experiments/multimodal-jev/audio2tool-native-processor-v1/run','--output','/output/run']
    binds=[f'{source}:/source:ro',f'{root}:/output:rw'];devices=['/dev/davinci6','/dev/davinci7']
    argv=['docker','create']
    for v in binds:argv+=['-v',v]
    for v in devices:argv+=['--device',v]
    argv += [IMAGE,*cmd]
    plan=dict(argv=argv,image=IMAGE,automatic_retry=False,physical_devices=[6,7],wall_seconds=3600,weight_updates=0,
              source_root=str(source),source_commit='fixture',source_files=pins)
    put(root/'intent.json',plan);intent_hash=file_hash(root/'intent.json')
    put(root/'launch.json',dict(container_id=cid,intent_sha256=intent_hash,source_commit='fixture',image=IMAGE))
    inspect=dict(Id=cid,Image=IMAGE,State=dict(Status='exited',ExitCode=0,OOMKilled=False,Running=False,Restarting=False,Dead=False,Error=''),
        Config=dict(Cmd=cmd,Env=[]),HostConfig=dict(Devices=[dict(PathOnHost=d) for d in devices],NanoCpus=16*10**9,
        Memory=128*1024**3,ShmSize=8*1024**3,NetworkMode='host',Privileged=False,Binds=binds))
    inspection=tmp_path/'inspection.json';put(inspection,inspect)
    identity=dict(preflight_only=False,targets_read=False,weight_updates=0,engine=gate.ENGINE,versions=gate.VERSIONS,
        model=gate.MODEL,asset_audit_sha256=gate.ASSET_AUDIT_SHA,context_result_sha256=gate.CONTEXT_SHA,
        manifest_sha256=gate.MANIFEST_SHA,sampling=dict(temperature=0.,max_tokens=1,n=1,seed=17),
        full_roster_records=30733,boundary_cases=12,sources=pins)
    put(run/'identity.json',identity);put(run/'terminal.json',dict(exit_code=0))
    cases=[];records=[]
    for i in range(12):
        case=dict(case='case'+str(i),id='id'+str(i),decision_id='d'+str(i),input_tokens=123,
                  input_ids_sha256='b'*64,audio_receipts=[{}]);cases.append(({},case))
        record={k:case[k] for k in ['case','id','decision_id','input_tokens','input_ids_sha256']}
        record.update(audio_count=1,input_matches_processor=True,token_ids=[7],raw_text='token',finish_reason='length',engine_seconds=.1)
        put(run/(case['case']+'.json'),record);records.append(record)
    put(run/'result.json',dict(status='passed',model_loaded=True,targets_read=False,weight_updates=0,
                              identity_sha256=file_hash(run/'identity.json'),records=records))
    # Real boundary source and waveform checks are separately verified; this is
    # a synthetic success/failure envelope test, never NPU evidence.
    monkeypatch.setattr(gate,'boundaries',lambda *a:([],{},cases))
    args=[root,intent_hash,inspection,file_hash(inspection),cid,context,release,b'fixture']
    return args,put


def test_registered_successful_twelve_case_envelope(tmp_path,monkeypatch):
    args,_=fixture(tmp_path,monkeypatch)
    assert gate.require_capacity(*args)['container_id']=='a'*64


@pytest.mark.parametrize('fault',['running','failed','oom','device','command','resources','preflight','missing_case',
                                  'token_ids','identity','source','wrong_intent','wrong_inspection'])
def test_invalid_capacity_cannot_admit_full_generation(tmp_path,monkeypatch,fault):
    args,put=fixture(tmp_path,monkeypatch);root=args[0];run=root/'run';inspection=args[2]
    if fault in ('running','failed','oom','device','command','resources'):
        x=json.loads(inspection.read_text())
        if fault=='running':x['State'].update(Status='running',Running=True)
        if fault=='failed':x['State']['ExitCode']=1
        if fault=='oom':x['State']['OOMKilled']=True
        if fault=='device':x['HostConfig']['Devices'][0]['PathOnHost']='/dev/davinci4'
        if fault=='command':x['Config']['Cmd'][-1]='/output/other'
        if fault=='resources':x['HostConfig']['Memory']=1024
        put(inspection,x);args[3]=file_hash(inspection)
    if fault in ('preflight','identity'):
        x=json.loads((run/'identity.json').read_text())
        if fault=='preflight':x['preflight_only']=True
        else:x['model']='other'
        put(run/'identity.json',x)
        r=json.loads((run/'result.json').read_text());r['identity_sha256']=file_hash(run/'identity.json');put(run/'result.json',r)
    if fault in ('missing_case','token_ids'):
        r=json.loads((run/'result.json').read_text())
        if fault=='missing_case':r['records'].pop()
        else:
            r['records'][0]['input_ids_sha256']='c'*64;put(run/'case0.json',r['records'][0])
        put(run/'result.json',r)
    if fault=='source':(root/'source'/gate.RUNNER).write_text('changed')
    if fault=='wrong_intent':args[1]='d'*64
    if fault=='wrong_inspection':args[3]='d'*64
    with pytest.raises(ValueError):gate.require_capacity(*args)
