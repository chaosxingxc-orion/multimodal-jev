import json

import pytest

from multimodal_jev.audio2tool_acceptance import audit_ledger
from multimodal_jev.audio2tool_inference import Ledger,accept_generation,digest
from types import SimpleNamespace


def fixture(tmp_path):
    audio=dict(path='public/tier1_direct_audios/query_00001/speaker_01_x.wav',sha256='a'*64,bytes=526)
    row=dict(id='one',tier=1,input=dict(audios=[audio]))
    budget=dict(id='one',input_tokens=2,frames=[2])
    receipt=dict(**audio,conversion=dict(source_rate=24000,source_channels=1,source_samples=241,
        target_rate=16000,target_samples=161,conversion='channel mean then scipy.signal.resample_poly when needed'),
        right_zero_padding_samples=159,actor_samples=320,expected_frames=2,actor_waveform_sha256='b'*64)
    header=dict(frames=241,sample_rate=24000,channels=1,sample_width_bytes=2,manifest_sha256=audio['sha256'])
    ledger=Ledger(tmp_path/'ledger');attempt=ledger.begin(row,{'prompt':'fixed'},[receipt],[1,2])
    response=SimpleNamespace(finished=True,prompt_token_ids=[1,2],outputs=[
        SimpleNamespace(text='malformed JSON retained',token_ids=[3],finish_reason='stop')])
    ledger.finish(attempt,accept_generation([response],[1,2]),.1)
    ledger.complete([row])
    prediction=json.loads((ledger.root/'predictions.jsonl').read_text())
    return ledger,row,budget,prediction,{audio['path']:header}


@pytest.mark.parametrize('mutation',[None,'missing','extra','projection','prompt','ids','receipts','order','raw_text','cutoff','error','nonfinite','token','coordinated_text_edit'])
def test_independent_audit_rejects_ledger_or_projection_changes(tmp_path,mutation):
    ledger,row,budget,prediction,headers=fixture(tmp_path)
    expected=lambda r,b:('fixed',[1,2])
    ap=ledger.root/'attempts'/(digest('one')+'.json');rp=ledger.root/'records'/(digest('one')+'.json')
    a=json.loads(ap.read_text());r=json.loads(rp.read_text())
    if mutation=='missing':rp.unlink()
    if mutation=='extra':(ledger.root/'records/extra.json').write_text('{}')
    if mutation=='projection':prediction['raw_text']='repaired JSON'
    if mutation=='prompt':expected=lambda r,b:('new prompt',[1,2])
    if mutation=='ids':expected=lambda r,b:('fixed',[2,1])
    if mutation=='receipts':
        a['audio_receipts'][0]['conversion']['target_samples']=160;r['audio_receipts']=a['audio_receipts'];ap.write_text(json.dumps(a))
    if mutation=='order':budget['id']='other'
    if mutation=='raw_text':r['raw_text']='silently repaired'
    if mutation=='cutoff':r['finish_reason']='length';r['length_limited']=True
    if mutation=='error':r['error']='engine died';prediction['error']='engine died'
    if mutation=='nonfinite':r['seconds']=float('nan')
    if mutation=='token':r['token_ids']=[True]
    if mutation=='coordinated_text_edit':r['raw_text']='repaired';prediction['raw_text']='repaired'
    if mutation!='missing':rp.write_text(json.dumps(r))
    args=(ledger.root,[row],[budget],[prediction],headers,expected,lambda tokens:'malformed JSON retained')
    if mutation is None:
        report=audit_ledger(*args)
        assert report['records']==1 and report['length_limited']==0 and report['generation_seconds']==.1
    else:
        with pytest.raises(ValueError):audit_ledger(*args)


@pytest.mark.parametrize('mutation',[None,'running','oom','command','device','source_mount','preflight','image','privileged'])
def test_full_container_envelope(mutation):
    from multimodal_jev.audio2tool_acceptance import IMAGE,check_container
    cid='a'*64
    argv=['docker','create','--device','/dev/davinci6:/dev/davinci6','--device','/dev/davinci7:/dev/davinci7',
          '-v','/source:/source:ro','-e','OFFLINE=1',IMAGE,'timeout','86400s','python','runner.py']
    plan=dict(image=IMAGE,argv=argv,automatic_retry=False,physical_devices=[6,7])
    item=dict(Id=cid,Image=IMAGE,State=dict(Status='exited',ExitCode=0,Running=False,Restarting=False,Dead=False,Error='',OOMKilled=False),
        HostConfig=dict(NanoCpus=16*10**9,Memory=128*1024**3,ShmSize=8*1024**3,NetworkMode='host',Privileged=False,
            Devices=[dict(PathOnHost='/dev/davinci6'),dict(PathOnHost='/dev/davinci7')],Binds=['/source:/source:ro']),
        Config=dict(Env=['OFFLINE=1'],Cmd=argv[argv.index(IMAGE)+1:]))
    if mutation=='running':item['State']['Status']='running'
    if mutation=='oom':item['State']['OOMKilled']=True
    if mutation=='command':item['Config']['Cmd'][-1]='different.py'
    if mutation=='device':item['HostConfig']['Devices'][0]['PathOnHost']='/dev/davinci0'
    if mutation=='source_mount':item['HostConfig']['Binds']=['/other:/source:ro']
    if mutation=='preflight':argv.append('--preflight-only');item['Config']['Cmd'].append('--preflight-only')
    if mutation=='image':item['Image']='sha256:'+'b'*64
    if mutation=='privileged':item['HostConfig']['Privileged']=True
    if mutation is None:check_container(item,plan,cid)
    else:
        with pytest.raises(ValueError):check_container(item,plan,cid)


def test_targets_not_accessed_before_successful_acceptance(monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
    import score_audio2tool_native as cli
    def reject(_):raise ValueError('incomplete actor')
    monkeypatch.setattr(cli,'accept',reject)
    monkeypatch.setattr(cli,'file_hash',lambda _:pytest.fail('target accessed before acceptance'))
    with pytest.raises(ValueError,match='incomplete actor'):cli.score(SimpleNamespace(targets='unread'))
