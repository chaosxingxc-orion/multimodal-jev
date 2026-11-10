import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from multimodal_jev.mmar_inference import accept_response, capacity_indices, ids_hash, validate_input
from multimodal_jev.mmar_inference import admit_capacity, ENGINE, IMAGE, INPUT_SHA, PROCESSOR_RESULT_SHA
from multimodal_jev.baseline import file_hash


def response(ids=(1,2), tokens=(3,), text='', reason='stop', finished=True):
    return [NS(prompt_token_ids=list(ids), finished=finished,
               outputs=[NS(token_ids=list(tokens), text=text, finish_reason=reason, stop_reason=None)])]


def test_empty_eos_and_raw_text_are_preserved():
    result = accept_response(response(), [1,2], 256)
    assert result['raw_text'] == '' and result['token_ids'] == [3]
    raw = '```json\n{"answer":"A"}\n```'
    assert accept_response(response(text=raw), [1,2], 256)['raw_text'] == raw
    limited = accept_response(response(tokens=range(256), reason='length'), [1,2], 256)
    assert limited['length_limited']


@pytest.mark.parametrize('kwargs', [dict(ids=(1,3)), dict(tokens=()), dict(tokens=(-1,)),
    dict(tokens=(True,)), dict(reason='length'), dict(reason='error'), dict(finished=False),
    dict(text=None), dict(tokens=range(257))])
def test_reject_protocol_or_budget_changes(kwargs):
    with pytest.raises(ValueError):
        accept_response(response(**kwargs), [1,2], 256)


def test_boundary_selection_uses_both_maxima_and_stable_ties():
    prior = [dict(input_tokens=t, audio=dict(actor_samples=a)) for t,a in [(100,50),(500,30),(200,80),(500,80)]]
    assert capacity_indices(prior) == [1,2]
    assert capacity_indices([prior[0]]) == [0]


def test_actual_inputs_must_match_accepted_processor():
    receipt = dict(actor_samples=160)
    prior = dict(audio=receipt, frames=[1], input_tokens=2, input_ids_sha256=ids_hash([1,2]),
                 prompt_sha256=hashlib.sha256(b'prompt').hexdigest())
    validate_input(dict(prompt='prompt'), receipt, [1,2], [1], prior)
    for request,audio,ids,frames in [(dict(prompt='changed'),receipt,[1,2],[1]),
                                   (dict(prompt='prompt'),dict(actor_samples=320),[1,2],[1]),
                                   (dict(prompt='prompt'),receipt,[2,1],[1]),
                                   (dict(prompt='prompt'),receipt,[1,2],[2])]:
        with pytest.raises(ValueError, match='differs'):
            validate_input(request,audio,ids,frames,prior)


@pytest.fixture
def capacity_fixture(tmp_path, monkeypatch):
    """Synthetic container/result envelope; not real NPU acceptance."""
    from multimodal_jev import audio2tool_engine
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    rows=[dict(id='a'),dict(id='b')]
    prior=[dict(id=r['id'],input_tokens=t,audio=dict(actor_samples=a),
                input_ids_sha256=ids_hash([1,2]),prompt_sha256='prompt'+r['id'])
           for r,t,a in zip(rows,[100,200],[400,300])]
    paths=['src/multimodal_jev/mmar_omni.py','src/multimodal_jev/mmar_inference.py',
           'src/multimodal_jev/sfc_audio.py','src/multimodal_jev/omni_waveform_boundary.py',
           'tools/jev/infer_mmar_omni.py']
    repo=Path(__file__).resolve().parents[1];pins={}
    for name in paths:
        dst=tmp_path/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_bytes((repo/name).read_bytes());pins[name]=file_hash(dst)
    audit=dict(model='synthetic-model',files=[]);write(tmp_path/'asset-audit.json',audit)
    audit_sha=file_hash(tmp_path/'asset-audit.json')
    monkeypatch.setattr(audio2tool_engine,'ASSET_AUDIT_SHA',audit_sha)
    argv=['docker','create','--device','/dev/davinci6','--device','/dev/davinci7',
          '-v','/synthetic:/source:ro','-e','TEST=1',IMAGE,'python3','runner','--capacity-only']
    write(tmp_path/'intent.json',dict(argv=argv,image=IMAGE,source_files=pins))
    intent=file_hash(tmp_path/'intent.json');cid='a'*64
    write(tmp_path/'launch.json',dict(container_id=cid,intent_sha256=intent))
    item=dict(Id=cid,Image=IMAGE,State=dict(Status='exited',ExitCode=0,OOMKilled=False,Error='',Running=False),
        HostConfig=dict(Privileged=False,NanoCpus=16*10**9,Memory=128*1024**3,ShmSize=8*1024**3,
            NetworkMode='host',Devices=[dict(PathOnHost='/dev/davinci6'),dict(PathOnHost='/dev/davinci7')],
            Binds=['/synthetic:/source:ro']),
        Config=dict(Env=['TEST=1'],Cmd=argv[argv.index(IMAGE)+1:],Entrypoint=['/opt/multimodal-jev/runtime-entrypoint.sh']))
    write(tmp_path/'inspection.json',item)
    identity=dict(engine=ENGINE,sampling=dict(temperature=0.,max_tokens=1,n=1,seed=17),
        inputs_sha256=INPUT_SHA,processor_result_sha256=PROCESSOR_RESULT_SHA,parameters_frozen=True,
        targets_read=False,selected_ids=['a','b'],asset_audit_sha256=audit_sha,
        model=audit['model'],model_files=[],versions=audio2tool_engine.VERSIONS,source_files=pins)
    write(tmp_path/'run/identity.json',identity)
    records=[]
    (tmp_path/'run/failures').mkdir()
    for i,r in enumerate(prior):
        attempt=dict(id=r['id'],audio=r['audio'],input_ids_sha256=r['input_ids_sha256'],
                     prompt_sha256=r['prompt_sha256'],
                     request_sha256=hashlib.sha256(json.dumps(rows[i],sort_keys=True).encode()).hexdigest())
        write(tmp_path/'run/attempts'/f'{i:04d}.json',attempt)
        write(tmp_path/'run/records'/f'{i:04d}.json',dict(attempt=attempt,seconds=.9,
            responses=[dict(finished=True,prompt_token_ids=[1,2],outputs=[dict(text='x',token_ids=[4],finish_reason='length',stop_reason=None)])]))
        records.append(dict(**attempt,token_ids=[4],finish_reason='length',error=None,
                            raw_text='x',stop_reason=None,length_limited=True,seconds=1.))
    (tmp_path/'run/predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    write(tmp_path/'run/result.json',dict(status='passed',mode='capacity',weight_updates=0,records=2,
        targets_read=False,model_files_unchanged=True,identity_sha256=file_hash(tmp_path/'run/identity.json'),
        predictions_sha256=file_hash(tmp_path/'run/predictions.jsonl')))
    write(tmp_path/'run/terminal.json',dict(exit_code=0))
    def call():
        return admit_capacity(tmp_path,intent,tmp_path/'inspection.json',file_hash(tmp_path/'inspection.json'),rows,prior)
    return tmp_path,write,call


def test_synthetic_capacity_envelope_accepted(capacity_fixture):
    _,_,call=capacity_fixture
    assert call()['container_id']=='a'*64


@pytest.mark.parametrize('target,key,value',[
    ('State','Status','running'),('State','ExitCode',1),('State','OOMKilled',True),
    ('HostConfig','Memory',1024),('HostConfig','Devices',[dict(PathOnHost='/dev/davinci0')]),
    ('Config','Entrypoint',['bypass']),('Config','Cmd',['different']),('Config','Env',[]),
])
def test_capacity_rejects_actual_container_drift(capacity_fixture,target,key,value):
    root,write,call=capacity_fixture
    path=root/'inspection.json';item=json.loads(path.read_text());item[target][key]=value;write(path,item)
    with pytest.raises(ValueError,match='capacity container'):
        call()


@pytest.mark.parametrize('mutation',['missing_row','changed_ids','text_budget','modified_source','changed_raw','failure'])
def test_capacity_rejects_result_or_source_drift(capacity_fixture,mutation):
    root,write,call=capacity_fixture
    if mutation=='modified_source':
        (root/'source/src/multimodal_jev/mmar_omni.py').write_text('changed')
    elif mutation=='changed_raw':
        p=root/'run/records/0000.json';v=json.loads(p.read_text());v['responses'][0]['outputs'][0]['text']='different';write(p,v)
    elif mutation=='failure':
        write(root/'run/failures/0000.json',dict(error='failed'))
    elif mutation=='text_budget':
        p=root/'run/identity.json';v=json.loads(p.read_text());v['sampling']['max_tokens']=256;write(p,v)
        p=root/'run/result.json';v=json.loads(p.read_text());v['identity_sha256']=file_hash(root/'run/identity.json');write(p,v)
    else:
        p=root/'run/predictions.jsonl';rows=[json.loads(l) for l in p.open()]
        if mutation=='missing_row':rows.pop()
        else:rows[0]['input_ids_sha256']='changed'
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        p=root/'run/result.json';v=json.loads(p.read_text());v['predictions_sha256']=file_hash(root/'run/predictions.jsonl');write(p,v)
    with pytest.raises(ValueError):call()
