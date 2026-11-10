"""Candidate context plumbing; synthetic processor, no task-quality claims."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_candidate_context import load_candidate
from multimodal_jev.sfc_prompt_candidates import stage
from test_sfc_prompt_candidates import proposal
from test_sfc_jev_shards import fixture as input_fixture
from test_sfc_jev import load_tool
from test_sfc_teacher_wire import fixture as wire_fixture,response as wire_response,update as wire_update
from multimodal_jev import sfc_teacher_wire as wire

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))


def staged(tmp_path):
    payload,context=proposal();p=tmp_path/'proposal.json';f=tmp_path/'feedback.json'
    p.write_bytes(canonical(payload));f.write_bytes(canonical(context))
    root=tmp_path/'staged';stage(p,f,root,file_hash(p),file_hash(f))
    return dict(staging=root,staging_hash=file_hash(root/'staged.json'),candidate_id='p1')


def test_staged_instruction_bound_without_semantic_approval(tmp_path):
    spec=staged(tmp_path);text,identity=load_candidate(**spec)
    assert text==proposal()[0]['candidates'][0]['instruction']
    assert identity['semantic_review']=='pending' and identity['candidate_accepted'] is False
    (spec['staging']/'p1.txt').write_text('changed')
    with pytest.raises(ValueError,match='changed'):load_candidate(**spec)


def test_verified_teacher_stage_loads_without_losing_wire_custody(tmp_path,monkeypatch):
    root,payload=wire_fixture(tmp_path,monkeypatch);intent_hash=wire_response(root,payload)
    candidates=root/'candidates';wire.stage_wire(root,candidates,intent_hash)
    spec=dict(staging=candidates,staging_hash=file_hash(candidates/'staged.json'),candidate_id='p1')
    text,identity=load_candidate(**spec)
    assert text==payload['candidates'][0]['instruction'] and identity['candidate_accepted'] is False
    assert identity['semantic_review']=='pending'
    wire_update(root/'response/request.json',lambda d:d.update(model='changed'))
    with pytest.raises(ValueError,match='request differs'):load_candidate(**spec)


def test_rehashed_teacher_instruction_cannot_override_original_wire(tmp_path,monkeypatch):
    root,payload=wire_fixture(tmp_path,monkeypatch);intent_hash=wire_response(root,payload)
    candidates=root/'candidates';wire.stage_wire(root,candidates,intent_hash)
    (candidates/'p1.txt').write_text('A different generic instruction.')
    wire_update(candidates/'staged.json',lambda d:d['candidates'][0].update(
        instruction_sha256=file_hash(candidates/'p1.txt'),characters=len((candidates/'p1.txt').read_text())))
    with pytest.raises(ValueError,match='verified teacher'):
        load_candidate(candidates,file_hash(candidates/'staged.json'),'p1')


@pytest.mark.parametrize('change',['receipt','id','duplicate','parent'])
def test_candidate_identity_cannot_be_rebound(tmp_path,change):
    spec=staged(tmp_path)
    if change=='receipt':spec['staging_hash']='wrong'
    if change=='id':spec['candidate_id']='../../other'
    if change=='duplicate':
        p=spec['staging']/'staged.json';r=json.loads(p.read_text());r['candidates']*=2
        atomic_json(p,r);spec['staging_hash']=file_hash(p)
    if change=='parent':
        p=spec['staging']/'staged.json';r=json.loads(p.read_text());r['parent_prompt_sha256']='other-parent'
        atomic_json(p,r);spec['staging_hash']=file_hash(p)
    with pytest.raises(ValueError):load_candidate(**spec)


@pytest.mark.parametrize('overflow',[False,True])
def test_candidate_processor_full_coverage_binding_and_parent_separation(tmp_path,monkeypatch,overflow):
    m=load_tool('audit_sfc_jev_context');v=load_tool('verify_sfc_jev_context')
    spec=staged(tmp_path);release,_=input_fixture(tmp_path)
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    root=tmp_path/'audit';out=root/'run';out.mkdir(parents=True)
    by_audio={}
    for p in release.glob('*.requests.jsonl'):
        for row in map(json.loads,p.read_text().splitlines()):
            by_audio.update(zip(row['input']['context']['audios'],row['audio_sha256']))
    original_hash=m.file_hash
    monkeypatch.setattr(m,'file_hash',lambda p:by_audio[str(p)] if str(p) in by_audio else original_hash(p))
    monkeypatch.setattr(m,'load_sfc_audio',lambda p:([0]*32000,{}))
    monkeypatch.setattr(m.importlib.metadata,'version',lambda _: 'synthetic')
    rendered=[]
    class Tensor:
        def __init__(self,data):self.data=data
        def __getitem__(self,key):return Tensor(self.data[key])
        def sum(self,axis):return self
        def tolist(self):return self.data
    class Processor:
        tokenizer=NS(convert_tokens_to_ids=lambda _:9)
        feature_extractor=NS(n_samples=480000,hop_length=160)
        @classmethod
        def from_pretrained(cls,*a,**k):return cls()
        def apply_chat_template(self,messages,**kwargs):
            rendered.append(messages);return json.dumps(messages)
        def __call__(self,**kwargs):
            n=len(kwargs['audio']);ids=[1,9,2]*n
            if overflow:ids += [1]*3073
            return dict(input_ids=Tensor([ids]),input_features=object(),feature_attention_mask=Tensor([200]*n))
    monkeypatch.setitem(sys.modules,'torch',NS(set_num_threads=lambda _:None,isfinite=lambda _:NS(all=lambda:True)))
    monkeypatch.setitem(sys.modules,'transformers',NS(Qwen3OmniMoeProcessor=Processor))
    monkeypatch.setitem(sys.modules,'transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe',
        NS(Qwen3OmniMoeProcessorKwargs=NS(_defaults={'audio_kwargs':{'truncation':False}})))
    result=m.run(model,release,out,candidate=spec)
    assert len(rendered)==72 and result['all_inputs_fit'] is not overflow
    text,_=load_candidate(**spec)
    assert all(ms[0]['content'].startswith(text+'\n\n') for ms in rendered)
    assert any(len(ms)==4 for ms in rendered)  # Supplied assistant history and both audio turns retained.
    identity=json.loads((out/'identity.json').read_text())
    assert identity['prompt']==result['prompt'] and identity['targets_read'] is False
    assert sum(s['decisions'] for s in result['splits'].values())==72
    # Bind the actual checked code files, with a synthetic Docker identity.
    repo=Path(__file__).parents[1]
    for name in identity['source_files']:
        p=root/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((repo/name).read_bytes())
    atomic_json(out/'terminal.json',dict(exit_code=0));atomic_json(root/'launch.json',dict(container_id='cid',image='image'))
    monkeypatch.setattr(v,'inspect',lambda _:dict(Id='cid',Image='image',State=dict(Status='exited',ExitCode=0),HostConfig=dict(Devices=[])))
    accepted=v.verify(root,candidate=spec)
    assert accepted['prompt']['instruction_sha256']==file_hash(spec['staging']/'p1.txt')
    assert accepted['all_inputs_fit'] is not overflow
    with pytest.raises(ValueError,match='parent context'):v.verify(root)
    result['prompt']['instruction_sha256']='different';atomic_json(out/'result.json',result)
    with pytest.raises(ValueError,match='exact staged'):v.verify(root,candidate=spec)
