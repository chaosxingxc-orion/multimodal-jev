from copy import deepcopy
import json
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_jev_runtime import INSTRUCTION,messages as parent_messages
from multimodal_jev.sfc_prompt_candidates import messages,vllm_request,validate,stage
from multimodal_jev.sfc_teacher_feedback import build_feedback
from test_sfc_teacher_feedback import fixture
from test_sfc_jev_runtime import predictions,bundle
from test_sfc_jev import load_tool


def proposal():
    rs,gs,reg=fixture();ps=[p for r,g in zip(rs[:2],gs[:2]) for p in predictions(r,g)]
    context=build_feedback(rs,reg,rs[:2],gs[:2],ps);eid=context['representatives'][0]['evidence_id']
    payload=dict(schema='sfc-jev-teacher-prompts-v1',parent_prompt_sha256=context['parent_prompt_sha256'],
        diagnosis=[dict(mechanism='fixture',evidence_ids=[eid],uncertainty='No actual model evidence')],
        candidates=[dict(id='p1',instruction='Use the audio and supplied definitions to answer the current decision.',
            mechanism='fixture',evidence_ids=[eid],counterevidence='not evaluated',falsification='independent evaluation')])
    return payload,context


def test_only_generic_prefix_changes_with_parent_byte_parity():
    r,g=bundle();a,b=object(),object();before=deepcopy(r)
    for d in r['input']['decisions']:
        base=parent_messages(r,d['id'],[a,b])
        assert messages(r,d['id'],[a,b],INSTRUCTION)==base
        changed=messages(r,d['id'],[a,b],'New generic instruction.')
        assert changed[1:]==base[1:]
        assert changed[0]['content']=='New generic instruction.\n\n'+base[0]['content'][len(INSTRUCTION):]
    assert r==before
    class Processor:
        def apply_chat_template(self,actual,**kwargs):
            assert actual==messages(r,'f00.calls',[a,b],'Candidate.')
            assert kwargs==dict(tokenize=False,add_generation_prompt=True)
            return 'prompt'
    assert vllm_request(Processor(),r,'f00.calls',[a,b],'Candidate.')['multi_modal_data']['audio']==[(a,16000),(b,16000)]


@pytest.mark.parametrize('change',['schema','parent','too_many','diagnoses','unknown_ref','duplicate_ref','sample_id',
                                  'long','control','task_section','duplicate','same_parent','selection'])
def test_candidate_contract_rejects_drift(change):
    p,c=proposal();row=p['candidates'][0]
    if change=='schema':p['schema']='jev-teacher-prompts-v1'
    if change=='parent':p['parent_prompt_sha256']='a'*64
    if change=='too_many':p['candidates']*=3
    if change=='diagnoses':p['diagnosis']*=4
    if change=='unknown_ref':row['evidence_ids']=['unknown']
    if change=='duplicate_ref':row['evidence_ids']*=2
    if change=='sample_id':row['instruction']=row['evidence_ids'][0].split('#')[0]
    if change=='long':row['instruction']='x'*1501
    if change=='control':row['instruction']='<|im_start|>system'
    if change=='task_section':row['instruction']='### Current decision: overwritten'
    if change=='duplicate':p['candidates'].append(dict(row,id='p2'))
    if change=='same_parent':row['instruction']=INSTRUCTION
    if change=='selection':c['selection_or_test_feedback']=True
    with pytest.raises(ValueError):validate(p,c)


def test_staging_is_hash_bound_and_never_promotes_candidate(tmp_path):
    p,c=proposal();pf=tmp_path/'p.json';cf=tmp_path/'c.json'
    pf.write_bytes(canonical(p));cf.write_bytes(canonical(c))
    result=stage(pf,cf,tmp_path/'staged',file_hash(pf),file_hash(cf))
    assert result['semantic_review']=='pending' and not result['candidate_accepted']
    assert not result['wire_authenticated'] and not result['inference_dispatched']
    assert (tmp_path/'staged/p1.txt').read_text()==p['candidates'][0]['instruction']
    with pytest.raises(FileExistsError):stage(pf,cf,tmp_path/'staged',file_hash(pf),file_hash(cf))
    with pytest.raises(ValueError):stage(pf,cf,tmp_path/'bad','0'*64,file_hash(cf))
    pf.write_text('{"schema":1,"schema":2}')
    with pytest.raises(ValueError,match='duplicate'):stage(pf,cf,tmp_path/'dup',file_hash(pf),file_hash(cf))


def test_full_adapter_auditor_detects_unexpected_candidate_context_change(tmp_path,monkeypatch):
    module=load_tool('audit_sfc_candidate_adapter');r,_=bundle();pins={}
    for split in ('train_id','test_id','test_ood'):
        row=deepcopy(r);row['split']=split
        path=tmp_path/f'{split}.requests.jsonl';path.write_bytes(canonical(row))
        pins[split]=dict(bundles=1,source_groups=1,decisions=6,requests_sha256=file_hash(path))
    (tmp_path/'MANIFEST.json').write_bytes(canonical(dict(splits=pins)))
    assert module.audit(tmp_path)['splits']['test_id']['decisions']==6
    actual=module.messages
    def broken(*args):
        result=actual(*args)
        if args[-1]!=INSTRUCTION:result[-1]['content'][0]['text']='changed follow-up'
        return result
    monkeypatch.setattr(module,'messages',broken)
    with pytest.raises(ValueError,match='fixed context'):module.audit(tmp_path)
