import hashlib
import json
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_candidate_execution import execution_prompt, context_spec, REVIEW_CHECKS
from multimodal_jev.sfc_jev_runtime import INSTRUCTION
from multimodal_jev import sfc_teacher_wire as wire
from test_sfc_teacher_wire import fixture, response
from test_sfc_candidate_context import staged


def reviewed(tmp_path,monkeypatch):
    root,payload=fixture(tmp_path,monkeypatch);ih=response(root,payload)
    staging=root/'candidates';wire.stage_wire(root,staging,ih)
    review=root/'semantic-review.json'
    record=dict(schema='sfc-jev-candidate-semantic-review-v1',candidate_id='p1',
        staging_sha256=file_hash(staging/'staged.json'),instruction_sha256=file_hash(staging/'p1.txt'),
        decision='eligible_for_evaluation',reviewer='synthetic test reviewer',rationale='synthetic test only',
        checks={k:True for k in REVIEW_CHECKS})
    review.write_bytes(canonical(record))
    cfg=dict(prompt_id='p1',instruction_sha256=record['instruction_sha256'],candidate=dict(
        staging=str(staging),staging_hash=record['staging_sha256'],candidate_id='p1',
        review=str(review),review_sha256=file_hash(review)))
    return cfg,payload


def test_reviewed_wire_candidate_and_parent_are_separate(tmp_path,monkeypatch):
    cfg,payload=reviewed(tmp_path,monkeypatch)
    text,identity=execution_prompt(cfg)
    assert text==payload['candidates'][0]['instruction']
    assert identity['candidate_accepted'] is False  # Engineering eligibility is not efficacy.
    assert context_spec(cfg)['staging_hash']==cfg['candidate']['staging_hash']
    parent=dict(prompt_id='parent',instruction_sha256=hashlib.sha256(INSTRUCTION.encode()).hexdigest())
    assert execution_prompt(parent)==(INSTRUCTION,None) and context_spec(parent) is None
    parent['candidate']=cfg['candidate']
    with pytest.raises(ValueError):execution_prompt(parent)


@pytest.mark.parametrize('change',['pending','check','instruction','review_hash','reviewer','parent','id','wire'])
def test_review_cannot_be_rebound_or_inferred(tmp_path,monkeypatch,change):
    cfg,_=reviewed(tmp_path,monkeypatch);spec=cfg['candidate'];path=Path(spec['review']);r=json.loads(path.read_text())
    if change=='pending':r['decision']='pending'
    if change=='check':r['checks']['no_noul_gating']=False
    if change=='instruction':r['instruction_sha256']='c'*64
    if change=='reviewer':r['reviewer']=''
    if change=='parent':cfg['instruction_sha256']='d'*64
    if change=='id':cfg['prompt_id']='p2'
    if change=='wire':(Path(spec['staging']).parent/'response/response.sse').write_text('changed')
    path.write_bytes(canonical(r));spec['review_sha256']=file_hash(path)
    if change=='review_hash':spec['review_sha256']='e'*64
    with pytest.raises(ValueError):execution_prompt(cfg)


def test_local_staging_does_not_masquerade_as_teacher_candidate(tmp_path):
    spec=staged(tmp_path)
    cfg=dict(prompt_id='p1',instruction_sha256=file_hash(Path(spec['staging'])/'p1.txt'),
        candidate={**spec,'review':str(tmp_path/'unused'),'review_sha256':'a'*64})
    with pytest.raises(ValueError,match='teacher response'):execution_prompt(cfg)


def candidate_shard(tmp_path,monkeypatch):
    from test_sfc_jev_shard_runtime import fixture as parent_fixture
    from multimodal_jev.sfc_jev_shards import make_plan,load_shard
    cfg,_,_=parent_fixture(tmp_path)
    candidate,_=reviewed(tmp_path,monkeypatch);cfg.update(candidate)
    plan=make_plan(tmp_path/'release','train_id',12,partition_path=tmp_path/'partition.json',
        partition_hash=file_hash(tmp_path/'partition.json'),part='learning')
    path=Path(cfg['shard_plan']);path.write_bytes(canonical(plan));cfg['shard_plan_sha256']=file_hash(path)
    cfg['split']='train_id';rows=load_shard(plan,'shard-0000')
    audit=Path(cfg['context_audit']);run=audit/'run';identity=json.loads((run/'identity.json').read_text())
    _,prompt=execution_prompt(cfg);identity['prompt']=prompt
    for name in ('src/multimodal_jev/sfc_prompt_candidates.py','src/multimodal_jev/sfc_candidate_context.py'):
        dest=audit/'source'/name;dest.write_bytes((Path(__file__).parents[1]/name).read_bytes())
        identity['source_files'][name]=file_hash(dest)
    (run/'identity.json').write_bytes(canonical(identity))
    checks=[dict(id=r['id'],decision_id=d['id'],context_overflow=False,input_tokens=3,
        input_ids_sha256=hashlib.sha256(canonical([1,2,3])).hexdigest(),prompt_sha256=hashlib.sha256(b'prompt').hexdigest(),
        expected_frames=[4000]*len(r['audio_sha256']),observed_frames=[4000]*len(r['audio_sha256']))
        for r in rows for d in r['input']['decisions']]
    path=run/'train_id.context.jsonl';path.write_bytes(b''.join(map(canonical,checks)))
    result=dict(status='passed',all_inputs_fit=True,identity_sha256=file_hash(run/'identity.json'),prompt=prompt,
        splits={'train_id':dict(records_sha256=file_hash(path))})
    (run/'result.json').write_bytes(canonical(result));cfg['context_result_sha256']=file_hash(run/'result.json')
    acceptance=Path(cfg['context_acceptance']);acceptance.write_bytes(canonical(dict(
        result_sha256=cfg['context_result_sha256'],container_id='cpu-container',all_inputs_fit=True,prompt=prompt)))
    cfg['context_acceptance_sha256']=file_hash(acceptance)
    return cfg,rows,{(r['id'],r['decision_id']):r for r in checks}


@pytest.mark.parametrize('change',[None,'parent_context','test','renderer'])
def test_candidate_shard_requires_own_context_and_train_partition(tmp_path,monkeypatch,change):
    from multimodal_jev.sfc_jev_shard_runtime import select_inputs
    cfg,rows,checks=candidate_shard(tmp_path,monkeypatch)
    if change=='parent_context':
        path=Path(cfg['context_acceptance']);value=json.loads(path.read_text());value.pop('prompt')
        path.write_bytes(canonical(value));cfg['context_acceptance_sha256']=file_hash(path)
    if change=='test':
        path=Path(cfg['shard_plan']);value=json.loads(path.read_text());value['split']='test_id'
        path.write_bytes(canonical(value));cfg['shard_plan_sha256']=file_hash(path)
    if change=='renderer':
        (Path(cfg['context_audit'])/'source/src/multimodal_jev/sfc_prompt_candidates.py').write_text('changed')
    if change is None:
        actual,plan,evidence=select_inputs(cfg)
        assert actual==rows and evidence==checks and plan['partition']['part']=='learning'
    else:
        with pytest.raises(ValueError):select_inputs(cfg)
