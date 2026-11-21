import json
from pathlib import Path
import pytest
import torch
from test_cached_context_batches import fixture
from multimodal_jev.cached_context_batches import CachedContextBatches
from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.stage1_candidate_layout import epoch_layouts, build_view
from multimodal_jev.stage1_training_layout import Stage1TrainingLayout


def test_actual_stage1_batch_rebinds_layout_on_prefetch_and_fresh_resume(monkeypatch):
    pop,task,epoch,prepare=fixture(monkeypatch)
    epoch=[dict(b,real_queries=sum(not slot['padding'] for slot in b['slots'])) for b in epoch]
    batches=epoch+[dict(b,epoch=1) for b in epoch]
    design=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/next-simple-jev-v1.json').read_bytes())
    cfg=design['training'];ids=list(pop)
    req=compile_candidates('audio',[dict(id=q['id'],definition=q['description']) for q in task['schema']['candidates']],
                           task['thinking']['requirements'],semantics='exclusive_event')
    plan=dict(source_identity='r',stage1_objective='o',
              arms={'training':dict(epochs=[dict(assignment_sha256=content_hash(epoch_layouts(ids,e,cfg))) for e in range(2)],
                                    real_exposures=2*len(ids))},
              panels={k:vars(build_view(req,layout=k,seed=cfg['permutation_seed'])) for k in cfg['layouts']})
    def collect(start):
        layout=Stage1TrainingLayout(ids,task,design=design,plan=plan,release_sha256='r',objective='o')
        with CachedContextBatches(batches,pop,task,prepare,start_update=start,stop_update=4,global_batch=2,slot_request=layout) as stream:
            out=[]
            for step in range(start,4):
                items,meta=stream.consume(step,'policy')
                from tools.reports.audit_context_infonce_training import check_batch
                record=dict(records=meta,rank_indices=[[0],[1]],policy_id='policy')
                check_batch(record,batches[step],pop,task,policy='policy',width=2,world=2,slot_request=layout)
                for item,m in zip(items,meta,strict=True):
                    y=item['labels'][0];names=m['target']['candidate_ids']
                    assert names[torch.where(y)[0].item()]==pop[m['id']]['intent']
                    out.append((m['id'],tuple(names),item['real_request']))
            return out
    full=collect(0)
    assert full[4:]==collect(2)
    for uid in ids:
        views=[names for k,names,real in full if k==uid and real]
        assert len(views)==2 and views[0]==views[1][::-1]
    plan['arms']['training']['epochs'][0]['assignment_sha256']='0'*64
    with pytest.raises(ValueError,match='assignment'):collect(0)


def test_text_view_rejects_misbound_native_and_unknown_order():
    from dataclasses import replace
    from multimodal_jev.stage1_layout_cache import LayoutTextView
    from multimodal_jev.context_decision_contract import PREFIX_LAYOUT
    from multimodal_jev.encoded_audio_cache import canonical
    req=compile_candidates('u.wav',[dict(id=k,definition=k) for k in ('on','off')],['listen'],semantics='exclusive_event')
    requests=dict(native=req,both=replace(req,questions=req.questions[::-1]))
    panels={k:canonical(r.model_input()) for k,r in requests.items()}
    for p in panels.values():del p['audio']
    enc=dict(audio_token_id=9,decision_token_id=8,prompt_layout=PREFIX_LAYOUT,max_tokens=20)
    spec=dict(schema='jev-stage1-layout-text-v1',processor_sha256='p',encoding=enc,templates={
        k:dict(panel=p,prefix=[i],suffix=[8,i,8]) for i,(k,p) in enumerate(panels.items())})
    view=LayoutTextView(spec,requests)
    ids=torch.tensor([[0,9,9,8,0,8]])
    feat=torch.ones(1,2)
    class Cache:
        def read(self,identity):
            assert identity['request']==canonical(req.model_input())
            return dict(input_ids=ids,attention_mask=torch.ones_like(ids),input_features=feat,input_features_mask=feat),torch.tensor([[3,5]]),{}
    ident=dict(request=canonical(requests['both'].model_input()),processor_sha256='p',context_encoding=enc)
    inputs,pos,_=view.read(Cache(),ident)
    assert inputs['input_ids'].tolist()==[[1,9,9,8,1,8]] and pos.tolist()==[[3,5]]
    assert inputs['input_features'] is feat
    with pytest.raises(ValueError,match='unknown candidate'):
        view.read(Cache(),dict(ident,request=dict(ident['request'],history=['changed'])))
    ids[0,1]=7
    with pytest.raises(ValueError,match='native cache'):view.read(Cache(),ident)
