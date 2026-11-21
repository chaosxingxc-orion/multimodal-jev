import json
import pytest
import torch

from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_decision_contract import DECISION_TOKEN, render_prefix_context_request
from multimodal_jev.stage1_candidate_layout import build_view


def test_independent_layout_interventions_bind_supervision_and_gradients_to_semantic_ids():
    ids=('volume_up','volume_down','play_music','play_radio')
    req=compile_candidates('train-audio',[{'id':k,'definition':'meaning of '+k} for k in ids],
                           ['judge the spoken intent'],semantics='exclusive_event',history=['same history'])
    signal=torch.tensor([2.,-1.,.1,.3],requires_grad=True)
    native=build_view(req,layout='native',seed=317)
    assert (native.system,native.history,native.readouts)==render_prefix_context_request(req)
    for layout in ('native','definitions','readouts','both'):
        v=build_view(req,layout=layout,seed=317)
        qs=json.loads(v.system.rsplit('\n',1)[1])['questions']
        actual=tuple(json.loads(line)['decision_id'] for line in v.readouts.splitlines() if line!=DECISION_TOKEN)
        assert actual==v.readout_ids and sorted(actual)==sorted(ids)
        assert {q['id']:q['definition'] for q in qs}=={k:'meaning of '+k for k in ids}
        assert (tuple(q['id'] for q in qs)!=ids)==(layout in ('definitions','both'))
        assert (actual!=ids)==(layout in ('readouts','both'))
        assert v.history==native.history
        z=signal[[ids.index(k) for k in actual]][list(v.canonical_positions)]
        assert torch.equal(z,signal)
        assert torch.equal(torch.autograd.grad(z[1],signal)[0],torch.tensor([0.,1.,0.,0.]))
    with pytest.raises(ValueError):build_view(req,layout='gold_first',seed=317)


def test_counterbalanced_two_epochs_without_duplicate_exposures():
    from multimodal_jev.stage1_candidate_layout import epoch_layouts
    ids=['a','b','c','d','e']
    c=dict(layouts=['native','definitions'],assignment_seed=317,permutation_seed=317)
    a,b=(epoch_layouts(ids,ep,c) for ep in (0,1))
    assert a==epoch_layouts(ids[::-1],0,c) and a.keys()==b.keys()==set(ids)
    assert all({a[k],b[k]}==set(c['layouts']) for k in ids)
    with pytest.raises(ValueError):epoch_layouts(ids+['b'],1,c)


def test_training_renderer_cache_and_target_share_order_after_semantic_permutation():
    from multimodal_jev.context_candidate_task import bind_candidate_targets
    from multimodal_jev.encoded_audio_cache import binding, key_for
    from multimodal_jev.stage1_candidate_layout import training_request
    from multimodal_jev.context_infonce import loss
    req=compile_candidates('same-audio',[dict(id=k,definition=k) for k in ('left','right','stop','other')],
                           ['judge the intent'],semantics='exclusive_event')
    alt=training_request(req,layout='both',seed=317)
    # Choose a moved label only in the test oracle, never in layout compilation.
    gold=next(a.id for a,b in zip(req.questions,alt.questions) if a.id!=b.id)
    signal={q.id:(3. if q.id==gold else -2.) for q in req.questions}
    keys=[];values=[]
    for view in (req,alt):
        keys.append(key_for(binding(request=view,audio_sha256='a'*64,processor_sha256='b'*64,preprocessing=None)))
        target=bind_candidate_targets(view,semantics='exclusive_event',exclusive_event=gold)
        z=torch.tensor([[signal[k] for k in target['candidate_ids']]])
        labels=torch.tensor([target['labels']])
        values.append(loss(z,labels,torch.ones_like(labels),torch.ones(1),torch.ones(1,dtype=torch.bool),
                           global_real_requests=1)[0])
    assert keys[0]!=keys[1]
    torch.testing.assert_close(*values)
    # A stale positional target silently penalizes the correct prediction.
    stale=torch.tensor([[q.id==gold for q in req.questions]])
    wrong=loss(z,stale,torch.ones_like(stale),torch.ones(1),torch.ones(1,dtype=torch.bool),
               global_real_requests=1)[0]
    assert wrong > values[1]+4
