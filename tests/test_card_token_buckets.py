import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from multimodal_jev.card_token_buckets import BucketCursor, OnlineBuckets, bind_buckets
from multimodal_jev.card_contrast import bind_sample_counts, Directions
from multimodal_jev.context_audio_collate import collate_context_audio
from multimodal_jev.card_training_acoustics import collate as unique_collate
from multimodal_jev.intent_triplet_training import OnlineTripletStream
from multimodal_jev.class_logit_ema import ClassLogitEMA


def config():
    return json.loads(Path('configs/portable/operations/card-stage1-token-buckets-v1.json').read_bytes())


@pytest.mark.parametrize('limit, width', [(112,32), (113,24), (144,24), (145,16), (192,16), (193,8)])
def test_width_follows_dynamic_batch_max(limit, width):
    c=config()
    bind_buckets(c)
    ids=list(range(1024))
    cursor=BucketCursor([ids], c['window']['token_buckets'], 2)
    batch, epoch, bucket, end, longest = cursor.take(lambda uid:limit)
    assert len(batch)==width*2 and longest==limit and not end


def test_public_recipe_changes_bins_draws_and_replays_cursor():
    c=config()
    c['token_bucketing']['buckets']=[dict(max_tokens=9,batch_size_per_rank=3),
                                     dict(max_tokens=None,batch_size_per_rank=1)]
    c['contrast_samples'].update(positive_recordings=2,negative_cards=4,negative_recordings=1)
    bind_buckets(c)
    bind_sample_counts(c)
    assert Directions.from_objective(c['window']['triplet_objective']).width==8
    spec=c['window']['token_buckets']
    ids=list(range(23))
    cur=BucketCursor([ids],spec,2)
    batch,*_=cur.take(lambda uid:8 if uid%2 else 20)
    assert batch==[0,2]
    recovered=BucketCursor([ids],spec,2,cur.state())
    got=batch.copy()
    while cur.epoch==0:
        # New draws move all remaining anchors into the short bucket.
        expected=cur.take(lambda uid:5)
        assert expected==recovered.take(lambda uid:5)
        got.extend(expected[0])
    assert sorted(got)==ids and len(expected[0])<6


@pytest.mark.parametrize('collate',[collate_context_audio, unique_collate])
def test_right_pad_to_actual_batch_maximum_and_preserve_markers(collate):
    items=[]
    for i,n in enumerate([115,137,123]):
        ids=torch.tensor([[9]+[13]*(n-2)+[8]])
        items.append(dict(request_id=str(i),policy_id='p',positions=torch.tensor([[n-1]]),
            inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),input_features=torch.zeros(1,1,100),
                        input_features_mask=torch.tensor([[1]+[0]*99]))))
    inp,pos=collate(items,decision_token_id=8,policy_id='p',audio_token_id=9,pad_token_id=0,
                    audio_chunk_size=100,audio_infer_window=100)
    assert inp['input_ids'].shape==(3,137)
    for i,n in enumerate([115,137,123]):
        assert pos[i].item()==n-1
        assert torch.equal(inp['input_ids'][i,:n],items[i]['inputs']['input_ids'][0])
        assert inp['attention_mask'][i,:n].all() and not inp['attention_mask'][i,n:].any()
        assert not inp['input_ids'][i,n:].any()


def test_stream_draws_actual_pairs_and_pads_only_rank_tail():
    c=config()
    bind_sample_counts(c)
    bind_buckets(c)
    w=c['window']
    w['class_logit_ema']['min_shared_coordinates']=2
    ids=[f'{k}/{i}' for k in ('x','y') for i in range(4)]
    pop={uid:dict(id=uid,intent=uid[0],role='learning',official_split='train') for uid in ids}
    batches=[dict(epoch=0,slots=[dict(id=u,weight=1) for u in ids])]
    specs=dict(labels=['x','y'],arms={'common':[0,1]},catalog=[dict(prefix=[11],suffix=[8]),
              dict(prefix=[12]*112,suffix=[8])])
    layout=SimpleNamespace(assignments={uid:['common'] for uid in ids})
    cfg=dict(instruction_objective=w['triplet_objective'], mixed={'seed':7},class_logit_ema=w['class_logit_ema'],
             triplet_prefetch=w['triplet_prefetch'],token_buckets=w['token_buckets'], batch_config={'epochs':1})
    def audio(uid):
        n=ids.index(uid)+1
        return {'input_ids':torch.full((1,n),9), 'input_features':torch.zeros(1,1,100),
                'input_features_mask':torch.ones(1,100)}
    ranks=[]
    for rank in range(3):
        s=OnlineTripletStream(batches,pop,None,layout,specs,cfg,rank,3,dict(audio_token_id=9,decision_token_id=8))
        s.packed=SimpleNamespace(read=audio,entries={u:[0,0,0,0,ids.index(u)+1] for u in ids})
        ema=ClassLogitEMA(specs['labels'],**w['class_logit_ema'])
        online=OnlineBuckets(s,ema,0,8)
        try:
            step,groups,count,version,info=next(online.queue)
            assert count==8 and info['end'] and len(groups)==3 and all(len(g[0])==14 for g in groups)
            ranks.append((sum(g[1] for g in groups),info))
            with pytest.raises(StopIteration):next(online.queue)
        finally:
            online.queue.close()
    assert sum(r[0] for r in ranks)==8
    assert max(info['local_max'] for _,info in ranks)==ranks[0][1]['max_tokens']
    assert len({r[1]['bucket'] for r in ranks})==1


def test_rebudget_reuses_pack_only_for_same_authenticated_inputs(tmp_path):
    from multimodal_jev.baseline import file_hash
    from tools.reports.card_stage1_window import verify_pack_inputs
    roots = [tmp_path/'prior', tmp_path/'current']
    receipts = []
    for root in roots:
        root.mkdir()
        proof = {'population': 200, 'policy_id': 'same'}
        for name, key, field in [('card-cache.json', 'cache_sha256', 'layout_plan_sha256'),
                                 ('single.json', 'single_sha256', 'plan')]:
            path = root/name
            path.write_text(json.dumps({field: root.name, 'tokens': [17, 29]}))
            proof[key] = file_hash(path)
        receipts.append(proof)
    verify_pack_inputs(*receipts, *roots, rebind_plan=True)
    path = roots[1]/'single.json'
    path.write_text(json.dumps({'plan': 'current', 'tokens': [17, 30]}))
    with pytest.raises(ValueError, match='bytes differ'):
        verify_pack_inputs(*receipts, *roots, rebind_plan=True)
    receipts[1]['single_sha256'] = file_hash(path)
    with pytest.raises(ValueError, match='beyond its schedule'):
        verify_pack_inputs(*receipts, *roots, rebind_plan=True)


def test_attention_mask_reaches_backbone_and_head():
    from multimodal_jev.context_decision_model import ContextDecisionModel

    class Backbone(torch.nn.Module):
        def forward(self, **kwargs):
            self.received = kwargs['attention_mask'].clone()
            return SimpleNamespace(last_hidden_state=torch.zeros(2, 6, 3))

    class Head(torch.nn.Module):
        def forward(self, hidden, positions, mask, return_logits=False):
            self.received = mask.clone()
            return hidden.new_zeros((2, 1))

    backbone, head = Backbone(), Head()
    model = ContextDecisionModel(backbone, head, 8)
    ids = torch.tensor([[1, 8, 0, 0, 0, 0], [1, 2, 3, 4, 8, 0]])
    mask = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0]])
    model(dict(input_ids=ids, attention_mask=mask), torch.tensor([[1], [4]]))
    assert torch.equal(backbone.received, mask)
    assert torch.equal(head.received, mask.bool())
