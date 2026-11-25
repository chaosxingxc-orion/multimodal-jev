import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from multimodal_jev.card_contrast import Directions, KIND
from multimodal_jev.class_logit_ema import ClassLogitEMA
from multimodal_jev.context_infonce import validate_objective
from multimodal_jev.context_objective_registry import resolve
from multimodal_jev.intent_triplet_sampling import AnchorSampler
from multimodal_jev.intent_triplet_training import OnlineTripletStream, update
from tools.reports.collect_card_first_updates import summarize


CONFIG = Path('configs/portable/operations/card-stage1-bidirectional-v1.json')
WINDOW = json.loads(CONFIG.read_bytes())['window']
OBJ = WINDOW['triplet_objective']


def population(labels, n=2):
    return {uid: dict(id=uid, audio=uid, audio_sha256='fixture', intent=label,
                      role='learning', official_split='train')
            for label in labels for i in range(n) for uid in [f'{label}/{i}']}


@pytest.mark.parametrize('aw,cw', [(1.,0.), (0.,1.), (.35,.65)])
def test_logit_objective_matches_separate_cross_entropies_and_gradients(aw,cw):
    obj = dict(OBJ, fixed_audio_weight=aw, fixed_card_weight=cw,
               positive_recordings_per_anchor=3, negative_cards_per_anchor=2,
               negative_recordings_per_anchor=4)
    validate_objective(obj)
    d = Directions.from_objective(obj)
    loss = resolve(KIND)
    n = 1 + (2 if aw else 0) + (7 if cw else 0)
    z = torch.linspace(-.4,.5,2*n).reshape(2,n).requires_grad_()
    got = loss(z,d)
    def compare(pos, negatives):
        scores = z[:, [pos, *negatives]] / obj['temperature']
        return torch.nn.functional.cross_entropy(scores, torch.zeros(2,dtype=torch.long),reduction='none')
    ref = 0
    if aw:
        ref += aw * compare(0,[1,2])
    if cw:
        start = 3 if aw else 1
        ref += cw * torch.stack([compare(p,range(start,start+4))
                                 for p in [0,*range(n-3,n)]]).mean(0)
    torch.testing.assert_close(got[:,0], ref)
    grads = torch.autograd.grad(got[:,0].sum(),z)[0]
    torch.testing.assert_close(grads, torch.autograd.grad(ref.sum(),z)[0])
    assert (grads[:,0]<0).all()
    if cw:
        assert (grads[:,-3:]<0).all()
    assert (grads[:,1:(-3 if cw else n)]>0).all()
    torch.testing.assert_close(loss(z+3.,d),got)  # no fixed threshold
    with pytest.raises(ValueError):
        loss(z[:,:-1],d)


def test_sampling_with_and_without_replacement_and_train_admission():
    labels=list('abcdef')
    pop=population(labels,6)
    q={a:{b:1/5 for b in labels if a!=b} for a in labels}
    for replace in [False,True]:
        d=Directions.from_objective(dict(OBJ,sample_with_replacement=replace))
        sampler=AnchorSampler(pop,labels,d.cards,12,contrast=d)
        positive,cards,negative=sampler.draw_contrast('b/0',1,13,q)
        assert (positive,cards,negative)==sampler.draw_contrast('b/0',1,13,q)
        assert all(x.startswith('b/') and x!='b/0' for x in positive)
        assert all(pop[x]['intent']==lab!='b' for lab,x in negative) and 'b' not in cards
        if not replace:
            assert len(set(positive))==len(set(cards))==len(set(c for c,_ in negative))==5
        else:
            q['b']=dict.fromkeys(q['b'],1e-16)
            q['b']['a']=1.-4e-16
            _,cards,negative=sampler.draw_contrast('b/0',1,13,q)
            assert cards==['a']*5 and [c for c,_ in negative]==['a']*5
    with pytest.raises(ValueError, match='without replacement'):
        AnchorSampler(population(labels,2),labels,5,12,contrast=Directions.from_objective(
            dict(OBJ,sample_with_replacement=False)))
    with pytest.raises(ValueError,match='TRAIN'):
        AnchorSampler(dict(pop,held=dict(pop['a/0'], id='held', official_split='test')),labels,5,12)


class Scorer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale=torch.nn.Parameter(torch.tensor(.1))

    def forward(self,inputs,positions,return_logits=False):
        label = inputs['input_ids'][:,0] % 10
        sound = inputs['input_features'][:,0,0].float()
        raw = ((label==sound).float()[:,None]*2.-1.)*self.scale
        return raw if return_logits else raw.sigmoid()

    def clip_grad_norm_(self,n):
        self.applied_clip = n
        return torch.nn.utils.clip_grad_norm_(self.parameters(),n)


def test_real_render_collate_update_parity_audit_and_ema_on_padded_batch(tmp_path):
    # Only one alternative class and one other positive recording. All five draws must survive.
    labels=['a','b']
    pop=population(labels)
    class Layout:
        assignments={uid:['first' if uid=='a/0' else 'other'] for uid in pop}
        def __call__(self,audio,uid,epoch):
            return audio
    calls=[]
    def prepare(item):
        uid,_=item
        calls.append(uid)
        return SimpleNamespace(cache_hit=True,inputs=dict(input_ids=torch.tensor([[9]]),
            input_features=torch.full((1,1,100), float(labels.index(pop[uid]['intent']))),
            input_features_mask=torch.tensor([[1]+[0]*99])))
    col=dict(pad_token_id=0,audio_token_id=9,decision_token_id=8,audio_chunk_size=100,audio_infer_window=100)
    cfg=dict(stage1_runtime={"grad_clip": .25}, instruction_objective=OBJ, microbatch=2,stop_update=1,mixed={'seed':3},
             triplet_prefetch=WINDOW['triplet_prefetch'], parity_probability_atol=1e-6,
             class_logit_ema=dict(WINDOW['class_logit_ema'],min_shared_coordinates=2))
    batch=dict(epoch=0,slots=[dict(id='a/0',weight=1),dict(id='b/0',weight=0)],real_queries=1)
    spec=dict(labels=labels,arms=dict(first=[0,1],other=[2,3]),
              catalog=[dict(prefix=[token],suffix=[8]) for token in (10,11,20,21)])
    ema=ClassLogitEMA(labels,**cfg['class_logit_ema'])
    stream=OnlineTripletStream([batch],pop,prepare,Layout(),spec,cfg,0,1,col)
    model=Scorer()
    opt=torch.optim.Adam(model.parameters(),lr=.01)
    dist.init_process_group('gloo',init_method='file://'+str(tmp_path/'init'),rank=0,world_size=1)
    try:
        stream.begin_epoch(0,ema)
        _,groups,count,version=next(stream.queue)
        rows,_,recordings,_=groups[0]
        assert len(rows)==16 and len({r['request_id'] for r in rows})==16
        assert len(calls)==sum(len(g[2]) for g in groups)
        assert all(uid in recordings for uid in ('a/0','a/1'))
        def identity(row):
            i=row['inputs']
            return int(i['input_ids'][0,0]),int(i['input_features'][0,0,0])
        assert list(map(identity,rows))==[(10,0)]+[(11,0)]*5+[(10,1)]*5+[(10,0)]*5
        metrics, parity=update(model,opt,groups,count,cfg,col,'test','cpu',1,4,ema)
        assert model.applied_clip == .25
        assert model.scale.item()>.1 and parity['max_abs']==0 and parity['inference_passes']==4
        assert version==0 and ema.updates==1
        assert ema.observation_counts.tolist()==[[1,5],[5,0]]
        torch.testing.assert_close(ema.values,torch.tensor([[.1,-.1],[-.1,0.]]))
        summary_cfg=dict(physical_devices=[0],global_batch=1,instruction_objective=OBJ)
        record=dict(metrics=metrics,real_queries=1,update=1,learning_rate=.01,preparation_seconds=.01,total_seconds=.1,peak_allocated=0,
                    optimizer_audit={'head':dict(numel=1,state='initialized',step=1)})
        summary=summarize([record],1,summary_cfg)
        assert summary['loss']==pytest.approx(summary['recomputed_loss'])
        metrics['jev/fixed_card_nce']+=1.
        with pytest.raises(ValueError,match='components'):
            summarize([record],1,summary_cfg)
    finally:
        stream.close()
        dist.destroy_process_group()
