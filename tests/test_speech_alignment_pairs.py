import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import torch.distributed as dist
from multimodal_jev.baseline import file_hash
from multimodal_jev.card_contrast import bind_sample_counts, Directions
from multimodal_jev.class_logit_ema import ClassLogitEMA
from multimodal_jev.intent_triplet_training import OnlineTripletStream, update
from multimodal_jev.speech_alignment_pairs import build, Partners
from multimodal_jev.speech_pair_readout import extract, counter_names, tally
from multimodal_jev.card_training_acoustics import ENGINE
from multimodal_jev.context_decision_model import ContextDecisionModel
from multimodal_jev.context_decision_head import ContextDecisionHead
from test_card_speech_contrast import CountingBackbone

SETTINGS=dict(positive_cycle=['repeat','same_intent'],min_shared_words=2,control_arm='rules')
RULES=[dict(labels=['up','down'])]


def data():
    forms=dict(a=('up','turn it up'), p=('up','turn it up'), q=('up','louder'),
               n=('down','turn it down'), nn=('down','less volume'), x=('other','turn it off'),
               y=('other','turn it on'), m=('up','change volume'), z=('other','change volume'))
    pop={k:dict(id=k,role='learning',official_split='train',known_pending_source=False,audio=k,
                audio_sha256=k,intent=v[0]) for k,v in forms.items()}
    return pop,{k:v[1] for k,v in forms.items()}


def test_official_supervision_no_self_no_ambiguous_or_unrelated_controls():
    pop,texts=data()
    view=build(pop,texts,RULES,SETTINGS)
    index=Partners(view,pop)
    draws=[index.draws('a',epoch,317) for epoch in (0,1)]
    assert set(draws)=={('repeat','p','n'),('same_intent','q','n')}
    assert index.draws('m',0,3)==(None,None,None)
    assert index.draws('x',0,3)[2] is None
    assert index.draws('a',1,317)==draws[1]
    pop['p']['audio_sha256']='a'
    assert all(Partners(build(pop,texts,RULES,SETTINGS),pop).draws('a',e,317)[1]!='p' for e in range(2))
    pop['p']['official_split']='test'
    with pytest.raises(ValueError):
        build(pop,texts,RULES,SETTINGS)


def test_mixed_groups_ties_and_padding_have_independent_correctness_counts():
    c=json.loads(Path('configs/portable/operations/card-stage1-speech-alignment-211.json').read_bytes())
    bind_sample_counts(c); d=Directions.from_objective(c['window']['triplet_objective'])
    groups=[([None]*10,1,None,None,'repeat'),([None]*8,1,None,None,None),
            ([None]*10,1,None,None,'same_intent'),([None]*10,0,None,None,'repeat')]
    raw=torch.zeros(38)
    # g(A),h(A),g(N),g(P),h(N),h(P)
    raw[[0,1,4,7,8,9]]=torch.tensor([1.,0.,0.,1.,1.,0.])
    raw[18:28]=raw[:10];raw[18+8]=raw[18+4]  # N tied, so switch fails
    vectors=torch.ones(38,2);vectors[4]=-1     # second panel has tied similarities
    ix,counts=extract(raw,vectors,groups,d)
    assert len(ix)==32 and set(ix.tolist()).isdisjoint({8,9,26,27,36,37})
    t=tally(dict(zip(counter_names(),counts.tolist())),3)
    assert list(t['repeat'].values())==[1,1,1,1,1,1]
    assert list(t['same_intent'].values())==[1,0,1,0,0,0]
    # Distinct extreme logits can produce an identical deployed probability: do not count a win.
    raw[0:2]=torch.tensor([80.,60.])
    _,tied=extract(raw,vectors,groups,d)
    assert tally(dict(zip(counter_names(),tied.tolist())),3)['repeat']['cards_switched']==0
    _,empty=extract(raw[:8],vectors[:8],groups[1:2],d)
    assert empty.sum()==0


def test_pinned_stream_to_optimizer_two_audio_relations_and_ema(tmp_path):
    pop,texts=data()
    f=tmp_path/'index.json';f.write_text(json.dumps(build(pop,texts,RULES,SETTINGS)))
    c=json.loads(Path('configs/portable/operations/card-stage1-speech-alignment-211.json').read_bytes())
    bind_sample_counts(c); w=c['window']; labels=['up','down','other']
    cfg=dict(instruction_objective=w['triplet_objective'],triplet_prefetch=w['triplet_prefetch'],
             stage1_runtime={'grad_clip':1.0}, mixed={'seed':317}, speech_partners=dict(path=str(f),sha256=file_hash(f)),
             training_acoustic_engine=ENGINE,microbatch=3,parity_probability_atol=1e-5,class_logit_ema=dict(
             w['class_logit_ema'],min_shared_coordinates=2))
    class Layout:
        assignments={u:['examples','examples'] for u in pop}
        def __call__(self,a,uid,epoch): return uid
    torch.manual_seed(42)
    sounds={u:torch.randn(1,4,100) for u in pop}
    def prepare(req):
        return SimpleNamespace(cache_hit=True,inputs=dict(input_ids=torch.tensor([[8]]),
            input_features=sounds[req[0]],input_features_mask=(torch.arange(100)<8)[None].long()))
    col=dict(pad_token_id=0,audio_token_id=8,decision_token_id=9,audio_chunk_size=100,audio_infer_window=100)
    spec=dict(labels=labels,arms=dict(rules=[0,1,2],examples=[3,4,5]),catalog=[dict(prefix=[1,i],suffix=[9]) for i in (10,11,12,14,15,16)])
    stream=OnlineTripletStream([],pop,prepare,Layout(),spec,cfg,0,1,col)
    ema=ClassLogitEMA(labels,**cfg['class_logit_ema'])
    model=ContextDecisionModel(CountingBackbone().eval(),ContextDecisionHead(4,2,12),9).to(torch.bfloat16)
    model.clip_grad_norm_=lambda bound:torch.nn.utils.clip_grad_norm_(model.parameters(),bound)
    opt=torch.optim.SGD(model.parameters(),lr=.001)
    dist.init_process_group('gloo',init_method='file://'+str(tmp_path/'gloo'),rank=0,world_size=1)
    try:
        kinds=set()
        for epoch in (0,1):
            slots=[dict(id='a',weight=1),dict(id='x',weight=1),dict(id='a',weight=0)]
            groups=stream.render_slots(epoch,epoch,slots,ema.distribution())
            assert [len(g[0]) for g in groups]==[10,8,10]
            assert int(groups[0][0][0]['inputs']['input_ids'][0,1])==10
            assert int(groups[1][0][0]['inputs']['input_ids'][0,1])==16
            kind=groups[0][4];kinds.add(kind)
            metrics,_=update(model,opt,groups,2,cfg,col,'policy','cpu',1,0,ema)
            assert tally(metrics,2)[kind]['panels']==1 and metrics['actor/grad_norm']>0
        assert kinds=={'repeat','same_intent'} and ema.updates==2
        assert ema.observation_counts.diag().sum()==4
    finally:
        dist.destroy_process_group()


def test_single_endpoint_uses_independent_stage1_proof_without_a_stage2(tmp_path):
    from tools.reports.prepare_paired_card_evaluation import single_endpoint
    def emit(name, value):
        path=tmp_path/name
        path.write_text(json.dumps(value))
        return dict(path=str(path),sha256=file_hash(path))
    card=emit('cards.json',{})
    train=emit('training.json',dict(model='model',base_id='base',instruction_objective=dict(kind='speech'),
               token_buckets={},batch_config=dict(epochs=2),single_card=card))
    manifest=emit('endpoint-manifest.json',dict(policy_id='P',contract={}))
    inp=emit('inputs.json',dict(cards=card))
    proof=dict(passed=True,container_exited=True,exit_code=0,host='host',policy_id='P',policy='/remote/trained',
               epoch_end_updates=[3,7],updates=7,config_sha256=train['sha256'],inputs_sha256=inp['sha256'],
               manifest_sha256=manifest['sha256'])
    r=emit('receipt.json',proof)
    c=dict(deployment=dict(host='host'),old_parent=emit('old.json',{}),
           stage1_only=dict(receipt=r['path'],objective='speech'))
    bound,parent,policies,training_args,mounts=single_endpoint(c)
    assert bound['cards']==card and parent['source_policy_id']=='P'
    assert policies=={'stage1':'P'} and training_args=={} and mounts=={'/remote/trained'}
    with pytest.raises(ValueError):
        single_endpoint(dict(c,stage2_updates=5))
    emit('receipt.json',dict(proof,policy_id='old'))
    with pytest.raises(ValueError):
        single_endpoint(c)
