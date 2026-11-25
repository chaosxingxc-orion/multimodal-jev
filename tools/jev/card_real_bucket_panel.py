"""Freeze real TRAIN pair plans for shape measurements, without writing to source caches."""
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.card_acoustic_pack import PackedAcoustics
from multimodal_jev.card_contrast import Directions
from multimodal_jev.class_logit_ema import ClassLogitEMA
from multimodal_jev.intent_triplet_sampling import AnchorSampler
from multimodal_jev.stage1_mixed_schedule import schedule


def read(path, sha):
    if file_hash(path) != sha:
        raise ValueError(f'changed source: {path}')
    return json.loads(Path(path).read_bytes())


def inputs(c):
    entry = Path(c['mixed']['entry'])
    release = Path(c['release'])
    manifest = read(release/'MANIFEST.json', c['mixed']['real_release_sha256'])
    if file_hash(release/'learning.jsonl') != manifest['files']['learning.jsonl']:
        raise ValueError('changed TRAIN membership')
    real = [json.loads(l) for l in (release/'learning.jsonl').read_text().splitlines()]
    adm = read(entry/'admission.json',c['release_sha256'])
    extra = read(entry/'synthetic.json',adm['roster_sha256'])
    population = {r['id']:r for r in real+extra}
    if len(population) != len(real)+len(extra) or len(population)!=c['batch_config']['source_queries']:
        raise ValueError('TRAIN cardinality differs')
    cards = read(c['single_card']['path'], c['single_card']['sha256'])
    plan = read(entry/'card-plan.json',cards['plan'])
    return population,cards,plan,len(real),len(extra)


def prepare(spec,c,out):
    pop,cards,plan,n_real,n_synth = inputs(c)
    batches, manifest = schedule(pop,c['batch_config'],c['mixed']['seed'])
    if manifest != read(Path(c['mixed']['entry'])/'mixed-schedule.json',c['schedule_sha256']):
        raise ValueError('cannot reproduce source order')
    idx=read(c['packed_acoustics']['path'],c['packed_acoustics']['sha256'])['entries']
    if idx.keys()!=pop.keys() or any(idx[k][3]!=r['audio_sha256'] for k,r in pop.items()):
        raise ValueError('audio identities differ')
    cards_len=[len(t['prefix'])+len(t['suffix']) for t in cards['catalog']]
    label_index={k:i for i,k in enumerate(cards['labels'])}
    epoch=spec['epoch']
    q=ClassLogitEMA(cards['labels'],**c['class_logit_ema']).distribution()
    report=dict(real=n_real,synthetic=n_synth,epoch=epoch,ema_updates=0,schedule=manifest,
        card_sha256=c['single_card']['sha256'],acoustic_index_sha256=c['packed_acoustics']['sha256'],recipes={})
    for name,counts in spec['recipes'].items():
        pos,neg_cards,neg_audio=counts
        d=Directions.from_objective(dict(c['instruction_objective'],positive_recordings_per_anchor=pos,
            negative_cards_per_anchor=neg_cards,negative_recordings_per_anchor=neg_audio))
        sampler=AnchorSampler(pop,cards['labels'],neg_cards,c['mixed']['seed'],contrast=d)
        def draw(uid,step):
            pairs,_=d.pairs(uid,pop[uid]['intent'],*sampler.draw_contrast(uid,epoch,step,q))
            return [(a,cards['arms'][plan['assignments'][card][epoch]][label_index[label]]) for a,label,card in pairs]
        groups=[]; hist=Counter()
        for step,b in enumerate(batches):
            if b['epoch']!=epoch:continue
            for s in b['slots']:
                if not s['weight']:continue
                uid=s['id']
                lens=[idx[a][4]+cards_len[t] for a,t in draw(uid,step)]
                hist.update(lens);groups.append((max(lens),uid,step))
        if len(groups)!=len(pop) or len({uid for _,uid,_ in groups})!=len(pop) or hist.total()!=len(pop)*d.width:
            raise ValueError('incomplete full TRAIN sweep')
        groups.sort()
        ah=Counter(g[0] for g in groups)
        def summary(h):
            vals=sorted(h.elements())
            return dict(n=len(vals),quantiles={str(q):vals[math.ceil(len(vals)*q)-1] for q in spec['quantiles']},
                        at_or_below_short=sum(v for k,v in h.items() if k<=spec['cutoff_diagnostic']),
                        histogram=dict(sorted(h.items())))
        panels=[];prev=0
        for band in spec['bands']:
            limit=groups[math.ceil(len(groups)*band)-1][0]
            choices=[g for g in groups if prev<g[0]<=limit]
            peak=next(g for g in choices if g[0]==limit)
            others=random.Random(f"{spec['seed']}|{name}|{band}").sample([g for g in choices if g!=peak],max(spec['widths'])-1)
            panels.append(dict(upper=limit,lower=prev+1,population=len(choices),anchors=[
                dict(expected=n,uid=uid,pairs=draw(uid,step)) for n,uid,step in [peak,*others]]))
            prev=limit
        report['recipes'][name]=dict(pairs=summary(hist),anchors=summary(ah),
            bands=[{k:v for k,v in row.items() if k!='anchors'} for row in panels])
        (out/f'panel-{name}.json').write_text(json.dumps(panels))
    (out/'distribution.json').write_text(json.dumps(report))


def render(c, panel, width, audio_token, marker):
    import torch
    pop,cards,_,_,_=inputs(c)
    pack=PackedAcoustics(c['packed_acoustics'],pop,audio_token)
    templates=[tuple(torch.tensor(t[k]) for k in ('prefix','suffix')) for t in cards['catalog']]
    cache={};rows=[]
    for i,g in enumerate(panel['anchors'][:width]):
        lengths=[]
        for j,(uid,t) in enumerate(g['pairs']):
            if uid not in cache:cache[uid]=pack.read(uid)
            x=cache[uid];pre,post=templates[t]
            ids=torch.cat((pre,x['input_ids'][0],post))[None]
            pos=(ids[0]==marker).nonzero().T.contiguous()
            rows.append(dict(request_id=f'{i}/{j}',policy_id='probe',positions=pos,
                inputs=dict(x,input_ids=ids,attention_mask=torch.ones_like(ids))))
            lengths.append(ids.shape[1])
        if max(lengths)!=g['expected']:raise ValueError('rendered and indexed shapes differ')
    return rows,len(cache)


def measure(spec,c,wrapped,base,processor,model,device):
    import statistics
    import time
    import torch
    from multimodal_jev.qwen3_asr_provider import Qwen3ASR
    from multimodal_jev.card_training_acoustics import collate
    from multimodal_jev.card_contrast import components
    lane=spec['real_lane'];recipe=lane['recipe'];width=lane['width']
    panels=json.loads(Path(lane['panel']).read_bytes())
    pos,cards,audio=spec['recipes'][recipe]
    d=Directions.from_objective(dict(c['instruction_objective'],positive_recordings_per_anchor=pos,
        negative_cards_per_anchor=cards,negative_recordings_per_anchor=audio))
    col=Qwen3ASR.collation(base,processor,model)
    opt=torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],lr=c['lr_schedule']['peak_lr'])
    for band,panel in enumerate(panels):
        rows,owners=render(c,panel,width,col['audio_token_id'],col['decision_token_id'])
        inp,positions=collate(rows,'probe',**col)
        assert inp['input_ids'].shape==(width*d.width,panel['upper'])
        inp={k:v.to(device) for k,v in inp.items()};positions=positions.to(device)
        times=[]
        for i in range(spec['warmup']+spec['repeats']):
            opt.zero_grad(set_to_none=True);torch.npu.synchronize();torch.npu.reset_peak_memory_stats()
            start=time.monotonic()
            logits=wrapped(inp,positions,return_logits=True).reshape(width,d.width)
            loss=components(logits,d)[:,0].mean();loss.backward()
            norm=wrapped.clip_grad_norm_(1.);opt.step();torch.npu.synchronize()
            dt=time.monotonic()-start
            assert torch.isfinite(loss).item() and torch.isfinite(norm).item()
            if i>=spec['warmup']:times.append(dt)
        print('BAND '+json.dumps(dict(recipe=recipe,band=band,width=width,tokens=panel['upper'],owners=owners,
            times=times,rate=width/statistics.median(times),allocated=torch.npu.max_memory_allocated(),
            reserved=torch.npu.max_memory_reserved(),policy=c['source_policy_id'])),flush=True)
        del rows, inp, positions, logits, loss, norm


if __name__=='__main__':
    prepare(json.loads(Path(sys.argv[1]).read_bytes()),json.loads(Path(sys.argv[2]).read_bytes()),Path(sys.argv[3]))
