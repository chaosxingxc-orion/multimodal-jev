"""Stage1 supervised contrast across matched recordings, with a distinct wrong-card probe."""
import math
import torch
from torch.nn import functional as F
from .context_audio_collate import collate_context_audio
from .intent_triplet_sampling import sample_epoch
from .ordered_prefetch import OrderedPrefetch


def components(markers, logits, temperature, margin, negative_cards_per_anchor=None):
    # Within each anchor group: A/a, B/b, C/c (three matches), C/a (mismatch).
    a, b, c = F.normalize(markers[:, :3].float(), dim=-1).unbind(dim=1)
    nce = F.softplus(((a*c).sum(-1) - (a*b).sum(-1)) / temperature)
    matched, wrong = logits[:, :3].float(), logits[:, 3:].float()
    if negative_cards_per_anchor is None:
        gap = matched.sigmoid() - wrong.sigmoid()
    else:
        if wrong.shape[1] != negative_cards_per_anchor:
            raise ValueError('sampled card count differs from objective')
        gap = matched[:, :1].sigmoid() - wrong.sigmoid()
    separation = (margin['margin'] - gap).relu().mean(-1)
    hi = margin['decision_threshold'] + margin['margin']/2
    lo = margin['decision_threshold'] - margin['margin']/2
    anchor = ((math.log(hi/(1-hi))-matched).relu().mean(-1)
              + (wrong-math.log(lo/(1-lo))).relu().mean(-1))/2
    total = nce + margin['gap_weight']*separation + margin['anchor_weight']*anchor
    return torch.stack((total, nce, separation, anchor), dim=1)


class TripletStream:
    def __init__(self, batches, population, preparer, layout, spec, config, rank, world, collation):
        self.batches, self.population, self.preparer = batches, population, preparer
        self.layout, self.spec, self.rank, self.world = layout, spec, rank, world
        self.label_index = {label:i for i,label in enumerate(spec['labels'])}
        self.templates = [(torch.tensor(t['prefix']), torch.tensor(t['suffix'])) for t in spec['catalog']]
        self.config = config
        self.collation = collation
        self.queue = None
        self.partners = {}
        self.negative_cards = config['instruction_objective'].get('negative_cards_per_anchor')

    def begin_epoch(self, first, distribution=None):
        epoch = self.batches[first]['epoch']
        if epoch and distribution is None:
            raise ValueError('missing previous epoch confusion report')
        if self.queue is not None:
            self.queue.close()
        self.partners = {a:(p,n) for a,p,n in sample_epoch(self.population.values(),
            seed=self.config['mixed']['seed'],epoch=epoch,distribution=distribution,
            negative_cards_per_anchor=self.negative_cards)}
        stop = next((i for i in range(first+1,self.config['stop_update'])
                     if self.batches[i]['epoch']!=epoch),self.config['stop_update'])
        self.queue = OrderedPrefetch(self.prepare,range(first,stop),**self.config['triplet_prefetch'])

    def render(self, acoustic, template, uid):
        prefix,suffix = self.templates[template]
        sound = acoustic['input_ids'][0]
        ids = torch.cat((prefix,sound[sound==self.collation['audio_token_id']],suffix))[None,:]
        positions = (ids[0]==self.collation['decision_token_id']).nonzero().T.contiguous()
        return dict(inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),
                         input_features=acoustic['input_features'],input_features_mask=acoustic['input_features_mask']),
                    positions=positions,request_id=uid)

    def prepare(self, index):
        batch = self.batches[index]
        epoch = batch['epoch']
        slots = batch['slots'][self.rank::self.world]
        groups = []
        for ordinal, slot in enumerate(slots):
            a = slot['id']
            p,n = self.partners[a]
            if self.negative_cards is not None:
                n, wrong_labels = n
            recording_ids = (a,p,n)
            inputs = {}
            for uid in recording_ids:
                row = self.population[uid]
                request = self.layout(row['audio'], uid, epoch)
                prepared = self.preparer((request, row['audio_sha256']))
                if not prepared.cache_hit:
                    raise ValueError('published acoustic cache required')
                inputs[uid] = prepared.inputs
            rows = []
            pairs = [(uid, self.population[uid]['intent'], uid) for uid in recording_ids]
            if self.negative_cards is None:
                pairs.append((n, self.population[a]['intent'], a))
            else:
                # Use the anchor's template variant for both its true and its wrong cards.
                pairs.extend((a, label, a) for label in wrong_labels)
            for j,(audio,label,card) in enumerate(pairs):
                acoustic = inputs[audio]
                column = self.label_index[label]
                arm = self.layout.assignments[card][epoch]
                rows.append(self.render(acoustic,self.spec['arms'][arm][column],f'{index}/{self.rank}/{ordinal}/{j}'))
            groups.append((rows,slot['weight'],recording_ids))
        return index,groups,batch['real_queries']

    def close(self):
        if self.queue is not None:
            self.queue.close()


def update(model, optimizer, groups, global_count, config, collation, policy, device, world, parity_passes):
    import torch.distributed as dist
    micro = config['microbatch']
    objective = config['instruction_objective']
    stats = torch.zeros(5,device=device)
    parity = dict(inference_passes=0,compared=0,max_abs=0.)
    optimizer.zero_grad(set_to_none=True)
    for start in range(0,len(groups),micro):
        chunk = groups[start:start+micro]
        rows = [dict(row,policy_id=policy) for group,_,_ in chunk for row in group]
        inputs,positions = collate_context_audio(rows,policy_id=policy,**collation)
        inputs = {k:v.to(device=device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
        positions = positions.to(device)
        probes = len(range(start//micro,parity_passes,len(groups)//micro))
        probe = probes > 0
        if probe:
            with torch.no_grad():
                references = [model(inputs,positions).detach() for _ in range(probes)]
        raw, markers = model(inputs,positions,return_logits=True,return_marker=True)
        if probe:
            error = max((raw.sigmoid()-reference).abs().max().item() for reference in references)
            if not 0 <= error <= config['parity_probability_atol']:
                raise ValueError(f'probability parity failed: {error}')
            parity.update(inference_passes=parity['inference_passes']+probes,compared=parity['compared']+probes*len(chunk),
                          max_abs=max(error,parity['max_abs']))
        k = objective.get('negative_cards_per_anchor')
        width = 3 + (k if k is not None else 1)
        terms = components(markers.reshape(len(chunk),width,-1),raw.reshape(len(chunk),width),
                           objective['temperature'],objective['probability_margin'],k)
        weight = terms.new_tensor([w for _,w,_ in chunk])
        sums = (terms*weight[:,None]).sum(0)
        (sums[0]*world/global_count).backward()
        stats[:4] += sums.detach()
        stats[4] += weight.sum()
    dist.all_reduce(stats)
    if stats[4].item()!=global_count or not torch.isfinite(stats).all():
        raise ValueError('global count or objective invalid')
    norm = model.clip_grad_norm_(1.)
    if not torch.isfinite(norm):
        raise ValueError('nonfinite gradient')
    optimizer.step()
    metrics = dict(zip(('actor/pg_loss','jev/infonce','jev/gap','jev/anchor'),
                       (stats[:4]/global_count).tolist(),strict=True))
    metrics['actor/grad_norm'] = norm.item()
    return metrics, parity


def confusion_distribution(labels, sums, counts, exponent):
    """Symmetric class confusion with Zipf mass by rank, splitting ties fairly."""
    if (len(set(labels)) != len(labels) or len(labels)<2 or sums.shape != (len(labels),len(labels))
            or counts.shape != (len(labels),) or not (counts>0).all()
            or not torch.isfinite(sums).all() or not math.isfinite(exponent) or exponent <= 0):
        raise ValueError('complete measured confusion table and positive tail exponent required')
    means = sums.double()/counts[:,None]
    if ((means < 0) | (means > 1)).any():
        raise ValueError('each measured mean must be a probability')
    confusion = (means + means.T)/2
    result = {}
    for i,label in enumerate(labels):
        alternatives = [j for j in range(len(labels)) if j != i]
        values = confusion[i,alternatives]
        scores,order = values.sort(descending=True)
        ranks = torch.arange(1,len(scores)+1,dtype=torch.float64)
        weights = ranks.pow(-exponent)
        for tied in scores.unique():
            group = scores == tied
            weights[group] = weights[group].mean()
        weights /= weights.sum()
        result[label] = {labels[alternatives[j]]:w for j,w in zip(order.tolist(),weights.tolist(),strict=True)}
    return result


@torch.no_grad()
def mine_epoch(model, stream, config, epoch, device, policy, *, bootstrap=False):
    """All real TRAIN rows, all cards; every rank makes equal FSDP forward calls."""
    import torch.distributed as dist
    from collections import Counter
    labels = stream.spec['labels']
    ids = sorted(stream.population)
    if bootstrap:
        import random
        from collections import defaultdict
        by_class = defaultdict(list)
        for uid in ids:
            by_class[stream.population[uid]['intent']].append(uid)
        rng = random.Random(config['mixed']['seed'])
        limit = config['epoch_mining']['bootstrap_per_class']
        ids = sorted(uid for label in labels for uid in rng.sample(by_class[label],min(limit,len(by_class[label]))))
    rank, world = stream.rank, stream.world
    n = len(labels)
    sums = torch.zeros((n,n),dtype=torch.float64)
    counts = torch.zeros(n,dtype=torch.long)
    width = config['epoch_mining']['forward_rows']
    for offset in range(0,len(ids),world):
        actual = offset+rank < len(ids)
        uid = ids[min(offset+rank,len(ids)-1)]
        record = stream.population[uid]
        req = stream.layout(record['audio'],uid,epoch)
        prepared = stream.preparer((req, record['audio_sha256']))
        if not prepared.cache_hit:
            raise ValueError('epoch mining requires the frozen audio cache')
        arm = config['epoch_mining']['card_arm']
        values = []
        for start in range(0,n,width):
            rows = []
            for j in range(start,min(n,start+width)):
                ix = stream.spec['arms'][arm][j]
                rows.append(dict(stream.render(prepared.inputs,ix,str(j)),policy_id=policy))
            inputs,pos = collate_context_audio(rows,policy_id=policy,**stream.collation)
            inputs = {k:v.to(device=device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
            values.append(model(inputs,pos.to(device)).flatten().float().cpu())
        if actual:
            gold = stream.label_index[record['intent']]
            sums[gold] += torch.cat(values)
            counts[gold] += 1
    # The table is small: gather FP64 CPU totals without requiring a FP64 HCCL kernel.
    tables = [None]*world
    dist.all_gather_object(tables,(sums,counts))
    sums = sum(t[0] for t in tables)
    counts = sum(t[1] for t in tables)
    truth = Counter(stream.population[uid]['intent'] for uid in ids)
    if counts.tolist() != [truth[label] for label in labels]:
        raise ValueError('epoch mining denominator differs from frozen TRAIN')
    return dict(epoch=epoch,labels=labels,counts=counts.tolist(),probability_sums=sums.tolist(),
                distribution=confusion_distribution(labels,sums,counts,config['epoch_mining']['zipf_exponent']),pairs=int(counts.sum())*n)
