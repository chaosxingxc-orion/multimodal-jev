"""Stage1 supervised contrast across matched recordings, with a distinct wrong-card probe."""
import math
import torch
from torch.nn import functional as F
from .context_audio_collate import collate_context_audio
from .card_contrast import KIND, Directions
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
        self.packed = None
        if 'packed_acoustics' in config:
            from .card_acoustic_pack import PackedAcoustics
            self.packed = PackedAcoustics(config['packed_acoustics'],population,collation['audio_token_id'])
        self.negative_cards = config['instruction_objective'].get('negative_cards_per_anchor')
        objective = config['instruction_objective']
        self.directions = Directions.from_objective(objective) if objective.get('kind') == KIND else None
        if self.directions and 'class_logit_ema' not in config:
            raise ValueError('card contrast requires online class-logit EMA')


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

    def group_plan(self, a, epoch, index, distribution):
        if self.directions:
            draws = self.sampler.draw_contrast(a, epoch, index, distribution)
            return self.directions.pairs(a, self.population[a]['intent'], *draws)
        p, n = (self.partners[a] if distribution is None else self.sampler.draw(a, epoch, index, distribution))
        if self.negative_cards is not None:
            n, wrong_labels = n
        pairs = [(uid, self.population[uid]['intent'], uid) for uid in (a, p, n)]
        if self.negative_cards is None:
            pairs.append((n, self.population[a]['intent'], a))
        else:
            pairs.extend((a, label, a) for label in wrong_labels)
        coords = (self.population[a]['intent'], *wrong_labels) if distribution is not None else None
        return pairs, coords

    def prepare(self, index, distribution=None):
        batch = self.batches[index]
        epoch = batch['epoch']
        slots = batch['slots'][self.rank::self.world]
        groups = []
        batch_inputs = {}
        for ordinal, slot in enumerate(slots):
            pairs, coords = self.group_plan(slot['id'], epoch, index, distribution)
            recording_ids = tuple(dict.fromkeys(audio for audio, _, _ in pairs))
            inputs = {}
            for uid in recording_ids:
                if self.packed is not None:
                    if uid not in batch_inputs:
                        batch_inputs[uid] = self.packed.read(uid)
                    inputs[uid] = batch_inputs[uid]
                    continue
                row = self.population[uid]
                request = self.layout(row['audio'], uid, epoch)
                prepared = self.preparer((request, row['audio_sha256']))
                if not prepared.cache_hit:
                    raise ValueError('published acoustic cache required')
                inputs[uid] = prepared.inputs
            rows = []
            for j,(audio,label,card) in enumerate(pairs):
                acoustic = inputs[audio]
                column = self.label_index[label]
                arm = self.layout.assignments[card][epoch]
                rows.append(self.render(acoustic,self.spec['arms'][arm][column],f'{index}/{self.rank}/{ordinal}/{j}'))
            group = (rows, slot['weight'], recording_ids)
            if coords is not None:
                group += (coords,)
            groups.append(group)
        return index,groups,batch['real_queries']

    def close(self):
        if self.queue is not None:
            self.queue.close()


class OnlineTripletStream(TripletStream):
    def __init__(self, *args):
        super().__init__(*args)
        from .intent_triplet_sampling import AnchorSampler
        self.sampler = AnchorSampler(self.population, self.spec['labels'], self.negative_cards,
                                     self.config['mixed']['seed'], contrast=self.directions)
        self.max_lag = self.config['class_logit_ema']['max_staleness_updates']
        if self.config['triplet_prefetch']['ahead'] > self.max_lag:
            raise ValueError('prefetch depth exceeds configured sampler lag')

    def begin_epoch(self, first, ema, pending=None):
        if self.queue is not None:
            self.queue.close()
        if ema.updates != first:
            raise ValueError('sampler and optimizer update differ')
        self.publish(ema)
        epoch = self.batches[first]['epoch']
        stop = next((i for i in range(first+1,self.config['stop_update'])
                     if self.batches[i]['epoch'] != epoch), self.config['stop_update'])
        if pending is not None and pending[0] != first-1:
            raise ValueError('recovered prefetched batch has the wrong EMA version')
        items = ((i, *(pending if i == first and pending is not None else self.snapshot)) for i in range(first, stop))
        self.queue = OrderedPrefetch(self.prepare_snapshot, items, **self.config['triplet_prefetch'])

    def publish(self, ema):
        self.snapshot = ema.updates, ema.distribution()

    def prepare_snapshot(self, item):
        index, version, distribution = item
        if not 0 <= index - version <= self.max_lag:
            raise ValueError('invalid sampler snapshot version')
        return (*self.prepare(index, distribution), version)


def update(model, optimizer, groups, global_count, config, collation, policy, device, world, parity_passes, class_ema=None):
    import torch.distributed as dist
    micro = config['microbatch']
    objective = config['instruction_objective']
    size = len(class_ema.labels) if class_ema is not None else 0
    fields = (1 + objective['negative_cards_per_anchor']) if class_ema is not None else 0
    directions = Directions.from_objective(objective) if objective.get('kind') == KIND else None
    metric_names = ('actor/pg_loss','jev/infonce','jev/gap','jev/anchor')
    if directions:
        from .context_objective_registry import resolve
        contrast_loss = resolve(KIND)
        metric_names = directions.metrics
        if class_ema is None:
            raise ValueError('card contrast requires online class-logit EMA')
        fields = directions.observed
    stat_width = len(metric_names) + 1
    reduction = torch.zeros(stat_width + world * len(groups) * fields * 2, device=device, dtype=torch.float32)
    if class_ema is not None:
        from .class_logit_ema import pack_observations, global_tables
        slots = reduction[stat_width:].view(world, len(groups), fields, 2)[dist.get_rank()]
    stats = reduction[:stat_width]
    parity = dict(inference_passes=0,compared=0,max_abs=0.)
    optimizer.zero_grad(set_to_none=True)
    collate = collate_context_audio
    if 'training_acoustic_engine' in config:
        from .card_training_acoustics import ENGINE, collate as unique_collate
        if not directions or config['training_acoustic_engine'] != ENGINE:
            raise ValueError('unregistered Stage1 acoustic engine')
        collate = unique_collate
    for start in range(0,len(groups),micro):
        chunk = groups[start:start+micro]
        rows = [dict(row,policy_id=policy) for group,*_ in chunk for row in group]
        inputs,positions = collate(rows,policy_id=policy,**collation)
        inputs = {k:v.to(device=device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
        positions = positions.to(device)
        probes = len(range(start//micro,parity_passes,len(groups)//micro))
        probe = probes > 0
        if probe:
            with torch.no_grad():
                parity_inputs, parity_positions = inputs,positions
                if 'training_acoustic_engine' in config:
                    native, parity_positions = collate_context_audio(rows,policy_id=policy,**collation)
                    parity_inputs = {k:v.to(device=device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in native.items()}
                    parity_positions = parity_positions.to(device)
                references = [model(parity_inputs,parity_positions).detach() for _ in range(probes)]
                del parity_inputs,parity_positions
        if directions:
            raw = model(inputs,positions,return_logits=True)
            if 'tower_witness' in config:
                digest = raw.detach().float().cpu().flatten().tolist()
            logits = raw.reshape(len(chunk), directions.width)
            terms = contrast_loss(logits, directions)
            pack_observations(slots[start:start+len(chunk)], logits[:, :fields], chunk, class_ema.label_index,
                              paired_coordinates=True)
        else:
            raw, markers = model(inputs,positions,return_logits=True,return_marker=True)
            if class_ema is not None:
                pack_observations(slots[start:start+len(chunk)], raw, chunk, class_ema.label_index)
            k = objective.get('negative_cards_per_anchor')
            width = 3 + (k if k is not None else 1)
            terms = components(markers.reshape(len(chunk),width,-1),raw.reshape(len(chunk),width),
                               objective['temperature'],objective['probability_margin'],k)
        if probe:
            error = max((raw.sigmoid()-reference).abs().max().item() for reference in references)
            if not 0 <= error <= config['parity_probability_atol']:
                raise ValueError(f'probability parity failed: {error}')
            parity.update(inference_passes=parity['inference_passes']+probes,compared=parity['compared']+probes*len(chunk),
                          max_abs=max(error,parity['max_abs']))
        weight = terms.new_tensor([group[1] for group in chunk])
        sums = (terms*weight[:,None]).sum(0)
        (sums[0]*world/global_count).backward()
        stats[:-1] += sums.detach()
        stats[-1] += weight.sum()
    dist.all_reduce(reduction)
    if class_ema is not None:
        reduction = reduction.cpu()
        stats = reduction[:stat_width]
        sums, counts = global_tables(reduction[stat_width:], size)
        if int(counts.diag().sum()) != global_count or int(counts.sum()) != global_count * fields:
            raise ValueError('anchor observation denominator differs from real TRAIN batch')
        class_ema.update(sums, counts)
    if stats[-1].item()!=global_count or not torch.isfinite(stats).all():
        raise ValueError('global count or objective invalid')
    norm = model.clip_grad_norm_(config['stage1_runtime']['grad_clip'])
    if not torch.isfinite(norm):
        raise ValueError('nonfinite gradient')
    optimizer.step()
    metrics = dict(zip(metric_names,
                       (stats[:-1]/global_count).tolist(),strict=True))
    metrics['actor/grad_norm'] = norm.item()
    if 'tower_witness' in config:
        parity['pair_logits'] = digest
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
