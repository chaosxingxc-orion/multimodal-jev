"""Deterministic TRAIN-only Stage1 difficulty view; immutable inputs stay shared.

This changes the Stage2 optimization distribution intentionally. No inverse
propensity claim, no selection/test feedback and no synthetic false labels.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from .baseline import file_hash
from .context_reference_dataset import digest
from .context_topology_batch_plan import derive
from .span_class_spread import class_spread_indices

SCHEMA='jev-stage2-difficulty-view-v1'


def validate_options(c):
    if c.get('mode')=='paired_train_v1':
        from .stage2_paired_sampling import validate_options as validate_paired
        return validate_paired(c)
    if set(c)!={'seed','decision_threshold','probability_margin','easy_fraction','min_easy_per_class'}:
        raise ValueError('explicit complete difficulty sampling options required')
    if (type(c['seed']) is not int or c['seed']<0 or type(c['min_easy_per_class']) is not int
            or c['min_easy_per_class']<1):raise ValueError('seed and positive easy class floor required')
    for k in ('decision_threshold','probability_margin','easy_fraction'):
        if type(c[k]) not in (int,float) or not math.isfinite(c[k]):raise ValueError('finite sampling values required')
    if not (0<=c['probability_margin']<1 and c['probability_margin']/2<c['decision_threshold']<1-c['probability_margin']/2 and 0<c['easy_fraction']<1):
        raise ValueError('sampling values outside allowed range')


def classify(prob, gold, c):
    if (len(prob)<2 or type(gold) is not int or not 0<=gold<len(prob)
            or any(type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1 for p in prob)):
        raise ValueError('complete finite probabilities and trusted gold required')
    best_wrong=max(p for i,p in enumerate(prob) if i!=gold)
    gap=prob[gold]-best_wrong
    ranking_correct=gap>0  # a tied top score is not a confidently correct answer
    exact=all((p>=c['decision_threshold'])==(i==gold) for i,p in enumerate(prob))
    anchored=prob[gold]>=c['decision_threshold']+c['probability_margin']/2 and best_wrong<=c['decision_threshold']-c['probability_margin']/2
    bucket='error' if not ranking_correct or not exact else 'boundary' if gap<c['probability_margin'] or not anchored else 'easy'
    return dict(bucket=bucket,ranking_correct=ranking_correct,exact_set=exact,anchored=anchored,
        gold_probability=prob[gold],best_wrong_probability=best_wrong,gold_margin=gap)


def classify_row(item):
    row,prob,gold,options=item
    return dict(id=row['id'],intent=row['intent'],**classify(prob,gold,options))


def sample(rows, candidate_ids, probabilities, options, *, execution=None, timings=None):
    validate_options(options)
    if options.get('mode')=='paired_train_v1':
        from .stage2_paired_sampling import sample as sample_paired
        return sample_paired(rows,candidate_ids,probabilities,options,execution=execution,timings=timings)
    if len(set(candidate_ids))!=len(candidate_ids) or len(candidate_ids)<2:raise ValueError('unique candidate order required')
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(probabilities) or not rows:raise ValueError('exact full TRAIN score coverage required')
    easy=defaultdict(list);difficult=[]
    def key(uid):return hashlib.sha256(json.dumps([options['seed'],uid]).encode()).hexdigest()
    candidate_index={name:i for i,name in enumerate(candidate_ids)}
    jobs=[]
    for r in sorted(rows,key=lambda r:r['id']):
        if (r.get('role')!='learning' or r.get('official_split')!='train'
                or r.get('known_pending_source') is not False or r['intent'] not in candidate_index):
            raise ValueError('admitted clean learning rows only; selection/test forbidden')
        p=probabilities[r['id']]
        if len(p)!=len(candidate_ids):raise ValueError('candidate coverage differs')
        jobs.append(({k:r[k] for k in ('id','intent')},p,candidate_index[r['intent']],options))
    if execution is None:
        diagnostics=[classify_row(job) for job in jobs]
    else:
        from .context_execution import map_cpu
        diagnostics,stats=map_cpu(classify_row,jobs,workers=execution['cpu_workers'],
            min_parallel=execution['min_parallel'],chunk=execution['chunk'],window=execution['window'])
        if timings is not None:timings['classify']={k:v for k,v in stats.items() if k!='task_seconds'}
    for d in diagnostics:
        if d['bucket']=='easy':easy[d['intent']].append(d['id'])
        else:difficult.append(d['id'])
    for group in easy.values():group.sort(key=lambda uid:(key(uid),uid))
    selected_easy=set()
    for group in easy.values():selected_easy.update(group[:options['min_easy_per_class']])
    # Ratio is a target; class floors can exceed it. Never duplicate a sample.
    target=min(sum(map(len,easy.values())),max(len(selected_easy),math.ceil(len(difficult)*options['easy_fraction']/(1-options['easy_fraction']))))
    remaining=sorted((uid for group in easy.values() for uid in group if uid not in selected_easy),key=lambda uid:(key(uid),uid))
    selected_easy.update(remaining[:target-len(selected_easy)])
    chosen=set(difficult)|selected_easy
    selected=sorted(chosen,key=lambda uid:(key(uid),uid))
    for d in diagnostics:d['selected']=d['id'] in chosen
    counts=Counter(d['bucket'] for d in diagnostics)
    classes=sorted({r['intent'] for r in rows})
    per_class={label:dict(population=0,selected=0,buckets=dict(error=0,boundary=0,easy=0),selected_easy=0) for label in classes}
    for d in diagnostics:
        item=per_class[d['intent']];item['population']+=1;item['selected']+=int(d['selected'])
        item['buckets'][d['bucket']]+=1;item['selected_easy']+=int(d['selected'] and d['bucket']=='easy')
    gaps=sorted(d['gold_margin'] for d in diagnostics)
    def quantile(q):
        position=q*(len(gaps)-1);low=int(position);high=min(low+1,len(gaps)-1)
        return gaps[low]+(gaps[high]-gaps[low])*(position-low)
    quality=dict(top1_strict=sum(d['ranking_correct'] for d in diagnostics)/len(rows),
        exact_set=sum(d['exact_set'] for d in diagnostics)/len(rows),
        margin_coverage=sum(d['gold_margin']>=options['probability_margin'] for d in diagnostics)/len(rows),
        anchor_coverage=sum(d['anchored'] for d in diagnostics)/len(rows),
        gap_quantiles={str(q):quantile(q) for q in (0,.05,.25,.5,.75,.95,1)})
    return selected,diagnostics,dict(population=len(rows),selected=len(selected),buckets=dict(counts),
        selected_easy=len(selected_easy),selected_difficult=len(difficult),
        actual_easy_fraction=len(selected_easy)/len(selected),requested_easy_fraction=options['easy_fraction'],
        class_floor_applied=True,replacement=False,all_difficult_retained=True,
        no_difficult_samples=not difficult,quality=quality,
        by_class=per_class,
        distribution='uniform_over_selected_view_not_unbiased_full_train')


def make_schedule(rows, selected_ids, batch_config, seed):
    plan=derive(batch_config);lookup={r['id']:r for r in rows}
    if len(lookup)!=len(rows) or any(r.get('role')!='learning' or r.get('official_split')!='train' or r.get('known_pending_source') is not False for r in rows):raise ValueError('unique clean learning population required')
    if (len(selected_ids)!=len(set(selected_ids)) or not set(selected_ids)<=set(lookup)
            or len(selected_ids)!=batch_config['source_queries']):raise ValueError('view population differs from topology')
    ordered=[lookup[uid] for uid in sorted(selected_ids)];batches=[]
    for epoch in range(batch_config['epochs']):
        indices=class_spread_indices([r['intent'] for r in ordered],seed=seed,epoch=epoch)
        real=[dict(id=ordered[i]['id'],weight=1.,padding=False) for i in indices];slots=list(real)
        while len(slots)%plan['global_batch']:
            slots.append(dict(real[(len(slots)-len(real))%len(real)],weight=0.,padding=True))
        for start in range(0,len(slots),plan['global_batch']):
            group=slots[start:start+plan['global_batch']]
            batches.append(dict(epoch=epoch,index=start//plan['global_batch'],slots=group,real_queries=sum(not s['padding'] for s in group)))
    return batches


def publish(root, *, rows, task, reference, reference_binding, options, batch_config):
    from .local_checkpoint import commit_json
    candidates=[x['id'] for x in task['schema']['candidates']]
    if reference_binding['manifest_sha256']!=reference.manifest_sha256 or reference_binding['identity']!=reference.identity or reference_binding['population']!=reference.population:raise ValueError('reference binding differs')
    if reference.identity['candidate_ids']!=candidates or set(reference.population)!={r['id'] for r in rows}:
        raise ValueError('same task and full TRAIN reference required')
    probs={r['id']:reference.read_sample(r['id'])['prob'] for r in rows}
    selected,diagnostics,summary=sample(rows,candidates,probs,options)
    config=dict(batch_config,source_queries=len(selected));plan=derive(config)
    if config['inference_instances']!=2 or plan['dp_replicas']!=2:raise ValueError('Stage2 DP2 plus two engines required')
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    payload=dict(selected_ids=selected,diagnostics=diagnostics,summary=summary)
    commit_json(root/'selection.json',payload)
    commit_json(root/'batches.json',make_schedule(rows,selected,config,options['seed']))
    manifest=dict(schema=SCHEMA,reference_manifest_sha256=reference.manifest_sha256,
        reference_identity=reference.identity,release_sha256=reference.identity['release_sha256'],
        full_learning_rows_sha256=digest(rows),options=options,batch_config=config,batch_plan=plan,
        selected_ids_sha256=digest(selected),files={n:file_hash(root/n) for n in ('selection.json','batches.json')},
        reference_storage='reuse_full_readonly_dataset',training_dispatch_allowed=False)
    commit_json(root/'MANIFEST.json',manifest)
    return dict(root=str(root),manifest_sha256=file_hash(root/'MANIFEST.json'),summary=summary,batch_plan=plan)


def load(root, manifest_sha256, *, rows, task, reference_binding, batch_config):
    root=Path(root)
    if any((root/n).is_symlink() for n in ('MANIFEST.json','selection.json','batches.json')) or file_hash(root/'MANIFEST.json')!=manifest_sha256:
        raise ValueError('pinned regular training view required')
    m=json.loads((root/'MANIFEST.json').read_text())
    if (m['schema']!=SCHEMA or m['files'].keys()!={'selection.json','batches.json'}
            or any(file_hash(root/n)!=h for n,h in m['files'].items())
            or m['full_learning_rows_sha256']!=digest(rows) or m['batch_config']!=batch_config
            or m['batch_plan']!=derive(batch_config)
            or m['reference_manifest_sha256']!=reference_binding['manifest_sha256']
            or m['reference_identity']!=reference_binding['identity']
            or set(reference_binding['population'])!={r['id'] for r in rows}
            or m['reference_identity']['candidate_ids']!=[x['id'] for x in task['schema']['candidates']]):
        raise ValueError('training view source/reference/configuration differs')
    validate_options(m['options'])
    if m['release_sha256']!=reference_binding['identity']['release_sha256']:raise ValueError('view release differs')
    data=json.loads((root/'selection.json').read_text());ids=data['selected_ids']
    if digest(ids)!=m['selected_ids_sha256']:raise ValueError('selected IDs changed')
    batches=json.loads((root/'batches.json').read_text())
    if batches!=make_schedule(rows,ids,batch_config,m['options']['seed']):raise ValueError('view schedule differs')
    chosen=set(ids)
    return [r for r in rows if r['id'] in chosen],batches,m


def training_inputs(c, rows, task, *, reference=None, execution=None):
    """Shared trainer opt-in; full source remains available for mirror checks."""
    if c.get('training_dataset') is not None:
        from .stage2_dataset_release import load_training_dataset
        if reference is None or execution is None:
            raise ValueError('authenticated full reference and explicit execution required')
        data=load_training_dataset(c,rows=rows,task=task,reference=reference,execution=execution)
        return data.population,data.batches,data.consumer
    if c.get('training_view') is None:
        from .context_replica_schedule import load_schedule
        population={r['id']:r for r in rows}
        batches,m=load_schedule(c['schedule'],c['schedule_sha256'],release_sha256=c['release_sha256'],population=population,batch_config=c['batch_config'],parent_root=c['parent_schedule'],parent_manifest_sha256=c['parent_schedule_sha256'],parent_release_sha256=c['schedule_release_sha256'])
        return population,batches,m
    view=c['training_view']
    if set(view)!={'root','manifest_sha256'} or c['stage']!='task':raise ValueError('Stage2 view only')
    selected,batches,m=load(view['root'],view['manifest_sha256'],rows=rows,task=task,reference_binding=c['reference_dataset'],batch_config=c['batch_config'])
    if m['release_sha256']!=c['release_sha256'] or c['schedule_sha256']!=view['manifest_sha256']:raise ValueError('view release or schedule binding differs')
    return {r['id']:r for r in selected},batches,m
