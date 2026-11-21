"""Compare pre-registered Stage1/direct/TPO endpoints with identical frozen audio inputs."""
from pathlib import Path
import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.model_provider import resolve_provider, require_same_provider
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from multimodal_jev.context_policy_snapshot import load_policy
from multimodal_jev.cached_context_preparation import processor_cache_identity
from tools.reports.run_native_seal_endpoint import load, read, score


def run(spec):
    out=Path(spec['output'])
    out.mkdir(parents=True,exist_ok=False)
    parent=read(spec['parent'])
    if (file_hash(spec['parent']) != spec['parent_sha'] or
            checkpoint_identity(parent['model'])[0]!=parent['base_id']):
        raise ValueError('training parent or base changed')
    c=dict(spec['endpoint'],base_id=parent['base_id'],model=parent['model'],contract=parent['source_contract'],
           snapshot=parent['source_policy'],physical_devices=spec['devices'],policy_id=parent['source_policy_id'])
    require_same_provider(parent,c['cache'])
    if 'model_provider' in c:
        require_same_provider(parent,c)
    if 'model_provider' in parent:
        c['model_provider'] = parent['model_provider']
    provider = resolve_provider(c)
    c['cache']['processor']['policy']=c['snapshot']
    stage1=read_execution_bundle(c['snapshot'],expected_policy_id=c['policy_id'],
        expected_contract=c['contract'],expected_base_id=c['base_id'])
    if 'calibration' in spec:
        from tools.reports.calibrate_tpo_probabilities import cached_source
        assets, candidates, encode = cached_source(spec['calibration'], parent)
    else:
        assets, candidates, encode = load(c)
    # Each container owns one card and one frozen dataset. Reuse model construction,
    # audited input tensors and candidate layout over all three endpoints.
    width=spec['batch']
    index=spec['shard']
    assets=assets[index::spec['shards']]
    processor=processor_cache_identity(c['snapshot'])
    rows = [] if 'calibration' in spec else [(asset, encode(asset)[0]) for asset in assets]
    model,device,base,handle=provider.load_reference(c,stage1,out)
    tokenizer_processor = provider.processor(stage1['processor'])
    collate = provider.collation(base,tokenizer_processor,model)
    arms=spec.get('arms',['stage1','direct','tpo'])
    checkpoints=spec.get('checkpoints', {})
    external=spec.get('audited_endpoints', {})
    if not arms or len(set(arms))!=len(arms) or set(arms)-{'stage1','direct','tpo',*checkpoints,*external}:
        raise ValueError('invalid evaluation endpoints')
    sources={'stage1':(c['snapshot'],c['policy_id'])} if 'stage1' in arms else {}
    for arm in (a for a in arms if a!='stage1'):
        if arm in external:
            pin = external[arm]
            root = Path(pin['root'])
            receipt_path = root/'receipt.json'
            if file_hash(receipt_path) != pin['receipt_sha256']:
                raise ValueError('registered endpoint receipt changed')
            receipt = read(receipt_path)
            manifest = root/'policy/MANIFEST.json'
            if (receipt['endpoint'] != pin['policy_id'] or receipt['passed'] is not True
                    or receipt['window'] != 'formal' or receipt['updates'] != pin['updates']
                    or file_hash(manifest) != receipt['manifest_sha256']):
                raise ValueError('registered endpoint differs from audited training')
            sources[arm] = (str(manifest.parent), pin['policy_id'])
            continue
        if arm in checkpoints:
            proof = checkpoints[arm]
            path = Path(spec['trained'])/'tpo'/f'policy-{arm}'
            if file_hash(path/'MANIFEST.json') != proof['manifest_sha256']:
                raise ValueError('audited checkpoint manifest changed')
            sources[arm] = (str(path), proof['policy_id'])
            continue
        root=Path(spec['trained'])/arm
        receipt=read(root/'receipt.json')
        if (receipt['binding']!=spec['binding'] or not receipt['passed'] or receipt['arm']!=arm or
                receipt['window']!='formal' or receipt['updates']!=spec['updates'] or
                file_hash(root/'policy/MANIFEST.json')!=receipt['manifest_sha256']):
            raise ValueError('unverified terminal policy')
        sources[arm]=(str(root/'policy'),receipt['endpoint'])
    if 'choice' in spec:
        from tools.reports.evaluate_choice_tasks import evaluate
        evaluate(spec,c,parent,model,device,tokenizer_processor,collate,rows,sources,provider)
        if spec.get('choice_only'):
            return dict(dataset=c['cache']['dataset'],shard=index,recordings=len(rows))
        if spec.get('full_ontology_only'):
            from tools.reports.evaluate_choice_tasks import with_baseline
            sources=with_baseline(spec,sources)
    for arm,(path,pid) in sources.items():
        if processor_cache_identity(path)!=processor:
            raise ValueError('changed processor invalidates paired evaluation')
        load_policy(path,model,expected_policy_id=pid,expected_contract=c['contract'],expected_base_id=c['base_id'])
        predictions=[]
        from contextlib import nullcontext
        from itertools import islice
        from multimodal_jev.ordered_prefetch import OrderedPrefetch
        newly_encoded = not rows
        stream = OrderedPrefetch(lambda asset: (asset, encode(asset)[0]), assets,
                                 workers=spec['io_workers'], ahead=spec['prefetch']) if newly_encoded else nullcontext(rows)
        with torch.inference_mode(), stream as queue:
            inputs = iter(queue)
            while group := list(islice(inputs, width)):
                if newly_encoded:
                    rows.extend(group)
                items=[dict(row,request_id=asset['id'],policy_id=pid) for asset,row in group]
                z=provider.forward(model,items,device=device,dtype=torch.bfloat16,policy_id=pid,**collate)
                prob=z.float().sigmoid().cpu()
                if not predictions:
                    single=provider.forward(model,items[:1],device=device,dtype=torch.bfloat16,
                        policy_id=pid,**collate).float().sigmoid().cpu()
                    if not torch.equal(prob[:1],single):
                        raise ValueError('batched endpoint differs from singleton')
                if not torch.isfinite(prob).all():
                    raise ValueError('nonfinite endpoint probabilities')
                predictions.extend(dict(id=item['request_id'],prob=v) for item,v in zip(items,prob.tolist(),strict=True))
        commit_json(out/(arm+'.json'),dict(policy_id=pid,records=predictions))
    dataset = 'source_train_calibration' if 'calibration' in spec else c['cache']['dataset']
    return dict(dataset=dataset,shard=index,recordings=len(assets))


def collect_predictions(shards, filename, assets):
    records, policies = {}, set()
    for part in shards:
        if isinstance(part, dict):
            from tools.reports.run_native_seal_endpoint import bound
            publication = bound(part)
        else:
            publication = read(Path(part)/filename)
        policies.add(publication['policy_id'])
        for row in publication['records']:
            if row['id'] in records:
                raise ValueError('overlapping evaluation shards')
            records[row['id']] = row
    if len(policies) != 1 or set(records) != {asset['id'] for asset in assets}:
        raise ValueError('incomplete endpoint populations')
    return records, policies.pop()


def collect(spec):
    # score() checks the complete ordered test population and independent sealed gold.
    from tools.reports.prepare_native_seal_cache import plan
    if 'calibration_fit' in spec:
        from tools.reports.calibrate_tpo_probabilities import collect as calibrate
        if file_hash(spec['parent']) != spec['parent_sha']:
            raise ValueError('calibration parent changed')
        return calibrate(spec, read(spec['parent']))
    result={}
    thresholds = None
    if 'thresholds' in spec:
        thresholds = read(spec['thresholds'])
        if (thresholds['schema'] != 'jev-source-train-harness-thresholds-v1'
                or thresholds['roster_sha256'] != spec['calibration_roster_sha256']):
            raise ValueError('frozen source TRAIN calibration required')
    for dataset,c in spec['endpoints'].items():
        jobs=plan(c['cache'])
        assets=[dict(id=r['id'],audio_sha256=r['audio_sha256']) for _,r,_ in jobs]
        candidates=[q.id for q in jobs[0][2].questions]
        if 'probability_panels' in spec:
            from tools.reports.score_probability_panels import score_panel
            inferred = spec.get('inferred_arms', {}).get(dataset, spec['arms'])
            predictions = {arm: collect_predictions(spec['shards'][dataset], arm+'.json', assets)
                           for arm in inferred}
            for arm, proof in spec.get('reused_predictions', {}).get(dataset, {}).items():
                predictions[arm] = collect_predictions(proof['shards'], arm+'.json', assets)
                if predictions[arm][1] != proof['policy_id']:
                    raise ValueError('historical prediction policy differs')
            result[dataset] = score_panel(c['cache'], jobs, predictions, spec['probability_panels'][dataset],
                                          thresholds=thresholds['arms'] if thresholds else None)
            continue
        semantic_protocol = spec.get('semantic_scoring', {}).get(dataset)
        if semantic_protocol is not None:
            from tools.reports.score_semantic_tpo import score as score_semantic
            noul, choice, policies = {}, {}, {}
            for arm in spec['arms']:
                noul[arm], policies[arm] = collect_predictions(spec['shards'][dataset], arm+'.json', assets)
                choice[arm], choice_policy = collect_predictions(spec['shards'][dataset], arm+'-choices.json', assets)
                if policies[arm] != choice_policy:
                    raise ValueError('Noul/Choice policy mismatch')
            result[dataset] = score_semantic(c['cache'], jobs, noul, choice, policies, semantic_protocol)
            continue
        result[dataset]={}
        prior = ['prior'] if 'choice_baseline' in spec else []
        noul_arms = [] if spec.get('choice_only') else [
            *spec.get('arms', ['stage1', 'direct', 'tpo']),
            *(prior if spec.get('full_ontology_only') else [])]
        for arm in noul_arms:
            records={};policies=set()
            for part in spec['shards'][dataset]:
                publication=read(Path(part)/(arm+'.json')); policies.add(publication['policy_id'])
                for r in publication['records']:
                    if r['id'] in records:raise ValueError('overlapping evaluation shards')
                    records[r['id']]=r
            if len(policies)!=1 or set(records)!={a['id'] for a in assets}:
                raise ValueError('incomplete endpoint populations')
            c['point']={'step':spec['steps'].get(arm,spec['steps']['tpo'])}
            metric=score(c,assets,candidates,[records[a['id']] for a in assets])['metrics']
            result[dataset][arm]=dict(policy_id=next(iter(policies)),**{k:metric[k] for k in
                ('queries','top1_macro_f1','top1_event_accuracy','mean_top1_probability',
                 'candidate_binary_nll','candidate_brier')})
        if 'choice' in spec:
            from tools.reports.run_native_seal_endpoint import bound
            from multimodal_jev.choice_metrics import choice_metrics
            seal=c['cache']['seal']; base=Path(seal['root'])
            manifest=bound(dict(path=base/'MANIFEST.json',sha256=seal['sha256']))
            rel='scoring/'+dataset+'/targets.json'
            gold=bound(dict(path=base/rel,sha256=manifest['files'][rel]))
            choices={}
            for arm in [*spec['arms'],*prior]:
                ordered={}; ids=set()
                for part in spec['shards'][dataset]:
                    pub=read(Path(part)/(arm+'-choices.json'));ids.add(pub['policy_id'])
                    for row in pub['records']:
                        if row['id'] in ordered:raise ValueError('duplicate Choice prediction')
                        ordered[row['id']]=row
                if len(ids)!=1 or set(ordered)!={a['id'] for a in assets}:raise ValueError('incomplete Choice results')
                if spec.get('full_ontology_only'):
                    from multimodal_jev.choice_metrics import full_ontology_metrics
                    if ids!={result[dataset][arm]['policy_id']}:raise ValueError('Noul/Choice policy mismatch')
                    choices[arm]=dict(policy_id=next(iter(ids)),**full_ontology_metrics(assets,gold,
                        [ordered[a['id']] for a in assets],candidates))
                    continue
                choices[arm]=dict(policy_id=next(iter(ids)),**choice_metrics(assets,gold,
                    [ordered[a['id']] for a in assets],candidates))
                if spec.get('choice_only'):
                    # Compare frozen policies on the same label-free candidate order for each recording.
                    roster = [(a['id'], ordered[a['id']]['slate'], ordered[a['id']]['proposal']) for a in assets]
                    if arm == spec['arms'][0]:
                        matched_roster = roster
                    elif matched_roster != roster:
                        raise ValueError('Variable Choice proposals differ across policies')
                    by_length = choices[arm]['by_length']
                    hits = sum(row['hits'] for row in by_length.values())
                    if hits != choices[arm]['coverage_hits'] or hits == 0:
                        raise ValueError('Expected auditable gold-present strata')
                    choices[arm]['false_none_rate'] = sum(
                        row['false_none_rate'] * row['hits'] for row in by_length.values() if row['hits']) / hits
                if 'paired_lengths' in spec:
                    from multimodal_jev.context_choice import lengths
                    panel_results={}
                    for k in lengths(dict(candidate_lengths=spec['paired_lengths']),len(candidates)):
                        samples=[dict(r) for r in (ordered[a['id']] for a in assets)]
                        for row in samples:row.update(row['panels'][str(k)])
                        metric=choice_metrics(assets,gold,samples,candidates)
                        panel_results[str(k)]=dict(**metric['by_length'][str(k)],by_proposal=metric['by_proposal'])
                    choices[arm]['paired_lengths']=panel_results
            result[dataset]['choices']=choices
    if 'probability_panels' in spec:
        result = dict(schema='jev-audited-probability-panels-v1', datasets=result,
                      seals={ds: c['cache']['seal']['sha256'] for ds,c in spec['endpoints'].items()})
        if thresholds:
            result['thresholds_sha256'] = file_hash(spec['thresholds'])
    elif spec.get('publish_seals'):
        result = dict(schema='jev-audited-tpo-evaluation-v1', datasets=result,
                      seals={dataset: c['cache']['seal']['sha256']
                             for dataset, c in spec['endpoints'].items()})
    commit_json(Path(spec['output']),result)
    return result


def endpoint_deltas(metrics, populations, policies, source):
    """Full-population paired point estimates, not a significance test."""
    from multimodal_jev.context_evaluation_contract import delta
    fields=('top1_macro_f1','top1_event_accuracy','candidate_binary_nll','candidate_brier')
    if (set(metrics)!=set(populations) or source not in populations or len(populations)<2 or
            set(policies)!={'stage1','direct','tpo'} or len(set(policies.values()))!=3):
        raise ValueError('distinct complete endpoints and all declared datasets required')
    differences={}
    for dataset,arms in metrics.items():
        if set(arms)!=set(policies):
            raise ValueError('incomplete paired comparison')
        values={}
        for arm,row in arms.items():
            if row['policy_id']!=policies[arm] or row['queries']!=populations[dataset]['recordings']:
                raise ValueError('endpoint identity or full population differs')
            values[arm]={k:row[k] for k in fields}
            if any(values[arm][k] is None or not 0<=values[arm][k]<=1
                   for k in ('top1_macro_f1','top1_event_accuracy')):
                raise ValueError('defined top-1 results required')
        differences[dataset]={name:delta(values[base],values['tpo'],fields)
                              for name,base in [('from_stage1','stage1'),('projection_increment','direct')]}
    def passes(key):
        return (differences[source][key]['top1_macro_f1']>0 and all(
                row[key]['top1_macro_f1']>=0 for ds,row in differences.items() if ds!=source))
    return dict(populations=populations,policy_ids=policies,differences=differences,
                observed_joint_gain_from_stage1=passes('from_stage1'),
                observed_joint_increment_from_projection=passes('projection_increment'),
                scope='Fixed endpoint and seed; signed raw metric differences. Negative NLL/Brier differences favor TPO.')


def compare_collected(cfg):
    """Post-score consumer of the existing runner's independently collected receipts."""
    eval_ops,train_ops=Path(cfg['evaluation_operations']),Path(cfg['training_operations'])
    for op in (eval_ops,train_ops):
        terminal=read(op/'terminal.json')
        if terminal!={'exit_code':0,'status':'declared_stages_finished'}:
            raise ValueError('training and evaluation must have finished successfully')
    seal_path=Path(cfg['seal']['root'])/'MANIFEST.json'
    if file_hash(seal_path)!=cfg['seal']['sha256']:
        raise ValueError('evaluation seal changed')
    pop={dataset:{k:spec[k] for k in ('recordings','candidates','split')}
         for dataset,spec in read(seal_path)['datasets'].items()}
    policies={'stage1':cfg['stage1_policy']}
    for arm in ('direct','tpo'):
        receipt=read(train_ops/(arm+'-collected.json'))
        if (not receipt['passed'] or receipt['arm']!=arm or receipt['window']!='formal' or
                receipt['binding']!=cfg['binding'] or receipt['updates']!=cfg['updates']):
            raise ValueError('formal endpoint lineage differs')
        policies[arm]=receipt['endpoint']
    scores=eval_ops/'score-collected.json'
    return dict(endpoint_deltas(read(scores),pop,policies,cfg['source_dataset']),
                seal_sha256=cfg['seal']['sha256'],metrics_sha256=file_hash(scores))


if __name__=='__main__':
    import argparse
    import json
    from multimodal_jev.local_config import load_config
    parser=argparse.ArgumentParser(description='Compare independently collected endpoint scores')
    parser.add_argument('--config',required=True)
    parser.add_argument('--local',required=True)
    parser.add_argument('--after-evaluation',action='store_true',
                        help='wait on the exact evaluation driver before collecting')
    args=parser.parse_args()
    cfg,_=load_config(args.config,args.local)
    if args.after_evaluation:
        from tools.reports.finalize_context_infonce import wait_exact
        ops=Path(cfg['evaluation_operations'])
        wait_exact(read(ops.with_suffix('.started.json')),ops/'terminal.json')
    result=compare_collected(cfg)
    if 'output' in cfg:
        path=Path(cfg['output'])
        if path.exists():
            if read(path)!=result:raise ValueError('existing report binding differs')
        else:commit_json(path,result)
    print(json.dumps(result,indent=2,allow_nan=False))
