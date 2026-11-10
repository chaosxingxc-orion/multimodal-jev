#!/usr/bin/env python3
"""Reaccept every registered shard, merge raw decisions, then score the population."""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_jev_shards import verify_plan,population,digest
from multimodal_jev.sfc_jev_shard_runtime import read,digest as value_digest
from verify_sfc_jev_shard import accept
from evaluate_sfc_jev import evaluate


def collect(plan_path,plan_hash,roster_path,roster_hash):
    """No targets, output directories or scoring until full acceptance succeeds."""
    if file_hash(plan_path)!=plan_hash or file_hash(roster_path)!=roster_hash:
        raise ValueError('registered plan or completed-run roster changed')
    plan=read(plan_path);verify_plan(plan);roster=read(roster_path)
    if (set(roster)!={'schema','plan_sha256','runs'} or roster['schema']!='sfc-jev-completed-shard-roster-v1'
            or roster['plan_sha256']!=plan_hash
            or [r['shard_id'] for r in roster['runs']]!=[s['id'] for s in plan['shards']]):
        raise ValueError('exact complete ordered shard roster required')
    all_rows=population(plan['release'],plan['split'])
    wanted={uid for s in plan['shards'] for uid in s['ordered_bundle_ids']}
    rows=[r for r in all_rows if r['id'] in wanted];by_id={r['id']:r for r in rows}
    if len(rows)!=plan['bundles']:raise ValueError('complete registered population required')
    records={};receipts=[];roots=set();containers=set();actor=None
    for shard,entry in zip(plan['shards'],roster['runs'],strict=True):
        if set(entry)!={'shard_id','root','config_sha256','launch_sha256'}:
            raise ValueError('exact shard run reference required')
        root=Path(entry['root']).resolve()
        if root in roots:raise ValueError('duplicate shard run root')
        roots.add(root)
        if (file_hash(root/'config.json')!=entry['config_sha256']
                or file_hash(root/'launch.json')!=entry['launch_sha256']):
            raise ValueError('registered shard launch/config changed')
        receipt=accept(root)  # Reinspect exact Docker states and all raw outputs.
        cfg=read(root/'config.json');identity=read(root/'run/identity.json')
        if (cfg['shard_plan_sha256']!=plan_hash or cfg['shard_id']!=shard['id']
                or cfg['split']!=plan['split'] or Path(cfg['shard_plan']).resolve()!=Path(plan_path).resolve()
                or receipt['shard_id']!=shard['id'] or receipt['shard_plan_sha256']!=plan_hash
                or receipt['split']!=plan['split'] or receipt['bundles']!=shard['bundles']
                or receipt['decisions']!=shard['decisions'] or receipt['source_groups']!=len(shard['source_groups'])
                or receipt['prompt_id']!='parent' or receipt['instruction_sha256']!=cfg['instruction_sha256']
                or receipt['container_id'] in containers):
            raise ValueError('shard is from a different population, prompt or container')
        containers.add(receipt['container_id'])
        # Require all runtime settings and pins to agree, not just the prompt
        # label. Only the shard ID is allowed to differ within one population.
        current=dict(config={k:v for k,v in cfg.items() if k!='shard_id'},
                     source_files=identity['source_files'],model_files=identity['model_files'],versions=identity['versions'])
        if actor is None:actor=current
        elif current!=actor:raise ValueError('mixed actor/runtime/source identities across shards')
        path=root/'run/predictions.jsonl'
        if file_hash(path)!=receipt['predictions_sha256']:raise ValueError('accepted predictions changed')
        shard_rows=[by_id[uid] for uid in shard['ordered_bundle_ids']]
        if digest(shard_rows)!=shard['requests_sha256']:raise ValueError('shard inputs differ from full population')
        expected=[(r['id'],d['id']) for r in shard_rows for d in r['input']['decisions']]
        observed=[]
        with path.open() as source:
            for line in source:
                p=json.loads(line);key=(p['example_id'],p['decision_id']);observed.append(key)
                if key in records or p['example_id'] not in by_id:
                    raise ValueError('duplicate or unknown decision during merge')
                if p['request_sha256']!=value_digest(by_id[p['example_id']]) or type(p['raw_text']) is not str:
                    raise ValueError('raw prediction input binding changed')
                records[key]=p
        if observed!=expected:raise ValueError('shard has missing, extra or reordered decisions')
        receipts.append(receipt)
    keys=[(r['id'],d['id']) for r in rows for d in r['input']['decisions']]
    if set(records)!=set(keys) or len(records)!=plan['decisions']:
        raise ValueError('full decision population missing or duplicated')
    # Restore release order for official scoring, retaining every raw field.
    return plan,rows,[records[k] for k in keys],receipts,actor


def score(plan_path,plan_hash,roster_path,roster_hash,official_scorer,scorer_hash,output):
    output=Path(output)
    if output.exists():raise FileExistsError('preserve prior merge/scoring attempt')
    plan,rows,predictions,receipts,actor=collect(plan_path,plan_hash,roster_path,roster_hash)
    if file_hash(official_scorer)!=scorer_hash:raise ValueError('registered official scorer changed')
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'engineering-acceptance.json',dict(plan_sha256=plan_hash,roster_sha256=roster_hash,
        shards=receipts,actor_identity=actor,bundles=len(rows),decisions=len(predictions),targets_read=False))
    path=output/'predictions.jsonl'
    with path.open('xb') as handle:
        for prediction in predictions:handle.write(canonical(prediction))
    try:
        release=Path(plan['release']);split=plan['split'];manifest=read(release/'MANIFEST.json')
        # First gold read occurs after all shards have passed complete acceptance.
        target=release/f'{split}.targets.jsonl'
        if file_hash(target)!=manifest['splits'][split]['targets_sha256']:
            raise ValueError('official targets changed')
        scoring_release=release
        if plan['partition'] is not None:
            # Only the registered complete TRAIN partition may be scored here.
            # Preserve release order and lossless native provenance for the bridge.
            wanted={r['id'] for r in rows};targets=[]
            with target.open() as source:
                for line in source:
                    t=json.loads(line)
                    if t['id'] in wanted:targets.append(t)
            if [t['id'] for t in targets]!=[r['id'] for r in rows]:
                raise ValueError('complete aligned TRAIN partition targets required')
            scoring_release=output/'scoring-population';scoring_release.mkdir()
            pin=dict(bundles=len(rows),decisions=len(predictions),source_groups=plan['source_groups'])
            for kind,values in (('requests',rows),('targets',targets)):
                dest=scoring_release/f'{split}.{kind}.jsonl'
                with dest.open('xb') as handle:
                    for value in values:handle.write(canonical(value))
                pin[kind+'_sha256']=file_hash(dest)
            atomic_json(scoring_release/'MANIFEST.json',dict(source_manifest_sha256=plan['manifest_sha256'],
                partition=plan['partition'],splits={split:pin},scope='Complete registered TRAIN prompt partition, not whole TRAIN.'))
        # The strict scorer accepts exactly four semantic fields. Preserve the
        # complete raw engineering records above and project without text edits.
        scoring_predictions=output/'scoring-predictions.jsonl'
        with scoring_predictions.open('xb') as handle:
            for prediction in predictions:
                handle.write(canonical({k:prediction[k] for k in
                    ('example_id','decision_id','request_sha256','raw_text')}))
        report=evaluate(scoring_release,split,scoring_predictions,Path(official_scorer),output/'score',scorer_hash)
        result=dict(schema='sfc-jev-complete-shard-score-v1',plan_sha256=plan_hash,roster_sha256=roster_hash,
            split=split,partition=plan['partition'],shards=len(receipts),bundles=len(rows),decisions=len(predictions),
            predictions_sha256=file_hash(path),scoring_predictions_sha256=file_hash(scoring_predictions),
            metrics_sha256=file_hash(output/'score/metrics.json'),
            merger_sha256=file_hash(__file__),counts=report['counts'],weight_updates=0,
            scope='All registered shards reaccepted, raw outputs merged in release order and scored. '
                  'Whole official split for TEST; only the named complete prompt partition for TRAIN. '
                  'No averaging of shard percentages, output repairs, or task-quality acceptance threshold.')
        atomic_json(output/'result.json',result)
    except BaseException as exc:
        atomic_json(output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('plan','roster','official-scorer','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('plan-hash','roster-hash','scorer-hash'):p.add_argument('--'+k,required=True)
    a=p.parse_args()
    print(json.dumps(score(a.plan,a.plan_hash,a.roster,a.roster_hash,a.official_scorer,a.scorer_hash,a.output),indent=2))
