"""Produce a compact offline TPO training target release on verified local disk."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_reference_dataset import digest, validate_record
from multimodal_jev.context_tpo import Objective, target
from multimodal_jev.local_checkpoint import commit_file, commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.tpo_view import schedule


def verified(path, expected):
    if file_hash(path) != expected:
        raise ValueError(f'content identity changed: {path}')
    return path


def read(path):
    return json.loads(path.read_bytes())


def build(cfg):
    dst = Path(cfg['output'])
    check_paths({'target_view': dst})
    v, r, s = (Path(cfg[k]) for k in ('view', 'reference', 'release'))
    vm = read(verified(v/'MANIFEST.json', cfg['view_sha256']))
    spec = vm['contract']
    rm = read(verified(r/'MANIFEST.json', spec['reference_manifest_sha256']))
    ref = rm['identity']
    if ref != spec['reference_identity'] or ref['policy_id'] != spec['endpoint']['policy_id']:
        raise ValueError('stale Stage1 reference')
    release = read(verified(s/'MANIFEST.json', ref['release_sha256']))
    records, pop = [read(verified(r/n, rm['files'][n])) for n in ('records.json','population.json')]
    ids = read(verified(v/'selection.json', vm['files']['selection.json']))['selected_ids']
    task = read(verified(s/'tasks.json', release['files']['tasks.json']))['intent']
    candidates = [i['id'] for i in task['schema']['candidates']]
    if candidates != ref['candidate_ids'] or len(set(ids)) != len(ids):
        raise ValueError('candidate order or sample population differs')
    by_file = defaultdict(set)
    with verified(Path(cfg['metadata']), cfg['metadata_sha256']).open() as src:
        for line in src:
            row = json.loads(line)
            for item in row['recordings']:
                by_file[item['file']].add((row['slurp_id'], row['intent']))
    source = {}
    with verified(s/'learning.jsonl', release['files']['learning.jsonl']).open() as src:
        for line in src:
            row = json.loads(line)
            if (row['official_split'] != 'train' or row['role'] != 'learning' or
                    row['known_pending_source'] is not False or row['id'] in source):
                raise ValueError('only unique clean TRAIN rows admitted')
            source[row['id']] = row
    if set(pop) != set(source) or digest(pop) != ref['population_sha256']:
        raise ValueError('reference population mismatch')
    rows, probs = {}, []
    for i, uid in enumerate(ids):
        row = source[uid]
        record = records[pop[uid]]
        validate_record(record, ref)
        if pop[uid] != record['request_id']:
            raise ValueError('stale input binding')
        links = by_file[Path(row['audio']).name]
        group, intent = next(iter(links)) if len(links) == 1 else (None, None)
        rows[uid] = dict(index=i, gold=candidates.index(row['intent']),
                         group=group if intent == row['intent'] else None,
                         audio_sha256=row['audio_sha256'], request_key=record['request_id'])
        gold = rows[uid]['gold']
        strongest = max((j for j in range(len(candidates)) if j != gold), key=record['prob'].__getitem__)
        rows[uid].update(competitor=strongest,ref_gap=record['prob'][gold]-record['prob'][strongest])
        probs.append(record['prob'])
    counts = Counter(row['gold'] for row in rows.values())
    if len(counts) != len(candidates):
        raise ValueError('view missing a macro-recall class')
    gold = torch.tensor([rows[u]['gold'] for u in ids])
    utility = torch.tensor([len(ids)/(len(counts)*counts[rows[u]['gold']]) for u in ids])
    batches = schedule(rows, **cfg['schedule'])
    if 'rewards' in cfg:
        from multimodal_jev.context_tpo_rewards import RewardSpec, compile_targets
        if 'objective' in cfg:
            raise ValueError('reward projection and direct objectives are mutually exclusive')
        logq, audit = compile_targets(torch.tensor(probs), rows, gold, utility, batches, RewardSpec(**cfg['rewards']))
        mode = dict(schema='jev-tpo-reward-view-v1', rewards=cfg['rewards'], reward_audit=audit)
    else:
        opts = Objective(**cfg['objective'])
        logq = target(torch.tensor(probs), gold, utility, opts.temperature)
        mode = dict(schema='jev-tpo-view-v1', objective=cfg['objective'])
    if not torch.isfinite(logq).all():
        raise ValueError('nonfinite projection target')
    dst.mkdir(parents=True, exist_ok=False)
    commit_file(dst/'targets.pt', lambda f: torch.save(logq, f))
    commit_json(dst/'rows.json', rows)
    commit_json(dst/'batches.json', batches)
    result = dict(**mode, source_policy=ref['policy_id'],
                  view_sha256=cfg['view_sha256'], reference_sha256=spec['reference_manifest_sha256'],
                  release_sha256=ref['release_sha256'], official_metadata_sha256=cfg['metadata_sha256'],
                  candidate_ids=candidates, schedule=cfg['schedule'],
                  class_counts=dict(counts), recordings=len(ids), updates=len(batches),
                  paired_exposures=sum(slot['paired'] for b in batches for slot in b['slots']),
                  contrast_exposures=sum(slot['confusion'] for b in batches for slot in b['slots']),
                  files={n:file_hash(dst/n) for n in ('targets.pt','rows.json','batches.json')})
    commit_json(dst/'MANIFEST.json', result)
    return {k:result[k] for k in ('recordings','updates','paired_exposures','source_policy')}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('local')
    args = p.parse_args()
    print(json.dumps(build(load_config(args.config,args.local)[0])))
