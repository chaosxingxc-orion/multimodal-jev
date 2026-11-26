"""Subset a pinned mined TRAIN roster, keep every hard example, compile two fixed targets."""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import torch
from multimodal_jev.baseline import file_hash
from multimodal_jev.frozen_probability import validate
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json, commit_file, sync_directory
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.build_card_tpo_targets import checked_file
from tools.reports.build_tpo_gap_targets import shuffled_batches
from tools.reports.build_tpo_bidirectional_targets import compile_schedule


def build(c):
    src, dst = Path(c['pool']), Path(c['output'])
    check_paths({'calibrated TRAIN view':dst})
    checked_file(src/'MANIFEST.json', c['pool_sha256'])
    m=json.loads((src/'MANIFEST.json').read_bytes())
    mined=Path(c['mined_manifest'])
    checked_file(mined,c['mined_sha256'])
    origin=json.loads(mined.read_bytes())
    scale=c['frozen_probability']
    validate(scale)
    if (origin['pool_sha256'],origin['checkpoint'],origin['slope'],origin['bias']) != (
            c['pool_sha256'],m['card_input']['checkpoint_sha256'],scale['a'],scale['b']):
        raise ValueError('mined population and frozen scale differ')
    data=mined.with_name('training.jsonl')
    checked_file(data,origin['report']['files'][data.name])
    for name in ('reference.pt','rows.json','cards.json','calibration.json'):
        checked_file(src/name,m['files'][name])
    population=json.loads((src/'rows.json').read_bytes())
    mined_rows=[json.loads(line) for line in data.read_bytes().splitlines()]
    table={r['id']:r for r in mined_rows}
    if len(table)!=len(mined_rows) or any(not r['selected'] or r['audio_sha256']!=population[r['id']]['audio_sha256']
        or r['gold']!=population[r['id']]['intent'] or r['group']!=population[r['id']]['group'] for r in mined_rows):
        raise ValueError('mining identity differs')
    hard={uid for uid,r in table.items() if r['stratum']!='replay'}
    fraction=Fraction(str(c['replay_fraction']))
    if not 0 < fraction < 1: raise ValueError('invalid replay fraction')
    quota=len(hard)*fraction/(1-fraction)
    if quota.denominator!=1: raise ValueError('cannot realize exact ratio without dropping hard examples')
    def key(uid): return hashlib.sha256(f"{c['seed']}:{uid}".encode()).digest()
    buckets=defaultdict(list)
    for uid in sorted(table.keys()-hard,key=key):
        buckets[(table[uid]['gold'],table[uid]['synthetic'])].append(uid)
    groups={table[k]['group'] for k in hard}
    replay=[]; deferred=[]
    for layer in itertools.zip_longest(*(buckets[k] for k in sorted(buckets))):
        for uid in layer:
            if uid is None or len(replay)==quota: continue
            if table[uid]['group'] in groups: deferred.append(uid)
            else: replay.append(uid); groups.add(table[uid]['group'])
    replay.extend(deferred[:int(quota)-len(replay)])
    if len(replay)!=quota: raise ValueError('replay shortfall')
    selected=sorted(hard|set(replay))
    rows={k:dict(population[k],index=i,stratum=table[k]['stratum'],ref_gap=table[k]['gap']) for i,k in enumerate(selected)}
    ref=torch.load(src/'reference.pt',weights_only=True,map_location='cpu')[[population[k]['index'] for k in selected]]
    reward=m['bidirectional_contrast']['reward']
    batches=shuffled_batches(rows,m['schedule'])
    (rank,prob),batches,counts,audit,order=compile_schedule(
        dict(schedule=m['schedule'],reward=reward,updates=len(batches),frozen_probability=scale),
        rows,batches,selected,dict(zip(selected,ref.tolist(),strict=True)))
    evidence=dict(hard=len(hard),replay=len(replay),source_groups=len({r['group'] for r in rows.values()}),
        recordings=len(selected),strata=dict(Counter(r['stratum'] for r in rows.values())),mined_sha256=c['mined_sha256'])
    dst.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dst.parent) as temp:
        stage=Path(temp)/'targets';stage.mkdir()
        for name,val in [('rows',rows),('batches',batches),('population-audit',evidence)]:
            commit_json(stage/(name+'.json'),val)
        for name in ['cards.json','calibration.json']: shutil.copyfile(src/name,stage/name)
        for name,t in [('targets',rank),('probability-targets',prob),('reference',ref)]:
            commit_file(stage/(name+'.pt'),lambda f:torch.save(t,f))
        files={p.name:file_hash(p) for p in stage.iterdir()}
        manifest=dict(m,files=files,view_sha256=files['rows.json'],pool_manifest_sha256=c['pool_sha256'],
            full_score_roster_sha256=m['view_sha256'],recordings=len(selected),updates=len(batches),batch_real=counts,
            sampling=dict(replay_fraction=c['replay_fraction'],seed=c['seed'],mined_sha256=c['mined_sha256']),
            real_exposures=sum(counts),strata=evidence['strata'],frozen_probability=scale,
            bidirectional_contrast=dict(reward=reward,batch_order_sha256=order,audit=audit))
        commit_json(stage/'MANIFEST.json',manifest)
        if dst.exists():
            for f in stage.iterdir(): checked_file(dst/f.name,file_hash(f))
        else: stage.rename(dst);sync_directory(dst.parent)
    return evidence


if __name__=='__main__':
    config=load_config(*sys.argv[1:])[0]
    torch.set_num_threads(config['cpu_threads'])
    print(json.dumps(build(config)))
