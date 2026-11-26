"""Publish a derived TPO view, reprojecting the fixed reference in the new batches."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.card_stratified_sampling import sample
from multimodal_jev.context_execution import map_io
from multimodal_jev.local_checkpoint import commit_file, commit_json, sync_directory
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.build_card_tpo_targets import checked_file, target_rank_counts
from tools.reports.build_tpo_bidirectional_targets import compile_schedule
from tools.reports.build_tpo_gap_targets import shuffled_batches


def read(p):
    return json.loads(p.read_bytes())


def build(c):
    src, dst = Path(c['pool']), Path(c['output'])
    check_paths({'stratified TRAIN targets': dst})
    checked_file(src/'MANIFEST.json', c['pool_sha256'])
    m = read(src/'MANIFEST.json')
    required = {'rows.json', 'batches.json', 'population-audit.json', 'reference.pt', 'targets.pt',
                'cards.json', 'calibration.json'}
    if set(m['files']) != required or m['schema'] != 'jev-tpo-reward-view-v1' or src.resolve()==dst.resolve():
        raise ValueError('expected a distinct, complete frozen card TPO pool')
    map_io(lambda kv: checked_file(src/kv[0], kv[1]), m['files'].items(), workers=c['execution']['io_workers'])
    checked_file(Path(c['score_audit']), m['reference_sha256'])
    score = read(Path(c['score_audit']))
    if (score['passed'] is not True or score['container_exited'] is not True or
            (score['labels'], score['policy_id'], score['checkpoint_sha256'], score['full_sha256']) !=
            (m['candidate_ids'], m['source_policy'], m['card_input']['checkpoint_sha256'], m['view_sha256'])):
        raise ValueError('score and candidate pool lineage differ')
    policy = c['sampling']
    torch.set_num_threads(c['cpu_threads'])
    source_rows = read(src/'rows.json')
    ordered = sorted(source_rows.values(), key=lambda r:r['index'])
    ref = torch.load(src/'reference.pt', map_location='cpu', weights_only=True)
    if (ref.shape != (m['recordings'], len(m['candidate_ids'])) or len(ordered) != len(ref) or
            not torch.isfinite(ref).all() or (ref < 0).any() or (ref > 1).any() or
            any(i != r['index'] or source_rows[r['id']] != r or
                m['candidate_ids'][r['gold']] != r['intent'] for i,r in enumerate(ordered))):
        raise ValueError('reference tensor and recording indices differ')
    rivals = ref.clone()
    gold = torch.tensor([r['gold'] for r in ordered])
    rivals.scatter_(1,gold[:, None],-1)
    if rivals.argmax(1).tolist() != [r['competitor'] for r in ordered]:
        raise ValueError('directed competitors differ from fixed scores')
    high = (rivals > policy['high_probability']).sum(1).tolist()
    ix = sample(ordered, high, policy)
    selected = [ordered[i] for i in ix]
    rows = {r['id']: dict(r, index=j) for j,r in enumerate(selected)}
    calibration = read(src/'calibration.json')
    if ({r['group'] for r in selected} & {r['source_group'] for r in calibration} or
            {r['audio_sha256'] for r in selected} & {r['audio_sha256'] for r in calibration}):
        raise ValueError('selected rows overlap isolated calibration')
    reference = ref[ix]
    batches = shuffled_batches(rows, m['schedule'])
    reward = m['bidirectional_contrast']['reward']
    targets, batches, counts, audit, order = compile_schedule(
        dict(schedule=m['schedule'], reward=reward, updates=len(batches)), rows, batches,
        list(rows), dict(zip(rows, reference.tolist(), strict=True)))
    strata = dict(Counter(r['stratum'] for r in selected))
    evidence = dict(pool_recordings=len(ref), selected=len(rows), strata=strata,
        source_groups=len({r['group'] for r in selected}), gold_labels=len({r['gold'] for r in selected}),
        error_directions=len({(r['gold'],r['competitor']) for r in selected if r['stratum']=='error'}),
        many_high_rivals=sum(high[i]>=policy['many_rivals'] for i in ix),
        source_types=dict(Counter('synthetic' if r['synthetic'] else 'real' for r in selected)),
        target_rank=target_rank_counts(targets, gold[ix]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.stratified-',dir=dst.parent) as tmp:
        stage = Path(tmp)/'targets'
        stage.mkdir()
        for name, value in [('rows',rows),('batches',batches),('population-audit',evidence)]:
            commit_json(stage/(name+'.json'),value)
        for name in ('cards.json','calibration.json'):
            with (src/name).open('rb') as stream:
                commit_file(stage/name, lambda out: shutil.copyfileobj(stream,out))
        for name,tensor in [('reference',reference),('targets',targets)]:
            commit_file(stage/(name+'.pt'), lambda out: torch.save(tensor,out))
        files = {k:file_hash(stage/k) for k in sorted(required)}
        manifest = dict(m, files=files, view_sha256=files['rows.json'], pool_manifest_sha256=c['pool_sha256'],
            full_score_roster_sha256=m['view_sha256'], sampling=policy, recordings=len(rows), updates=len(batches),
            real_exposures=sum(counts), batch_real=counts, strata=strata,
            bidirectional_contrast=dict(reward=reward,batch_order_sha256=order,audit=audit))
        commit_json(stage/'MANIFEST.json', manifest)
        if dst.exists():
            for k in ['MANIFEST.json', *files]:
                checked_file(dst/k,file_hash(stage/k))
        else:
            stage.rename(dst)
            sync_directory(dst.parent)
    return evidence


if __name__ == '__main__':
    print(json.dumps(build(load_config(*sys.argv[1:])[0])))
