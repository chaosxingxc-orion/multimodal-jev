"""Prepare a bucketed actor plan using only the audited Stage1 acoustic index."""
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.card_acoustic_pack import PackedAcoustics
from multimodal_jev.card_tpo_schedule import check_batches, make_batches
from multimodal_jev.local_checkpoint import commit_json


def prepare(config, stage1, data, entry, manifest, world):
    if set(config) != {'buckets', 'lookahead_batches', 'acoustic_mount', 'acoustic_index',
                       'acoustic_engine'} or manifest.get('actor_pair_margin') or manifest['schema'] != 'jev-tpo-reward-view-v1':
        raise ValueError('bucketed Stage2 requires the fixed, recording-wise objective')
    from multimodal_jev.card_training_acoustics import ENGINE
    if config['acoustic_engine'] != ENGINE:
        raise ValueError('unsupported Stage2 acoustic engine')
    # The submit host may see another mount of the same content. Both must match
    # the Stage1 receipt, and the runtime never consumes the submit host path.
    source = config['acoustic_mount']
    index = Path(config['acoustic_index'])
    pin = stage1['packed_acoustics']
    token = manifest['card_input']['acoustic_common']['context_encoding']['audio_token_id']
    rows, cards, batches = (json.loads((data/name).read_bytes())
                            for name in ('rows.json', 'cards.json', 'batches.json'))
    selected = [cards['catalog'][i] for i in cards['arms'][manifest['card_input']['arm']]]
    if [c['id'] for c in selected] != manifest['candidate_ids']:
        raise ValueError('frozen Stage2 candidate order differs')
    if any(c.get('id') is None or not c['prefix'] or not c['suffix'] for c in selected):
        raise ValueError('missing compiled Stage2 card tokens')
    prompt = max(len(c['prefix'])+len(c['suffix']) for c in selected)
    packing = PackedAcoustics(dict(pin, path=str(index)), rows, token, allow_subset=True)
    lengths = {uid: packing.entries[uid][4]+prompt for uid in rows}
    design = {k: config[k] for k in ('buckets', 'lookahead_batches')}
    epochs = manifest['schedule']['epochs']
    updates = make_batches(batches, lengths, design, world, epochs)
    counts = check_batches(updates, lengths, design, world, epochs)
    if len(updates) < 2:
        raise ValueError('bucketed Stage2 requires two finite updates')
    # Run two real batches with largest effective forward footprints on the finite
    # actor; formal order stays deterministic and follows the shuffled buckets.
    pressure = sorted(range(len(updates)), key=lambda i: (
        updates[i]['microbatch']*updates[i]['max_tokens'], updates[i]['local_batch'], -i), reverse=True)
    first = pressure[0]
    second = max((i for i in pressure if i != first), key=lambda i: (
        updates[i]['max_tokens'], updates[i]['bucket'] != updates[first]['bucket'], -i))
    payload = dict(target_sha=file_hash(data/'MANIFEST.json'), world=world, epochs=epochs,
                   lengths=lengths, design=design, counts=counts, updates=updates, finite=[first, second],
                   acoustic_engine=ENGINE, cache=dict(pin, path='/jev-stage2-acoustics/index.json'))
    commit_json(entry/'optimizer-buckets.json', payload)
    return f'{source}:/jev-stage2-acoustics'


def load(entry, sha, target_sha, world, epochs):
    path = entry/'optimizer-buckets.json'
    if file_hash(path) != sha:
        raise ValueError('Stage2 optimizer plan changed')
    plan = json.loads(path.read_bytes())
    if (plan['target_sha'], plan['world'], plan['epochs']) != (target_sha, world, epochs):
        raise ValueError('Stage2 optimizer plan lineage differs')
    return plan
