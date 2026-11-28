"""Export the same pre-card speech vectors used by the Stage1 loss."""
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_file, commit_json
from tools.reports.card_scoring_benchmark import pinned


def run(c):
    import torch
    from torch.nn import functional as F
    from multimodal_jev.card_acoustic_pack import PackedAcoustics
    from multimodal_jev.card_training_acoustics import frozen_frames, pool_speech
    from multimodal_jev.context_policy_artifacts import read_execution_bundle
    from tools.reports.context_transfer_macro_parity import load_reference_model
    from multimodal_jev.ordered_prefetch import OrderedPrefetch

    out = Path(c['output'])
    out.mkdir()
    rows = pinned(c['roster'])['rows']
    population = {r['id']:r for r in rows}
    rank, world, width = c['shard'], c['shards'], c['batch']
    if (len(population) != len(rows) or not 0 <= rank < world <= len(rows) or width < 1):
        raise ValueError('invalid acoustic roster or shard')
    token = c['audio_token_id']
    pack = PackedAcoustics(c['index'], population, token)
    binding = c['policy']
    if file_hash(Path(binding['snapshot'])/'MANIFEST.json') != c['policy_sha256']:
        raise ValueError('speech reference checkpoint changed')
    bundle = read_execution_bundle(binding['snapshot'], expected_policy_id=binding['policy_id'],
                                   expected_contract=binding['contract'], expected_base_id=binding['base_id'])
    model, device, _, handle = load_reference_model(binding, bundle, out)
    ids = sorted(sorted(population)[rank::world], key=lambda uid:(pack.entries[uid][6], uid))

    def prepare(group):
        tensors = [pack.read(uid) for uid in group]
        lengths = torch.tensor([t['input_ids'].shape[-1] for t in tensors])
        width = max(t['input_features'].shape[-1] for t in tensors)
        feat, mask = (torch.cat([F.pad(t[key], (0,width-t[key].shape[-1])) for t in tensors])
                      for key in ('input_features', 'input_features_mask'))
        padded = torch.arange(int(lengths.max()))[None, :]
        valid = padded < lengths[:, None]
        index = ((lengths.cumsum(0)-lengths)[:, None]+padded).masked_fill(~valid, 0)
        return feat, mask, index, valid, int(lengths.sum())

    groups = (ids[i:i+width] for i in range(0,len(ids),width))
    queue = OrderedPrefetch(prepare,groups,**c['prefetch'])
    vectors = []
    try:
        with torch.inference_mode():
            for feat, mask, index, valid, expected in queue:
                frames = frozen_frames(model.backbone, feat.to(device), mask.to(device))
                if len(frames) != expected:
                    raise ValueError('cached audio tokens and projected frames differ')
                speech = pool_speech(frames, index.to(device), valid.to(device))
                vectors.append(model.decision_head.speech_vectors(speech).float().cpu())
    finally:
        queue.close()
    vectors = torch.cat(vectors)
    if not torch.isfinite(vectors).all() or not (vectors.norm(dim=-1) > 0).all():
        raise ValueError('empty or nonfinite speech readout')
    import numpy as np
    path = out/'vectors.npy'
    commit_file(path, lambda f: np.save(f, vectors.numpy(),allow_pickle=False))
    result = dict(schema='jev-card-speech-vectors-v1', policy_id=binding['policy_id'],
                  policy_sha256=c['policy_sha256'], roster_sha256=c['roster']['sha256'],
                  acoustic_sha256=c['index']['sha256'], shard=rank, shards=world, ids=ids,
                  vectors_sha256=file_hash(path), width=vectors.shape[1], recordings=len(ids))
    commit_json(out/'receipt.json',result)
    return result
