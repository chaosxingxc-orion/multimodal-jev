"""Derive a ranking target from audited fixed TPO logits; preserve the training view."""
import argparse
import json
import math
from pathlib import Path
import shutil

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_file, commit_json
from multimodal_jev.local_config import load_config


def build(c):
    src, dst = Path(c['baseline']), Path(c['output'])
    manifest_path = src / 'MANIFEST.json'
    if file_hash(manifest_path) != c['baseline_sha256']:
        raise ValueError('baseline manifest changed')
    m = json.loads(manifest_path.read_bytes())
    if m['schema'] != 'jev-tpo-view-v1' or any(
        file_hash(src / name) != digest for name, digest in m['files'].items()
    ):
        raise ValueError('baseline files changed')
    rows = sorted(json.loads((src / 'rows.json').read_bytes()).values(), key=lambda row: row['index'])
    base = torch.load(src / 'targets.pt', map_location='cpu', weights_only=True)
    if ([r['index'] for r in rows] != list(range(m['recordings']))
        or base.shape != (m['recordings'], len(m['candidate_ids']))
        or not torch.isfinite(base).all()):
        raise ValueError('invalid target tensor or row order')
    margin, cap = math.log(c['gold_odds_ratio']), c['max_correction_per_side']
    if not (0 < margin < math.inf and 0 < cap < math.inf):
        raise ValueError('invalid ranking repair bounds')
    idx = torch.arange(len(rows))
    gold = torch.tensor([r['gold'] for r in rows])
    mask = torch.nn.functional.one_hot(gold, base.shape[1]).bool()
    scores = base.double()
    wrong = scores.masked_fill(mask, -torch.inf).amax(1)
    gap = scores[idx, gold] - wrong
    delta = ((margin - gap).clamp_min(0) / 2).clamp_max(cap)
    changed = delta > 0
    trial = torch.minimum(scores, (wrong - delta)[:, None])
    trial[idx, gold] = scores[idx, gold] + delta
    after = trial[idx, gold] - trial.masked_fill(mask, -torch.inf).amax(1)
    torch.testing.assert_close(after, gap + 2 * delta, atol=1e-12, rtol=0)
    result = base.clone()
    result[changed] = trial[changed].log_softmax(1).to(base.dtype)
    after_saved = result[idx, gold] - result.masked_fill(mask, -torch.inf).amax(1)
    torch.testing.assert_close(after_saved.double(), after, atol=2e-6, rtol=0)
    torch.testing.assert_close(result.double().logsumexp(1), torch.zeros_like(gap), atol=1e-6, rtol=0)
    if not torch.equal(result[~changed], base[~changed]):
        raise ValueError('targets already satisfying the margin changed')
    dst.mkdir(parents=True)
    for name in ('rows.json', 'batches.json'):
        shutil.copyfile(src / name, dst / name)
    commit_file(dst / 'targets.pt', lambda f: torch.save(result, f))
    m['ranking_repair'] = dict(baseline_sha256=c['baseline_sha256'], gold_odds_ratio=c['gold_odds_ratio'],
                               max_correction_per_side=cap, modified_rows=int(changed.sum()),
                               cap_reached=int((delta == cap).sum()),
                               previously_strictly_correct_target_lost=int(((gap > 0) & (after_saved <= 0)).sum()))
    m['files']['targets.pt'] = file_hash(dst / 'targets.pt')
    commit_json(dst / 'MANIFEST.json', m)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('local')
    args = parser.parse_args()
    build(load_config(args.config, args.local)[0])
