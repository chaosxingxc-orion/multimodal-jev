"""Read the sealed full TRAIN panel, without sampling or treating endpoint scores as training gradients."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(root, temperatures, margin):
    saved = json.loads((root/'audit.json').read_bytes())
    receipt = json.loads((root/'collected/score/receipt.json').read_bytes())
    raw = (root/'entry/full.json').read_bytes()
    if not saved['passed'] or hashlib.sha256(raw).hexdigest() != saved['full_sha256'] or any(
            saved[k] != receipt[k] for k in ('full_sha256', 'policy_id', 'policy_sha256', 'admission_sha256')):
        raise ValueError('invalid scored population')
    full = json.loads(raw)
    gold = {r['id']:full['labels'].index(r['intent']) for r in full['rows']}
    above = equal = below = gap_ok = 0
    shares = {t:[] for t in temperatures}
    gaps, positive_scores, rival_scores = [], [], []
    for i, worker in enumerate(receipt['workers']):
        for block in worker['blocks']:
            data = (root/'collected/score'/f'worker-{i}'/block['name']).read_bytes()
            if hashlib.sha256(data).hexdigest() != block['sha256']:
                raise ValueError('wrong score block')
            records = json.loads(data)['payload']['records']
            p = np.asarray([r['prob'] for r in records],dtype=np.float64)
            if p.shape != (len(records),len(full['labels'])) or not ((0<p)&(p<1)).all():
                raise ValueError('invalid recorded probabilities')
            idx = np.arange(len(records)), [gold.pop(r['id']) for r in records]
            true = p[idx].copy()
            z = np.log(p) - np.log1p(-p)
            z[idx] = -np.inf
            p[idx] = 0
            rival = p.max(1)
            above += int((rival>true).sum()); equal += int((rival==true).sum()); below += int((rival<true).sum())
            gap_ok += int((true-rival >= margin).sum())
            positive_scores.extend(true.tolist()); rival_scores.extend(rival.tolist())
            gaps.extend((true-rival).tolist())
            z -= z.max(1, keepdims=True)
            for t in shares:
                shares[t].extend((1/np.exp(z/t).sum(1)).tolist())
    if gold or above+equal+below != saved['recordings']:
        raise ValueError('incomplete population')
    return dict(recordings=saved['recordings'], candidates=len(full['labels']), full_sha256=saved['full_sha256'],
                policy_id=saved['policy_id'], higher_wrong=above, tied=equal, higher_gold=below,
                probability_margin=margin, gap_satisfied=gap_ok, median_gap=float(np.median(gaps)),
                median_true_probability=float(np.median(positive_scores)),
                median_strongest_wrong_probability=float(np.median(rival_scores)),
                # Conditional on all wrong candidates; excludes gold; no claim to replay sampled updates.
                median_strongest_fraction_among_all_wrong={t:float(np.median(v)) for t,v in shares.items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--temperature',type=float,nargs='+',required=True)
    p.add_argument('--margin',type=float,required=True);a=p.parse_args()
    if not all(np.isfinite(t) and t>0 for t in a.temperature) or not 0 < a.margin < 1:
        p.error('positive temperatures and probability margin in (0,1) required')
    print(json.dumps(audit(a.root,a.temperature,a.margin),indent=2,allow_nan=False))
