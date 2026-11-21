"""Compare finite cold/warm outputs against frozen predictions; never score gold."""
import argparse
import json
from pathlib import Path

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.run_cached_history_probe import read_bound


def compare(config, result, frozen, threshold):
    records = {r['id']: r for r in frozen['records']}
    if not 0 < threshold < 1:
        raise ValueError('frozen probability threshold required')
    passes = []
    for run in result['passes']:
        rows = []
        if len(run['records']) != len(config['inputs']):
            raise ValueError('complete probe output required')
        for entry, values in zip(config['inputs'], run['records'], strict=True):
            ref = records[entry['id']]
            if ref['policy_id'] != config['policy_id']:
                raise ValueError('frozen policy/output differs')
            probs, logits = zip(*values, strict=True)
            # Prediction records use one list per request, not batch dimensions.
            p, l = ref['prob'], ref['logits']
            if p and isinstance(p[0], list): p = p[0]
            if l and isinstance(l[0], list): l = l[0]
            if len(values)!=len(p) or len(p)!=len(l):
                raise ValueError('complete candidate output required')
            rows.append(dict(id=entry['id'],
                max_prob_abs=max(abs(a-b) for a,b in zip(probs,p,strict=True)),
                max_logit_abs=max(abs(a-b) for a,b in zip(logits,l,strict=True)),
                threshold_flips=sum((a>=threshold)!=(b>=threshold) for a,b in zip(probs,p,strict=True)),
                top1_changed=max(range(len(probs)),key=probs.__getitem__)!=max(range(len(p)),key=p.__getitem__)))
        passes.append(dict(pass_index=run['pass_index'],seconds=run['seconds'],records=rows,
                           exact_from_cold=run['exact'],stats=run['stats']))
    return dict(passes=passes, formal_admitted=False,
                scope='Finite numerical diagnostic only; no performance gate or target scoring')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    result=compare(read_bound(c['probe_config']),read_bound(c['probe_result']),
                   read_bound(c['frozen_predictions']),c['threshold'])
    commit_json(Path(c['output']),dict(result,provenance=provenance))
    print(json.dumps(result,indent=2))
