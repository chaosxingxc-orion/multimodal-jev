"""Read-only ranking credit coverage on full TRAIN and on authenticated real GRPO groups."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
import multiprocessing as mp
from pathlib import Path

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.jev_train_confusions import pinned, run as train_population


def counts(q, wins, group_size):
    """q is the probability of choosing gold, irrespective of canonical left/right."""
    import numpy as np
    out = Counter()
    for name, mask in [('wrong', q < .5), ('tied', q == .5), ('right', q > .5)]:
        prob = q[mask]
        out[name+'_pairs'] = len(prob)
        out[name+'_expected_all_wrong'] = float(np.sum((1-prob)**group_size))
        if wins is not None:
            out[name+'_all_wrong'] = int((wins[mask] == 0).sum())
            out[name+'_all_correct'] = int((wins[mask] == group_size).sum())
    return out


def limit_threads():
    import torch
    torch.set_num_threads(1)


def recorded(job):
    import numpy as np
    import torch
    path, digest, audit, size = job
    rows = torch.load(BytesIO(pinned(path, digest)), weights_only=True, map_location='cpu')
    total=Counter(); wrong_audio=0; any_silent=0
    for r in rows:
        if not r['real_request']:continue
        credit=r['dual_credit']; br=credit['branches']['ranking']; b=credit['binding']
        if (b['group_size']!=size or b['update']!=audit['update'] or b['algorithm']!=audit['algorithm']
                or sum(b['labels'])!=1 or not r['observed'].all()):
            raise ValueError('wrong sampled panel')
        left=br['probability'].numpy(); target=br['target'].numpy(); rewards=br['rewards']
        q=np.where(target,left,1-left)
        if (rewards.shape!=(len(b['labels'])-1,size) or not np.isfinite(q).all()
                or not torch.equal(rewards, (br['actions']==br['target'][:,None]).float())):
            raise ValueError('inconsistent ranking actions')
        # Determine rank sign from original pair logits, avoiding subtraction cancellation at ties.
        w=rewards.sum(-1).numpy()
        wrong_audio+=int((q<.5).any());any_silent+=int(((q<.5)&(w==0)).any())
        total.update(counts(q,w,size));total['real_exposures']+=1
    components=audit['branches']['ranking']
    # The prior component audit includes all slots, including padding on tails;
    # cross-check full batches, and count only actual exposures in this report.
    if total['real_exposures']!=audit['real_queries'] or (not audit['padding_slots'] and
            sum(total[f'{k}_{s}'] for k in ('wrong','right','tied') for s in ('all_wrong','all_correct'))
            !=components['zero_signal_components']):
        raise ValueError('sample totals differ from existing formal audit')
    total.update(wrong_exposures=wrong_audio,wrong_exposures_with_silent_pair=any_silent)
    return dict(total)


def run(c, provenance):
    import numpy as np
    audit=json.loads(pinned(Path(c['audit']),c['audit_sha256']))
    trans=json.loads(pinned(Path(c['transport']),audit['transport_sha256']))
    if (audit['status']!='stage2_view_formal_outputs_audited' or audit['experiment_kind']!='formal_source'
            or audit['container_id']!=trans['container_id'] or trans['bad_pins'] or trans['bad_package']):
        raise ValueError('audited formal source run required')
    size=c['group_size']; root=Path(c['collection']); files=trans['files']
    cfg=json.loads(pinned(root/'config.json',files['config.json']))
    if cfg['reference_dataset']['manifest_sha256']!=c['reference_sha256'] or size!=cfg['stage2_objective']['group_size']:
        raise ValueError('source reference or rollout group size mismatch')
    jobs=[]
    for i,row in enumerate(audit['updates']):
        if row['update']!=i:raise ValueError('missing update')
        name=f'run/groups-{i:04d}.pt'
        jobs.append((root/name,files[name],row,size))
    if len(jobs)!=audit['formal_contract']['updates']:raise ValueError('incomplete formal run')
    with ProcessPoolExecutor(max_workers=c['cpu_workers'],mp_context=mp.get_context('spawn'),initializer=limit_threads) as pool:
        steps=list(pool.map(recorded,jobs,chunksize=c['chunk_size']))
    actual=Counter()
    for s in steps:actual.update(s)
    if actual['real_exposures']!=audit['real_exposures']:raise ValueError('incomplete exposures')
    initial=Counter()
    def visit(_,gold,labels,p):
        pg=p[labels.index(gold)]
        v=np.asarray([v for k,v in zip(labels,p,strict=True) if k!=gold])
        if not 0<pg<1 or not ((v>0)&(v<1)).all():raise ValueError('interior independent sigmoid probabilities required')
        a=pg*(1-v); q=a/(a+v*(1-pg))
        initial.update(counts(q,None,size));initial['recordings']+=1
    source=train_population(c,observe=visit)
    result=dict(schema='jev-ranking-signal-coverage-v1',group_size=size,initial_from_cached_sigmoid=initial,
        observed_formal_groups=actual,updates=steps,source_policy=source['policy_id'],
        audit_sha256=c['audit_sha256'],reference_sha256=c['reference_sha256'],
        tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),provenance=provenance,model_calls=0)
    commit_json(c['output'],result)
    return {k:result[k] for k in ('initial_from_cached_sigmoid','observed_formal_groups')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();print(json.dumps(run(*load_config(a.config,a.local)),indent=2))
