"""Separate directional evidence in two candidates from selection across the native panel."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.jev_train_confusions import pinned


def score(c):
    if c['schema']!='jev-aligned-cue-probe-v1' or not 0<c['minimum_probability_margin']<1:
        raise ValueError('explicit contrast and probability margin required')
    setup=c['package'];out=Path(setup['output']);seal=Path(setup['seal'])
    terminal=json.loads((out/'terminal.json').read_bytes())
    if terminal['exit_code']!=0:raise ValueError('paired endpoint run failed')
    audit=json.loads(pinned(out/'paired-score.json',terminal['score_sha256']))
    m=json.loads(pinned(seal/'MANIFEST.json',audit['seal_sha256']))
    def item(rel):
        return json.loads(pinned(seal/rel,m['files'][rel]))
    pairs=item('scoring/pairs.json');targets={r['id']:r for r in item('scoring/targets.json')}
    report={}
    for arm,proof in audit['arms'].items():
        labels=item(f'inference/{arm}/panel.json')['candidate_ids']
        a,b=(labels.index(c[k]) for k in ('positive','contrast'))
        preds=json.loads(pinned(out/'hf/run'/f'hf-{arm}-collected.json',proof['predictions_file_sha256']))
        probs={r['id']:r['prob'] for r in preds['records']}
        if probs.keys()!=targets.keys() or preds['policy_id']!=audit['policy_id']:raise ValueError('incomplete predictions')
        counts={}; margins=[]
        for pair in pairs:
            up,down=pair['up'],pair['down']
            if targets[up]['gold']!=labels[a] or targets[down]['gold']!=labels[b]:
                raise ValueError('pair direction mismatch')
            x,y=probs[up],probs[down]
            left,right=x[a]-x[b],y[b]-y[a]
            margins.append(min(left,right))
            for group in ('all','same_speaker' if pair['same_speaker'] else 'different_speakers',
                          '/'.join(pair['tokens'])):
                k=counts.setdefault(group,Counter())
                k['pairs']+=1
                k['moves_in_expected_direction']+=left+right>0
                k['both_beat_contrast']+=min(left,right)>0
                k['both_clear_margin']+=min(left,right)>=c['minimum_probability_margin']
                k['both_top1']+=max(range(len(x)),key=x.__getitem__)==a and max(range(len(y)),key=y.__getitem__)==b
        if counts['all']['both_top1']!=proof['direction_both_correct']:raise ValueError('paired baseline changed')
        report[arm]=dict(groups=counts,median_weaker_pair_gap=median(margins))
    return dict(policy=audit['policy_id'],seal=audit['seal_sha256'],margin=c['minimum_probability_margin'],
                arms=report,scope='source TRAIN, paired lexical frames; not isolated word detection or held-out transfer')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True);a=p.parse_args()
    cfg,provenance=load_config(a.config,a.local);res=dict(**score(cfg),provenance=provenance)
    dest=Path(cfg['package']['output']).parent/'aligned-cue-score-v1.json'
    commit_json(dest,res);print(json.dumps(res,indent=2))
