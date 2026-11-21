"""Separate strict ranking, empty/multiple threshold outputs and margin failures on frozen TRAIN."""
import argparse
from collections import Counter
import json
from pathlib import Path

from multimodal_jev.local_config import load_config
from tools.reports.jev_train_confusions import pinned


def breakdown(view):
    root = Path(view['view'])
    manifest = json.loads(pinned(root/'MANIFEST.json', view['manifest_sha256']))
    selection = json.loads(pinned(root/'selection.json', manifest['files']['selection.json']))
    threshold = manifest['contract']['options']['decision_threshold']
    counts = Counter()
    for r in selection['diagnostics']:
        if not r['ranking_correct']:
            k = 'wrong_or_tied'
        elif not r['exact_set']:
            k = 'rank_correct_empty' if r['gold_probability'] < threshold else 'rank_correct_multiple'
        else:
            k = r['bucket']
        counts[k] += 1
    s = selection['summary']
    if (counts['wrong_or_tied']+counts['rank_correct_empty']+counts['rank_correct_multiple'] != s['buckets']['error']
            or counts['boundary']!=s['buckets']['boundary'] or counts['easy']!=s['buckets']['easy']
            or sum(counts.values())!=s['population']):
        raise ValueError('diagnostics and frozen sampling summary disagree')
    return dict(counts=dict(counts), selected_easy=s['selected_easy'], population=s['population'],
                selected=s['selected'], threshold=threshold, margin=manifest['contract']['options']['probability_margin'],
                view_sha256=view['manifest_sha256'],policy_id=manifest['contract']['endpoint']['policy_id'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    args=p.parse_args()
    c,_=load_config(args.config,args.local)
    print(json.dumps(breakdown(c['train_view']),indent=2))
