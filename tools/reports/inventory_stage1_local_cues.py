"""Discover one contiguous local edit in otherwise identical clean TRAIN wordings."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import json

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.build_jev_boundary_panel import source_rows


def local_edits(rows, allowed, max_changed_words, min_shared_words):
    # Compare distinct wordings, not all microphone copies of each sentence.
    labels = {label for pair in allowed for label in pair}
    phrases = defaultdict(list)
    for row in rows:
        if row['gold'] in labels:
            phrases[(row['gold'], tuple(row['text'].split()))].append(row)
    frames = defaultdict(list)
    for label, words in sorted(phrases):
        for start in range(len(words)+1):
            for size in range(min(max_changed_words, len(words)-start)+1):
                if len(words)-size >= min_shared_words:
                    frames[words[:start],words[start+size:]].append((label,words,words[start:start+size]))
    pairs = []
    seen = set()
    for (prefix,suffix), group in frames.items():
        for left,right in combinations(group,2):
            a,x,edit1 = left
            b,y,edit2 = right
            if x==y or (a!=b and frozenset((a,b)) not in allowed):
                continue
            if edit1 and edit2 and (edit1[0]==edit2[0] or edit1[-1]==edit2[-1]):
                continue
            key=tuple(sorted(((a,x),(b,y))))
            if key in seen: continue
            seen.add(key)
            pairs.append(dict(labels=[a,b],frame=' '.join((*prefix,'<edit>',*suffix)),
                              edits=[' '.join(edit1),' '.join(edit2)],
                              kind='preserve' if a==b else 'flip',
                              references=[min(r['id'] for r in phrases[a,x]),min(r['id'] for r in phrases[b,y])]))
    return pairs


def run(c, provenance):
    if c['schema']!='stage1-local-cues-v1':
        raise ValueError('unsupported cue protocol')
    _,rows=source_rows(c)
    pairs=local_edits(rows,{frozenset(p) for p in c['intent_pairs']},c['max_changed_words'],c['min_shared_words'])
    stats=Counter(p['kind'] for p in pairs)
    by_pair=Counter(' / '.join(sorted(p['labels'])) for p in pairs if p['kind']=='flip')
    result=dict(source_recordings=len(rows),distinct_wording_pairs=dict(stats),flip_by_native_label=dict(by_pair),
                max_changed_words=c['max_changed_words'],min_shared_words=c['min_shared_words'],
                scope='offline contiguous orthographic edit candidates; no audio frame alignment or individual causal annotation',
                provenance=provenance,pairs=pairs)
    commit_json(Path(c['output']),result)
    return {k:v for k,v in result.items() if k not in ('pairs','provenance')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();print(json.dumps(run(*load_config(a.config,a.local)),indent=2))
