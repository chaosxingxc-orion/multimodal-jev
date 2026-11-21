"""Deterministic source TRAIN contrasts. Transcript matching is offline only."""
from collections import defaultdict
from hashlib import sha256


def rank(seed, key):
    return sha256(f'{seed}:{key}'.encode()).digest()


def split_panel(rows, cfg):
    """Hold out the union of all example speakers, source groups and wording."""
    ordered = sorted(rows, key=lambda r: (rank(cfg['seed'], r['id']), r['id']))
    examples = {}
    fixed = cfg.get('example_text')
    if fixed is None:
        labels = defaultdict(set)
        for row in ordered:
            labels[row['text']].add(row['gold'])
        rule = cfg['examples']
        if not 0 < rule['min_words'] <= rule['max_words']:
            raise ValueError('bounded automatic support wording length required')
        for label in cfg['boundaries']:
            examples[label] = next((r for r in ordered if r['gold'] == label
                                    and labels[r['text']] == {label}
                                    and rule['min_words'] <= len(r['text'].split()) <= rule['max_words']
                                    and all(r[k] for k in ('speaker', 'source', 'group'))), None)
    else:
        for label, text in fixed.items():
            examples[label] = next((r for r in ordered if r['gold'] == label and r['text'] == text), None)
    for label in examples:
        if not examples[label]:
            raise ValueError(f'no admitted TRAIN example for {label}')
    def keys(row):
        return {('speaker',row['speaker']), ('group',row['group']), ('source',row['source']),
                ('audio',row['audio_sha256']), ('text',row['text'])}
    excluded = set().union(*(keys(r) for r in examples.values()))
    pool = [r for r in ordered if not keys(r) & excluded]
    chosen = {}
    for gold in cfg['boundaries']:
        subset = [r for r in pool if r['gold']==gold]
        used = set()
        for r in subset:
            if r['text'] not in used:
                chosen[r['id']] = (r,'calendar')
                used.add(r['text'])
                if len(used) == cfg['calendar_each']:
                    break
        if len(used) != cfg['calendar_each']:
            raise ValueError('insufficient distinct primary wordings')
    groups = defaultdict(lambda: defaultdict(list))
    for r in pool:
        if r['gold'] not in ('audio_volume_up','audio_volume_down'):
            continue
        words = r['text'].split()
        for a,b in cfg['direction_pairs']:
            word = a if r['gold']=='audio_volume_up' else b
            if word in words and words.count(a)+words.count(b)==1:
                frame = (a,b,tuple('_' if w==word else w for w in words))
                groups[frame][word].append(r)
    pairs = []
    for (a,b,frame), variants in sorted(groups.items()):
        left, right = variants[a], variants[b]
        for _ in range(cfg['pair_limit_per_frame']):
            x = next((r for r in left if r['id'] not in chosen),None)
            if x is None: break
            eligible = [r for r in right if r['id'] not in chosen and r['audio_sha256'] != x['audio_sha256']]
            if not eligible: break
            y = min(eligible, key=lambda r:r['speaker'] != x['speaker'])
            for r in (x,y):
                chosen[r['id']] = (r, 'direction')
            pairs.append(dict(up=x['id'],down=y['id'],frame=' '.join(frame),tokens=[a,b],
                              same_speaker=x['speaker']==y['speaker']))
    if not pairs:
        raise ValueError('no positive/negative direction pairs')
    others = [r for r in pool if r['gold'] not in {*cfg['boundaries'],'audio_volume_up','audio_volume_down'}]
    if len(others)<cfg['background']: raise ValueError('insufficient background')
    for r in others[:cfg['background']]:
        chosen[r['id']] = (r,'background')
    selected = sorted(chosen.values(),key=lambda item:item[0]['id'])
    return examples, selected, pairs
