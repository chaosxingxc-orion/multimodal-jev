"""Score-blind partner index for admitted TRAIN; request text is never changed."""
from collections import defaultdict
import hashlib
import random
import unicodedata

SCHEMA = 'jev-train-speech-pairs-v1'
KINDS = ('repeat', 'same_intent')


def wording(s):
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())


def build(population, texts, rules, settings):
    if set(settings) != {'positive_cycle','min_shared_words','control_arm'} or (
            settings['control_arm'] != 'rules' or not isinstance(settings['positive_cycle'],list) or set(settings['positive_cycle']) != set(KINDS)
            or type(settings['min_shared_words']) != int or settings['min_shared_words'] < 1):
        raise ValueError('explicit positive cycle and minimum shared wording required')
    if set(population) != set(texts):
        raise ValueError('source annotation roster differs')
    texts = {u:wording(t) for u,t in texts.items()}
    groups = defaultdict(set)
    audio_labels = defaultdict(set)
    for u,r in population.items():
        if (u!=r['id'] or r['role']!='learning' or r['official_split']!='train' or r['known_pending_source']
                or not texts[u] or not r['audio_sha256']):
            raise ValueError('expected admitted source TRAIN and official text')
        groups[texts[u]].add(r['intent'])
        audio_labels[r['audio_sha256']].add(r['intent'])
    # A multi-label wording or identical file with conflicting labels cannot certify a semantic pair.
    keys = sorted(t for t in groups if len(groups[t]) == 1)
    index = {t:i for i,t in enumerate(keys)}
    row_keys = {u:index.get(t,-1) if len(audio_labels[population[u]['audio_sha256']]) == 1 else -1
                for u,t in sorted(texts.items())}
    by_pattern = defaultdict(list)
    rivals = defaultdict(set)
    for rule in rules:
        for c in rule['labels']:
            rivals[c].update(k for k in rule['labels'] if k!=c)
    eligible_keys = set(row_keys.values())
    for i,t in enumerate(keys):
        if i not in eligible_keys:
            continue
        tokens = tuple(t.split())
        if len(tokens)-1 >= settings['min_shared_words']:
            for j in range(len(tokens)):
                by_pattern[j,tokens[:j]+tokens[j+1:]].append(i)
    links = defaultdict(set)
    labels = [next(iter(groups[t])) for t in keys]
    for ids in by_pattern.values():
        if len(ids)<2:
            continue
        by_label = defaultdict(list)
        for i in ids:
            by_label[labels[i]].append(i)
        for c,members in by_label.items():
            others = [v for k in rivals[c] for v in by_label.get(k,())]
            if others:
                for i in members:
                    links[i].update(others)
    return dict(schema=SCHEMA, settings=settings, rows=row_keys,
                rivals={str(i): sorted(v) for i,v in sorted(links.items())})


class Partners:
    def __init__(self, view, population):
        if view['schema']!=SCHEMA or view['rows'].keys()!=population.keys():
            raise ValueError('speech partner index does not cover the training population')
        self.keys = view['rows']
        self.cycle = view['settings']['positive_cycle']
        self.arm = view['settings']['control_arm']
        self.labels = {u:r['intent'] for u,r in population.items()}
        self.sounds = {u:r['audio_sha256'] for u,r in population.items()}
        self.members = defaultdict(list)
        self.by_label = defaultdict(set)
        for u,k in sorted(self.keys.items()):
            if k >= 0:
                self.members[k].append(u)
                self.by_label[self.labels[u]].add(k)
        self.rivals = {int(k):v for k,v in view['rivals'].items()}
        self.by_label = {k:sorted(v) for k,v in self.by_label.items()}
        self.pos = {k:i for v in self.by_label.values() for i,k in enumerate(v)}
        self.phase = {u:int.from_bytes(hashlib.sha256(u.encode()).digest()[:4],'big') for u in self.keys}

    def draws(self, uid, epoch, seed):
        """Return observed positive type/uid plus a rule-linked one-word substitution, if available."""
        key = self.keys[uid]
        if key < 0:
            return None,None,None
        rng = random.Random(f'{seed}|speech-relations|{uid}|{epoch}')
        other_forms = self.by_label[self.labels[uid]]
        other_recs = [v for v in self.members[key] if self.sounds[v]!=self.sounds[uid]]
        kind = self.cycle[(epoch+self.phase[uid])%len(self.cycle)]
        if not other_recs:
            kind = 'same_intent'
        if len(other_forms)==1:
            kind = 'repeat'
        pos = None
        if kind=='repeat' and other_recs:
            pos=rng.choice(other_recs)
        elif kind=='same_intent' and len(other_forms)>1:
            idx = rng.randrange(len(other_forms)-1)
            other = other_forms[idx+(idx>=self.pos[key])]
            pos=rng.choice(self.members[other])
        rivals=self.rivals.get(key)
        neg = rng.choice(self.members[rng.choice(rivals)]) if rivals else None
        if pos is not None and self.sounds[pos]==self.sounds[uid]:
            # A file may also be listed under a different same-class wording. Search alternatives
            # only on that exceptional path so the normal O(1) different-wording draw stays cheap.
            choices = [v for form in other_forms if form != key for v in self.members[form]
                       if self.sounds[v] != self.sounds[uid]]
            if choices:
                pos = rng.choice(choices)
            elif other_recs:
                kind,pos = 'repeat',rng.choice(other_recs)
            else:
                pos = None
        if neg is not None and self.sounds[neg] in (self.sounds[uid], self.sounds.get(pos)):
            neg=None
        return kind if pos is not None else None,pos,neg
