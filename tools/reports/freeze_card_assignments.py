"""Freeze TRAIN-only support decisions, template variants and per-epoch exposures."""
import hashlib
import json
import random
from collections import defaultdict
from collections import Counter
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.intent_card_template import compile_intent_card_request
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.build_slurp_intent_cards import read
from tools.reports.review_knowledge_card_transfer import coverage


def normalize(text):
    return ' '.join(text.casefold().split())


def compile_plan(c, output, design_sha, *, extension=None):
    bank = json.loads(read(Path(c['bank']), c['bank_sha256']))
    pins = bank['pins']
    release = Path(c['release'])
    manifest = json.loads(read(release/'MANIFEST.json', pins['clean_manifest']))
    rows = [json.loads(line) for line in read(release/'learning.jsonl', manifest['files']['learning.jsonl']).splitlines()]
    task = json.loads(read(release/'tasks.json', manifest['files']['tasks.json']))['intent']
    truth = json.loads(read(Path(c['truth']), pins['truth']))
    roster = json.loads(read(Path(c['roster']), pins['roster']))
    calibration = json.loads(read(Path(c['calibration']), c['calibration_sha256']))
    texts = {'slurp:'+rec['file']: normalize(ann['sentence'])
             for line in read(Path(c['annotations']), pins['annotations']).splitlines()
             for ann in [json.loads(line)] for rec in ann['recordings']}
    blocked = [{r[k] for r in calibration} for k in ('source_group','audio_sha256')]
    if set(truth) != {r['id'] for r in roster} or any(r['source_group'] in blocked[0] or
            r['audio_sha256'] in blocked[1] for r in roster):
        raise ValueError('card development pool overlaps calibration')
    pool = {r['id']:r for r in roster}
    support_roster = list(roster)
    release_sha = pins['clean_manifest']
    if extension:
        extra_rows, definitions, release_sha = extension
        extra_ids = set(definitions)
        for label, spec in definitions.items():
            task['candidate_pairs'][label] = spec['pair']
            task['schema']['candidates'].append(dict(id=label,description=spec['activation']))
            bank['cards'].append(dict(id=label, activation=spec['activation'], examples=[], contrast_with=None))
        for row in extra_rows:
            texts[row['id']] = normalize(row['annotation_text'])
            if row['intent'] in extra_ids:
                support_roster.append(row)
                pool[row['id']] = row
        rows.extend(extra_rows)
        commit_json(output/'mixed-task.json',task)
    labels = [x['id'] for x in bank['cards']]
    # Draw from distinct, short official TRAIN annotations, including rare classes.
    admitted = {r['id']:r for r in rows}
    inventory = defaultdict(dict)
    text_labels = defaultdict(set)
    for item in support_roster:
        uid = item['id']
        row = admitted[uid]
        if any(row[k] != item[k] for k in ('source_group','intent','audio_sha256')):
            raise ValueError('support roster changed')
        text = texts[uid]
        text_labels[text].add(row['intent'])
        inventory[row['intent']].setdefault(text, uid)
    supports = []
    support_keys = set()
    def keys(group, audio, text):
        return {('group',group),('audio',audio),('text',text)}
    for card in bank['cards']:
        label = card['id']
        options = sorted((text,uid) for text,uid in inventory[label].items()
                         if text_labels[text]=={label} and len(text.split())<=c['max_support_words'])
        rng = random.Random(f"{c['seed']}|pool|{label}")
        rng.shuffle(options)
        chosen = []
        groups = set()
        for text,uid in options:
            r = pool[uid]
            if r['source_group'] in groups: continue
            groups.add(r['source_group'])
            chosen.append(dict(id=uid,text=text,group=r['source_group'],audio_sha256=r['audio_sha256']))
            if len(chosen)==c['max_supports_per_class']: break
        if not chosen:
            chosen = card['examples']
        card['examples'] = chosen
        values = []
        for sample in chosen:
            row = pool[sample['id']]
            if (row['intent'] != label or row['source_group'] != sample['group'] or
                    row['audio_sha256'] != sample['audio_sha256'] or texts[sample['id']] != normalize(sample['text'])):
                raise ValueError('support differs from independent TRAIN annotation')
            values.append(keys(row['source_group'],row['audio_sha256'],texts[row['id']]))
            support_keys.update(values[-1])
        supports.append(values)
    commit_json(output/'sampled-bank.json',bank)
    rankings = []
    for draw in range(c['support_panels']):
        panel_ranks = []
        for label,values in zip(labels,supports,strict=True):
            order = list(range(len(values)))
            random.Random(f"{c['seed']}|panel|{draw}|{label}").shuffle(order)
            panel_ranks.append(order)
        rankings.append(panel_ranks)
    # Invert support membership so conflict decisions do not need gold or a label search.
    conflicts = {}
    for i, values in enumerate(supports):
        for j, terms in enumerate(values):
            for term in terms:
                conflicts.setdefault(term,set()).add((i,j))
    template = json.loads(Path(c['template']).read_bytes())
    def panel(arm, indices=None):
        d = compile_intent_card_request('audio',task,bank,template,arm,example_indices=indices).model_input()
        del d['audio']
        return d
    panels = dict(native=panel('original'),rules=panel('definitions_rules'))
    design = json.loads(read(output/'design.json', design_sha))
    epochs = design['stage1']['batch_config']['epochs']
    assignments, variants, roles = {}, {}, {}
    review = Counter()
    for row in rows:
        uid = row['id']
        if row['role'] != 'learning' or row['official_split'] != 'train' or row['known_pending_source']:
            raise ValueError('unadmitted Stage1 row')
        terms = keys(row['source_group'],row['audio_sha256'],texts[uid])
        excluded = set().union(*(conflicts.get(k,()) for k in terms))
        draw = random.Random(f"{c['seed']}|examples|{uid}").randrange(len(rankings))
        indices = tuple(next((j for j in order if (i,j) not in excluded),None) for i,order in enumerate(rankings[draw]))
        if indices not in variants:
            name = 'examples-'+str(len(variants))
            panels[name] = panel('definitions_rules_example', dict(zip(labels,indices,strict=True)))
            variants[indices] = name
        pair = ['rules',variants[indices]]
        if hashlib.sha256(f"{c['seed']}:{row['source_group']}".encode()).digest()[0] % 2:
            pair.reverse()
        assignments[uid] = [pair[e % len(pair)] for e in range(epochs)]
        review['alternate'] += sum(x is not None and x!=order[0] for x,order in zip(indices,rankings[draw]))
        review['omitted'] += indices.count(None)
        if terms & support_keys:
            roles[uid] = 'support-or-related'
        elif uid in truth:
            roles[uid] = 'card-development'
        else:
            roles[uid] = 'stage1-train-only'
    report = coverage(bank,truth,labels)
    # Round two: count both sides of shared lexical cues on TRAIN labels.
    draft = json.loads(Path(c['review']).read_bytes())
    words = {w:{label:len({texts[r['id']] for r in roster if r['intent']==label and w in texts[r['id']].split()})
                for label in classes} for w,classes in draft['word_checks'].items()}
    # Round three: switch all examples and reverse the complete panel, restoring ID order.
    alt = {label:len(v)-1 for label,v in zip(labels,supports,strict=True)}
    alternate = panel('definitions_rules_example',alt)
    reversed_task = dict(task,schema=dict(task['schema'],candidates=list(reversed(task['schema']['candidates']))))
    reversed_panel = compile_intent_card_request('audio',reversed_task,bank,template,'definitions_rules_example',
                                                example_indices=alt).model_input()
    if reversed_panel['questions'][::-1] != alternate['questions'] or alternate['questions']==panels['rules']['questions']:
        raise ValueError('replacement or permutation failed')
    plan = dict(schema='jev-stage1-card-plan-v1',release_sha256=release_sha, design_sha256=design_sha,
                panels=panels,assignments=assignments)
    commit_json(output/'card-plan.json',plan)
    commit_json(output/'roles.json',roles)
    commit_json(output/'card-review.json',dict(population=len(rows),exposures=epochs*len(rows),panels=len(panels),
        supports=sum(map(len,supports)),conflicts=review,roles=Counter(roles.values()),pair_review=report,
        words=words,supports_by_class=dict(zip(labels,map(len,supports))),seed=c['seed'],sampled_panels=len(rankings),permutation_roundtrip=True,bank=c['bank_sha256'],plan=file_hash(output/'card-plan.json')))
