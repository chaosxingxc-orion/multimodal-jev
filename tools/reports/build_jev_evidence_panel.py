"""CPU design fixture: closed-world availability questions, not slot extraction."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_decision_contract import PREFIX_LAYOUT, render_prefix_context_request
from multimodal_jev.decision_evidence import adjudicate
from multimodal_jev.local_checkpoint import commit_json, sync_directory
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.baseline import file_hash
from tools.reports.build_jev_boundary_panel import source_rows


def scan_calendar(rows, rules):
    """A frozen grammar identifies only questions to which a day-existence control applies."""
    offsets = rules['day_offsets']
    if offsets != {'today': 0, 'tomorrow': 1} or not rules['patterns']:
        raise ValueError('two explicit day offsets and a frozen grammar required')
    patterns = [re.compile(pattern) for pattern in rules['patterns']]
    if any('day' not in pattern.groupindex for pattern in patterns):
        raise ValueError('every calendar pattern must capture its day')
    labels = defaultdict(set)
    for row in rows:
        labels[row['text']].add(row['gold'])
    counts = Counter()
    selected = []
    for row in sorted(rows, key=lambda r: r['id']):
        if row['gold'] != rules['native_intent']:
            continue
        counts['native_intent_recordings'] += 1
        days = {match['day'] for pattern in patterns if (match := pattern.fullmatch(row['text']))}
        if not days:
            counts['outside_grammar'] += 1
        elif len(days) != 1 or len(labels[row['text']]) != 1:
            counts['ambiguous'] += 1
        else:
            day = next(iter(days))
            if day not in offsets:
                raise ValueError('calendar grammar captured an unregistered day')
            counts[day] += 1
            selected.append((row, offsets[day]))
    for day in offsets:
        counts.setdefault(day, 0)
    if not selected:
        raise ValueError(f'no eligible calendar recordings: {dict(counts)}')
    return selected, dict(train_recordings=len(rows), **counts)


def build(c, provenance=None):
    if c['schema'] not in ('jev-evidence-calendar-v1', 'jev-evidence-calendar-train-v1'):
        raise ValueError('explicit diagnostic protocol required')
    _, rows = source_rows(c)
    scan = c['schema'] == 'jev-evidence-calendar-train-v1'
    worlds={'forward':(0,), 'reverse':(1,), 'missing':(0,1), 'irrelevant':(0,)}
    if scan:
        selected, population = scan_calendar(rows, c['calendar'])
        vocabulary = c['calendar']['context']
        contexts = {arm: vocabulary['convention'] + ' ' + vocabulary['meeting_by_day'][str(options[0])]
                    if len(options) == 1 else vocabulary['convention'] + ' ' + vocabulary['missing']
                    for arm, options in worlds.items()}
        contexts['irrelevant'] += ' ' + vocabulary['unrelated_missing']
    else:
        selected = [(r, int(c['wordings'][r['text']] == 'tomorrow')) for r in rows
                    if r['gold'] == 'calendar_query' and r['text'] in c['wordings']]
        if {r['text'] for r, _ in selected} != set(c['wordings']) or set(c['wordings'].values()) != {'today','tomorrow'}:
            raise ValueError('both admitted wordings and day offsets must be observed')
        population, contexts = {}, c['contexts']
    present = {r['text'] for r, _ in selected}
    queries={'intent':'Is this a request to consult the calendar?',
             'availability':'Under the supplied complete facts, is there a meeting on the day requested?',
             'intent_sufficient':'Is there enough information to decide the calendar intent?',
             'availability_sufficient':'Is there enough information to decide whether that day has a meeting?'}
    if set(contexts)!=set(worlds):raise ValueError('complete fixed counterfactual required')
    out=Path(c['output']);check_paths({'design':out});out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    with TemporaryDirectory(dir=out.parent) as tmp:
        root=Path(tmp)/'data';root.mkdir();files={};counts=Counter()
        def write(rel,obj):
            p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);commit_json(p,obj);files[rel]=file_hash(p)
        roster=[{k:r[k] for k in ('id','audio','audio_sha256')} for r,_ in selected]
        for arm,options in worlds.items():
            facts=contexts[arm]
            req=compile_candidates('audio',[dict(id=k,definition=v) for k,v in queries.items()],[facts],semantics='candidate_applicability')
            system,_,readouts=render_prefix_context_request(req)
            write(f'inference/{arm}/panel.json',dict(system=system,readouts=readouts,questions=[asdict(q) for q in req.questions],
                requirements=list(req.requirements),version=req.version,candidate_ids=list(queries),prompt_layout=PREFIX_LAYOUT,history=[]))
            write(f'inference/{arm}/rows.json',roster)
            labels=[]
            for r,day in selected:
                a=adjudicate([day==x for x in options],exhaustive=True)
                labels.append(dict(id=r['id'],audio_sha256=r['audio_sha256'],group=r['group'],speaker=r['speaker'],gold={
                    'intent':True,'availability':a.answer,'intent_sufficient':True,'availability_sufficient':a.sufficient}))
                counts['availability_unknown' if a.sufficient is False else 'availability_known']+=1
            write(f'scoring/{arm}.json',labels)
        commit_json(root/'MANIFEST.json',dict(schema='jev-native-evaluation-seal-v1',prompt_layout=PREFIX_LAYOUT,
            datasets={a:dict(recordings=len(selected),candidates=len(queries),split='source_train_synthetic_context') for a in worlds},
            target_tuning_allowed=False,threshold=.5,files=files,
            diagnostic='calendar_fact_availability', population=population, provenance=provenance,
            clean_sha256=c['clean']['sha256'], native_sha256=c['native']['sha256'],
            scope='Hypothetical two-day schedule; no real-calendar claims, no acoustic evidence removal, not approved for training.'))
        root.rename(out)
        sync_directory(out.parent)
    return dict(recordings=len(selected),contexts=len(worlds),wordings=len(present),**counts,population=population,
                manifest_sha256=file_hash(out/'MANIFEST.json'),model_calls=0,npu_executed=False,observed_errors=None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();print(json.dumps(build(*load_config(a.config,a.local)),indent=2))
