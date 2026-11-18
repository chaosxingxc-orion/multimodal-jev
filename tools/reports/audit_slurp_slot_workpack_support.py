"""Reconcile current semantic proposals with the frozen learning population.

No new annotations, automatic propagation, or training admission. Source
reference text is never emitted or sent to a model by this report.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def reconcile(population,sources,coordinates,types):
    recording={r['id']:r for r in population}
    if len(recording)!=len(population):raise ValueError('Duplicate recording')
    if any(r['role']!='learning' or r['official_split']!='train' or r['known_pending_source'] for r in population):
        raise ValueError('Only admitted clean learning population allowed')
    groups=defaultdict(set);source_groups=defaultdict(set);bindings=defaultdict(set)
    for r in population:groups[r['source_group']].add(r['id'])
    source_ids=set()
    for s in sources:
        sid=str(s['source_id'])
        if sid in source_ids:raise ValueError('Duplicate source')
        source_ids.add(sid)
        if s['optimization_allowed'] or s['semantic_annotation'] is not None:
            raise ValueError('Expected reference-only workpack')
        for b in s['recording_bindings']:
            uid=b['recording_id']
            if uid not in recording:raise ValueError('Source outside frozen population')
            r=recording[uid]
            if (b['source_group']!=r['source_group'] or b['audio_sha256']!=r['audio_sha256']
                    or b['source_target_sha256']!=r['source_target_sha256']
                    or sid not in [str(x) for x in b['all_source_ids']]
                    or s['native_intent']!=r['intent']):raise ValueError('Source/audio/target binding changed')
            bindings[sid].add(uid);source_groups[r['source_group']].add(sid)
    if set().union(*bindings.values())!=set(recording) or set(source_groups)!=set(groups):
        raise ValueError('Incomplete recording or group coverage')
    index={};by_type={t:Counter() for t in types};supports={t:defaultdict(set) for t in types}
    for c in coordinates:
        sid,t=str(c['source_id']),c['definition_id'];key=(sid,t)
        if sid not in source_ids or t not in by_type or key in index:raise ValueError('Invalid coordinate')
        if c['optimization_allowed'] or c['admitted_value'] is not None:
            raise ValueError('Reference coordinate unexpectedly admitted')
        index[key]=c;by_type[t][c['status']]+=1
        supports[t][c['status']].update(bindings[sid])
    all_reviewed_sources={sid for sid in source_ids if all((sid,t) in index for t in types)}
    all_current_sources={sid for sid in source_ids if all(index.get((sid,t),{}).get('status')=='current_reference_proposal' for t in types)}
    columns=[]
    for t in types:
        counts=by_type[t];known=sum(counts.values())
        columns.append(dict(type=t,source_status=dict(counts),unassessed_sources=len(source_ids)-known,
            recording_support_by_status={k:len(v) for k,v in supports[t].items()},
            formally_admitted_coordinates=0,
            note='Recording support is descriptive evidence coverage, not propagated labels or independent sources.'))
    merged=[]
    for group,sids in sorted(source_groups.items()):
        if len(sids)>1:
            merged.append(dict(source_group=group,source_ids=sorted(sids),recordings=len(groups[group]),
                intents=sorted({recording[uid]['intent'] for uid in groups[group]})))
    coordinate_counts=Counter(c['status'] for c in coordinates)
    return dict(recordings=len(recording),source_groups=len(groups),source_ids=len(source_ids),
        source_recording_links=sum(len(v) for v in bindings.values()),
        merged_source_groups=len(merged),extra_source_ids=sum(len(v)-1 for v in source_groups.values()),
        merged_group_details=merged,types=len(types),total_source_type_coordinates=len(source_ids)*len(types),
        total_group_type_coordinates=len(groups)*len(types),reviewed_coordinates=len(index),
        unassessed_coordinates=len(source_ids)*len(types)-len(index),coordinate_status=dict(coordinate_counts),
        sources_with_any_all_type_reference=len(all_reviewed_sources),
        sources_with_all_types_current_proposals=len(all_current_sources),
        current_proposal_fraction=coordinate_counts['current_reference_proposal']/(len(source_ids)*len(types)),
        admitted_coordinates=0,by_type=columns,
        limits=['Source IDs and deduplicated source groups are different denominators.',
            'Current compatible proposals are not independently reviewed semantic truth.',
            'Missing/stale/unavailable coordinates must not become negative examples.',
            'No proposal is propagated across merged sources or recordings by this audit.'])


def run(release,workpack):
    release,workpack=Path(release),Path(workpack)
    m=read(release/'MANIFEST.json');report=read(workpack/'report.json')
    if sha(release/'learning.jsonl')!=m['files']['learning.jsonl']:raise ValueError('Learning release changed')
    for name,h in report['files'].items():
        if sha(workpack/name)!=h:raise ValueError('Workpack changed')
    for name,h in report['pins'].items():
        if sha(name)!=h:raise ValueError('Workpack source dependency changed')
    ontology=read(workpack/'ontology.json')
    types=[r['entity_type'] for r in ontology['slots']]
    result=reconcile(rows(release/'learning.jsonl'),rows(workpack/'sources.jsonl'),
        rows(workpack/'coordinates.jsonl'),types)
    if (result['recordings']!=40513 or result['source_ids']!=report['source_utterances']
            or result['reviewed_coordinates']!=report['slot_reference_coordinates']
            or result['coordinate_status']!=report['coordinate_status_counts']):
        raise ValueError('Summary population differs')
    result.update(schema='slurp-slot-workpack-learning-support-v1',training_dispatch_allowed=False,
        sources={str(p):sha(p) for p in (release/'MANIFEST.json',release/'learning.jsonl',
            workpack/'report.json',workpack/'sources.jsonl',workpack/'coordinates.jsonl',workpack/'ontology.json')},
        tool_sha256=sha(__file__),scope='Full TRAIN identity and existing proposal coverage only; no new semantic labels or model results.')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('release','workpack','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();r=run(a.release,a.workpack)
    with Path(a.output).open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('by_type','sources','merged_group_details')},indent=2))
