"""TRAIN-only native attribute projection dataset, never semantic-presence gold.

One task-wide menu, shared event definitions, explicit group membership. Input
construction accepts no per-example targets. Outputs are candidates only.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from multimodal_jev.context_decision_contract import ApplicabilityQuestion, ContextDecisionRequest

RELEASE_SHA='38e4273d9328e4f02db081315e46f97a17628b974be7ba175254882e95421482'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def catalog(task):
    pairs=task['candidate_pairs'];events=task['schema']['candidates']
    if (len({c['id'] for c in events})!=len(events) or set(pairs)!={c['id'] for c in events}
            or not pairs or any(type(v)!=list or len(v)!=2 or any(type(x)!=str or not x for x in v) for v in pairs.values())
            or len({tuple(v) for v in pairs.values()})!=len(pairs)):
        raise ValueError('Complete distinct native event pairs required')
    nodes=[]
    for index,family in enumerate(('scenario','action')):
        for value in sorted({v[index] for v in pairs.values()}):
            nodes.append(dict(id=family+':'+value,family=family,value=value,
                              members=sorted(k for k,v in pairs.items() if v[index]==value)))
    nodes += [dict(id='final:'+c['id'],family='final',value=c['id'],members=[c['id']]) for c in events]
    return dict(schema='context-native-attribute-catalog-candidate-v1',events=events,nodes=nodes,
                requirements=list(task['thinking']['requirements']),
                semantics='membership_of_exclusive_benchmark_event_not_mention_detection',
                supervision='deterministic_projection_of_native_Final_not_independent_process_annotation')


def request(audio, menu, *, arm):
    if arm not in ('direct','q_input','q_credit'):raise ValueError('Known comparison arm required')
    nodes=[n for n in menu['nodes'] if arm!='direct' or n['family']=='final']
    questions=tuple(ApplicabilityQuestion(n['id'],
        'This group consists exactly of event IDs: '+json.dumps(n['members'],separators=(',',':'))+'.',
        'Is the benchmark-assigned target event a member of this defined group?') for n in nodes)
    rules=(*menu['requirements'],
           'Exactly one benchmark event is assigned. Judge membership independently for each supplied group. '
           'Group membership is not semantic mention detection. Do not infer semantic absence from an unassigned event.',
           'Shared complete benchmark event catalog: '+json.dumps(menu['events'],ensure_ascii=False,separators=(',',':')))
    return ContextDecisionRequest(audio,questions,rules,version='jev-native-attribute-projection-candidate-v1')


def targets(menu, event):
    if event is not None and event not in {e['id'] for e in menu['events']}:
        raise ValueError('Unknown assigned event')
    return dict(candidate_ids=[n['id'] for n in menu['nodes']],
                labels=[None if event is None else event in n['members'] for n in menu['nodes']],
                supervised=[event is not None]*len(menu['nodes']))


def build(root, output):
    root=Path(root);output=Path(output)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(root/'MANIFEST.json')!=RELEASE_SHA:raise ValueError('Frozen release changed')
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for name in ('tasks.json','learning.jsonl','learning.audio.jsonl'):
        if sha(root/name)!=manifest['files'][name]:raise ValueError('TRAIN asset changed')
    task=json.loads((root/'tasks.json').read_text())['intent'];menu=catalog(task)
    rows=[json.loads(line) for line in (root/'learning.jsonl').read_text().splitlines()]
    audio=[json.loads(line) for line in (root/'learning.audio.jsonl').read_text().splitlines()]
    if len(rows)!=40513 or len(audio)!=len(rows) or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Full clean TRAIN population required')
    byid={a['id']:a for a in audio}
    if set(byid)!={r['id'] for r in rows}:raise ValueError('Audio identity differs')
    groups={};counts=Counter()
    for r in rows:
        if (r['role']!='learning' or r['official_split']!='train' or r['known_pending_source'] is not False
                or task['candidate_pairs'][r['intent']]!=[r['scenario'],r['action']]
                or byid[r['id']]['audio_sha256']!=r['audio_sha256']):
            raise ValueError('Native label or admitted audio identity differs')
        if groups.setdefault(r['source_group'],r['intent'])!=r['intent']:raise ValueError('Conflicting source group')
        counts[r['intent']]+=1
    if set(counts)!=set(task['candidate_pairs']):raise ValueError('Incomplete event coverage')
    if output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve()):
        raise ValueError('Independent candidate destination required')
    output.mkdir(exist_ok=False)
    menu_id=digest(menu)
    (output/'catalog.json').write_text(json.dumps(menu,ensure_ascii=False,indent=2)+'\n')
    positive=Counter();slices=Counter()
    with (output/'inputs.jsonl').open('x') as inputs,(output/'targets.jsonl').open('x') as labels:
        for r in rows:
            input_row=dict(id=r['id'],audio=r['audio'],audio_sha256=r['audio_sha256'],catalog_sha256=menu_id)
            t=targets(menu,r['intent'])
            for uid,y in zip(t['candidate_ids'],t['labels'],strict=True):positive[uid]+=y
            a=sum(p[0]==r['scenario'] for p in task['candidate_pairs'].values())
            b=sum(p[1]==r['action'] for p in task['candidate_pairs'].values())
            structural='both_attributes_ambiguous' if a>1 and b>1 else 'at_least_one_attribute_unique'
            slices[structural]+=1
            inputs.write(json.dumps(input_row,separators=(',',':'))+'\n')
            labels.write(json.dumps(dict(id=r['id'],input_sha256=digest(input_row),source_group=r['source_group'],
                candidate_ids=t['candidate_ids'],labels=t['labels'],supervised=t['supervised'],
                structural_slice=structural,truth_basis=menu['supervision']),separators=(',',':'))+'\n')
    report=dict(status='candidate_dataset_only_not_training_or_evaluation_admitted',training_dispatch_allowed=False,
        evaluation_dispatch_allowed=False,release=str(root),release_sha256=RELEASE_SHA,catalog_sha256=menu_id,
        recordings=len(rows),source_groups=len(groups),families=dict(Counter(n['family'] for n in menu['nodes'])),
        positive_counts=dict(positive),structural_slices=dict(slices),
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()},tool_sha256=sha(Path(__file__)),
        limits=['TRAIN only; selection and TEST targets not parsed','native membership not semantic presence',
                'Q-input and Q-credit inputs identical; reward configuration not yet frozen',
                'Direct and Q arms share Final nodes/catalog but new renderer requires new baseline',
                '124 readouts not processor/context/NPU admitted','parallel readouts not temporal steps'])
    (output/'MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=build(a.release,a.output)
    print(json.dumps({k:v for k,v in r.items() if k!='positive_counts'},indent=2))
