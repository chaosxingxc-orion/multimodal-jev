#!/usr/bin/env python3
"""Bind authored reference judgements to learning sources without training admission."""
import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_hash(value):
    return hashlib.sha256((json.dumps(value,sort_keys=True,ensure_ascii=False)+'\n').encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--reviews',type=Path,required=True);p.add_argument('--definitions',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.release/'MANIFEST.json')!=a.manifest_sha256:raise ValueError('release identity mismatch')
    m=read(a.release/'MANIFEST.json');source=Path(m['source']);sm=read(source/'MANIFEST.json')
    mp=source.parent/'metadata/train.jsonl'
    pins={str(a.release/'MANIFEST.json'):a.manifest_sha256,str(a.release/'learning.jsonl'):m['files']['learning.jsonl'],
        str(source/'learning.targets.jsonl'):sm['files']['learning.targets.jsonl'],str(mp):sm['identity']['input_hashes']['clean_metadata_sha256'],
        str(a.reviews):sha(a.reviews),str(a.definitions):sha(a.definitions),str(a.policy):sha(a.policy)}
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('changed source '+path)
    defs={x['entity_type']:x for x in read(a.definitions)['types']}
    policies={x['entity_type']:x for x in read(a.policy)['policies']}
    if read(a.policy)['source_definition_sha256']!=sha(a.definitions):raise ValueError('stale policy')
    learning=rows(a.release/'learning.jsonl');targets={r['id']:r for r in rows(source/'learning.targets.jsonl')}
    by_source=defaultdict(list)
    for row in learning:
        if row['role']!='learning' or row['official_split']!='train' or row['known_pending_source']:raise ValueError('learning only')
        target=targets[row['id']]
        if content_hash(target)!=row['source_target_sha256']:raise ValueError('target binding mismatch')
        for sid in target['source_ids']:by_source[str(sid)].append(row)
    authored=list(csv.DictReader(a.reviews.open(),delimiter='\t'))
    keys=[(r['source_id'],r['entity_type']) for r in authored]
    if len(set(keys))!=len(keys):raise ValueError('duplicate review key')
    wanted={r['source_id'] for r in authored}
    if not wanted<=by_source.keys():raise ValueError('review outside admitted learning')
    metadata={str(r['slurp_id']):r for r in rows(mp) if str(r['slurp_id']) in wanted}
    decisions=[];bindings=[]
    for row in authored:
        sid,t=row['source_id'],row['entity_type'];status=row['status']
        if status not in ('positive','negative','unavailable','conflict') or t not in defs:raise ValueError('invalid decision')
        value=json.loads(row['proposed_value'])
        if (status=='positive' and value is not True) or (status=='negative' and value is not False) or (status=='unavailable' and value is not None) or (status=='conflict' and type(value) is not bool):raise ValueError('invalid value/status')
        if not row['evidence'].strip() or not row['limitation'].strip():raise ValueError('evidence and scope required')
        native=any(e['type']==t for e in metadata[sid]['entities'])
        if status in ('positive','negative') and native!=value:raise ValueError('disagreement must be conflict')
        if status=='conflict' and native==value:raise ValueError('conflict must identify actual native disagreement')
        d=dict(source_id=sid,entity_type=t,status=status,proposed_semantic_value=value,native_annotated=native,
            evidence=row['evidence'],limitation=row['limitation'],reference=metadata[sid]['sentence'],
            metadata_record_sha256=content_hash(metadata[sid]),definition_sha256=content_hash(defs[t]),policy_sha256=content_hash(policies[t]),
            decision_stage='reference_adjudication_not_audio_admission',optimization_allowed=False)
        decisions.append(d)
        for recording in by_source[sid]:
            target=targets[recording['id']]
            if any(e['type']==t for e in target['native_target']['entities'])!=native:raise ValueError('recording/source annotation disagreement')
            bindings.append(dict(source_id=sid,entity_type=t,recording_id=recording['id'],source_group=recording['source_group'],
                audio_sha256=recording['audio_sha256'],source_target_sha256=recording['source_target_sha256'],
                review_sha256=content_hash(d),all_recording_source_ids=target['source_ids'],
                complete_source_review=all((str(s),t) in keys for s in target['source_ids']),status=status,
                optimization_allowed=False))
    a.output.mkdir(parents=True)
    for name,items in [('decisions',decisions),('recording-bindings',bindings)]:
        (a.output/(name+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in items))
    report=dict(schema='slurp-slot-reference-adjudication-v1',pins=pins,implementation_sha256=sha(__file__),
        source_utterances=len(wanted),reference_decisions=len(decisions),recording_decision_bindings=len(bindings),
        unique_recordings=len({r['recording_id'] for r in bindings}),unique_source_groups=len({r['source_group'] for r in bindings}),
        reference_status_counts=dict(Counter(x['status'] for x in decisions)),binding_status_counts=dict(Counter(x['status'] for x in bindings)),
        files={name:sha(a.output/name) for name in ('decisions.jsonl','recording-bindings.jsonl')},
        scope='Purposefully selected boundary diagnostic, not representative prevalence or training population. Reviewed reference semantics only; no audio listening, source-group propagation, native relabelling, or optimization admission.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
