#!/usr/bin/env python3
"""Audit actual published controls independently of the rule rendering builder."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_training import load_reviewed_bundles


def verify(root):
    manifest=json.loads((root/'MANIFEST.json').read_text())
    if manifest['schema']!='jev-span-rule-controls-v1' or manifest['training_admitted'] or manifest['evaluation_admitted']:
        raise ValueError('development-only release required')
    for name,digest in manifest['files'].items():
        if file_hash(root/name)!=digest:raise ValueError('published data changed')
    for name,digest in manifest['source_sha256'].items():
        if file_hash(name)!=digest:raise ValueError('builder source changed')
    bundles,_=load_reviewed_bundles(manifest['parent_release'],manifest['parent_manifest_sha256'],['date','time','timeofday'])
    by_id={b['id']:b for b in bundles}
    rows=[json.loads(s) for s in (root/'learning.jsonl').read_text().splitlines()]
    groups=defaultdict(dict);counts=Counter();used=set()
    for row in rows:
        p=row['provenance'];b=by_id[p['recording_id']]
        if row['split']!='learning' or row['request']['audio']!=b['request'].audio:
            raise ValueError('source or split differs')
        if p['source_group']!=b['source_group'] or p['audio_sha256']!=b['audio_sha256'] or p['decision_ids']!=b['decision_ids']:
            raise ValueError('source provenance differs')
        keys=[q.id for q in b['request'].questions];positions=[keys.index(k) for k in p['dependency_keys']]
        if len(positions)!=2 or positions[0]>=positions[1] or not all(b['supervised'][i] for i in positions):
            raise ValueError('unreviewed or malformed dependencies')
        a,c=[b['targets'][i] for i in positions]
        if p['dependency_targets']!=[a,c]:raise ValueError('dependency labels differ')
        first,second=p['dependency_keys'];op=p['rule']['op']
        A=dict(op='fact',key=first);B=dict(op='fact',key=second)
        if op=='all':expected=a and c;rule=dict(op='all',args=[A,B])
        elif op=='any':expected=a or c;rule=dict(op='any',args=[A,B])
        elif op=='except':expected=a and not c;rule=dict(op='except',base=A,exception=B)
        else:raise ValueError('unregistered operation')
        if rule!=p['rule']:raise ValueError('rule dependencies differ')
        arm=row['arm'];target={'final_only':[expected],'structured':[a,c,expected],'reordered':[c,a,expected]}[arm]
        if row['targets']!=target or row['supervised']!=[True]*len(target):raise ValueError('derived labels differ')
        request=row['request']
        if set(request)!={'audio','questions','final_id','requirements','version'}:
            raise ValueError('unexpected metadata in input')
        if request['final_id']!='final' or len(request['questions'])!=len(target):raise ValueError('input structure differs')
        pair=(p['recording_id'],p['rule_id'])
        if arm in groups[pair]:raise ValueError('duplicate arm')
        groups[pair][arm]=row
        counts[(p['rule_id'],expected)]+=int(arm=='final_only');used.add(p['recording_id'])
    for arms in groups.values():
        if set(arms)!={'final_only','structured','reordered'}:raise ValueError('missing matched arm')
        direct=arms['final_only']['request'];structured=arms['structured']['request'];reordered=arms['reordered']['request']
        if direct['questions'][-1]!=structured['questions'][-1] or direct['questions'][-1]!=reordered['questions'][-1]:
            raise ValueError('final question changed across arms')
        if direct['requirements']!=structured['requirements'] or direct['requirements']!=reordered['requirements']:
            raise ValueError('thinking requirements changed across arms')
        if [q['schema'] for q in structured['questions'][:2]]!=[q['schema'] for q in reversed(reordered['questions'][:2])]:
            raise ValueError('reordered facts differ')
    expected=sum(3*sum(all(b['supervised'][i] for i in pair) for pair in ((0,1),(0,2),(1,2))) for b in bundles)
    if len(groups)!=expected or len(rows)!=expected*3 or len(rows)!=manifest['records']:
        raise ValueError('incomplete reviewed population')
    distributions=[]
    for rule in sorted({r for r,v in counts}):
        nfalse,ntrue=counts[(rule,False)],counts[(rule,True)]
        distributions.append(dict(rule_id=rule,false=nfalse,true=ntrue,majority_accuracy=max(nfalse,ntrue)/(nfalse+ntrue)))
    return dict(root=str(root),passed=True,manifest_sha256=file_hash(root/'MANIFEST.json'),
        recordings=len(used),derived_rows=len(rows),matched_triplets=len(groups),
        distributions=distributions,training_admitted=False,evaluation_admitted=False,
        scope='Independent truth/provenance/arm audit of learning-only controls; no model score or generalization evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=verify(args.root);atomic_json(args.output,result);print(json.dumps(result,indent=2))
