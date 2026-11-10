#!/usr/bin/env python3
"""Project all pinned release targets separately from actor inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path

from multimodal_jev.audio2tool_inventory import registry
from multimodal_jev.audio2tool_input import REGISTRY_SHA256
from multimodal_jev.audio2tool_score import load_official
from multimodal_jev.audio2tool_targets import project
from multimodal_jev.baseline import atomic_json,file_hash


def run(args):
    for k in ('audit','inputs'):
        if file_hash(getattr(args,k))!=getattr(args,k+'_hash'):raise ValueError('registered evidence changed')
    audit=json.loads(args.audit.read_text());official=load_official(args.official_metrics)
    names={t['tool_name'] for t in registry(args.registry.read_bytes(),REGISTRY_SHA256)}
    inputs=[json.loads(l) for l in args.inputs.open()];index={r['id']:r for r in inputs}
    if len(index)!=30733 or len(inputs)!=30733:raise ValueError('full input roster required')
    targets=[];stats={}
    if len(audit['source_sha256'])!=8:raise ValueError('all pinned release files required')
    for relative,pin in sorted(audit['source_sha256'].items()):
        p=args.root/relative
        if not p.resolve().is_relative_to(args.root.resolve()) or file_hash(p)!=pin:
            raise ValueError('release target source changed')
        tier=int(Path(relative).parts[1][4]);rows=json.loads(p.read_text());counts=Counter()
        for row in rows:
            target=project(row,tier,official,names);inp=index[target['id']]
            if inp['source_query_idx']!=target['source_query_idx'] or inp['tier']!=tier:
                raise ValueError('target/input release identity differs')
            source_paths=(['public/'+t['path'] for t in row['chat_history']] if tier==7 else ['public/'+row['path']])
            if source_paths!=[a['path'] for a in inp['input']['audios']+inp['provenance']['post_decision_audios']]:
                raise ValueError('target/input audio mapping differs')
            targets.append(target);counts['records']+=1
            counts[target['full_call_argument_coverage']]+=1
            counts['primary_consistency:'+str(target['primary_encoding_consistent'])]+=1
            for a in target['additional_annotations']:counts['extra:'+a['argument_evidence']]+=1
        stats[str(tier)]=dict(counts)
    if len(targets)!=30733 or {t['id'] for t in targets}!=set(index):raise ValueError('full target coverage required')
    targets.sort(key=lambda r:(r['tier'],r['release_id']))
    with (args.output/'targets.jsonl').open('x') as f:
        for target in targets:f.write(json.dumps(target)+'\n')
    (args.output/'targets.jsonl').chmod(0o600)
    atomic_json(args.output/'result.json',dict(status='projected',records=len(targets),tiers=stats,
        target_sha256=file_hash(args.output/'targets.jsonl'),input_sha256=args.inputs_hash,
        source_audit_sha256=args.audit_hash,official_metrics_sha256=file_hash(args.official_metrics),
        registry_sha256=REGISTRY_SHA256,runner_sha256=file_hash(__file__),
        actor_inputs_modified=False,model_loaded=False,
        scope='Full scorer-only targets with structured primary and raw official extra-tool labels; incomplete extra arguments explicit, no fabricated empty dicts. Not complete strict multi-call gold or model scores.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','registry','audit','inputs','official-metrics','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('audit','inputs'):p.add_argument('--'+k+'-hash',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False,mode=0o700)
    try:run(a)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0))
