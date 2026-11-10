#!/usr/bin/env python3
"""Full external-test Jev release with separate, partial supervision."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile
import time

from multimodal_jev.audio2tool_acceptance import TARGETS_SHA
from multimodal_jev.audio2tool_inference import INPUT_SHA,digest
from multimodal_jev.audio2tool_input import REGISTRY_SHA256
from multimodal_jev.audio2tool_jev import task_bank,request,target,verify_gold
from multimodal_jev.audio2tool_score import load_official
from multimodal_jev.baseline import atomic_json,file_hash


def export(args):
    start=time.monotonic()
    if file_hash(args.inputs)!=INPUT_SHA or file_hash(args.targets)!=TARGETS_SHA or file_hash(args.registry)!=REGISTRY_SHA256:
        raise ValueError('registered complete native inputs, projected targets and registry required')
    official=load_official(args.official_metrics);bank=task_bank(args.registry.read_bytes())
    repo=Path(__file__).resolve().parents[2]
    sources=['tools/jev/prepare_audio2tool_jev.py','src/multimodal_jev/audio2tool_jev.py',
        'src/multimodal_jev/audio2tool_input.py','src/multimodal_jev/audio2tool_inventory.py',
        'src/multimodal_jev/audio2tool_score.py','src/multimodal_jev/audio2tool_targets.py',
        'src/multimodal_jev/audio2tool_inference.py']
    identity=dict(dataset='audio2tool-jev',split='external_test',inputs_sha256=INPUT_SHA,targets_sha256=TARGETS_SHA,
        registry_sha256=REGISTRY_SHA256,task_bank_sha256=digest(bank),official_metrics_sha256=file_hash(args.official_metrics),
        source_files={n:file_hash(repo/n) for n in sources})
    release=digest(identity);args.destination.mkdir(parents=True,exist_ok=True);final=args.destination/release
    if final.exists():raise FileExistsError('preserve prior immutable release')
    stage=Path(tempfile.mkdtemp(prefix='.audio2tool-jev-',dir=args.destination))
    # Retain failure artifacts explicitly; do not silently retry or delete them.
    atomic_json(stage/'identity.json',identity);atomic_json(stage/'task-bank.json',bank)
    counts=Counter();missing=Counter();available=Counter();ids=set();complete=0
    try:
        with args.inputs.open() as inputs,args.targets.open() as targets, \
                (stage/'external_test.requests.jsonl').open('x') as requests, \
                (stage/'external_test.targets.jsonl').open('x') as golds:
            os.chmod(stage/'external_test.targets.jsonl',0o600)
            for index,(line,tline) in enumerate(zip(inputs,targets,strict=True)):
                native,native_target=json.loads(line),json.loads(tline)
                row=request(native,bank);gold=target(row,native_target,bank,official)
                answers=verify_gold(row,gold,bank,official)
                if row['id'] in ids:raise ValueError('duplicate release ID')
                ids.add(row['id']);counts[row['tier']]+=1;complete+=not gold['unavailable_decisions']
                for d in bank['decisions']:
                    if answers[d['id']]['status']=='available':available[d['decision_type']]+=1
                    else:missing[d['decision_type']]+=1
                requests.write(json.dumps(row,ensure_ascii=False)+'\n');golds.write(json.dumps(gold,ensure_ascii=False)+'\n')
                if index%1000==0:
                    requests.flush();golds.flush()
                    atomic_json(stage/'progress.json',dict(done=index+1,heartbeat_unix=time.time()))
        if (len(ids)!=30733 or complete!=25969 or counts!={1:4292,2:6320,3:4292,4:4278,5:4292,6:4292,7:2146,8:821}
                or len(bank['decisions'])!=457):
            raise ValueError('full population and annotation coverage changed')
        manifest=dict(release_id=release,identity=identity,records=len(ids),counts_by_tier=dict(counts),
            decisions_per_record=len(bank['decisions']),decisions=len(ids)*len(bank['decisions']),
            complete_argument_records=complete,incomplete_argument_records=len(ids)-complete,
            available_decisions_by_type=dict(available),unavailable_decisions_by_type=dict(missing),
            requests_sha256=file_hash(stage/'external_test.requests.jsonl'),targets_sha256=file_hash(stage/'external_test.targets.jsonl'),
            task_bank_file_sha256=file_hash(stage/'task-bank.json'),elapsed_seconds=time.monotonic()-start,
            full_calls_reconstruction='Exact ordered typed reconstruction for all 25969 complete records; partial records preserve original evidence and explicit unavailable call-list supervision.',
            scope='Full external-test data conversion only. No task/prompt selection from model results, inference or training. Typed extensions are project protocols; sequence annotation order is not verified execution chronology.')
        atomic_json(stage/'MANIFEST.json',manifest);atomic_json(stage/'terminal.json',dict(exit_code=0))
        os.rename(stage,final)
        return final,manifest
    except BaseException as exc:
        atomic_json(stage/'terminal.json',dict(exit_code=1,error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('inputs','targets','registry','official-metrics','destination'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args();path,manifest=export(args)
    print(json.dumps(dict(release=str(path),records=manifest['records'],decisions=manifest['decisions'])))
