#!/usr/bin/env python3
"""Audit completed frozen TEST comparison; never consumes partial results."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from multimodal_jev.baseline import file_hash
from multimodal_jev.fsc_independent_audit import score, transitions, require_same_transitions
from compare_omni_test import load, compare


def run(root):
    read = lambda p: json.loads(p.read_text())
    if read(root/'evaluation-terminal.json')['exit_code'] != 0:
        raise ValueError('completed successful full TEST queue required')
    protocol = root/'test-protocol.json'
    registered = read(protocol)
    recorded = read(root/'test-comparison.json')
    recomputed = compare(root)
    if recorded != recomputed:
        raise ValueError('frozen queue comparison does not reproduce')
    outcomes = {}; reports = []; inspections = {}
    for key in registered['prompts']:
        child = root/key
        identity,requests,indexed,result = load(child/'run',protocol,key)
        launch=read(child/'launch.json'); cid=launch['container_id']
        if not re.fullmatch('[0-9a-f]{64}',cid):raise ValueError('exact container ID required')
        inspection=json.loads(subprocess.check_output(['ssh','npu','docker inspect '+cid],text=True))[0]
        if (inspection['Id']!=cid or inspection['Image']!=registered['image']
                or inspection['State']['Status']!='exited' or inspection['State']['ExitCode']!=0
                or inspection['State']['OOMKilled']):
            raise ValueError('actual successful original container required')
        argv=read(child/'intent.json')['argv']
        if inspection['Config']['Cmd']!=argv[argv.index(registered['image'])+1:]:
            raise ValueError('actual command differs from original intent')
        target=Path(identity['config']['requests']).with_name('test.targets.jsonl')
        if file_hash(target)!=registered['targets_sha256']:raise ValueError('targets changed')
        targets=[json.loads(line) for line in target.open()]
        audit=score(requests,targets,list(indexed.values()))
        entry=next(r for r in recorded['comparisons'] if r['id']==key)
        if audit['source_groups']!=3793 or audit['metrics']!=entry['metrics']:
            raise ValueError('independent raw scoring differs')
        if read(child/'score.json')['replicates']!=[audit['metrics']]:
            raise ValueError('child score differs')
        outcomes[key]=audit['outcomes'];inspections[key]=inspection
        reports.append(dict(id=key,container_id=cid,predictions_sha256=result['predictions_sha256'],
                            metrics=audit['metrics']))
    for report in reports:
        paired=transitions(outcomes['parent'],outcomes[report['id']])
        entry=next(r for r in recorded['comparisons'] if r['id']==report['id'])
        require_same_transitions(paired,entry['paired_to_parent'])
        report['paired_to_parent']=paired
    return dict(schema='fsc-independent-test-comparison-audit-v1',
        protocol_sha256=file_hash(protocol),comparison_sha256=file_hash(root/'test-comparison.json'),
        reports=reports,actual_inspections=inspections,source_groups=3793,requests_per_prompt=22758,
        scope='Completed queue admission, actual container state and independent raw JSON/count/pair audit. No inference, output repair, new metric or TEST-driven prompt change. Frozen source snapshot authentication remains a separate check.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=run(args.root)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
