#!/usr/bin/env python3
"""Collect exact successful formal audit bytes and independently rehash evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

repository=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repository/'src'))
from multimodal_jev.resident_service_admission import IMAGE


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit-tag',required=True)
    p.add_argument('--state-dir',type=Path,default=repository/'.local/simple-final-v1')
    a=p.parse_args()
    if not re.fullmatch(r'stage2-[a-z0-9_-]+',a.audit_tag):p.error('explicit stage2 audit tag required')
    local=a.state_dir;tag=a.audit_tag
    launch=json.loads((local/(tag+'-launch.json')).read_text())
    observed=json.loads((local/(tag+'-observed.json')).read_text())
    if (launch['image']!=IMAGE or observed['image']!=IMAGE or observed['container_id']!=launch['container_id']
            or observed['state']['Status']!='exited' or observed['state']['ExitCode']!=0):
        raise ValueError('successful exact fixed-image audit observation required')
    script='''import hashlib,json
from pathlib import Path
r=Path(ROOT)
assert json.loads((r/'terminal.json').read_text())['exit_code']==0
assert json.loads((r/'run/terminal.json').read_text())['exit_code']==0
p=r/'run/state-verified.json';raw=p.read_text();v=json.loads(raw)
for path,digest in v['files'].items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
print(json.dumps(dict(raw=raw,verified_files=len(v['files']))))
'''.replace('ROOT',repr(launch['remote_root']))
    wire=json.loads(subprocess.check_output(['ssh','npu',shlex.join(['python3','-c',script])],text=True,timeout=60))
    report=json.loads(wire['raw']);updates=report['updates'];prefix=report.get('audit_mode')=='stopped_prefix'
    if (report.get('passed') is not True or report.get('stage')!='simple_stage2_rl'
            or report.get('audit_mode') not in ('stopped_prefix','terminal')
            or report.get('ranks')!=4 or updates!=launch['completed_updates']
            or type(updates) is not int or not 1<=updates<=634
            or report.get('complete_training') is not (not prefix)
            or (updates>=634 if prefix else updates!=634)
            or report.get('new_updates')!=updates or len(report.get('checks',[]))!=updates
            or set(report.get('policies',{}))!={str(i) for i in range(updates+1)}
            or any(report.get(k) is not True for k in ('exact_master_export','exact_peft_reload','exact_pre_update_recovery'))):
        raise ValueError('complete formal audit scope required')
    from multimodal_jev.simple_resident_selection import SELECTION_UPDATES
    if ([r.get('updates') for r in report.get('selection',[])]!=[s for s in SELECTION_UPDATES if s<=updates]
            or any(r.get('passed') is not True for r in report['selection'])
            or len(report.get('initial_state_checks',[]))!=4
            or any(c.get('passed') is not True for c in report['initial_state_checks'])):
        raise ValueError('all due selection points required')
    for i,check in enumerate(report['checks'],1):
        if (check.get('passed') is not True or check.get('requests')!=128 or check.get('updates')!=i
                or check.get('policy_id')!=report['policies'][str(i)]):
            raise ValueError('complete ordered update checks required')
    (local/(tag+'-verified.json')).write_text(wire['raw'])
    (local/(tag+'-file-check.json')).write_text(json.dumps(dict(passed=True,
        container_id=launch['container_id'],verified_files=wire['verified_files'],
        state_sha256=hashlib.sha256(wire['raw'].encode()).hexdigest()),indent=2)+'\n')
    print(json.dumps(dict(passed=True,updates=updates,audit_mode=report['audit_mode'],
        selection_points=len(report['selection']),verified_files=wire['verified_files'])))


if __name__=='__main__':main()
