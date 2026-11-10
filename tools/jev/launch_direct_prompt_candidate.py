#!/usr/bin/env python3
"""Launch one reviewed prompt with the exact completed parent actor snapshot."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from multimodal_jev.baseline import atomic_json,file_hash


def launch(parent, candidate, review_path, instruction, root):
    review=json.loads(review_path.read_text())
    entry=next(r for r in review['candidates'] if r['id']==candidate)
    prior=json.loads((parent/'launch.json').read_text())
    identity=json.loads((parent/'run/identity.json').read_text())
    if (review['parent_prompt_sha256']!=identity['instruction_sha256']
            or review['selection_or_test_items_read_for_review'] is not False
            or candidate not in review['evaluation_eligible']
            or entry['decision']!='eligible_for_selection_evaluation'
            or entry['instruction_sha256']!=file_hash(instruction)):
        raise ValueError('reviewed prompt identity required')
    cfg=json.loads((parent/'config.json').read_text())
    if cfg!=identity['config'] or cfg['route']!='audio-direct' or cfg['split']!='train':
        raise ValueError('unchanged direct TRAIN parent required')
    cid=prior['container_id']
    state=json.loads(subprocess.check_output(['ssh','npu','docker inspect '+cid],text=True))[0]
    if (state['Id']!=cid or state['Image']!=prior['image'] or state['State']['Status']!='exited'
            or state['State']['ExitCode']!=0):raise ValueError('parent not successfully complete')
    root.mkdir(parents=True,exist_ok=False);(root/'source').mkdir()
    archive=subprocess.Popen(['git','archive',prior['source_commit']],stdout=subprocess.PIPE)
    subprocess.run(['tar','-x','-C',str(root/'source')],stdin=archive.stdout,check=True)
    archive.stdout.close()
    if archive.wait():raise RuntimeError('snapshot export failed')
    (root/'instruction.txt').write_bytes(instruction.read_bytes())
    cfg['instruction']=str(root/'instruction.txt');atomic_json(root/'config.json',cfg)
    plan={k:v for k,v in prior.items() if k not in ('container_id','supervisor_pid')}
    argv=[a.replace(str(parent),str(root)).replace(parent.name,root.name) for a in prior['argv']]
    plan.update(argv=argv,root=str(root),config_sha256=file_hash(root/'config.json'),
                review_sha256=file_hash(review_path),candidate_id=candidate,
                instruction_sha256=file_hash(instruction),parent_container_id=cid)
    atomic_json(root/'launch-intent.json',plan)
    result=subprocess.run(['ssh','-o','ConnectTimeout=8','npu',shlex.join(argv)],capture_output=True,text=True,timeout=90)
    (root/'dispatch.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('dispatch failed; reconcile without retry')
    cid=result.stdout.strip()
    if len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid):raise ValueError('ambiguous dispatch ID')
    plan['container_id']=cid;atomic_json(root/'launch.json',plan)
    (root/'supervisor.py').write_bytes((parent/'supervisor.py').read_bytes())
    with (root/'supervisor.log').open('ab') as log:
        proc=subprocess.Popen([sys.executable,str(root/'supervisor.py')],stdout=log,stderr=log,start_new_session=True)
    plan['supervisor_pid']=proc.pid;atomic_json(root/'launch.json',plan)
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('parent','review','instruction','root'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--candidate',choices=['p1','p2'],required=True)
    a=p.parse_args();r=launch(a.parent,a.candidate,a.review,a.instruction,a.root)
    print(json.dumps({k:r[k] for k in ('container_id','candidate_id','root','source_commit')},indent=2))
