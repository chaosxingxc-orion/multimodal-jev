"""Compare the exposed Stage1 wait on all ranks for an equal number of real anchors."""
from pathlib import Path
import argparse
import json
import math
import shlex
import subprocess

from tools.reports.watch_stage1_first_updates import REMOTE


def window(handle, steps, world):
    run=Path(handle['remote_root'])/'formal'
    args=['python3','-c',REMOTE.replace('CONFIG',repr(dict(root=str(run),world=world,updates=steps,timeout=30)),1)]
    observed=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes',handle['host'],shlex.join(args)]))
    records=observed['records']
    if len(records)!=world*steps:
        raise ValueError('incomplete rank window')
    result=[]
    for i in range(steps):
        ranks=records[i*world:(i+1)*world]
        sizes={r['real_queries'] for r in ranks}
        if (len(sizes)!=1 or min(sizes)<1 or any(r['update']!=i+1 for r in ranks)
                or any(not math.isfinite(r[k]) or not 0<=r[k]<=r['total_seconds']
                       for r in ranks for k in ('preparation_seconds','total_seconds'))):
            raise ValueError('invalid distributed timing evidence')
        result.append(dict(step=i+1,anchors=sizes.pop(),max_wait=max(r['preparation_seconds'] for r in ranks),
                           max_step=max(r['total_seconds'] for r in ranks)))
    return dict(container_id=handle['container_id'],steps=result,anchors=sum(r['anchors'] for r in result),
                wait_seconds=sum(r['max_wait'] for r in result),step_seconds=sum(r['max_step'] for r in result))


def compare(reference, candidate, width, count, reference_count, receipt):
    proof=json.loads(receipt.read_bytes())
    if (proof['container_id']!=candidate['container_id'] or len(proof['updates'])!=count
            or any(r['ranks']!=width for r in proof['updates'])):
        raise ValueError('candidate formal startup receipt differs')
    if reference['host']!=candidate['host'] or reference['container_id']==candidate['container_id']:
        raise ValueError('distinct runs on the same host required')
    baseline=window(reference,reference_count,width)
    current=window(candidate,count,width)
    if (baseline['anchors']!=current['anchors'] or
          any(abs(x['max_wait']-y['max_prepare_seconds'])>1e-8 or x['anchors']!=y['real_queries']
              for x,y in zip(current['steps'],proof['updates']))):
        raise ValueError('exposure or collected timing differs')
    return dict(reference=baseline,candidate=current,passed=current['wait_seconds']<=baseline['wait_seconds'])


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('reference',type=Path)
    cli.add_argument('candidate',type=Path)
    cli.add_argument('receipt',type=Path)
    cli.add_argument('--ranks',type=int,required=True)
    cli.add_argument('--steps',type=int,required=True)
    cli.add_argument('--reference-steps',type=int,required=True)
    c=cli.parse_args()
    result=compare(*(json.loads(p.read_bytes()) for p in (c.reference,c.candidate)),c.ranks,c.steps,c.reference_steps,c.receipt)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
