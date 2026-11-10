#!/usr/bin/env python3
"""One-shot SFC TRAIN generation and official/strict scoring after SLURP jobs."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
import time

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.consumption import canonical
from evaluate_sfc import evaluate
from infer_sfc_omni_probe import select_inputs
from watch_omni_transfer_readiness import observe_dependency
from watch_qwen3_omni_probe import inspect,wait
from verify_omni_transfer_train import verify as verify_transfer


def accept_and_score(root,plan):
    run=root/'run';cfg=json.loads((root/'config.json').read_text())
    rows,preflight=select_inputs(cfg)
    result=json.loads((run/'result.json').read_text());identity=json.loads((run/'identity.json').read_text())
    launch=json.loads((root/'launch.json').read_text());item=inspect(launch['container_id'])
    if (item['Id']!=launch['container_id'] or item['Image']!=plan['image']
            or item['State']['Status']!='exited' or item['State']['ExitCode']!=0
            or json.loads((run/'terminal.json').read_text())['exit_code']!=0
            or result['status']!='passed' or result['weight_updates']!=0
            or result['model_files_unchanged'] is not True):
        raise ValueError('exact successful frozen SFC container and application required')
    audit=json.loads(Path(cfg['asset_audit']).read_text())
    if (identity['config']!=cfg or identity['versions']!=cfg['versions']
            or identity['model_files']!=audit['files']
            or identity['inputs_sha256']!=hashlib.sha256(canonical(rows)).hexdigest()
            or file_hash(run/'predictions.jsonl')!=result['predictions_sha256']):
        raise ValueError('SFC identity or predictions changed')
    for name,digest in identity['source_files'].items():
        if file_hash(root/'source'/name)!=digest or plan['inference_files'].get(name)!=digest:
            raise ValueError('inference snapshot changed')
    ids=[r['example_id'] for r in rows]
    predictions=[json.loads(line) for line in (run/'predictions.jsonl').open()]
    if (json.loads((run/'selection.json').read_text())['ids']!=ids
            or len(predictions)!=6 or [p['example_id'] for p in predictions]!=ids
            or result['requests']!=6 or result['predictions']!=6):
        raise ValueError('all six original TRAIN turns required')
    for row,p in zip(rows,predictions):
        if (p['split']!='train_id' or p['source_group']!=row['metadata']['source_group']
                or p['audio_count']!=len(row['input']['audios'])
                or p['audio_sha256']!=row['metadata']['audio_sha256']
                or p['input_sha256']!=hashlib.sha256(canonical(row['input'])).hexdigest()
                or p['hf_vllm_prompt_ids_equal'] is not True
                or type(p['input_tokens']) is not int or not 0<p['input_tokens']<=3072
                or len(p['token_ids'])>1024 or type(p['raw_text']) is not str
                or p['finish_reason'] not in ('stop','length')
                or type(p['length_limited']) is not bool or p['length_limited']!=(p['finish_reason']=='length')):
            raise ValueError('SFC input binding, token parity or budget failed')
    if result['length_limited']!=sum(p['length_limited'] for p in predictions):
        raise ValueError('SFC truncation count differs')
    # Labels are read only after complete inference acceptance, for these six turns.
    source=Path(preflight['release'])/'train_id.jsonl'
    targets=[json.loads(line) for line in source.open() if json.loads(line)['example_id'] in set(ids)]
    if [t['example_id'] for t in targets]!=ids:raise ValueError('paired TRAIN scoring subset required')
    manifest=root/'score-input.jsonl'
    with manifest.open('xb') as f:f.write(b''.join(map(canonical,targets)))
    score=evaluate(manifest,run/'predictions.jsonl',Path(plan['official_scorer']),root/'score',plan['official_scorer_sha256'])
    atomic_json(root/'acceptance.json',dict(container_id=item['Id'],requests=6,source_groups=4,weight_updates=0,
        score_sha256=file_hash(root/'score/metrics.json'),result_sha256=file_hash(run/'result.json'),
        strict_metrics=score['strict_metrics'],strict_conversation_metrics=score['strict_conversation_metrics'],
        scope='Six TRAIN turns, no quality threshold. Offline supplied history; not full SFC TEST or causal multi-audio efficacy.'))


def check_pins(root,plan):
    for name,digest in plan['files'].items():
        if file_hash(root/name)!=digest:raise ValueError('prepared SFC input changed: '+name)
    for name,digest in plan['inference_files'].items():
        if file_hash(root/'source'/name)!=digest:raise ValueError('prepared inference source changed')
    code=Path(plan['coordinator_source'])
    for name,digest in plan['coordinator_files'].items():
        if file_hash(code/name)!=digest:raise ValueError('coordinator source changed')
    if file_hash(plan['official_scorer'])!=plan['official_scorer_sha256']:
        raise ValueError('official SFC scorer changed')
    if {k:importlib.metadata.version(k) for k in plan['scorer_versions']}!=plan['scorer_versions']:
        raise ValueError('SFC scorer dependencies changed')


def dispatch_once(root,plan):
    if any((root/p).exists() for p in ('dispatch-intent.json','dispatch.log','launch.json','run')):
        raise FileExistsError('prior SFC dispatch; no replay')
    prep=json.loads((root/'preparation.json').read_text())
    if prep['physical_devices']!=[6,7] or prep['image']!=plan['image']:
        raise ValueError('registered image and NPU6/7 required')
    argv=prep['argv'];devices={argv[i+1].split(':')[0] for i,a in enumerate(argv[:-1]) if a=='--device'}
    if {d for d in devices if d.removeprefix('/dev/davinci').isdigit()}!={'/dev/davinci6','/dev/davinci7'}:
        raise ValueError('device mapping differs')
    for cid in subprocess.check_output(['ssh','npu','docker ps -q'],text=True,timeout=20).split():
        item=inspect(cid)
        if any(d['PathOnHost'] in ('/dev/davinci6','/dev/davinci7') for d in item['HostConfig'].get('Devices') or []):
            raise RuntimeError('reserved device is still occupied; no dispatch')
    with (root/'dispatch-intent.json').open('x') as f:
        json.dump(dict(started_unix=time.time(),automatic_retry=False),f)
    result=subprocess.run(['ssh','npu',shlex.join(argv)],capture_output=True,text=True,timeout=90)
    (root/'dispatch.log').write_text(result.stdout+result.stderr);cid=result.stdout.strip()
    if result.returncode or len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid):
        raise RuntimeError('SFC dispatch ambiguous; no automatic retry')
    atomic_json(root/'launch.json',dict(container_id=cid,image=plan['image'],source_commit=prep['source_commit'],physical_devices=[6,7]))
    state=wait(root,cid,plan['image'],time.monotonic()+3900)
    atomic_json(root/'docker-terminal.json',dict(container_id=cid,state=state))
    logs=subprocess.run(['ssh','npu','docker logs '+cid],capture_output=True,text=True,timeout=30)
    (root/'container.log').write_text(logs.stdout+logs.stderr)
    if state['Status']!='exited' or state['ExitCode']!=0:raise RuntimeError('SFC container failed')


def execute(root):
    if (root/'evaluation-terminal.json').exists() or (root/'dispatch-intent.json').exists():
        raise FileExistsError('preserve prior SFC attempt')
    plan=json.loads((root/'queue-plan.json').read_text());check_pins(root,plan)
    if set(plan['dependencies'])!={'native_test','transfer_train'}:raise ValueError('both registered predecessors required')
    deadline=time.monotonic()+95000
    while True:
        try:ready={k:observe_dependency(v) for k,v in plan['dependencies'].items()}
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,json.JSONDecodeError) as exc:
            atomic_json(root/'queue-progress.json',dict(phase='observation_retry',error_type=type(exc).__name__,heartbeat_unix=time.time()))
        else:
            atomic_json(root/'queue-progress.json',dict(phase='waiting_dependencies',ready=ready,heartbeat_unix=time.time()))
            if all(ready.values()):break
        if time.monotonic()>deadline:raise TimeoutError('observation expired; no restart')
        time.sleep(20)
    check_pins(root,plan)
    verify_transfer(Path(plan['dependencies']['transfer_train']['root']))
    select_inputs(json.loads((root/'config.json').read_text()))
    dispatch_once(root,plan)
    accept_and_score(root,plan)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    with (a.root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.root/'evaluation-terminal.json').exists() or (a.root/'dispatch-intent.json').exists():
            raise FileExistsError('existing SFC attempt')
        try:execute(a.root)
        except BaseException as exc:
            atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),automatic_retry=False));raise
        atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time()))
