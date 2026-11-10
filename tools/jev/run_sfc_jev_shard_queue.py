#!/usr/bin/env python3
"""Prepare and execute a complete parent SFC-Jev population, without replay."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import time
import zipfile

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.sfc_jev_shards import verify_plan
from multimodal_jev.sfc_jev_shard_runtime import select_inputs,check_probe,read
from infer_sfc_jev_omni_shard import SOURCE_FILES
from verify_sfc_jev_probe import accept as accept_probe
from verify_sfc_jev_shard import accept as accept_shard
from verify_sfc_jev_context import verify as verify_context
from watch_sfc_native_full import dispatch_once
from merge_sfc_jev_shards import score as merge_score


def code_files(source):
    return {str(p.relative_to(source)):file_hash(p)
            for folder in ('src','tools') for p in sorted((source/folder).rglob('*.py'))}


def committed_snapshot(source,commit):
    """Export committed code only; reject a different executing working tree."""
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','--verify',commit+'^{commit}'],text=True).strip()
    raw=subprocess.check_output(['git','-C',str(source),'archive','--format=zip',revision,'src','tools'])
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        files={}
        for entry in archive.infolist():
            path=Path(entry.filename)
            if (path.is_absolute() or '..' in path.parts or path.parts[0] not in ('src','tools')
                    or (entry.external_attr>>16)&0o170000==0o120000):
                raise ValueError('regular committed source paths required')
            if not entry.is_dir():files[entry.filename]=archive.read(entry)
    pins={name:hashlib.sha256(value).hexdigest() for name,value in files.items() if name.endswith('.py')}
    if not pins or code_files(source)!=pins:
        raise ValueError('executing source differs from requested committed snapshot')
    return revision,files,pins


def check_source(root,intent):
    source=root/'source'
    if (source.is_symlink() or source.resolve()!=Path(intent['source'])
            or code_files(source)!=intent['source_files']):
        raise ValueError('queue source changed')
    if Path(__file__).resolve()!=source/'tools/jev/run_sfc_jev_shard_queue.py':
        raise ValueError('execute the exported queue runner with its source/src on PYTHONPATH')
    import inspect
    for func in (select_inputs,check_probe,verify_plan,accept_probe,accept_shard,verify_context,
                 dispatch_once,merge_score,file_hash,atomic_json):
        path=Path(inspect.getsourcefile(func)).resolve()
        if not path.is_relative_to(source):
            raise ValueError('queue dependencies must load from exported source')


def check_template(cfg):
    path=Path(cfg['shard_plan'])
    if file_hash(path)!=cfg['shard_plan_sha256']:raise ValueError('registered population changed')
    plan=read(path);verify_plan(plan)
    if not plan['shards'] or cfg['shard_id']!=plan['shards'][0]['id']:
        raise ValueError('template must bind first shard of complete population')
    select_inputs(cfg);check_probe(cfg)
    accept_probe(Path(cfg['jev_probe']))
    context=verify_context(Path(cfg['context_audit']))
    if (context['container_id']!=cfg['context_container_id']
            or context['result_sha256']!=cfg['context_result_sha256'] or context['all_inputs_fit'] is not True):
        raise ValueError('exact successful complete CPU audit required')
    return plan


def prepare(root,template,template_hash,scorer,scorer_hash,source_commit):
    """Export a pinned commit locally; no NPU allocation or generation."""
    root=Path(root).resolve();template=Path(template)
    if file_hash(template)!=template_hash or file_hash(scorer)!=scorer_hash:
        raise ValueError('registered template or scorer changed')
    cfg=read(template);plan=check_template(cfg)
    if root.exists():raise FileExistsError('preserve previous queue preparation')
    revision,files,source_files=committed_snapshot(Path(__file__).resolve().parents[2],source_commit)
    versions={k:importlib.metadata.version(k) for k in ('opencc-python-reimplemented','regex')}
    root.mkdir(parents=True,exist_ok=False);source=root/'source';source.mkdir()
    for name,value in files.items():
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value);path.chmod(0o444)
    for path in sorted(source.rglob('*'),reverse=True):
        if path.is_dir():path.chmod(0o555)
    source.chmod(0o555)
    # Workspace documents are not shared with server175. Preserve the exact
    # accepted bytes in the shared queue root before pinning the actor config.
    evidence=root/'context-acceptance.json'
    evidence.write_bytes(Path(cfg['context_acceptance']).read_bytes())
    if file_hash(evidence)!=cfg['context_acceptance_sha256']:
        raise ValueError('context acceptance changed during preparation')
    cfg={**cfg,'context_acceptance':str(evidence)}
    atomic_json(root/'config-template.json',cfg)
    intent=dict(schema='sfc-jev-parent-complete-queue-v1',template_sha256=file_hash(root/'config-template.json'),
        shard_plan=cfg['shard_plan'],shard_plan_sha256=cfg['shard_plan_sha256'],
        shard_ids=[s['id'] for s in plan['shards']],bundles=plan['bundles'],decisions=plan['decisions'],
        split=plan['split'],partition=plan['partition'],physical_devices=[6,7],image=cfg['image'],
        source=str(source),source_commit=revision,source_files=source_files,wall_seconds_per_shard=86400,
        official_scorer=str(Path(scorer).resolve()),official_scorer_sha256=scorer_hash,
        scorer_versions=versions,
        automatic_retry=False,dispatched=False,
        scope='Complete parent population. Preparation does not allocate NPU, call a teacher or train weights.')
    atomic_json(root/'queue-plan.json',intent)
    return intent


def check_pins(root,intent):
    if (intent['schema']!='sfc-jev-parent-complete-queue-v1' or intent['physical_devices']!=[6,7]
            or intent['automatic_retry'] is not False or intent['wall_seconds_per_shard']!=86400
            or file_hash(root/'config-template.json')!=intent['template_sha256']
            or file_hash(intent['official_scorer'])!=intent['official_scorer_sha256']):
        raise ValueError('bounded registered queue identity required')
    check_source(root,intent)
    if {k:importlib.metadata.version(k) for k in intent['scorer_versions']}!=intent['scorer_versions']:
        raise ValueError('official scorer environment changed')
    cfg=read(root/'config-template.json')
    if (cfg['image']!=intent['image'] or cfg['shard_plan']!=intent['shard_plan']
            or cfg['shard_plan_sha256']!=intent['shard_plan_sha256']):
        raise ValueError('queue template differs')
    plan=check_template(cfg)
    if (intent['shard_ids']!=[s['id'] for s in plan['shards']]
            or any(intent[k]!=plan[k] for k in ('bundles','decisions','split','partition'))):
        raise ValueError('entire registered population required')
    return cfg,plan


def prepare_shard(root,intent,template,shard_id):
    if shard_id not in intent['shard_ids']:raise ValueError('registered shard required')
    cfg={**template,'shard_id':shard_id};select_inputs(cfg);check_probe(cfg)
    child=root/shard_id;child.mkdir(exist_ok=False)
    (child/'source').symlink_to(root/'source',target_is_directory=True)
    atomic_json(child/'config.json',cfg)
    name='sfc-jev-'+hashlib.sha256(str(root).encode()).hexdigest()[:12]+'-'+shard_id
    argv=['docker','run','-d','--pull=never','--name',name,'--network=host','--cpus=16','--memory=128g','--shm-size=8g']
    for d in ('davinci6','davinci7','davinci_manager','devmm_svm','hisi_hdc'):argv+=['--device',f'/dev/{d}:/dev/{d}']
    env=dict(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn',HCCL_SOCKET_IFNAME='bond0',
        GLOO_SOCKET_IFNAME='bond0',TP_SOCKET_IFNAME='bond0',HCCL_IF_BASE_PORT='63500',PYTHONPATH=str(root/'source/src'))
    for k,v in env.items():argv+=['-e',k+'='+v]
    # Explicit read-only evidence mounts; the exported source also stays read-only.
    mounts={Path(p) for p in ('/usr/local/Ascend/driver','/usr/local/Ascend/firmware','/usr/local/dcmi',
        '/etc/hccn.conf','/etc/ascend_install.info','/mnt/cephfs/xingchao/models','/mnt/cephfs/xingchao/datasets',
        cfg['jev_probe'],cfg['context_audit'])}
    for key in ('shard_plan','asset_audit'):mounts.add(Path(cfg[key]).parent)
    plan=read(cfg['shard_plan'])
    if plan['partition'] is not None:mounts.add(Path(plan['partition']['path']).parent)
    for path in sorted(mounts):argv+=['-v',f'{path}:{path}:ro']
    argv+=['-v',f'{root}:{root}:rw','-v',f'{root}/source:{root}/source:ro',cfg['image'],
        'timeout','--signal=TERM','--kill-after=30s','86400s','python3','-B',
        str(root/'source/tools/jev/infer_sfc_jev_omni_shard.py'),'--config',str(child/'config.json'),'--output',str(child/'run')]
    prep=dict(argv=argv,image=cfg['image'],physical_devices=[6,7],config_sha256=file_hash(child/'config.json'),
        source_files={n:file_hash(root/'source'/n) for n in SOURCE_FILES},wall_seconds=86400,automatic_retry=False)
    atomic_json(child/'preparation.json',prep)
    return child,prep


def execute(root,expected_plan_hash):
    root=Path(root).resolve()
    if file_hash(root/'queue-plan.json')!=expected_plan_hash:raise ValueError('registered queue plan changed')
    intent=read(root/'queue-plan.json')
    if any((root/n).exists() for n in ('dispatch-intent.json','evaluation-terminal.json','completed-runs.json','merged')):
        raise FileExistsError('previous queue attempt; reconcile without replay')
    if any((root/s).exists() for s in intent['shard_ids']):raise FileExistsError('prior shard preparation exists')
    cfg,plan=check_pins(root,intent)
    with (root/'dispatch-intent.json').open('x') as handle:
        handle.write(json.dumps(dict(queue_plan_sha256=expected_plan_hash,started_unix=time.time(),automatic_retry=False)))
        handle.flush()
        import os
        os.fsync(handle.fileno())
    completed=[]
    for index,shard_id in enumerate(intent['shard_ids']):
        check_pins(root,intent)
        atomic_json(root/'progress.json',dict(phase='preparing',done=index,total=len(intent['shard_ids']),shard_id=shard_id,heartbeat_unix=time.time()))
        child=root/shard_id
        try:
            child,prep=prepare_shard(root,intent,cfg,shard_id)
            atomic_json(root/'progress.json',dict(phase='running',done=index,total=len(intent['shard_ids']),
                shard_id=shard_id,child_root=str(child),heartbeat_unix=time.time()))
            dispatch_once(child,prep)  # Existing one-shot 6/7 occupancy check and exact-container observer.
            receipt=accept_shard(child);atomic_json(child/'acceptance.json',receipt)
        except BaseException as exc:
            if child.exists():
                atomic_json(child/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),
                    scope='Coordinator failure; consult exact Docker state before inferring model termination. No replay.'))
            raise
        atomic_json(child/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time()))
        completed.append(dict(shard_id=shard_id,root=str(child),config_sha256=file_hash(child/'config.json'),
                              launch_sha256=file_hash(child/'launch.json')))
        atomic_json(root/'completed-runs.json',dict(schema='sfc-jev-completed-shard-roster-v1',plan_sha256=cfg['shard_plan_sha256'],runs=completed))
        atomic_json(root/'progress.json',dict(phase='accepted',done=index+1,total=len(intent['shard_ids']),heartbeat_unix=time.time()))
    check_pins(root,intent)
    roster=root/'completed-runs.json'
    merge_score(Path(cfg['shard_plan']),cfg['shard_plan_sha256'],roster,file_hash(roster),
        Path(intent['official_scorer']),intent['official_scorer_sha256'],root/'merged')
    atomic_json(root/'progress.json',dict(phase='complete',done=len(completed),total=len(completed),heartbeat_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--template',type=Path);p.add_argument('--template-hash');p.add_argument('--official-scorer',type=Path)
    p.add_argument('--scorer-hash');p.add_argument('--source-commit');p.add_argument('--queue-plan-hash');a=p.parse_args()
    if a.prepare:
        if not all((a.template,a.template_hash,a.official_scorer,a.scorer_hash,a.source_commit)):
            p.error('all preparation pins including source commit required')
        print(json.dumps(prepare(a.root,a.template,a.template_hash,a.official_scorer,a.scorer_hash,a.source_commit),indent=2))
    else:
        if not a.queue_plan_hash:p.error('queue plan hash required')
        with (a.root/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (a.root/'evaluation-terminal.json').exists():raise FileExistsError('existing terminal')
            try:execute(a.root,a.queue_plan_hash)
            except BaseException as exc:
                atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),automatic_retry=False));raise
            atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time(),automatic_retry=False))
