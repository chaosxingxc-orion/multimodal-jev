"""Seal a card Stage1 resume from the audited epoch checkpoint and score table."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.card_contrast import TRIPLET_KINDS
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.source_snapshot import verify
from tools.reports.collect_card_first_updates import summarize
from tools.reports.context_macro_probe_operations import inspect_runtime
from tools.reports.native_seal_operations import run, verify_memory
from tools.reports.stop_card_stage1_at_epoch import endpoint, inspect, read, remote
from tools.reports.watch_stage1_first_updates import REMOTE


def prepare(c):
    parent = Path(c['parent'])
    upstream = parent/'package/entry'
    old_ops = read(parent/'package/operations-local.json')['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    launch = read(parent/'ops/formal-launch.json')
    resume = read(Path(c['handoff']))
    start = resume['updates']
    checkpoint_digest = resume['checkpoint_sha256']
    old_root = Path(launch['remote_root'])
    deployment = dict(old_ops, **c['deployment'])
    host = deployment['host']
    online = c.get('sampler') == 'class-logit-ema-v1'
    if online:
        sampler = resume['sampler_state']
        if sampler['kind'] != c['sampler'] or sampler['updates'] != start or 'scoring' in c:
            raise ValueError('online sampler handoff differs')
        evidence = dict(sampler_state=sampler)
    else:
        merged = Path(c['scoring'])
        audit = read(merged/'audit.json')
        confusion_path = merged/'confusion-epoch-1.json'
        confusion = read(confusion_path)
        if (audit['passed'] is not True or audit['container_exited'] is not True
                or audit['updates'] != start
                or not audit['checkpoint_sha256'] == checkpoint_digest == confusion['checkpoint_sha256']
                or file_hash(confusion_path) != audit['confusion_sha256']
                or sum(confusion['counts']) != audit['recordings']
                or audit['pairs'] != audit['recordings']*len(confusion['labels'])
                or confusion['pairs'] != audit['pairs'] or confusion['policy_id'] != resume['policy_id']
                or set(confusion['labels']) != confusion['distribution'].keys()
                or any(count <= 0 for count in confusion['counts'])):
            raise ValueError('merged confusion and epoch checkpoint must agree')
        evidence = dict(confusion_sha256=audit['confusion_sha256'])
    if (resume['passed'] is not True or resume['container_exited'] is not True
            or resume['container_id'] != launch['container_id']
            or file_hash(parent/'ops/formal-launch.json') != resume['launch_sha256']
            or not host == resume['host'] == launch['host']
            or deployment['image'] != old_ops['image']
            or inspect(launch)['State']['Status'] != 'exited'):
        raise ValueError('checkpoint, merged scoring and parent deployment must agree')
    raw = remote(host,'cat',str(old_root/'formal/config.json')).stdout
    train = json.loads(raw)
    devices = train['physical_devices']
    if (sha256(raw).hexdigest() != resume['config_sha256']
            or devices != resume['devices'] or devices != deployment['idle_devices']
            or train['instruction_objective']['kind'] not in TRIPLET_KINDS
            or not 0 < start < train['reference_endpoint']
            or start % train['checkpoint_interval']
            or train['reference_endpoint'] != train['checkpoint_interval'] * train['batch_config']['epochs']
            or train['stop_update'] not in (start, train['reference_endpoint'])
            or online != ('class_logit_ema' in train)
            or (online and train['class_logit_ema'] != sampler['config'])
            or (not online and train['batch_config']['source_queries'] != audit['recordings'])):
        raise ValueError('merged source population or frozen training contract differs')
    deployment['read_only_mounts'] = [*old_ops['read_only_mounts'],
                                     f'{old_root}/formal/run:/jev-resume',
                                     f'{old_root}/prepare:/jev-resume-prepared']
    old_window = read(upstream/'formal.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    verify(upstream/'source', manifest_sha256=old_window['source']['sha256'],
                          io_workers=old_window['io_workers'])
    parent_proof = endpoint(dict(updates=start, training=dict(sha256=resume['config_sha256']),
                                 hash_timeout_seconds=c['transfer_timeout_seconds'],
                                 completed_window=train['stop_update'] == start),train,launch)
    if any(resume[k] != v for k,v in parent_proof.items()):
        raise ValueError('live parent checkpoint differs from the handoff')

    root = Path(c['output'])
    entry = root/'entry'
    entry.mkdir(parents=True)
    passthrough = [name for name in old_ops['entry_pins']
                   if name not in ('formal.json', 'finite.json', 'prepare.json', 'source/FILES.json')]
    for name in passthrough:
        if file_hash(upstream/name) != old_ops['entry_pins'][name]:
            raise ValueError(f'parent package input differs: {name}')
        shutil.copyfile(upstream/name,entry/name)
    (entry/'parent-training.json').write_bytes(raw)
    if not online:
        shutil.copyfile(confusion_path,entry/'confusion.json')
    # Epoch resumes preserve the exact runtime that produced the optimizer checkpoint.
    # Pre-resume archival packages need their original, versioned migration tool.
    if not (upstream/'source/tools/reports/card_stage1_window.py').is_file() or (
            not online and 'def run_resume(' not in (upstream/'source/tools/reports/card_stage1_window.py').read_text()):
        raise ValueError('legacy package requires its original epoch-resume migration release')
    shutil.copytree(upstream/'source',entry/'source')
    verify(entry/'source', manifest_sha256=old_window['source']['sha256'], io_workers=old_window['io_workers'])
    window = {k:old_window[k] for k in ('phase','entry','work','io_workers',
              'finite_updates','parity_passes','master_addr','master_port')}
    window['source'] = old_window['source']
    window['resume'] = dict(start_update=start,parent_config_sha256=resume['config_sha256'],
                            checkpoint_sha256=checkpoint_digest,**evidence,
                            checkpoint_root='/jev-resume',prepared_root='/jev-resume-prepared')
    for role in ('finite','formal'):
        commit_json(entry/f'{role}.json',dict(schema='jev-local-macros-v1',
                    macros=dict(NATIVE_SEAL_CONTAINER_CONFIG=dict(window,role=role))))
    archive = root/'entry.tar'
    with tarfile.open(archive,'x') as tar:
        tar.add(entry,arcname='entry')
    jobs = [job for chain in old_ops['chains'] for group in chain for job in group
            if job['name'] in ('finite','formal')]
    if [job['name'] for job in jobs] != ['finite','formal'] or any(job['devices'] != devices for job in jobs):
        raise ValueError('expected the frozen eight-rank finite and formal allocation')
    names = [*passthrough,'parent-training.json',*(() if online else ('confusion.json',)),'finite.json','formal.json','source/FILES.json']
    deployment.update(archive=str(archive), entry_archive_sha256=file_hash(archive),
                      entry_pins={name:file_hash(entry/name) for name in names},
                      chains=[[[job] for job in jobs]])
    commit_json(root/'operations.json',deployment)
    return deployment, window, train


def observe(c, deployment, window, parent):
    ops = Path(deployment['operations'])
    handle = read(ops/'formal-launch.json')
    host, cid = handle['host'], handle['container_id']
    path = Path(handle['remote_root'])/'formal'
    resume = window['resume']
    start = resume['start_update']
    count = c['startup_updates']
    devices = parent['physical_devices']
    world = len(devices)
    timeout = c['startup_timeout_seconds']
    obj, = json.loads(remote(host,'docker','inspect',cid).stdout)
    inspect_runtime(obj,handle)
    verify_memory(obj,handle['argv'])
    observation = dict(root=str(path), world=world, start_update=start, updates=count, timeout=timeout)
    records = json.loads(remote(host,'python3','-c',REMOTE.replace('CONFIG',repr(observation),1),
                                timeout=timeout+10).stdout)['records']
    finite = read(ops/'finite-collected.json')
    raw = remote(host,'cat',str(path/'config.json')).stdout
    train = json.loads(raw)
    fields = ('physical_devices','batch_config','instruction_objective','reference_endpoint','lr_schedule',
              'schedule_sha256','source_policy_id','source_contract')
    if (len(records) != count*world or any(train[k] != parent[k] for k in fields)
            or train['stop_update'] != parent['reference_endpoint']
            or train['start_update'] != start or train['engineering_prefix'] is not False
            or train['training_dispatch_allowed'] is not True
            or ('sampler_state' in resume and train['class_logit_ema'] != parent['class_logit_ema'])
            or ('confusion_sha256' in resume and train['resume_confusion']['sha256'] != resume['confusion_sha256'])
            or train['pins'][f'/jev-resume/checkpoint-{start:04d}-committed.json'] != resume['checkpoint_sha256']
            or finite['passed'] is not True or finite['ranks'] != world):
        raise ValueError('formal resume configuration or finite admission differs')
    restored = json.loads(remote(host,'python3','-c',
        'import json,sys; print(json.dumps([json.load(open(p)) for p in sys.argv[1:]]))',
        *(str(path/'run'/f'rank-{rank}'/'restored.json') for rank in range(world))).stdout)
    if restored != [dict(exact_masters=True,exact_adam=True,completed_updates=start)]*world:
        raise ValueError('formal optimizer state was not restored on all ranks')
    evidence = {key:resume[key] for key in ('sampler_state','confusion_sha256') if key in resume}
    if 'sampler_state' in evidence and finite['sampler_state'] != evidence['sampler_state']:
        raise ValueError('all-rank sampler restoration proof differs')
    updates = [summarize(records[i*world:(i+1)*world],start+i+1,train) for i in range(count)]
    result = dict(passed=True, container_id=cid, image=obj['Image'], ranks=world,
                  start_update=start, total_updates=train['stop_update'], checkpoint_sha256=resume['checkpoint_sha256'],
                  **evidence, config_sha256=sha256(raw).hexdigest(),
                  finite=finite, updates=updates)
    commit_json(ops/'first-updates.json',result)
    print(json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('local',type=Path)
    args = parser.parse_args()
    c,_ = load_config(args.config,args.local)
    deployment, window, parent = prepare(c)
    with ThreadPoolExecutor(max_workers=1) as pool:
        milestone = None

        def started(role):
            nonlocal milestone
            if role == 'formal':
                milestone = pool.submit(observe,c,deployment,window,parent)

        exit_code = run(deployment,on_started=started)
        if milestone is not None:
            milestone.result()
        return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
