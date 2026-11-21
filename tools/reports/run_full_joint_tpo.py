"""Wait for the exact predecessor, run the sealed 181 chain, and audit collection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import threading
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from multimodal_jev.local_checkpoint import commit_file, commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.source_snapshot import verify
from tools.reports.build_tpo_view import read, verified
from tools.reports.context_macro_probe_operations import inspect_runtime
from tools.reports.finalize_context_infonce import wait_exact
from tools.reports.joint_tpo_window import check_update
from tools.reports.native_seal_operations import run as dispatch

HOST = '100.100.135.181'


def ssh(host, args, timeout=55):
    if host.rsplit('@',1)[-1] != HOST:
        raise ValueError('only host 181 is allowed')
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
                                    host,shlex.join(args)],timeout=timeout)


def checked_endpoint(receipt, mem, steps, binding):
    if (receipt['passed'] is not True or receipt['updates'] != steps or receipt['window'] != 'formal'
            or receipt['arm'] != 'tpo' or receipt['binding'] != binding
            or not re.fullmatch('[0-9a-f]{64}',receipt['endpoint'])
            or receipt['endpoint'] == binding['initial_policy']
            or hashlib.sha256(mem).hexdigest() != receipt['manifest_sha256']
            or json.loads(mem)['policy_id'] != receipt['endpoint']):
        raise ValueError('formal endpoint receipt does not match frozen run')


def await_predecessor(c, host, root):
    pre = Path(c['predecessor'])
    launch = read(verified(pre/'tpo-launch.json',c['predecessor_launch_sha256']))
    finite = read(verified(pre/'finite-collected.json',c['predecessor_finite_sha256']))
    binding = finite['binding']
    if (launch['host'] != host or binding['joint_sha'] != c['predecessor_joint_sha256']
            or finite['passed'] is not True or finite['updates'] != 2 or finite['window'] != 'finite'):
        raise ValueError('wrong predecessor binding')
    driver = read(verified(pre.with_suffix('.started.json'),c['predecessor_driver_sha256']))
    commit_json(root/'waiting.json',dict(driver=driver, container=launch['container_id'], expected_updates=c['predecessor_updates']))
    wait_exact(driver,pre/'terminal.json')
    terminal = read(pre/'terminal.json')
    receipt = read(pre/'tpo-collected.json')
    prior_root = launch['remote_root']
    if (terminal != {'exit_code':0,'status':'declared_stages_finished'}
            or read(pre/'tpo-wait.json') != {'exit_code':0}
            or receipt['anchor_sha256'] != finite['anchor_sha256']):
        raise ValueError('predecessor has not completed the required formal run')
    cid = launch['container_id']
    observed, = json.loads(ssh(host,['docker','inspect',cid]))
    inspect_runtime(observed,launch,terminal=True)
    inspect_runtime(read(pre/'tpo-inspect.json'),launch,terminal=True)
    if json.loads(ssh(host,['cat',prior_root+'/tpo/receipt.json'])) != receipt:
        raise ValueError('predecessor local and remote endpoint receipts differ')
    checked_endpoint(receipt,ssh(host,['cat',prior_root+'/tpo/policy/MANIFEST.json']),c['predecessor_updates'],binding)
    proof = dict(container_id=cid, terminal_sha256=file_hash(pre/'terminal.json'),
                 receipt_sha256=file_hash(pre/'tpo-collected.json'), receipt=receipt,
                 state=observed['State'])
    commit_json(root/'predecessor-released.json',proof)


def observe(stop, ops, c, root):
    n = 0
    while not stop.wait(c['ordinary_seconds']):
        n += 1
        roles = ('tpo','anchor','finite','finite-anchor')
        out = dict(observed_unix=time.time(),status='waiting')
        try:
            for role in roles:
                handle = ops/(role+'-launch.json')
                if handle.is_file():
                    h = read(handle)
                    states = json.loads(ssh(h['host'],['docker','inspect','--format','{{json .State}}',h['container_id']]))
                    out.update(status=states['Status'], role=role, container_id=h['container_id'], state=states)
                    if role in ('finite','tpo'):
                        path = h['remote_root']+'/'+role
                        cmd = 'from pathlib import Path; import sys; print(len(list(Path(sys.argv[1]).glob("update-*.json"))))'
                        out['committed_updates'] = int(ssh(h['host'],['python3','-c',cmd,path]))
                    break
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            out.update(status='observation_failed',error=repr(exc))
        commit_json(root/f'observation-{n:04d}.json',out)


def collect(c, deployment, ops, plan, root):
    steps = plan['updates']
    host = deployment['host']
    launch = read(ops/'tpo-launch.json')
    if launch['host'] != host or launch['remote_root'] != deployment['remote_root']:
        raise ValueError('formal launch does not match this deployment')
    cid = launch['container_id']
    observed, = json.loads(ssh(host,['docker','inspect',cid]))
    inspect_runtime(observed,launch,terminal=True)
    if read(ops/'tpo-wait.json') != {'exit_code':0}:
        raise ValueError('exact formal wait did not succeed')
    dest = root/'collected'
    dest.mkdir()
    policy_dirs = [f'policy-step-{s}' for s in plan['intermediate_export_updates']]+['policy']
    names = [f'update-{s:04d}.json' for s in range(1,steps+1)]
    names.extend(('result.json','receipt.json','supervision/terminal.json'))
    names.extend(f'rank-{r}/initial-parity.json' for r in range(len(plan['training_devices'])))
    names.extend(policy_dirs)
    remote_path = launch['remote_root']+'/tpo'
    # Omit activations, temporary rollouts, and container cache.
    def collect_tar(f):
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,
                        shlex.join(['tar','-cf','-','-C',remote_path,*names])],
                       stdout=f,check=True,timeout=c['collection_timeout_seconds'])
    archive = root/'formal-evidence.tar'
    commit_file(archive,collect_tar)
    with tarfile.open(archive) as tar:
        allowed_dirs=set(policy_dirs)
        for entry in tar:
            p=Path(entry.name)
            if (p.is_absolute() or '..' in p.parts or entry.name not in names and
                    p.parts[0] not in allowed_dirs or not (entry.isfile() or entry.isdir())):
                raise ValueError('unexpected collection member')
            if entry.isfile():
                target=dest/p
                target.parent.mkdir(parents=True,exist_ok=True)
                commit_file(target,lambda out:out.write(tar.extractfile(entry).read()))
    receipt=read(dest/'receipt.json')
    lifecycle=read(dest/'supervision/terminal.json')
    if (lifecycle['passed'] is not True or lifecycle['actor_exit_code']!=0
            or lifecycle['forced_cleanup'] is not False
            or len(lifecycle['services'])!=len(plan['inference_devices'])
            or any(x['exit_code']!=0 for x in lifecycle['services'])):
        raise ValueError('formal process lifecycle does not match two successful TP1 services')
    pkg = Path(deployment['archive']).parent/'entry'
    window = read(pkg/'tpo.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    bound=dict(joint_sha=file_hash(pkg/'joint.json'),source=window['source']['sha256'],
               target_sha=plan['target_sha'],parent_sha=plan['parent_sha'],
               world_size=len(plan['training_devices']),initial_policy=plan['source_policy'])
    checked_endpoint(receipt,(dest/'policy/MANIFEST.json').read_bytes(),steps,bound)
    if (receipt != read(ops/'tpo-collected.json')
            or receipt['anchor_sha256'] != file_hash(ops/'anchor-collected.json')):
        raise ValueError('formal receipt or fixed anchor changed on collection')
    # JSON copies use the same durable serialization as the original receipt.
    schedule = read(verified(pkg/'targets/batches.json',read(pkg/'targets/MANIFEST.json')['files']['batches.json']))
    parent = read(verified(pkg/'parent.json',plan['parent_sha']))
    previous = plan['source_policy']
    curve = []
    exposures = 0
    for s,(name,batch) in enumerate(zip(names[:steps],schedule,strict=True),1):
        row=read(dest/name)
        check_update(row,s,plan,bound,previous)
        ref=row['reference']
        activation=ref['next_policy']
        lr=parent['view_execution']['optimizer']['lr']*min(1.,s/plan['warmup_updates'])
        valid_next = (isinstance(activation,str) and re.fullmatch('[0-9a-f]{64}',activation)
                      and activation!=previous) if s<steps else activation is None
        if (row['real'] != sum(x['weight'] for x in batch['slots']) or row['learning_rate'] != lr
                or not valid_next):
            raise ValueError('formal exposure, schedule, or actor chain differs')
        previous=activation
        exposures+=row['real']
        curve.append(row)
    if exposures != plan['recordings']*plan['epochs'] or read(dest/'result.json')['policy']!=receipt['endpoint']:
        raise ValueError('formal endpoint result or exposure total differs')
    for rank in range(bound['world_size']):
        parity=read(dest/f'rank-{rank}/initial-parity.json')
        if (len(parity)!=4 or any(r['passed'] is not True or r['rows']!=plan['microbatch'] or
                                  r['threshold_flips']!=0 or not 0<=r['max_abs']<=parent['behavior_bounds']['probability_max_abs']
                                  for r in parity)):
            raise ValueError('initial parity audit failed')
    def policy_check(name):
        path=dest/name
        manifest=read(path/'MANIFEST.json')
        read_execution_bundle(path,expected_policy_id=manifest['policy_id'],
                              expected_contract=parent['source_contract'],expected_base_id=parent['base_id'])
        if manifest['policy_id']==plan['source_policy']:
            raise ValueError('trained policy unchanged')
        return name,dict(policy_id=manifest['policy_id'],manifest_sha256=file_hash(path/'MANIFEST.json'))
    with ThreadPoolExecutor(max_workers=min(c['io_workers'],len(policy_dirs))) as pool:
        checkpoints=dict(pool.map(policy_check,policy_dirs))
    commit_json(root/'curve.json',curve)
    # This duplicate tar is private transfer staging; durable collection is complete.
    archive.unlink()
    elapsed=math.fsum(x['seconds'] for x in curve)
    proof=dict(status='formal_training_collected_and_audited',container_id=cid,endpoint=receipt['endpoint'],
               terminal_state=observed['State'],recordings=plan['recordings'],updates=steps,
               real_exposures=exposures,training_step_seconds=elapsed,recordings_per_step_second=exposures/elapsed,
               curve_sha256=file_hash(root/'curve.json'),receipt_sha256=file_hash(dest/'receipt.json'),
               checkpoints=checkpoints,evaluation_complete=False)
    commit_json(root/'formal-audit.json',proof)
    return proof


def run(c, root):
    import torch
    torch.set_num_threads(c['io_workers'])
    dep = operations(c)
    if dep != read(root/'operations.json') or dep['host'].rsplit('@',1)[-1]!=HOST:
        raise ValueError('operations do not match the frozen 181 dispatch')
    entry=Path(c['package'])/'entry'
    plan=read(verified(entry/'joint.json',dep['entry_pins']['joint.json']))
    stated=read(entry/'tpo.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']['source']['sha256']
    ops=Path(dep['operations'])
    stop=threading.Event()
    ctl=dict(c['custody'],io_workers=c['io_workers'])
    if ctl['ordinary_seconds']!=900:
        raise ValueError('the registered ordinary observation interval is 900 seconds')
    observer=threading.Thread(target=observe,args=(stop,ops,ctl,root),daemon=True)
    try:
        verify(entry/'source',manifest_sha256=stated,io_workers=c['io_workers'])
        observer.start()
        await_predecessor(ctl,dep['host'],root)
        if dispatch(dep) != 0:
            raise ValueError('sealed device chain failed')
        result=collect(ctl,dep,ops,plan,root)
        commit_json(root/'terminal.json',dict(exit_code=0,**result))
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,restart=False,error=repr(exc)))
        raise
    finally:
        stop.set()


def operations(c):
    data = read(Path(c['package'])/'operations-local.json')
    return data['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('local')
    p.add_argument('--detach',action='store_true')
    p.add_argument('--resolved',action='store_true')
    a=p.parse_args()
    c=read(Path(a.config)) if a.resolved else load_config(a.config,a.local)[0]
    root=Path(c['custody']['root'])
    if a.detach:
        check_paths({'custody':root})
        root.mkdir(exist_ok=False,parents=True)
        dep=operations(c)
        commit_json(root/'operations.json',dep)
        # Freeze the resolved controller inputs separately from the caller's template/macros.
        commit_json(root/'config.json',c)
        source=Path(c['package'])/'entry/source'
        env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,(source/'src',source,source/'tools/jev'))))
        argv=[sys.executable,'-B','-m','tools.reports.run_full_joint_tpo',str(root/'config.json'),
              str(root/'config.json'),'--resolved']
        # Python resolves the working directory before PYTHONPATH; execute from the snapshot.
        with (root/'driver.log').open('x') as log:
            child=subprocess.Popen(argv,cwd=source,env=env,start_new_session=True,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT)
        record=dict(pid=child.pid,argv=argv,process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
        commit_json(root/'started.json',record)
        print(json.dumps(record))
    else:
        run(c,root)


if __name__=='__main__':
    main()
