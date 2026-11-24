"""Stop the owned Stage1 container after its epoch checkpoint is committed."""
import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import content_key
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.card_scoring_benchmark import pinned
from tools.reports.context_macro_probe_operations import inspect_runtime


def read(path):
    return json.loads(path.read_bytes())


def remote(host, *args, check=True, timeout=60):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                           host, shlex.join(args)], check=check, capture_output=True,
                          timeout=timeout)


def endpoint(c, train, launch):
    update = c['updates']
    expected = f'policy-{update:04d}'
    host = launch['host']
    formal = Path(launch['remote_root'])/'formal'
    root = formal/'run'
    commit_path = root/f'checkpoint-{update:04d}-committed.json'
    raw = remote(host, 'cat', str(commit_path)).stdout
    ckpt = json.loads(raw)
    sampler = ckpt.get('sampler_state')
    if 'class_logit_ema' in train:
        if (not isinstance(sampler, dict) or sampler.get('kind') != 'class-logit-ema-v1'
                or sampler.get('updates') != update or sampler.get('config') != train['class_logit_ema']
                or len(sampler.get('labels', [])) < 2
                or len(sampler['labels']) != len(sampler.get('anchor_counts', []))
                or sum(sampler['anchor_counts']) != (update // train['checkpoint_interval']) * train['batch_config']['source_queries']
                or any(len(sampler.get(key, '')) != 64 for key in
                       ('values_sha256', 'observed_sha256', 'counts_sha256', 'distribution_sha256'))):
            raise ValueError('committed endpoint lacks the expected full-epoch sampler state')
    joint = ckpt['joint_policy']
    policy = root/expected
    manifest_path = policy/'MANIFEST.json'
    manifest = json.loads(remote(host, 'cat', str(manifest_path)).stdout)
    world = len(train['physical_devices'])
    shards = {f'rank-{rank}/checkpoint-{update:04d}.pt' for rank in range(world)}
    hashes = {k: v for k, v in manifest.items() if k != 'policy_id'}
    from hashlib import sha256
    if (update % train['checkpoint_interval'] or
            (update != train['stop_update'] if c.get('completed_window')
             else update != train['checkpoint_interval'] or not update < train['stop_update'])
            or train['engineering_prefix'] is not False
            or ckpt['updates'] != update or ckpt['joint_policy_directory'] != expected
            or set(ckpt['rank_shards']) != shards
            or ckpt['execution_file'] != f'execution-{update:04d}.pt'
            or joint['source_policy_id'] != train['source_policy_id']
            or manifest['schema'] != 'context-policy-execution-v1'
            or manifest['policy_id'] != joint['policy_id']
            or content_key_policy(hashes) != manifest['policy_id']
            or manifest['contract'] != joint['contract']
            or joint['contract'] != train['source_contract']):
        raise ValueError('epoch checkpoint differs from the frozen training contract')

    expected_files = {
        formal/'config.json': c['training']['sha256'],
        commit_path: sha256(raw).hexdigest(),
        manifest_path: ckpt['joint_policy_manifest_sha256'],
        root/ckpt['execution_file']: ckpt['execution_sha256'],
        **{root/name: digest for name, digest in ckpt['rank_shards'].items()},
    }
    for name, digest in manifest['files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or not rel.parts:
            raise ValueError('invalid policy file path in the committed manifest')
        expected_files[policy/rel] = digest
    listing = remote(host, 'sha256sum', '--', *map(str, expected_files), timeout=c['hash_timeout_seconds'])
    actual = {Path(name): digest for digest, name in
              (line.split(maxsplit=1) for line in listing.stdout.decode().splitlines())}
    if actual != expected_files:
        raise ValueError('epoch checkpoint bytes differ from the commit')

    rank_records = [root/f'rank-{rank}'/f'update-{update:04d}.json' for rank in range(world)]
    output = remote(host, 'python3', '-c',
                    'import json,sys; print(json.dumps([json.load(open(p)) for p in sys.argv[1:]]))',
                    *map(str, rank_records))
    rows = json.loads(output.stdout)
    if len(rows) != world or any(r['update'] != update or r['checkpoint'] is not True
            or not math.isfinite(r['metrics']['actor/pg_loss']) for r in rows):
        raise ValueError('missing or invalid all-rank epoch update records')
    return dict(updates=update, policy=str(policy), policy_id=joint['policy_id'],
                manifest_sha256=ckpt['joint_policy_manifest_sha256'],
                checkpoint_sha256=expected_files[commit_path],
                rank_records_sha256=content_key(rows), devices=train['physical_devices'],
                **(dict(sampler_state=sampler) if sampler else {}))


def content_key_policy(value):
    from multimodal_jev.context_policy_snapshot import _hash
    return _hash(value)


def inspect(launch):
    host, cid = launch['host'], launch['container_id']
    observed, = json.loads(remote(host, 'docker', 'inspect', cid).stdout)
    inspect_runtime(observed, launch)
    if observed['State']['OOMKilled']:
        raise ValueError('Stage1 container was OOM killed')
    return observed


def run(c):
    train = pinned(c['training'])
    out = Path(c['output'])
    check_paths({'epoch handoff': out})
    launch = pinned(c['launch'])
    host, cid = launch['host'], launch['container_id']
    root = Path(launch['remote_root'])/'formal'/'run'
    update = c['updates']
    commit_path = root/f'checkpoint-{update:04d}-committed.json'
    final_records = [str(root/f'rank-{rank}'/f'update-{update:04d}.json')
                     for rank in range(len(train['physical_devices']))]
    if (not root.is_absolute() or launch['host'] != c['host'] or
            update != train['checkpoint_interval'] or update >= train['stop_update']):
        raise ValueError('epoch boundary must match the authorized frozen training run')
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + c['deadline_seconds']
    print(f'Waiting for all-rank checkpoint {update} from {cid}', flush=True)
    probe = 'import os,sys; sys.exit(0 if all(os.path.isfile(p) for p in sys.argv[1:]) else 1)'
    while True:
        state = inspect(launch)['State']
        ready = remote(host, 'python3', '-c', probe, str(commit_path), *final_records, check=False)
        if ready.returncode == 0:
            break
        if ready.returncode != 1:
            raise ValueError('epoch checkpoint readiness could not be checked')
        if state['Status'] != 'running' or time.monotonic() >= deadline:
            raise ValueError('owned training did not commit the requested epoch before the deadline')
        time.sleep(min(c['poll_seconds'], max(0, deadline-time.monotonic())))

    proof = endpoint(c, train, launch)
    # Holding the original training lease would deadlock this handoff. Only the
    # pinned container can be stopped, then its original owner releases the lease.
    state = inspect(launch)['State']
    if state['Status'] == 'running':
        later = str(root/'rank-0'/f'update-{update+1:04d}.json')
        progress = remote(host, 'python3', '-c', probe, later, check=False)
        if progress.returncode != 1:
            raise ValueError('next epoch may already have started; refusing to stop at the earlier boundary')
        commit_json(out/'stop-intent.json', dict(container_id=cid, **proof))
        remote(host, 'docker', 'stop', '--time', str(c['stop_timeout_seconds']), cid,
               timeout=c['stop_timeout_seconds']+30)
    else:
        raise ValueError('owned Stage1 container exited before the authorized stop')
    terminal = inspect(launch)
    state = terminal['State']
    if state['Status'] != 'exited' or state['Running'] or state['Pid'] != 0:
        raise ValueError('owned Stage1 container is still active')
    commit_json(out/'inspect.json', terminal)
    receipt = dict(proof, passed=True, container_exited=True, container_id=cid, host=host,
                   launch_sha256=c['launch']['sha256'],
                   config_sha256=c['training']['sha256'], exit_code=state['ExitCode'],
                   inspect_sha256=file_hash(out/'inspect.json'))
    commit_json(out/'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('local', type=Path)
    args = parser.parse_args()
    c, _ = load_config(args.config, args.local)
    print(json.dumps(run(c)), flush=True)


if __name__ == '__main__':
    main()
