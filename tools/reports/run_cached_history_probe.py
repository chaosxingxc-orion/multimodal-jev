"""One immutable cache diagnostic, exact container wait, compact collection.

Uses the existing device lease and a pinned prior launch as the deployment
template. No training, scoring, automatic retries, or formal admission.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.source_snapshot import freeze


def read_bound(binding):
    raw = Path(binding['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding['sha256']:
        raise ValueError('probe template evidence changed')
    return json.loads(raw)


def run(c, provenance):
    root = Path(c['output']); check_paths({'probe': root})
    root.mkdir(parents=True, exist_ok=False)
    commit_json(root/'operations.json', c); commit_json(root/'provenance.json', provenance)
    def remote(args, timeout=55):
        return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                               c['host'], shlex.join(args)], check=True, capture_output=True,
                              text=True, timeout=timeout).stdout
    try:
        with Path(c['lease']).open('r') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            old_launch = read_bound(c['prior_launch']); config = read_bound(c['prior_config'])
            if 'No running processes found in NPU '+str(c['device']) not in remote(['npu-smi', 'info']):
                raise ValueError('probe device occupied; preserve existing work')
            old_root = str(Path(config['output']).parent)
            new_root = c['remote_root']; old_source = c['prior_remote_source']; new_source = new_root+'/source'
            if (not new_root.startswith('/data/') or '..' in Path(new_root).parts
                    or new_root == old_root or new_root in old_source):
                raise ValueError('fresh local persistent remote probe root required')
            args = [s.replace(old_source, new_source).replace(old_root, new_root) for s in old_launch['argv']]
            args[args.index('--name')+1] = c['tag']
            if ('/dev/davinci'+str(c['device'])) not in args:
                raise ValueError('template physical device differs')
            image = old_launch['image']
            if json.loads(remote(['docker', 'image', 'inspect', image]))[0]['Id'] != image:
                raise ValueError('fixed image unavailable')
            source = freeze(c['source'], root/'source', patterns=['src/**/*.py', 'tools/**/*.py'], io_workers=4)
            commit_json(root/'source-receipt.json', source)
            config.update(output=new_root+'/run', runtime_id=source['manifest_sha256'],
                          align_cached_arithmetic=c['align_cached_arithmetic'])
            commit_json(root/'config.json', config)
            archive = root/'source.tar'
            with tarfile.open(archive, 'x') as tar:
                tar.add(root/'source', arcname='source')
            # Validate remote filesystem without writing to Ceph.
            storage = (Path(c['source'])/'src/multimodal_jev/local_runtime_storage.py').read_text()
            remote(['python3', '-c', storage+'\ncheck_paths({"probe": '+repr(new_root)+'})'])
            remote(['mkdir', new_root])
            remote(['mkdir', new_root+'/cache', new_root+'/home'])
            subprocess.run(['scp', str(archive), str(root/'config.json'), c['host']+':'+new_root+'/'], check=True, capture_output=True)
            remote(['tar', '-xf', new_root+'/source.tar', '-C', new_root])
            commit_json(root/'attempt.json', dict(argv=args, image=image, source=source))
            cid = remote(args).strip()
            commit_json(root/'launch.json', dict(container_id=cid, argv=args, image=image))
            remote(['docker', 'start', cid])
            code = int(remote(['docker', 'wait', cid], timeout=c['timeout_seconds']).strip())
            state = json.loads(remote(['docker', 'inspect', cid]))[0]['State']
            commit_json(root/'inspect.json', state)
            logs = subprocess.run(['ssh', c['host'], shlex.join(['docker','logs','--tail','160',cid])], capture_output=True, text=True)
            (root/'container.log').write_text(logs.stdout+logs.stderr)
            subprocess.run(['scp','-r',c['host']+':'+new_root+'/run',str(root/'collected')], check=True, capture_output=True)
            commit_json(root/'terminal.json', dict(exit_code=code, state=state,
                        status='finite_probe_collected_independent_comparison_required', formal_admitted=False))
            return dict(exit_code=code, container_id=cid, formal_admitted=False)
    except BaseException as exc:
        commit_json(root/'failure.json', dict(error=repr(exc), automatic_retry=False,
                    preserve_exact_container=True, formal_admitted=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--local', required=True)
    a = p.parse_args(); c, provenance = load_config(a.config, a.local)
    print(json.dumps(run(c, provenance), indent=2))
