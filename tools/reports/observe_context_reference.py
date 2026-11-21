"""Read committed TRAIN scoring batches on the original container."""

import argparse
import json
from pathlib import Path
import re

from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.stage2_view_runtime import pinned
from tools.reports.context_replica_training_operations import path as canonical
from tools.reports.observe_context_replica_training import handle, observe


def run(c, *, remote=None, now=None):
    prior = c['operations']
    binding = pinned(prior['path'], prior['sha256'])
    base = Path(canonical(binding['operations_root']))
    if Path(prior['path']) != base/'finalization/config.json':
        raise ValueError('exact reference operation configuration required')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', binding['tag']) or not re.fullmatch(
            r'(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9][A-Za-z0-9_.-]*', binding['host']):
        raise ValueError('canonical reference host and tag required')
    package = Path(canonical(binding['package']))
    root = canonical(binding['reference_job']['remote_root'])
    manifest = pinned(package/'PREPARED.json', binding['manifest_sha256'])
    build = pinned(package/'reference-build-config.json', manifest['files']['reference-build-config.json'])
    client = build['macro_client']
    if (manifest['remote_root'] != root or 'resume' in binding['reference_job']
            or build.get('resume') is not None or client['journal_root'] != root+'/reference-journal'
            or not re.fullmatch(r'[A-Za-z0-9_-]+', client['attempt_id'])):
        raise ValueError('original reference journal required; resumed blocks require separate accounting')
    total, batch_size = build['expected_population'], client['max_queries']
    if any(type(n) is not int or n < 1 for n in (total, batch_size)):
        raise ValueError('positive reference population and batch size required')
    observations = canonical(c['observations_root'])
    check_paths({'reference_observations': observations})
    _, expected = handle(binding)
    if (expected['host'], expected['remote_root'], expected['image']) != (binding['host'], root, manifest['image']):
        raise ValueError('reference container differs from prepared operation')
    source = Path(__file__).with_name('read_context_reference_progress.py').read_text()
    args = [client['journal_root']+'/'+client['attempt_id'], build['source_policy_id'], str(total), str(batch_size)]

    def read_progress(launch, read_remote):
        if launch != expected:
            raise ValueError('reference launch changed during observation')
        return json.loads(read_remote(['docker', 'exec', launch['container_id'], 'python', '-B', '-c', source, *args]))

    return observe(dict(binding, observations_root=observations), remote=remote, now=now,
                   read_progress=read_progress)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    a = p.parse_args()
    config, _ = load_config(a.config, a.local)
    print(json.dumps(run(config), indent=2))
