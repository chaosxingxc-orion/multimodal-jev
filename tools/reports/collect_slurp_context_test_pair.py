"""Collect success/failure first, then independently audit a frozen SLURP TEST pair.

Reuses the existing read-only terminal transport; no target labels or scores.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from tools.jev.collect_context_rl_selection import collect
from tools.reports.prepare_slurp_context_test_package import verify, sha, relocate
from tools.reports.launch_slurp_context_test_pair import command
from tools.reports.run_slurp_context_test_pair import configs, check_full
from tools.reports.slurp_context_test_runtime import require_parity


def read(path):return json.loads(Path(path).read_text())


def audit(package, digest, launch, observed, collection, transport):
    package, collection = Path(package), Path(collection)
    manifest = verify(package, digest);spec = read(package/'config.json')
    expected = command(manifest, spec)
    image_index = expected.index(manifest['image'])
    def values(flag):return [expected[i+1] for i,v in enumerate(expected) if v == flag]
    state = observed['State']
    if (launch['manifest_sha256'] != digest or launch['argv'] != expected
            or launch['config_sha256'] != sha(package/'config.json')
            or launch['remote_root'] != manifest['remote_root']
            or observed['Id'] != launch['container_id'] or observed['Image'] != manifest['image']
            or launch['image'] != manifest['image']
            or state['Status'] != 'exited' or state['ExitCode'] != 0 or state['OOMKilled']
            or observed['Config']['Entrypoint'] != values('--entrypoint')
            or observed['Config']['Cmd'] != expected[image_index+1:]
            or sorted(observed['HostConfig']['Binds']) != sorted(values('-v'))
            or observed['HostConfig']['NetworkMode'] != 'none'):
        raise ValueError('successful exact fixed-image runtime and mounts required')
    env = dict(v.split('=',1) for v in observed['Config']['Env'])
    if any(env.get(k) != v for k,v in (e.split('=',1) for e in values('-e'))):
        raise ValueError('actual runtime environment differs')
    devices = observed['HostConfig']['Devices']
    if sorted((d['PathOnHost'],d['PathInContainer'],d['CgroupPermissions']) for d in devices) != sorted(
            (*v.split(':'),'rwm') for v in values('--device')):
        raise ValueError('actual isolated device scope differs')
    if (transport['status'] != 'transport_complete_independent_audit_required'
            or transport['container_id'] != launch['container_id'] or transport['state'] != state
            or transport['image'] != manifest['image'] or transport['remote_root'] != manifest['remote_root']):
        raise ValueError('complete transport of same runtime required')
    actual = {}
    for p in collection.rglob('*'):
        if p.is_symlink():raise ValueError('collection symlink forbidden')
        if p.is_file():actual[str(p.relative_to(collection))] = sha(p)
    if actual != transport['files'] or sha(collection/'config.json') != launch['config_sha256']:
        raise ValueError('collected file population differs')
    root = collection/'evaluation'
    if read(root/'input.json') != spec or read(root/'terminal.json')['exit_code'] != 0:
        raise ValueError('successful input-bound pair runner required')
    local = relocate(spec, manifest['remote_root'], str(package.resolve()))
    pair, rows, receipts = configs(local)
    panels, results = {}, {}
    for role, local_config in pair.items():
        expected_config = relocate(local_config, str(package.resolve()), manifest['remote_root'])
        folder = root/role
        if read(folder/'config.json') != expected_config:
            raise ValueError('actual source policy configuration differs')
        panels[role] = require_parity(expected_config, folder, rows, receipts)
        if panels[role] != read(folder/'parity.json'):
            raise ValueError('parity summary differs')
        results[role] = check_full(folder, expected_config, rows, receipts)
        predictions = read(folder/'predictions.json')['records']
        for index, (prediction, receipt) in enumerate(zip(predictions, receipts, strict=True)):
            if read(folder/f'record-{index:05d}.json') != dict(prediction=prediction, input_receipt=receipt):
                raise ValueError('durable TEST prediction differs')
    result = read(root/'result.json')
    if (result['status'] != 'test_pair_inference_complete_independent_collection_pending'
            or result['source_manifest_sha256'] != spec['source_manifest_sha256']
            or result['panels'] != panels or result['policies'] != results or result['target_tuning'] is not False):
        raise ValueError('pair summary differs from actual outputs')
    return dict(status='slurp_test_pair_inference_independently_collected',container_id=launch['container_id'],
                state=state,package_manifest_sha256=digest,source_manifest_sha256=spec['source_manifest_sha256'],
                panels=panels,policies=results,
                predictions={role:dict(path=str((root/role/'predictions.json').resolve()),
                             sha256=sha(root/role/'predictions.json')) for role in results},
                scope='Source-selected pair and fixed runtime/inputs audited; target scoring remains separate.')


def main():
    p = argparse.ArgumentParser()
    for name in ('package','manifest-sha256','launch','output','report'):p.add_argument('--'+name,required=True)
    a = p.parse_args();launch = read(a.launch)
    transport = collect(launch,a.output)
    observed = json.loads(subprocess.check_output(['ssh','npu',shlex.join(
        ['docker','inspect',launch['container_id']])],text=True,timeout=60))[0]
    inspect_path = Path(a.output).with_suffix('.inspect.json')
    with inspect_path.open('x') as stream:json.dump(observed,stream,indent=2);stream.write('\n')
    # Failed collections remain on disk even when this independent audit rejects.
    report = audit(a.package,a.manifest_sha256,launch,observed,a.output,transport)
    report['runtime_inspect'] = dict(path=str(inspect_path.resolve()),sha256=sha(inspect_path))
    with Path(a.report).open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=report['status'],container_id=report['container_id'])))


if __name__ == '__main__':main()
