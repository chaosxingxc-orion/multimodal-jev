"""Bind a stopped attempt's admitted TRAIN macro prefix to a fresh reference job."""
from pathlib import Path
import shutil

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_reference_resume import archive_prefix, lineage, seal
from multimodal_jev.local_checkpoint import commit_file, commit_json
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_reference_shutdown_recovery import read, tree


PROJECT = Path(__file__).resolve().parents[2]
OVERLAY = {
    'resume_reference.py': PROJECT/'tools/reports/context_reference_resume_entry.py',
    'reference_resume_builder.py': PROJECT/'tools/reports/build_context_reference_dataset.py',
    'reference_resume_pool.py': PROJECT/'src/multimodal_jev/context_reference_resume.py',
}


def prepare_prefix(c, provenance, *, allow_empty=False):
    from tools.reports.context_cached_storage_operations import audit_container, reference_command
    from tools.reports.context_reference_job_operations import admission

    deployment = c['prior_operations']
    # Re-run the same endpoint TRAIN parity and original package audit.
    gate = admission(deployment)
    spec = deployment['reference_job']
    package = Path(spec['output'])
    stem = Path(deployment['operations_root'])/deployment['tag']
    collection = Path(str(stem)+'-collected')
    launch = read(str(stem)+'-launch.json')
    inspect = read(str(stem)+'-inspect.json')
    wait = read(str(stem)+'-wait.json')
    transport = read(str(stem)+'-collected.transport.json')
    state = inspect['State']
    manifest = read(package/'PREPARED.json')
    exit_code = wait['exit_code']
    if (type(exit_code) is not int or exit_code <= 0 or state.get('Error')
            or wait['container_id'] != launch['container_id']
            or launch['host'] != deployment['host'] or launch['remote_root'] != spec['remote_root']
            or launch['manifest_sha256'] != spec['manifest_sha256']
            or launch['config_sha256'] != file_hash(package/'config.json')
            or launch['layout'] != manifest['layout']
            or transport['container_id'] != launch['container_id'] or transport['state'] != state
            or transport['remote_root'] != spec['remote_root'] or transport['image'] != launch['image']
            or transport['status'] != 'transport_complete_independent_audit_required'
            or any(transport[key] for key in ('bad_pins', 'bad_package', 'launch_binding_mismatches'))
            or file_hash(str(stem)+'-collected.tar') != transport['archive_sha256']
            or tree(collection,c['io_workers']) != transport['files']):
        raise ValueError('exact stopped, independently collected reference attempt required')
    training = read(package/'config.json')
    storage = audit_container(reference_command(deployment,manifest,training),training,launch,
                              inspect,producer=True,expected_exit=exit_code)
    build = read(package/'reference-build-config.json')
    if file_hash(collection/'reference-build-config.json') != file_hash(package/'reference-build-config.json'):
        raise ValueError('collected reference builder config differs')
    journal = collection/'reference-journal'/build['macro_client']['attempt_id']
    endpoint = file_hash(package/'initial/endpoint-commit.json')
    scope = lineage(build, endpoint_sha256=endpoint)
    if allow_empty and not journal.is_symlink():
        # The exact archive was already checked above. An empty journal is
        # comparison evidence, never an admissible resume asset.
        attempts = sorted(journal.glob('batch-*'))
        if (not journal.exists() or journal.is_dir()) and (
                not attempts or (len(attempts) == 1 and attempts[0].name == 'batch-00000000'
                and attempts[0].is_dir() and not attempts[0].is_symlink()
                and not (attempts[0]/'COMPLETE.json').exists())):
            return dict(status='stopped_reference_has_no_commits', commits=0)
    prefix = seal(journal, scope)
    output = Path(c['output'])
    check_paths({'stopped reference prefix':output})
    output.mkdir(parents=True, exist_ok=False)
    commit_file(output/'reference-prefix.zip', lambda stream: archive_prefix(journal,prefix,stream))
    commit_json(output/'reference-prefix.json', prefix)
    proof = dict(schema='jev-reference-prefix-transport-v1', container_id=launch['container_id'],
                 source_host=deployment['host'], source_lease=deployment['lease'],
                 source_manifest_sha256=spec['manifest_sha256'], exit_code=exit_code,
                 prefix_sha256=file_hash(output/'reference-prefix.json'),
                 archive_sha256=file_hash(output/'reference-prefix.zip'),
                 commits=len(prefix['commits']), original_transport_sha256=file_hash(str(stem)+'-collected.transport.json'),
                 admission=gate, storage=storage, provenance=provenance, npu_executed=False)
    commit_json(output/'receipt.json', proof)
    return dict(status='stopped_reference_prefix_sealed', receipt_sha256=file_hash(output/'receipt.json'),
                commits=proof['commits'])


def bind_assets(spec, package, build, remote, *, prepare, legacy_service=False):
    required = {'prefix_root','receipt_sha256','replay_last'}
    if set(spec) != required or type(spec['replay_last']) is not bool:
        raise ValueError('explicit pinned stopped attempt and tail replay policy required')
    source = Path(spec['prefix_root'])
    if source.is_symlink() or file_hash(source/'receipt.json') != spec['receipt_sha256']:
        raise ValueError('stopped reference prefix receipt changed')
    receipt = read(source/'receipt.json')
    manifest = read(source/'reference-prefix.json')
    if (receipt['schema'] != 'jev-reference-prefix-transport-v1'
            or receipt['commits'] != len(manifest['commits'])
            or manifest.get('lineage') != lineage(build, endpoint_sha256=file_hash(package/'initial/endpoint-commit.json'))):
        raise ValueError('resume requires the exact same endpoint, TRAIN inputs and numerical runtime')
    files = dict(OVERLAY, **{name:source/name for name in ('reference-prefix.zip','reference-prefix.json')})
    if legacy_service:
        files['resume-service.py'] = PROJECT/'tools/reports/context_reference_replay_service.py'
    expected = {name:file_hash(path) for name,path in files.items()}
    if (expected['reference-prefix.json'] != receipt['prefix_sha256']
            or expected['reference-prefix.zip'] != receipt['archive_sha256']):
        raise ValueError('stopped reference prefix changed')
    for name,path in files.items():
        if prepare:
            shutil.copyfile(path,package/name)
        if file_hash(package/name) != expected[name]:
            raise ValueError('packaged resume asset differs: '+name)
    build['resume'] = dict(archive=remote+'/reference-prefix.zip',manifest=remote+'/reference-prefix.json',
                           manifest_sha256=expected['reference-prefix.json'],replay_last=spec['replay_last'])
    return expected


def audit_resumed(package, collection, total):
    build = read(package/'reference-build-config.json')
    resume = build['resume']
    prefix = collection/'reference-prefix.json'
    if file_hash(prefix) != resume['manifest_sha256']:
        raise ValueError('executed reference prefix manifest differs')
    expected = read(prefix)
    pool = build['macro_client']
    journal = collection/'reference-journal'/pool['attempt_id']
    actual = seal(journal, expected['lineage'])['commits']
    n = len(expected['commits'])
    replay = int(resume['replay_last'])
    if len(actual) != total or n > total or actual[:n-replay] != expected['commits'][:n-replay]:
        raise ValueError('resumed journal does not preserve pinned prefix')
    if replay:
        # Wall timings are fresh; record content must match the original commit.
        import json
        import zipfile
        path = f'batch-{n-1:08d}/COMPLETE.json'
        with zipfile.ZipFile(package/'reference-prefix.zip') as source:
            old = json.loads(source.read(path))
        new = read(journal/path)
        if new['result_sha256'] != old['result_sha256']:
            raise ValueError('replayed final macro differs from original')
    receipt = dict(manifest_sha256=resume['manifest_sha256'], reused_macros=n-replay,
                   replayed_macros=replay, new_macros=total-n)
    if read(collection/'reference-build/result.json')['resume'] != receipt:
        raise ValueError('reference resume execution receipt differs')
    return receipt


if __name__ == '__main__':
    import argparse
    import json
    from multimodal_jev.local_config import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_prefix(*load_config(args.config,args.local)),indent=2))
