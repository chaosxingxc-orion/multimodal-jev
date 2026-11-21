"""Recompute one committed reference macro from pinned TRAIN audio inputs."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_preparation import CachedContextPreparer, processor_cache_identity
from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache
from multimodal_jev.slurp_portable_release import load
from tools.reports.build_context_reference_dataset import audio_mirror, validate_endpoint
from tools.reports.context_dual_runtime_candidate import verify_runtime
from tools.reports.context_reference_input_evidence import AuditedPreparer, load as load_evidence
from tools.reports.context_replica_macro_client import connect_pool


def read(path):
    return json.loads(Path(path).read_text())


def run(build, replay):
    output = Path(build['output'])
    check_paths({'tail replay': output})
    output.mkdir(exist_ok=False)
    for name, digest in build['pins'].items():
        if file_hash(name) != digest:
            raise ValueError('replay dependency changed: ' + name)
    runtime = verify_runtime(build)
    pool_config = build['macro_client']
    if pool_config['runtime_identity'] != runtime:
        raise ValueError('replay runtime differs')
    bundle = read_execution_bundle(build['source_policy'], expected_policy_id=build['source_policy_id'],
                                   expected_contract=build['source_contract'], expected_base_id=build['base_id'])
    validate_endpoint(read(build['stage1_commit']), build, bundle)
    rows, task, _ = load(build['release'], build['release_sha256'], 'learning', purpose='optimize')
    processor = processor_cache_identity(build['source_policy'])
    evidence, _ = load_evidence(build, rows=rows, task=task, release_sha256=build['release_sha256'],
                                processor_sha256=processor)
    candidates = [dict(id=x['id'], definition=x['description']) for x in task['schema']['candidates']]
    def cache_miss():
        raise ValueError('offline reference input required')
    preparer = AuditedPreparer(CachedContextPreparer(cache_miss,
        ReadOnlyEncodedAudioCache(build['encoded_audio_cache']), processor_sha256=processor,
        decision_token_id=build['source_contract']['token']['id'], audio_token_id=build['audio_token_id'],
        max_tokens=build['max_tokens'], audio_mirror=audio_mirror(build), prompt_layout=build.get('prompt_layout')),
        evidence)
    ids = replay['sample_ids']
    if not 0 < len(ids) <= pool_config['max_queries'] or len(set(ids)) != len(ids):
        raise ValueError('exactly one bounded macro of distinct TRAIN IDs required')
    lookup = {r['id']: r for r in rows}
    requests = []
    for uid in ids:
        row = lookup[uid]
        req = compile_candidates(row['audio'], candidates, task['thinking']['requirements'], semantics='exclusive_event')
        value = preparer((req, row['audio_sha256']))
        if not value.cache_hit:
            raise ValueError('expected cached replay input')
        requests.append(dict(inputs=value.inputs, positions=value.positions))
    pool = connect_pool(pool_config, bundle=bundle)
    try:
        pool.score_macro(replay['batch_id'], requests, audio_token_id=build['audio_token_id'])
    finally:
        pool.close()
    commit_json(output/'terminal.json', dict(exit_code=0, batch_id=replay['batch_id'], scored=len(ids)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', required=True); p.add_argument('--replay', required=True)
    a = p.parse_args()
    run(read(a.build), read(a.replay))
