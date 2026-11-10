"""Coordinator-pinned actual Jev capacity admission before full generation."""
import json
import math
from pathlib import Path

from .audio2tool_acceptance import check_container
from .audio2tool_engine import ASSET_AUDIT_SHA, ENGINE, VERSIONS
from .audio2tool_jev_engine import CONTEXT_SHA, MANIFEST_SHA, boundaries
from .baseline import file_hash

RUNNER = 'tools/jev/probe_audio2tool_jev_engine.py'
MODEL = '/mnt/cephfs/xingchao/models/qwen3-omni-30b-a3b-instruct'
PREFLIGHT = Path('/mnt/cephfs/xingchao/experiments/multimodal-jev/audio2tool-jev-engine-preflight-v1')


def require_capacity(root, intent_hash, inspection, inspection_hash, container_id, context_run, release, registry):
    """Caller must obtain inspection from the exact live Docker handle.

    Hashes are external inputs pinned before dispatch, not values inferred from
    mutable output files. This function does not call SSH or trust a filename
    alone as proof of origin; independent final acceptance must re-inspect.
    """
    root = Path(root)
    if file_hash(root / 'intent.json') != intent_hash or file_hash(inspection) != inspection_hash:
        raise ValueError('externally pinned capacity intent and inspection required')
    read = lambda p:json.loads(Path(p).read_text())
    plan = read(root / 'intent.json'); observed = read(inspection)
    launch = read(root / 'launch.json')
    if (launch['container_id'] != container_id or launch['intent_sha256'] != intent_hash
            or launch['source_commit'] != plan['source_commit'] or launch['image'] != plan['image']):
        raise ValueError('capacity launch identity differs')
    check_container(observed, plan, container_id)
    if plan['wall_seconds'] != 3600 or plan['weight_updates'] != 0:
        raise ValueError('registered one-hour frozen capacity budget required')
    source = Path(plan['source_root'])
    expected = ['timeout', '--signal=TERM', '--kill-after=30s', '3600s', 'python3', '-B',
                '/source/' + RUNNER, '--root', '/mnt/cephfs/xingchao/datasets/audio2tool',
                '--model', MODEL, '--release', str(release),
                '--registry', '/mnt/cephfs/xingchao/datasets/audio2tool/tools_registry.csv',
                '--asset-audit', '/mnt/cephfs/xingchao/experiments/multimodal-jev/sfc-qwen3-omni-train6-v1/asset-audit.json',
                '--context-run', str(context_run), '--native-processor-run',
                '/mnt/cephfs/xingchao/experiments/multimodal-jev/audio2tool-native-processor-v1/run',
                '--output', '/output/run']
    if observed['Config']['Cmd'] != expected:
        raise ValueError('exact Jev capacity command required')
    binds = set(observed['HostConfig']['Binds'])
    if f'{source}:/source:ro' not in binds or f'{root}:/output:rw' not in binds:
        raise ValueError('capacity immutable source/output mounts required')
    for name, pin in plan['source_files'].items():
        if file_hash(source / name) != pin:
            raise ValueError('capacity source snapshot changed')
    # A pinned intent may not substitute an unrelated successful script. Require
    # the reviewed immutable preflight source's exact probe and dependencies.
    preflight = PREFLIGHT
    if file_hash(preflight / 'intent.json') != 'f5a3c9d312dea0c6a893af70c9074727d2f7ebcfb01c9bb2ab7a1b6b36022727':
        raise ValueError('accepted original probe source intent required')
    original = read(preflight / 'intent.json')
    for name, pin in original['source_files'].items():
        if plan['source_files'].get(name) != pin:
            raise ValueError('capacity probe source differs from accepted preflight')
    run = root / 'run'; result = read(run / 'result.json'); identity = read(run / 'identity.json')
    if (read(run / 'terminal.json') != {'exit_code':0} or result['status'] != 'passed'
            or result['model_loaded'] is not True or result['targets_read'] is not False or result['weight_updates'] != 0
            or result['identity_sha256'] != file_hash(run / 'identity.json')
            or identity['preflight_only'] is not False or identity['targets_read'] is not False or identity['weight_updates'] != 0
            or identity['engine'] != ENGINE or identity['versions'] != VERSIONS or identity['model'] != MODEL
            or identity['asset_audit_sha256'] != ASSET_AUDIT_SHA
            or identity['context_result_sha256'] != CONTEXT_SHA or identity['manifest_sha256'] != MANIFEST_SHA
            or identity['sampling'] != dict(temperature=0., max_tokens=1, n=1, seed=17)
            or identity['full_roster_records'] != 30733 or identity['boundary_cases'] != 12):
        raise ValueError('real complete frozen Jev capacity execution required')
    for name, pin in identity['sources'].items():
        if plan['source_files'].get(name) != pin:
            raise ValueError('capacity executed source changed')
    _, _, selected = boundaries(context_run, release, registry)
    cases = {case['case']:case for row, case in selected}
    if len(result['records']) != 12 or {r['case'] for r in result['records']} != set(cases):
        raise ValueError('all twelve real boundary responses required')
    for r in result['records']:
        c = cases[r['case']]
        if (r != read(run / (r['case'].replace(':', '-') + '.json'))
                or any(r[k] != c[k] for k in ('id', 'decision_id', 'input_tokens', 'input_ids_sha256'))
                or r['audio_count'] != len(c['audio_receipts']) or r['input_matches_processor'] is not True
                or len(r['token_ids']) != 1 or type(r['token_ids'][0]) is not int or r['token_ids'][0] < 0
                or type(r['raw_text']) is not str or r['finish_reason'] not in ('stop', 'length')
                or type(r['engine_seconds']) not in (int, float) or not math.isfinite(r['engine_seconds']) or r['engine_seconds'] <= 0):
            raise ValueError('exact one-token Jev capacity result differs')
    return dict(container_id=container_id, intent_sha256=intent_hash, inspection_sha256=inspection_hash,
                result_sha256=file_hash(run / 'result.json'), source_commit=plan['source_commit'])
