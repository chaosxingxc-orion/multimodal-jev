#!/usr/bin/env python3
"""Independent full-population reconstruction of new official intent release."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from multimodal_jev.baseline import file_hash
from multimodal_jev.slurp_training import content_hash


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


def verify(root, expected_hash):
    root = Path(root); manifest = read(root/'MANIFEST.json')
    if file_hash(root/'MANIFEST.json') != expected_hash or manifest['schema'] not in ('slurp-official-clean-intent-v1','slurp-official-clean-intent-v2'):
        raise ValueError('explicit publication identity required')
    if manifest['schema']=='slurp-official-clean-intent-v2':
        # Share only path/hash resolution, keep the independent population and
        # native-label reconstruction below separate from the production loader.
        from multimodal_jev.slurp_portable_release import config as resolve
        config,_,_=resolve(root,expected_hash)
    else:
        for name,digest in manifest['files'].items():
            if Path(name).name!=name or file_hash(root/name)!=digest:
                raise ValueError('publication file changed')
        config=read(root/'publication-config.json')
        for path,digest in config['pins'].items():
            if file_hash(path)!=digest:raise ValueError('dependency changed')
    source=Path(config['source'])
    original_manifest = read(source/'MANIFEST.json')
    pairs = read(root/'tasks.json')['intent']['candidate_pairs']
    quarantine = read(root/'quarantine.json')
    if len({r['id'] for r in quarantine}) != len(quarantine):
        raise ValueError('duplicate quarantine identity')
    groups = {}; waves = {}; summaries = {}
    for role in ('learning', 'selection'):
        originals = rows(source/(role+'.requests.jsonl'))
        native = {r['id']: r for r in rows(source/(role+'.targets.jsonl'))}
        for suffix in ('requests', 'targets'):
            name = role+'.'+suffix+'.jsonl'
            if file_hash(source/name) != original_manifest['files'][name]:
                raise ValueError('original source changed')
        packet_path, audit_path = config['reviews'][role]
        packet, audit = read(packet_path), read(audit_path)
        if file_hash(packet_path) != audit['packet_sha256']:
            raise ValueError('audit packet changed')
        pending = ({packet[int(i)]['request']['source_group'] for i in audit['pending']} if role == 'learning'
                   else {r['source_group'] for r in audit['records'] if r['status'] == 'pending'})
        excluded = {r['id']: r for r in quarantine if r['role'] == role}
        expected_pending = {r['id'] for r in originals if r['source_group'] in pending}
        if expected_pending != {uid for uid, r in excluded.items() if r['reason'] == 'known_pending_source'}:
            raise ValueError('pending source mask differs')
        if any(r['reason'] not in ('known_pending_source', 'invalid_audio_asset') for r in excluded.values()):
            raise ValueError('unregistered exclusion')
        actual = rows(root/(role+'.jsonl')); assets = rows(root/(role+'.audio.jsonl'))
        lookup = {r['id']: r for r in originals}
        if (len({r['id'] for r in actual}) != len(actual) or len(assets) != len(actual)
                or {r['id'] for r in actual} != lookup.keys()-excluded.keys()):
            raise ValueError('complete population differs')
        for row, asset in zip(actual, assets, strict=True):
            req, target = lookup[row['id']], native[row['id']]
            pair = [target['native_target']['scenario'], target['native_target']['action']]
            if (pairs[row['intent']] != pair or [row['scenario'], row['action']] != pair
                    or row['source_request_sha256'] != content_hash(req)
                    or row['source_target_sha256'] != content_hash(target)
                    or target['request_sha256'] != content_hash(req)
                    or row['role'] != role or row['official_split'] != 'train'
                    or row['source_group'] != req['source_group'] or row['audio'] != req['input']['audio']
                    or row['audio_sha256'] != req['audio_sha256'] or row['known_pending_source']
                    or asset['id'] != row['id'] or asset['audio_sha256'] != row['audio_sha256']
                    or asset['sample_rate'] != 16000 or asset['samples'] <= 0 or asset['peak'] <= 0):
                raise ValueError('native intent/source/audio reconstruction differs')
            if groups.setdefault(row['source_group'], role) != role or waves.setdefault(asset['waveform_sha256'], role) != role:
                raise ValueError('cross-role overlap')
        observed = dict(recordings=len(actual), source_groups=len({r['source_group'] for r in actual}),
                        classes=dict(sorted(Counter(r['intent'] for r in actual).items())),
                        audio_hours=sum(r['seconds'] for r in assets)/3600)
        if (any(observed[k]!=manifest['counts'][role][k] for k in ('recordings','source_groups','classes'))
                or not math.isclose(math.fsum(r['seconds'] for r in assets)/3600,
                    manifest['counts'][role]['audio_hours'],rel_tol=0,abs_tol=1e-9)):
            raise ValueError('reported counts differ')
        summaries[role] = observed
    return dict(passed=True, manifest_sha256=expected_hash, counts=summaries,
                excluded=len(quarantine), cross_role_source_or_decoded_audio_overlap=0,
                implementation_sha256=file_hash(__file__),
                scope='Independent native target/population reconstruction and saved audio receipt verification; '
                      'does not repeat waveform decode or certify semantic listening/NPU/training.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('immutable independent verification exists')
    result = verify(a.root, a.manifest_sha256)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'counts'}, indent=2))
