#!/usr/bin/env python3
"""Independent full source-to-derived comparison, without conversion helpers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical


def verify(release):
    manifest = json.loads((release/'MANIFEST.json').read_text())
    source = Path(manifest['identity']['source'])
    assert file_hash(source/'MANIFEST.json') == manifest['identity']['source_manifest_sha256']
    native = json.loads((source/'MANIFEST.json').read_text())
    reports = {}
    assert set(manifest['splits']) == set(native['splits']) == {'train_id','test_id','test_ood'}
    for split, pin in manifest['splits'].items():
        for root, pins in ((release, pin), (source, native['splits'][split])):
            for kind in ('requests','targets'):
                assert file_hash(root/f'{split}.{kind}.jsonl') == pins[kind+'_sha256']
        seen = set(); groups = set(); count = 0; questions = 0
        with (source/f'{split}.requests.jsonl').open() as nr, (source/f'{split}.targets.jsonl').open() as nt, \
                (release/f'{split}.requests.jsonl').open() as jr, (release/f'{split}.targets.jsonl').open() as jt:
            for lines in zip(nr,nt,jr,jt,strict=True):
                n,t,r,g = map(json.loads,lines)
                assert r['id'] == n['id'] == t['id'] == g['id'] and r['id'] not in seen
                assert set(r) == {'id','split','source_group','audio_sha256','input','protocol'}
                assert set(r['input']) == {'context','decisions'}
                assert r['input']['context'] == n['input']
                for field in ('split','source_group','audio_sha256'): assert r[field] == n[field]
                assert r['split'] == split
                assert g['native_target'] == t
                assert g['request_sha256'] == hashlib.sha256(canonical(r)).hexdigest()
                assert t['request_sha256'] == hashlib.sha256(canonical(n)).hexdigest()
                functions = json.loads(n['input']['messages'][0]['content'].split('### Available Functions:\n')[1].split('\n\n### Environment:\n')[0])
                decisions = r['input']['decisions']
                assert len(decisions) == 3*len(functions)
                assert set(g['answers']) == {d['id'] for d in decisions}
                restored = []
                for index, f in enumerate(functions):
                    ds = decisions[3*index:3*index+3]
                    assert [d['id'] for d in ds] == [f'f{index:02d}.{s}' for s in ('present','absent','calls')]
                    assert [d['decision_type'] for d in ds] == ['noul','noul','typed_call_list']
                    assert all(d['function'] == f for d in ds)
                    assert ds[2]['value_schema'] == dict(type='array',items=f['parameters'])
                    args = [c['arguments'] for c in t['supervision']['calls'] if c['name'] == f['name']]
                    a,b,c = [g['answers'][d['id']] for d in ds]
                    assert a is bool(args) and b is (not args)
                    # Canonical bytes distinguish bool/int and numeric/string values.
                    assert canonical(c) == canonical(args)
                    restored.extend(dict(name=f['name'],arguments=value) for value in c)
                assert Counter(map(canonical,restored)) == Counter(map(canonical,t['supervision']['calls']))
                seen.add(r['id']); groups.add(r['source_group']); count += 1; questions += len(decisions)
        assert count == pin['bundles'] == native['splits'][split]['requests']
        assert questions == pin['decisions'] and len(groups) == pin['source_groups']
        reports[split] = dict(bundles=count,decisions=questions,source_groups=len(groups))
    return dict(release=str(release.resolve()),manifest_sha256=file_hash(release/'MANIFEST.json'),
        verifier_sha256=file_hash(Path(__file__)),splits=reports,
        scope='Full independent source/derived input, provenance, type-sensitive answer and call-multiset comparison. No model inference.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--release',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); result=verify(a.release)
    with a.output.open('xb') as f: f.write(canonical(result))
    print(json.dumps(result,indent=2))
