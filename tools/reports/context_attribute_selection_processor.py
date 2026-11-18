"""Reuse hash-pinned population audit for the frozen held-out population.

Only the population count and declared scope differ from the TRAIN auditor.
No labels, model weights, metrics or selection decisions are consumed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import types

AUDIT_SHA='aa866c7d4c1109d2786ea4002edca61ae2c59f1cce157001d31118ec52d0d635'
VERIFY_SHA='98abacdf71cedaff5331c5729d57f73b983e97a6e2d4bec8ad4e62cb422f12e1'
SELECTION_SHA='85d17ce7a12d2e39efb9bdc676cdb24b02f2531ccb4c93b7adeba303485d0710'
RELEASE_SHA='38e4273d9328e4f02db081315e46f97a17628b974be7ba175254882e95421482'
RELEASE='/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/official-clean-intent-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_package(package):
    package=Path(package)
    pins=json.loads((package/'FILES.json').read_text())
    for name,h in pins.items():
        if sha(package/name)!=h:raise ValueError('Packaged source changed')
    candidate=package/'candidate-manifest.json'
    if sha(candidate)!=SELECTION_SHA:raise ValueError('Frozen selection manifest changed')
    m=json.loads(candidate.read_text())
    if m['release_manifest_sha256']!=RELEASE_SHA or m['recordings']!=9226:
        raise ValueError('Selection population differs')
    for name in ('inputs.jsonl','catalog.json'):
        if sha(package/name)!=m['files'][name]:raise ValueError('Selection input binding differs')
    if sha(package/'audio.jsonl')!=m['source_files']['selection.audio.jsonl']:
        raise ValueError('Selection audio population differs')
    return m


def derive(source_root, kind):
    """Narrow adaptations leave the original audit/verification files intact."""
    if kind not in ('audit','verify'):raise ValueError('Known audit module required')
    path=Path(source_root)/f'tools/reports/{kind}_context_attribute_population.py'
    if sha(path)!={'audit':AUDIT_SHA,'verify':VERIFY_SHA}[kind]:
        raise ValueError('Pinned population implementation changed')
    source=path.read_text()
    if kind=='audit':
        replacements=[('byid=validate_sources(inputs,metadata,40513)', 'byid=validate_sources(inputs,metadata,9226)'),
            ('total=40513','total=9226'),
            ('Actual complete clean TRAIN CPU inputs and lossless semantic fields. No selection, model, NPU capacity or efficacy admission.',
             'Actual complete clean selection CPU inputs and lossless semantic fields. No model, NPU capacity, score or efficacy admission.')]
    else:
        replacements=[("t['terminal']['completed']!=40513","t['terminal']['completed']!=9226"),
            ("source=Path(candidate['release'])",'source=Path('+repr(RELEASE)+')'),
            ("candidate['release_sha256']","candidate['release_manifest_sha256']"),
            ("source_manifest['files']['learning.audio.jsonl']","source_manifest['files']['selection.audio.jsonl']"),
            ('frozen TRAIN population','frozen selection population'),
            ('expected_count=40513','expected_count=9226'),
            ('Full TRAIN population,','Full selection population,')]
    for old,new in replacements:
        if source.count(old)!=1:raise ValueError('Population adaptation anchor changed: '+old)
        source=source.replace(old,new)
    module=types.ModuleType('attribute_selection_'+kind)
    module.__file__=str(path)
    exec(compile(source,str(path)+'[selection]','exec'),module.__dict__)
    return module


def run(package, output, mirror_root):
    package=Path(package);validate_package(package)
    # Publish verified replicas on local /data; Ceph is mounted readonly.
    from multimodal_jev.local_audio_mirror import LocalAudioMirror
    from multimodal_jev.baseline import atomic_json
    from multimodal_jev.ordered_prefetch import OrderedPrefetch
    mirror=LocalAudioMirror(mirror_root)
    rows=[json.loads(s) for s in (package/'inputs.jsonl').read_text().splitlines()]
    def materialize(row):
        path=mirror.publish(row['audio'],row['audio_sha256'])
        return path.stat().st_size
    total=0
    with OrderedPrefetch(materialize,rows,workers=2,ahead=4) as results:
        for count,size in enumerate(results,1):
            total+=size
            if count%256==0:atomic_json(Path(output)/'mirror-progress.json',dict(completed=count,total=9226,bytes=total))
    if count!=9226:raise ValueError('Incomplete selection audio')
    atomic_json(Path(output)/'mirror-result.json',dict(recordings=count,bytes=total,verified_encoded_audio=True))
    return derive(package,'audit').run(package,output,mirror_root)


def verify(collection, package, transport, anchors, source_root):
    validate_package(package)
    return derive(source_root,'verify').verify(collection,package,transport,anchors)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mirror-root');p.add_argument('--collection');p.add_argument('--transport');p.add_argument('--anchors')
    p.add_argument('--source-root',default='.')
    a=p.parse_args()
    if a.collection:
        result=verify(a.collection,a.package,a.transport,a.anchors,a.source_root)
        with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    else:run(a.package,a.output,a.mirror_root)
