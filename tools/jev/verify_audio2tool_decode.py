#!/usr/bin/env python3
"""CPU-only full-decode audit using coordinator-captured Docker inspections.

The coordinator must freshly inspect all four exact containers and pin the
evidence file before invoking this tool. A JSON file alone is not a trust root.
"""
import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path
import tempfile

from multimodal_jev.audio2tool_acceptance import check_container
from multimodal_jev.audio2tool_decode_acceptance import audit_records
from multimodal_jev.audio2tool_engine import ASSET_AUDIT_SHA, VERSIONS, prior_records
from multimodal_jev.audio2tool_inference import require_capacity as admit_native, write_once
from multimodal_jev.audio2tool_jev_capacity import MODEL, require_capacity as admit_jev
from multimodal_jev.baseline import file_hash

PREPARATION_SHA='10d876c333114956f8a952e062511f720ba7755c55a206ab4a78cb048b48c6f6'
PREFLIGHT_INTENT='e6c8839e63fd766641f41f06ec5c3337aa6f6516067a126128a8bcd79126ea6b'
PREFLIGHT_RESULT='01d64292294d6912a47cd46bf3abc3ad4e679320625c3bf6a13cddd926cba50f'
PREFLIGHT_ID='fdf7a9a736b0be8b40951908b886f1121c98898d199c7d9527fb8e421511aee1'


def read(p): return json.loads(Path(p).read_text())


def verify_preflight(item, plan):
    argv=plan['argv'];host=item['HostConfig'];state=item['State']
    if (item['Id']!=PREFLIGHT_ID or item['Image']!=plan['image']
            or state['Status']!='exited' or state['ExitCode']!=0 or state['OOMKilled']
            or host['NanoCpus']!=2*10**9 or host['Memory']!=20*1024**3
            or host['NetworkMode']!='none' or host['Devices'] or host['Privileged']
            or item['Config']['Cmd']!=argv[argv.index(plan['image'])+1:]
            or set(host['Binds'])!={argv[i+1] for i,v in enumerate(argv) if v=='-v'}
            or not all(argv[i+1] in item['Config']['Env'] for i,v in enumerate(argv) if v=='-e')):
        raise ValueError('actual successful original CPU preflight required')


def check_intent(root, prep, intent, launch):
    argv=prep['argv']+['--capacity-evidence',str(root/'capacity-evidence.json'),
                     '--capacity-evidence-hash',file_hash(root/'capacity-evidence.json')]
    if (intent!=dict(prep,argv=argv) or launch['intent_sha256']!=file_hash(root/'intent.json')
            or launch['image']!=prep['image'] or launch['source_commit']!=prep['source_commit']
            or launch['physical_devices']!=[6,7]):
        raise ValueError('exact prepared profile plus pinned capacity evidence required')


def capacities(root, snapshots, prep):
    """Recheck both captured-at-dispatch and fresh actual capacity containers."""
    base=root.parent;proof=read(root/'capacity-evidence.json')
    expected=dict(native=base/'audio2tool-engine-capacity-v1',jev=base/'audio2tool-jev-engine-capacity-v1')
    saved={};fresh={}
    with tempfile.TemporaryDirectory(prefix='decode-final-capacity-') as tmp:
        for family,prior in expected.items():
            p=proof[family];launch=read(prior/'launch.json');cid=launch['container_id']
            if p['root']!=str(prior) or p['container_id']!=cid:
                raise ValueError('original capacity root and exact container required')
            original=Path(p['inspection'])
            if original!=root/(family+'-capacity-inspection.json') or file_hash(original)!=p['inspection_sha256']:
                raise ValueError('dispatch capacity inspection changed')
            current=Path(tmp)/(family+'.json');write_once(current,snapshots[family])
            def admit(path,pin):
                if family=='native':
                    return admit_native(prior,path,pin,cid,base/'audio2tool-native-processor-v1/run')
                return admit_jev(prior,p['intent_sha256'],path,pin,cid,base/'audio2tool-jev-context-v1/run',
                    Path(prep['argv'][prep['argv'].index('--release')+1]),
                    Path(prep['argv'][prep['argv'].index('--registry')+1]).read_bytes())
            saved[family]=admit(original,p['inspection_sha256'])
            fresh[family]=admit(current,file_hash(current))
            if {k:v for k,v in saved[family].items() if k!='inspection_sha256'}!={k:v for k,v in fresh[family].items() if k!='inspection_sha256'}:
                raise ValueError('fresh capacity evidence differs from dispatch admission')
    if read(root/'run/identity.json')['capacity']!=saved:
        raise ValueError('actual profile capacity admission differs')
    return fresh


def envelope(root,evidence,evidence_hash):
    if file_hash(evidence)!=evidence_hash or file_hash(root/'preparation.json')!=PREPARATION_SHA:
        raise ValueError('externally pinned inspections and original profile preparation required')
    snapshots=read(evidence)
    if set(snapshots)!={'profile','preflight','native','jev'}:
        raise ValueError('all four fresh actual Docker inspections required')
    prep=read(root/'preparation.json');intent=read(root/'intent.json');launch=read(root/'launch.json')
    check_intent(root,prep,intent,launch)
    check_container(snapshots['profile'],intent,launch['container_id'])
    preflight=root.parent/'audio2tool-decode-preflight-v1'
    if file_hash(preflight/'intent.json')!=PREFLIGHT_INTENT or file_hash(preflight/'run/result.json')!=PREFLIGHT_RESULT:
        raise ValueError('original accepted real-processor preflight required')
    before=read(preflight/'intent.json');verify_preflight(snapshots['preflight'],before)
    if (prep['source_commit']!=before['source_commit'] or prep['source_files']!=before['source_files']
            or prep['source_root']!=str(preflight/'source')):
        raise ValueError('actual profile must use original preflight source')
    for name,pin in prep['source_files'].items():
        if file_hash(Path(prep['source_root'])/name)!=pin: raise ValueError('profile source changed')
    identity=read(root/'run/identity.json')
    for name,pin in identity['sources'].items():
        if prep['source_files'].get(name)!=pin: raise ValueError('executed source differs')
    for path,pin in identity['input_pins'].items():
        if file_hash(Path(path))!=pin: raise ValueError('profile input changed')
    return prep,preflight,capacities(root,snapshots,prep)


def verify(root,evidence,evidence_hash):
    prep,preflight,accepted_capacities=envelope(root,evidence,evidence_hash)
    if {k:importlib.metadata.version(k) for k in VERSIONS}!=VERSIONS:
        raise ValueError('fixed audit tokenizer runtime required')
    _,processor_identity=prior_records(root.parent/'audio2tool-native-processor-v1/run')
    for name,pin in processor_identity['processor_assets'].items():
        if file_hash(Path(MODEL)/name)!=pin: raise ValueError('original processor asset changed')
    audit=root.parent/'sfc-qwen3-omni-train6-v1/asset-audit.json'
    identity=read(root/'run/identity.json')
    if (file_hash(audit)!=ASSET_AUDIT_SHA or identity['asset_audit_sha256']!=ASSET_AUDIT_SHA
            or identity['model_files']!=read(audit)['files'] or identity['model']!=MODEL):
        raise ValueError('registered full model asset manifest required')
    from transformers import Qwen3OmniMoeProcessor
    processor=Qwen3OmniMoeProcessor.from_pretrained(MODEL,local_files_only=True)
    if file_hash(inspect.getfile(type(processor)))!=processor_identity['hf_processor_source_sha256']:
        raise ValueError('original HF processor implementation required')
    records=audit_records(root/'run',preflight/'run',lambda ids:processor.tokenizer.decode(ids,skip_special_tokens=True))
    return dict(schema='audio2tool-decode-profile-acceptance-v1',root=str(root),
        container_id=read(root/'launch.json')['container_id'],image=prep['image'],source_commit=prep['source_commit'],
        evidence_sha256=evidence_hash,intent_sha256=file_hash(root/'intent.json'),
        capacity_revalidation=accepted_capacities,records_audit=records,verifier_sha256=file_hash(__file__),
        weights_loaded=False,targets_read=False,weight_updates=0,
        limitations='Coordinator must obtain actual inspections of all four exact containers. Full model weights and waveforms are not reloaded; authenticated pinned execution verified them. Boundary observations do not estimate population throughput or task quality.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','evidence','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--evidence-hash',required=True);a=p.parse_args()
    if a.output.exists(): raise FileExistsError('preserve prior acceptance')
    write_once(a.output,verify(a.root,a.evidence,a.evidence_hash))
