#!/usr/bin/env python3
"""Complete static input/adapter coverage and recursive gold-conversion audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_jev import restore_bundle
from multimodal_jev.sfc_jev_runtime import messages
from multimodal_jev.sfc_jev_score import decode, native_text
from multimodal_jev.sfc import call_multiset, parse_calls


def audit(release):
    manifest=json.loads((release/'MANIFEST.json').read_text()); reports={}
    for split,pin in manifest['splits'].items():
        for kind in ('requests','targets'):
            if file_hash(release/f'{split}.{kind}.jsonl')!=pin[kind+'_sha256']:
                raise ValueError('release changed')
        counts=Counter(); hashes=hashlib.sha256(); maximum_chars=0
        with (release/f'{split}.requests.jsonl').open() as ri, (release/f'{split}.targets.jsonl').open() as ti:
            for rline,tline in zip(ri,ti,strict=True):
                request,gold=json.loads(rline),json.loads(tline)
                native=restore_bundle(request,gold)
                context=request['input']['context']; paths=context['audios']
                for decision in request['input']['decisions']:
                    rendered=messages(request,decision['id'],paths)
                    if [m['role'] for m in rendered]!=['system','user']+(['assistant','user'] if len(paths)==2 else []):
                        raise ValueError('conversation role loss')
                    actual=[c['audio'] for m in rendered if isinstance(m['content'],list) for c in m['content'] if c['type']=='audio']
                    if actual!=paths:raise ValueError('audio order changed')
                    if len(paths)==2 and rendered[2]!=context['messages'][1]:raise ValueError('supplied history changed')
                    encoded=canonical(rendered);hashes.update(encoded);maximum_chars=max(maximum_chars,len(encoded.decode()))
                    result=decode(decision,json.dumps(gold['answers'][decision['id']],ensure_ascii=False))
                    if not result['schema_valid']:raise ValueError((request['id'],decision['id'],result['errors']))
                    counts['decisions']+=1;counts[decision['decision_type']]+=1
                calls=native['supervision']['calls']
                if call_multiset(parse_calls(native_text(calls)))!=call_multiset(calls):raise ValueError('native bridge loss')
                counts['bundles']+=1;counts['two_audio_bundles']+=len(paths)==2
        if counts['bundles']!=pin['bundles'] or counts['decisions']!=pin['decisions']:
            raise ValueError('incomplete release coverage')
        reports[split]=dict(counts=dict(counts),rendered_messages_sha256=hashes.hexdigest(),maximum_message_chars=maximum_chars)
    return dict(release=str(release.resolve()),manifest_sha256=file_hash(release/'MANIFEST.json'),splits=reports,
        code_sha256={name:file_hash(Path(__file__).parents[2]/name) for name in
                     ('src/multimodal_jev/sfc_jev_runtime.py','src/multimodal_jev/sfc_jev_score.py','tools/jev/audit_sfc_jev_runtime.py')},
        scope='Full static message mapping and recursive conversion-schema audit. Audio paths stand in for waveforms; '
              'no tokenizer/processor, model inference, performance estimate or held-out prompt selection.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.release)
    with a.output.open('xb') as f:f.write(canonical(result))
    print(json.dumps(result,indent=2))
