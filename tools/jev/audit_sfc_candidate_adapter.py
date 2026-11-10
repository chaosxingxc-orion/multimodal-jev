#!/usr/bin/env python3
"""Check full static parent parity and fixed-context preservation for prompt candidates."""
import argparse
import hashlib
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_jev_runtime import INSTRUCTION,messages as baseline
from multimodal_jev.sfc_prompt_candidates import messages


def audit(release):
    manifest=json.loads((release/'MANIFEST.json').read_text());reports={}
    if set(manifest['splits'])!={'train_id','test_id','test_ood'}:raise ValueError('all official splits required')
    for split,pin in manifest['splits'].items():
        path=release/f'{split}.requests.jsonl'
        if file_hash(path)!=pin['requests_sha256']:raise ValueError('input changed')
        count=0;ids=set();groups=set();digest=hashlib.sha256()
        for line in path.open():
            row=json.loads(line);audio=row['input']['context']['audios']
            if row['split']!=split or row['id'] in ids:raise ValueError('duplicate or wrong-split request')
            ids.add(row['id']);groups.add(row['source_group'])
            for d in row['input']['decisions']:
                old=baseline(row,d['id'],audio);parent=messages(row,d['id'],audio,INSTRUCTION)
                if canonical(parent)!=canonical(old):raise ValueError('parent prompt differs')
                changed=messages(row,d['id'],audio,'ENGINEERING PREFIX SUBSTITUTION FIXTURE.')
                if (changed[1:]!=old[1:] or changed[0]['role']!='system'
                        or changed[0]['content']!='ENGINEERING PREFIX SUBSTITUTION FIXTURE.\n\n'+old[0]['content'][len(INSTRUCTION):]):
                    raise ValueError('candidate changed fixed context')
                digest.update(canonical(parent));count+=1
        if len(ids)!=pin['bundles'] or len(groups)!=pin['source_groups'] or count!=pin['decisions']:
            raise ValueError('full coverage required')
        reports[split]=dict(bundles=len(ids),decisions=count,source_groups=len(groups),parent_messages_sha256=digest.hexdigest())
    return dict(release=str(release.resolve()),manifest_sha256=file_hash(release/'MANIFEST.json'),splits=reports,
        adapter_sha256=file_hash(Path(__file__).parents[2]/'src/multimodal_jev/sfc_prompt_candidates.py'),
        auditor_sha256=file_hash(Path(__file__)),
        scope='Complete static parent-byte parity and synthetic generic-prefix substitution. No teacher response, '
              'waveform processing, tokenizer, generation or quality evaluation.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.release)
    with a.output.open('xb') as f:f.write(canonical(result))
    print(json.dumps(result,indent=2))
