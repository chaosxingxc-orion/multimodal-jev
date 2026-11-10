#!/usr/bin/env python3
"""Independently check complete context-audit coverage after exact container exit."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from watch_qwen3_omni_probe import inspect


def verify(root):
    launch=json.loads((root/'launch.json').read_text());item=inspect(launch['container_id'])
    if (item['Id']!=launch['container_id'] or item['Image']!=launch['image']
            or item['State']['Status']!='exited' or item['State']['ExitCode']!=0
            or item['HostConfig'].get('Devices')):
        raise ValueError('exact successful CPU container required')
    run=root/'run';result=json.loads((run/'result.json').read_text());identity=json.loads((run/'identity.json').read_text())
    if (json.loads((run/'terminal.json').read_text())['exit_code']!=0 or result['status']!='passed'
            or file_hash(run/'identity.json')!=result['identity_sha256'] or identity['targets_read'] is not False
            or identity['context_limit']!=4096 or identity['output_budget']!=1024):
        raise ValueError('successful input-only audit identity required')
    for name,digest in identity['source_files'].items():
        if file_hash(root/'source'/name)!=digest:raise ValueError('audit source changed')
    release=Path(identity['release']);manifest=json.loads((release/'MANIFEST.json').read_text())
    if file_hash(release/'MANIFEST.json')!=identity['manifest_sha256']:raise ValueError('release changed')
    if set(result['splits'])!=set(manifest['splits']) or set(result['splits'])!={'train_id','test_id','test_ood'}:
        raise ValueError('complete audit splits required')
    reports={}
    for split,pin in manifest['splits'].items():
        requests=release/f'{split}.requests.jsonl';context=run/f'{split}.context.jsonl';summary=result['splits'][split]
        if file_hash(requests)!=pin['requests_sha256'] or file_hash(context)!=summary['records_sha256']:
            raise ValueError('input or audit records changed')
        groups=set();ids=set();count=0;overflow=0;minimum=None;maximum=0
        with requests.open() as source,context.open() as audited:
            for line in source:
                row=json.loads(line)
                if row['id'] in ids or row['split']!=split:raise ValueError('duplicate/wrong split request')
                ids.add(row['id']);groups.add(row['source_group'])
                for decision in row['input']['decisions']:
                    actual=json.loads(next(audited))
                    if (actual['id']!=row['id'] or actual['decision_id']!=decision['id']
                            or len(actual['audio_token_lengths'])!=len(row['audio_sha256'])
                            or len(actual['expected_frames'])!=len(row['audio_sha256'])
                            or actual['observed_frames']!=actual['expected_frames']
                            or any(type(n) is not int or n<=0 for n in actual['audio_token_lengths'])
                            or type(actual['input_tokens']) is not int or actual['input_tokens']<=0
                            or type(actual['context_overflow']) is not bool
                            or actual['context_overflow']!=(actual['input_tokens']+1024>4096)):
                        raise ValueError('decision coverage, audio or context evidence differs')
                    for key in ('prompt_sha256','input_ids_sha256'):
                        value=actual[key]
                        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                            raise ValueError('invalid prompt/token digest')
                    count+=1;overflow+=actual['context_overflow'];maximum=max(maximum,actual['input_tokens'])
                    minimum=actual['input_tokens'] if minimum is None else min(minimum,actual['input_tokens'])
            if next(audited,None) is not None:raise ValueError('extra audit decisions')
        observed=dict(bundles=len(ids),decisions=count,source_groups=len(groups),context_overflow=overflow,
                      min_input_tokens=minimum,max_input_tokens=maximum)
        if (len(ids)!=pin['bundles'] or count!=pin['decisions'] or len(groups)!=pin['source_groups']
                or any(summary[k]!=v for k,v in observed.items())):
            raise ValueError('full population summary differs')
        reports[split]=observed
    fits=all(r['context_overflow']==0 for r in reports.values())
    if result['all_inputs_fit'] is not fits:raise ValueError('context fit claim differs')
    return dict(root=str(root),container_id=item['Id'],result_sha256=file_hash(run/'result.json'),
        verifier_sha256=file_hash(Path(__file__)),splits=reports,all_inputs_fit=fits,
        scope='Independent exact decision coverage and context summary check; uses recorded processor evidence, '
              'does not recompute features or establish vLLM parity/generation quality.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=verify(a.root)
    with a.output.open('xb') as f:f.write(canonical(result))
    print(json.dumps(result,indent=2))
