#!/usr/bin/env python3
"""Static full-release Omni template audit; no weights, decoding or gold scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from jinja2 import Environment
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_omni import messages


def audit(release, template_path):
    manifest=json.loads((release/'MANIFEST.json').read_text())
    template=Environment().from_string(json.loads(template_path.read_text())['chat_template'])
    reports={}
    for split,pin in manifest['splits'].items():
        path=release/(split+'.jsonl')
        if file_hash(path)!=pin['sha256']:raise ValueError('release changed')
        groups=set();levels=Counter();audio_counts=Counter();digest=hashlib.sha256();ids=set()
        for line in path.open():
            record=json.loads(line)
            # Gold is never supplied to message construction or the template.
            inputs=record['input'];paths=inputs['audios'];turns=messages(inputs,paths)
            rendered=template.render(messages=turns,add_generation_prompt=True)
            if rendered.count('<|audio_start|><|audio_pad|><|audio_end|>')!=len(paths):
                raise ValueError('audio placeholder loss or duplication')
            if not rendered.endswith('<|im_start|>assistant\n'):
                raise ValueError('unexpected generation prefix')
            for t in inputs['messages']:
                if t['content'] not in rendered:raise ValueError('official text or history changed by template')
            if len(paths)==2:
                first=rendered.index('<|audio_start|>');last=rendered.rindex('<|audio_start|>')
                history=rendered.index(inputs['messages'][1]['content'],first)
                if not first<history<last:raise ValueError('history chronology changed')
            uid=record['example_id']
            if uid in ids:raise ValueError('duplicate example')
            ids.add(uid);groups.add(record['metadata']['source_group'])
            levels[record['metadata']['level']]+=1;audio_counts[len(paths)]+=1
            digest.update(canonical(dict(example_id=uid,prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),audios=paths)))
        if len(ids)!=pin['records'] or len(groups)!=pin['source_groups']:raise ValueError('incomplete official split')
        reports[split]=dict(records=len(ids),source_groups=len(groups),by_level=dict(levels),
            by_audio_count=dict(audio_counts),rendered_input_inventory_sha256=digest.hexdigest(),manifest_sha256=pin['sha256'])
    return dict(schema='sfc-omni-static-input-audit-v1',release=str(release.resolve()),
        release_id=manifest['release_id'],template=str(template_path.resolve()),template_sha256=file_hash(template_path),
        adapter_sha256=file_hash(Path(__file__).parents[2]/'src/multimodal_jev/sfc_omni.py'),splits=reports,
        history_policy='official supplied assistant history, unchanged; not generated agent interaction',
        scope='Static Jinja rendering on complete published inputs. No processor feature/token expansion, context-fit validation, waveform loading, model inference or semantic scores.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('release','template','output'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args();report=audit(args.release,args.template)
    with args.output.open('xb') as f:f.write(canonical(report))
    print(json.dumps(report,indent=2))
