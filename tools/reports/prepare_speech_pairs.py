"""Compile pinned official TRAIN texts and the admitted real/synthetic roster into partner keys."""
import json
from pathlib import Path
from multimodal_jev.baseline import file_hash
from multimodal_jev.speech_alignment_pairs import build, wording


def checked(path, sha):
    path=Path(path)
    if file_hash(path)!=sha:
        raise ValueError('changed speech-pair source: '+path.name)
    return path


def compile_view(c):
    bank=json.loads(checked(c['bank'],c['bank_sha256']).read_bytes())
    manifest=Path(c['release'])/'MANIFEST.json'
    m=json.loads(checked(manifest,bank['pins']['clean_manifest']).read_bytes())
    learn=checked(manifest.with_name('learning.jsonl'),m['files']['learning.jsonl'])
    with learn.open() as stream:
        rows={x['id']:x for line in stream if (x:=json.loads(line))}
    texts={}
    with checked(c['annotations'],bank['pins']['annotations']).open() as stream:
        for line in stream:
            a=json.loads(line)
            for rec in a['recordings']:
                uid='slurp:'+rec['file']
                if uid in rows:
                    t=wording(a['sentence'])
                    if rows[uid]['intent']!=a['scenario']+'_'+a['action'] or texts.setdefault(uid,t)!=t:
                        raise ValueError('ambiguous real TRAIN annotation')
    source=Path(c['mixed']['data'])
    proof=json.loads(checked(source/'admission.json',c['mixed']['admission_sha256']).read_bytes())
    if proof['source_pins']['real']!=bank['pins']['clean_manifest']:
        raise ValueError('different real TRAIN release')
    for row in json.loads(checked(source/'synthetic.json',proof['roster_sha256']).read_bytes()):
        uid=row['id']
        if uid in rows:
            raise ValueError('duplicate training recording')
        rows[uid]=row
        texts[uid]=row['annotation_text']
    if len(rows)!=proof['population']:
        raise ValueError('incomplete TRAIN population')
    view=build(rows,texts,bank['semantic_rules'],c['semantic_sampling'])
    view['source']=dict(real=bank['pins']['clean_manifest'],annotations=bank['pins']['annotations'],
                        bank=c['bank_sha256'],synthetic=proof['roster_sha256'])
    return view,rows
