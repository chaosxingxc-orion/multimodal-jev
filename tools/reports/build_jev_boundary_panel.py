"""Create a label-free, stable-prefix evaluation seal and separate paired scoring data."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from multimodal_jev.baseline import file_hash
from multimodal_jev.boundary_panel import split_panel
from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_decision_contract import PREFIX_LAYOUT, render_prefix_context_request
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.jev_train_confusions import pinned


def source_rows(c):
    cr,nr = (Path(c[k]['root']) for k in ('clean','native'))
    cm,nm = (json.loads(pinned(root/'MANIFEST.json',c[k]['sha256'])) for root,k in ((cr,'clean'),(nr,'native')))
    if cm['label_basis']!='official_clean_v7' or cm['allowed_optimization_role']!='learning':
        raise ValueError('admitted native training labels required')
    def lines(root,name,digest):
        return map(json.loads,pinned(root/name,digest).splitlines())
    clean = {r['id']:r for r in lines(cr,'learning.jsonl',cm['files']['learning.jsonl'])}
    if len(clean)!=cm['counts']['learning']['recordings']: raise ValueError('incomplete learning set')
    task=json.loads(pinned(cr/'tasks.json',cm['files']['tasks.json']))['intent']
    split=nm['splits']['train']
    rows=[]
    for a,t in zip(lines(nr,'train.requests.jsonl',split['requests_sha256']),
                   lines(nr,'train.targets.jsonl',split['targets_sha256']),strict=True):
        if a['id']!=t['id'] or a['split']!='train':raise ValueError('native metadata join mismatch')
        if a['id'] not in clean: continue
        r=clean.pop(a['id'])
        if (r['role']!='learning' or r['official_split']!='train' or r['known_pending_source']
                or a['audio_sha256']!=r['audio_sha256'] or
                any(r[k]!=t['target'][k] for k in ('scenario','action'))):
            raise ValueError('untrusted source label')
        m=t['metadata']
        rows.append(dict(id=r['id'],audio=r['audio'],audio_sha256=r['audio_sha256'],gold=r['intent'],
                         group=r['source_group'],source=a['source_group'],speaker=m['speaker_id'],
                         text=' '.join(m['transcript_reference'].lower().split())))
    if clean:raise ValueError('missing TRAIN metadata')
    return task,rows


def build(c,provenance):
    automatic = c['schema'] == 'jev-boundary-panel-auto-train-v1'
    if (c['schema'] not in ('jev-boundary-panel-v1','jev-boundary-panel-auto-train-v1')
            or c['arms']!=['original','boundary','fewshot','reversed_boundary']
            or (automatic and ('example_text' in c or 'examples' not in c))):
        raise ValueError('fixed paired experiment arms required')
    out=Path(c['output']);check_paths({'seal':out})
    if out.exists():raise FileExistsError(out)
    task,rows=source_rows(c)
    examples,selection,pairs=split_panel(rows,c)
    roster=[{k:r[k] for k in ('id','audio','audio_sha256')} for r,_ in selection]
    targets=[dict(id=r['id'],audio_sha256=r['audio_sha256'],gold=r['gold'],group=r['group'],speaker=r['speaker'],
                  stratum=s) for r,s in selection]
    out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as tmp:
        root=Path(tmp)/'seal';root.mkdir()
        files={}
        def write(rel,value):
            path=root/rel;path.parent.mkdir(exist_ok=True,parents=True)
            commit_json(path,value);files[rel]=file_hash(path)
        datasets={}
        for arm in c['arms']:
            candidates=[dict(id=x['id'],definition=x['description']) for x in task['schema']['candidates']]
            requirements=list(task['thinking']['requirements'])
            if arm!='original':
                for x in candidates:
                    if x['id'] in c['boundaries']:x['definition']+=' '+c['boundaries'][x['id']]
            if arm=='fewshot':
                requirements.extend('Example: '+json.dumps(r['text'])+' belongs to '+k+'.' for k,r in examples.items())
            if arm=='reversed_boundary':candidates.reverse()
            req=compile_candidates('audio',candidates,requirements,semantics='exclusive_event')
            system,history,readouts=render_prefix_context_request(req)
            panel=dict(system=system,readouts=readouts,questions=[asdict(q) for q in req.questions],
                       requirements=req.requirements,version=req.version,candidate_ids=[q.id for q in req.questions],
                       prompt_layout=PREFIX_LAYOUT,history=[])
            write(f'inference/{arm}/panel.json',panel)
            write(f'inference/{arm}/rows.json',roster)
            datasets[arm]=dict(candidates=len(candidates),recordings=len(roster),split='source_train_diagnostic')
        write('scoring/targets.json',targets);write('scoring/pairs.json',pairs)
        write('support.json',{k:{n:r[n] for n in ('id','group','source','speaker','audio_sha256')} for k,r in examples.items()})
        write('design.json',dict(clean_manifest=c['clean']['sha256'],native_manifest=c['native']['sha256'],provenance=provenance))
        manifest=dict(schema='jev-native-evaluation-seal-v1',prompt_layout=PREFIX_LAYOUT,datasets=datasets,
                      target_tuning_allowed=False,threshold=.5,files=files)
        if automatic:
            manifest.update(diagnostic='automatic_class_boundary', clean_sha256=c['clean']['sha256'])
        commit_json(root/'MANIFEST.json',manifest);root.rename(out)
    return dict(seal_sha256=file_hash(out/'MANIFEST.json'),recordings=len(roster),pairs=len(pairs),
                same_speaker_pairs=sum(p['same_speaker'] for p in pairs),strata=dict(Counter(s for _,s in selection)),
                model_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();print(json.dumps(build(*load_config(a.config,a.local)),indent=2))
