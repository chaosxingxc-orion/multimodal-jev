"""Create portable, inference/gold-separated semantic Jev evaluation releases."""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import io
import json
from math import gcd
from pathlib import Path
import shutil
import subprocess
import tempfile
import os

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_decision_contract import ApplicabilityQuestion, ContextDecisionRequest, PREFIX_LAYOUT
from multimodal_jev.local_config import load_config


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n')


def source_files(c,name,pattern):
    spec=c['sources'][name]; root=Path(c['datasets'])/spec['directory']
    clone=Path(c['clones'])/spec['checkout']
    import fnmatch
    output=subprocess.check_output(['git','-C',clone,'ls-tree','-r','--name-only',spec['revision']],text=True)
    files=sorted(n for n in output.splitlines() if fnmatch.fnmatch(n,pattern))
    if not files:raise ValueError('no source shards')
    for rel in files:
        pointer=subprocess.check_output(['git','-C',clone,'show',spec['revision']+':'+rel],text=True)
        if ('oid sha256:'+file_hash(root/rel)) not in pointer.splitlines():raise ValueError('unbound source shard')
        yield root/rel


def prepare_banking(c, spec, root):
    """Bind speaker and intent from the same published Audio TEST record."""
    import pyarrow.parquet as pq
    import soundfile as sf

    view = spec['eval']
    dataset = view['dataset']
    source_root = Path(c['datasets']) / spec['directory']
    files = {kind: source_root / entry['path'] for kind, entry in view['files'].items()}
    digests = {kind: file_hash(path) for kind, path in files.items()}
    if any(digests[kind] != entry['sha256'] for kind, entry in view['files'].items()):
        raise ValueError('pinned audio source or ontology changed')
    with files['ontology'].open(encoding='utf-8-sig', newline='') as stream:
        classes = sorted({row['intent'] for row in csv.DictReader(stream)})
    if len(classes) != view['classes'] or any(not name.strip() for name in classes):
        raise ValueError('invalid published audio ontology')
    panel = dict(version='jev-semantic-answer-v1', requirements=[view['question']],
                 questions=[asdict(ApplicabilityQuestion(
                     label, label.replace('_', ' ').lower(), view['decision_question']))
                            for label in classes])
    ph = hashlib.sha256(json.dumps(panel, sort_keys=True).encode()).hexdigest()
    rows, gold, ids, hashes, rates = [], [], set(), set(), Counter()
    path = files['test']
    sample = pq.ParquetFile(path)
    audio_dir = root / 'audio'
    audio_dir.mkdir()
    for batch in sample.iter_batches(batch_size=16, columns=['audio', 'intent', 'user_id']):
        for row in batch.to_pylist():
            audio = row['audio']
            uid, label, speaker = audio['path'], row['intent'], row['user_id']
            raw = audio['bytes']
            digest = hashlib.sha256(raw).hexdigest()
            if (not uid or uid in ids or digest in hashes or label not in classes
                    or type(speaker) is not int):
                raise ValueError('duplicate recording or invalid official intent/speaker')
            info = sf.info(io.BytesIO(raw))
            if (info.format != 'WAV' or info.samplerate != 16000
                    or info.channels != 1 or info.frames <= 0):
                raise ValueError('unexpected audio format')
            ids.add(uid)
            hashes.add(digest)
            rates[info.samplerate] += 1
            rel = 'audio/' + digest + '.wav'
            (root / rel).write_bytes(raw)
            rows.append(dict(id=uid, audio=rel, audio_sha256=digest, panel=ph))
            gold.append(dict(id=uid, audio_sha256=digest, gold=label, group=str(speaker)))
    groups = len({item['group'] for item in gold})
    if (sample.metadata.num_rows, len(rows), groups) != tuple(view['population']) or set(
            item['gold'] for item in gold) != set(classes):
        raise ValueError('recording population, speaker roster or class coverage changed')
    dump(root / 'inference' / dataset / 'panels.json', {ph: panel})
    dump(root / 'inference' / dataset / 'rows.json', rows)
    dump(root / 'scoring' / dataset / 'targets.json', gold)
    dump(root / 'MANIFEST.json', dict(
        schema='jev-semantic-evaluation-seal-v1', prompt_layout=PREFIX_LAYOUT,
        threshold=.5, target_tuning_allowed=False,
        datasets={dataset: dict(split=view['split'], recordings=len(rows), candidates=len(classes),
                                group_count=groups, group_unit='speaker')},
        source=dict(repo=spec['id'], revision=spec['revision'], shards={
            entry['path']: digests[kind] for kind, entry in view['files'].items()},
                    source_rows=sample.metadata.num_rows, source_rates=dict(rates)),
        files={str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob('*.json'))}))


def prepare(c,name,root):
    if c['sources'][name]['eval'].get('format') == 'wolbanking_audio_test':
        return prepare_banking(c, c['sources'][name], root)
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly
    sib=name=='sib';spec=c['sources'][name]; view=spec['eval']; dataset=view['dataset']
    rows=[];gold=[];panels={};rates=Counter();inputs={};seen=set(); nsource=0
    for p in source_files(c,name,view['shards']):
        pf=pq.ParquetFile(p)
        inputs[p.name]=file_hash(p)
        if sib:classes=json.loads(pf.schema_arrow.metadata[b'huggingface'])['info']['features']['category']['names']
        cols=(['id','filename','speaker_id','category','audio','num_samples'] if sib else
              ['id','task_name','audio','question','choice_a','choice_b','choice_c','choice_d','answer_gt'])
        for batch in pf.iter_batches(batch_size=16,columns=cols):
            for r in batch.to_pylist():
                nsource+=1
                if not sib and r['task_name']!=view['task_name']:continue
                candidates=classes if sib else [r['choice_'+x] for x in 'abcd']
                ids=candidates if sib else list('ABCD')
                question=view['question'] if sib else r['question']
                if len({x.strip().casefold() for x in candidates})!=len(ids) or not all(x.strip() for x in candidates):
                    raise ValueError('ambiguous alternatives')
                target=classes[r['category']] if sib else ids[candidates.index(r['answer_gt'])]
                panel=dict(version='jev-semantic-answer-v1',requirements=[question],
                           questions=[asdict(ApplicabilityQuestion(i,d,'Does this answer the supplied question about the audio?'))
                                      for i,d in zip(ids,candidates,strict=True)])
                ph=hashlib.sha256(json.dumps(panel,sort_keys=True).encode()).hexdigest();panels[ph]=panel
                utterance=str(r['id']);audios=r['audio'] if sib else [r['audio']]
                if sib and not len(audios)==len(r['filename'])==len(r['speaker_id'])==len(r['num_samples']):
                    raise ValueError('unaligned recordings')
                for j,a in enumerate(audios):
                    uid=utterance+':'+r['filename'][j] if sib else utterance
                    if uid in seen:raise ValueError('duplicate recording ID')
                    seen.add(uid)
                    raw=a['bytes'];wave,rate=sf.read(io.BytesIO(raw),dtype='float32')
                    if wave.ndim!=1 or not len(wave) or not np.isfinite(wave).all():raise ValueError('invalid waveform')
                    if sib and (a['path']!=r['filename'][j] or len(wave)!=r['num_samples'][j] or rate!=16000):
                        raise ValueError('source audio identity differs')
                    rates[rate]+=1
                    if rate!=16000:
                        g=gcd(rate,16000);wave=resample_poly(wave,16000//g,rate//g)
                        payload=io.BytesIO();sf.write(payload,wave,16000,format='WAV',subtype='FLOAT');raw=payload.getvalue()
                    digest=hashlib.sha256(raw).hexdigest();rel='audio/'+digest+'.wav'
                    wav=root/rel;wav.parent.mkdir(exist_ok=True)
                    if not wav.exists():wav.write_bytes(raw)
                    rows.append(dict(id=uid,audio=rel,audio_sha256=digest,panel=ph))
                    gold.append(dict(id=uid,audio_sha256=digest,gold=target,group=utterance))
    if (nsource,len(rows),len(set(t['group'] for t in gold)))!=tuple(view['population']):
        raise ValueError('source or evaluation population changed')
    dump(root/'inference'/dataset/'panels.json',panels);dump(root/'inference'/dataset/'rows.json',rows)
    dump(root/'scoring'/dataset/'targets.json',gold)
    m=dict(schema='jev-semantic-evaluation-seal-v1',prompt_layout=PREFIX_LAYOUT,threshold=.5,target_tuning_allowed=False,
           datasets={dataset:dict(split=view['split'],recordings=len(rows),candidates=len(ids),utterance_groups=len(set(t['group'] for t in gold)))},
           source=dict(repo=spec['id'],revision=spec['revision'],shards=inputs,source_rows=nsource,source_rates=dict(rates)),
           files={str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*.json'))})
    dump(root/'MANIFEST.json',m)


def audit(root,dataset):
    from tools.reports.prepare_native_seal_cache import plan
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.context_choice import request
    from multimodal_jev.context_decision_contract import render_prefix_context_request
    jobs=plan(dict(seal=dict(root=str(root),sha256=file_hash(root/'MANIFEST.json')),dataset=dataset))
    m=json.loads((root/'MANIFEST.json').read_bytes());name=f'scoring/{dataset}/targets.json';p=root/name
    if file_hash(p)!=m['files'][name]:raise ValueError('scoring file changed')
    for (_,r,q),t in zip(jobs,json.loads(p.read_bytes()),strict=True):
        if any(t[k]!=r[k] for k in ('id','audio_sha256')) or file_hash(r['audio'])!=t['audio_sha256']:raise ValueError('identity differs')
        load_audio(r['audio']);choices=[x.id for x in q.questions]
        if choices.count(t['gold'])!=1:raise ValueError('invalid gold')
        qc=request(q,'open',list(range(len(choices))))
        if q.requirements[0] not in qc.requirements or len(qc.questions)!=len(choices)+1:raise ValueError('lost question or NONE')
        render_prefix_context_request(q);render_prefix_context_request(qc)
    return dict(recordings=len(jobs),manifest=file_hash(root/'MANIFEST.json'),model_calls=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('local');p.add_argument('name')
    a=p.parse_args();c=load_config(a.config,a.local)[0];view=c['sources'][a.name]['eval']
    dest=Path(c['datasets'])/view['release'];ds=view['dataset']
    if not dest.exists():
        with tempfile.TemporaryDirectory(dir=c['clones']) as tmp:
            work=Path(tmp)/'release';work.mkdir();prepare(c,a.name,work);audit(work,ds)
            dest.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(dir=dest.parent) as staging:
                stage=Path(staging)/'release';shutil.copytree(work,stage);audit(stage,ds);os.rename(stage,dest)
    print(json.dumps(audit(dest,ds)))


if __name__=='__main__':main()
