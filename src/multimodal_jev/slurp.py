"""SLURP -> auditable domain-held-out Jev manifests using native entity spans."""
from __future__ import annotations
from collections import Counter,defaultdict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unicodedata

from .task_schema import Candidate,DecisionSchema,JevExample


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda:handle.read(4*1024**2),b''):h.update(data)
    return h.hexdigest()


def encode(obj):
    return (json.dumps(obj,sort_keys=True,ensure_ascii=False)+'\n').encode()


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def slot_target(row,slot_type):
    """First occurrence defined by native token IDs, not substring matching."""
    entities=sorted((e for e in row['entities'] if e['type']==slot_type),key=lambda e:e['span'][0])
    if not entities:return None,[]
    tokens={t['id']:t['surface'] for t in row['tokens']}
    span=entities[0]['span']
    if not span or len(set(span))!=len(span) or span!=sorted(span) or any(i not in tokens for i in span):
        raise ValueError('invalid native token span')
    return ' '.join(tokens[i] for i in span),span


def speaker_index(metadata):
    speakers={}
    for row in metadata.values():
        for filename,record in row['recordings'].items():
            if filename in speakers and speakers[filename]!=record['usrid']:
                raise ValueError('conflicting speaker metadata')
            speakers[filename]=record['usrid']
    return speakers


def prepare_sources(root,config):
    names={'train':'train','valid':'devel','test':'test'}
    sources={split:root/'metadata'/f'{name}.jsonl' for split,name in names.items()}
    rows={split:[json.loads(line) for line in path.open()] for split,path in sources.items()}
    scenarios=sorted({r['scenario'] for r in rows['train']})
    if not set(config['heldout_scenarios']) < set(scenarios):
        raise ValueError('heldout scenarios must be a proper subset of train taxonomy')
    if not set(config['slot_types']) <= {e['type'] for r in rows['train'] for e in r['entities']}:
        raise ValueError('slot query types must come from train taxonomy')
    official_ids={split:{r['slurp_id'] for r in items} for split,items in rows.items()}
    for a,b in [('train','valid'),('train','test'),('valid','test')]:
        if official_ids[a]&official_ids[b]:raise ValueError('cross-split SLURP id overlap')
    return sources,rows,scenarios


def build(root,destination,config,progress=None):
    root,destination=Path(root).resolve(),Path(destination)
    sources,rows,scenarios=prepare_sources(root,config)
    hashes={s:sha(p) for s,p in sources.items()}
    hashes['metadata']=sha(root/'metadata/metadata.json')
    identity={'config':config,'source_sha256':hashes,
              'builder_sha256':sha(__file__),'schema_code_sha256':sha(Path(__file__).with_name('task_schema.py')),
              'source_root':str(root)}
    release=hashlib.sha256(encode(identity)).hexdigest()
    output=destination/release
    if output.exists():raise FileExistsError(f'immutable release exists: {output}')
    speakers=speaker_index(json.loads((root/'metadata/metadata.json').read_text()))
    schema=DecisionSchema('slurp.scenario.v1','choice',
        'Which topic domain best describes the spoken request? Choose using the domain descriptions.',
        tuple(Candidate(f'd{i:02d}',s.replace('_',' ')) for i,s in enumerate(scenarios)))
    ids={s:c.id for s,c in zip(scenarios,schema.candidates)}
    heldout=set(config['heldout_scenarios'])
    heldout_texts={normalize(r['sentence']) for items in rows.values() for r in items if r['scenario'] in heldout}
    evaluation_texts={normalize(r['sentence']) for split in ('valid','test') for r in rows[split]}
    exclusions=Counter();counts=Counter();slot_counts=Counter()
    split_speakers=defaultdict(set);seen_paths={};seen_hashes={}
    train_texts=set();eval_texts=set()
    native_labels={s:{r['intent'] for r in items} for s,items in rows.items()}
    destination.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.building-',dir=destination))
    streams={name:(stage/(name+'.jsonl')).open('wb') for name in ('train','valid','test','heldout-valid','heldout-test')}
    started=time.monotonic();processed=0
    try:
        for split,items in rows.items():
            for row in items:
                text=normalize(row['sentence'])
                reason=None
                if split=='train':
                    if row['scenario'] in heldout:reason='heldout_domain_train'
                    elif text in heldout_texts:reason='train_text_shared_with_heldout_domain'
                    elif text in evaluation_texts:reason='train_text_shared_with_evaluation'
                if reason:
                    exclusions[reason+'_utterances']+=1
                    exclusions[reason+'_recordings']+=len(row['recordings'])
                    continue
                partition=('heldout-' if row['scenario'] in heldout else '')+split
                if split=='train':train_texts.add(text)
                else:eval_texts.add(text)
                for recording in row['recordings']:
                    rel=recording['file'];audio=(root/'slurp_real'/rel).resolve()
                    if not audio.is_relative_to(root/'slurp_real'):raise ValueError('audio path escape')
                    if rel not in speakers:raise ValueError('speaker missing from metadata')
                    if rel in seen_paths:raise ValueError('repeated audio path')
                    seen_paths[rel]=split
                    audio_hash=sha(audio)
                    if audio_hash in seen_hashes and seen_hashes[audio_hash]!=split:
                        raise ValueError('audio bytes duplicated across splits')
                    seen_hashes[audio_hash]=split
                    split_speakers[partition].add(speakers[rel])
                    group=f'slurp:{row["slurp_id"]}:{rel}'
                    meta={'source_dataset':'slurp','source_group':group,'utterance_group':f'slurp:{row["slurp_id"]}',
                          'speaker_id':speakers[rel],'audio_sha256':audio_hash,'variant':'native',
                          'official_split':split,'partition':partition,'scenario':row['scenario'],
                          'native_intent':row['intent'],'native_action':row['action'],
                          'native_entities':row['entities'],'transcript_reference':row['sentence'],
                          'protocol':config['protocol']}
                    answer=ids[row['scenario']]
                    views=[('native',schema,answer)]
                    perm=tuple(sorted(schema.candidates,key=lambda c:hashlib.sha256((group+c.id).encode()).digest()))
                    views.append(('permutation',replace(schema,candidates=perm),answer))
                    remap={c.id:schema.candidates[(i+1)%len(schema.candidates)].id for i,c in enumerate(schema.candidates)}
                    views.append(('rebind',replace(schema,candidates=tuple(Candidate(remap[c.id],c.description) for c in schema.candidates)),remap[answer]))
                    for variant,current,target in views:
                        ex=JevExample(group+':'+variant,split,str(audio),current,target,{**meta,'variant':variant})
                        streams[partition].write(encode(ex.to_record()));counts[partition+'_decisions']+=1
                    for slot in config['slot_types']:
                        target,span=slot_target(row,slot)
                        question=f'What is the first explicitly mentioned {slot.replace("_"," ")} in the utterance? Return its spoken surface words, or null if none is annotated.'
                        current=DecisionSchema('slurp.slot.'+slot+'.v1','open_slot',question)
                        ex=JevExample(group+':slot.'+slot,split,str(audio),current,target,
                                      {**meta,'variant':'slot.'+slot,'native_token_span':span})
                        streams[partition].write(encode(ex.to_record()));counts[partition+'_decisions']+=1
                        slot_counts[f'{partition}:{slot}:{"missing" if target is None else "present"}']+=1
                    counts[partition+'_recordings']+=1;processed+=1
                    if progress and processed%100==0:
                        Path(progress).write_text(json.dumps({'recordings':processed,'elapsed_seconds':time.monotonic()-started,'heartbeat_unix':time.time()}))
        if train_texts&eval_texts:raise ValueError('residual train/evaluation transcript overlap')
        for stream in streams.values():stream.close()
        overlaps={f'{a}/{b}':len(sa&sb) for a,sa in split_speakers.items() for b,sb in split_speakers.items() if a<b}
        receipt={'release_id':release,'identity':identity,'counts':dict(counts),'exclusions':dict(exclusions),
                 'slot_support':dict(slot_counts),'speaker_overlap':overlaps,
                 'train_evaluation_normalized_transcript_overlap':0,'cross_split_audio_hash_overlap':0,
                 'native_intents_absent_from_train':{s:sorted(v-native_labels['train']) for s,v in native_labels.items()},
                 'manifests':{p.name:sha(p) for p in sorted(stage.glob('*.jsonl'))},
                 'elapsed_seconds':time.monotonic()-started,
                 'limits':['Scenario names are taxonomy descriptions, not human-authored business definitions.',
                           'Null denotes absent annotated entity, not acoustic unintelligibility.',
                           'Native intent/action/entity annotations retained; this projection is not official full SLU scoring.',
                           'Official speaker overlap is reported; no speaker-disjoint generalization claim.',
                           'All recordings of heldout-domain train utterances excluded; no synthetic SLURP audio.']}
        (stage/'MANIFEST.json').write_bytes(encode(receipt))
        os.rename(stage,output)
        return output,receipt
    except BaseException:
        for stream in streams.values():stream.close()
        shutil.rmtree(stage)
        raise
