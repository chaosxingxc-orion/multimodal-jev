"""Shared paired TRAIN compiler for the reference builder and actor."""
from contextlib import contextmanager
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_batches import CachedContextBatches
from multimodal_jev.cached_context_preparation import CachedContextPreparer, processor_cache_identity
from multimodal_jev.context_choice_inputs import identity
from multimodal_jev.model_provider import cache_namespace
from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache
from tools.reports.build_tpo_view import read, verified
from tools.reports.context_cached_reference_trainer import no_online_processor


def load(parent, data, plan):
    p=read(verified(parent,plan['parent_sha']))
    m=read(verified(data/'MANIFEST.json',plan['target_sha']))
    if (m['schema']!='jev-tpo-joint-view-v1' or m['source_policy']!=p['source_policy_id'] or
            m['release_sha256']!=p['release_sha256'] or p['physical_devices']!=plan['training_devices'] or
            m['updates']!=plan['updates'] or any(m['schedule'][k]!=plan[k] for k in ('global_batch','microbatch','seed','epochs'))):
        raise ValueError('invalid joint training binding')
    for name,sha in m['files'].items(): verified(data/name,sha)
    verified(Path(p['source_policy'])/'MANIFEST.json',plan['source_manifest_sha256'])
    release=Path(p['release']); files=read(verified(release/'MANIFEST.json',m['release_sha256']))['files']
    task=read(verified(release/'tasks.json',files['tasks.json']))['intent']
    if [i['id'] for i in task['schema']['candidates']]!=m['candidate_ids']:
        raise ValueError('candidate order changed')
    rows=read(data/'rows.json')
    import json
    with verified(release/'learning.jsonl',files['learning.jsonl']).open() as f:
        population={r['id']:r for line in f if (r:=json.loads(line))['id'] in rows}
    if set(rows)!=set(population):
        raise ValueError('paired population must come from clean TRAIN')
    return p,m,task,rows,population,read(data/'batches.json')


@contextmanager
def stream(c, task, rows, population, batches, processor, collate, *, provider, rank, start, stop):
    local=c['batch_config']['per_dp_replica_batch']
    batches=[dict(b,slots=b['slots'][rank*local:(rank+1)*local]) for b in batches]
    prep=CachedContextPreparer(no_online_processor,ReadOnlyEncodedAudioCache(c['encoded_audio_cache']),
        processor_sha256=processor_cache_identity(c['source_policy']),
        decision_token_id=collate['decision_token_id'],audio_token_id=collate['audio_token_id'],
        max_tokens=c['view_execution']['max_input_tokens'],prompt_layout=c['prompt_layout'],provider_namespace=cache_namespace(c),encode=provider.encode)
    compiler=provider.choice_compiler(processor,collate,c['view_execution']['max_input_tokens'])
    source=c['source_policy_id']
    with CachedContextBatches(batches,population,task,prep,start_update=start,stop_update=stop,global_batch=local) as cached:
        def consume(step,policy):
            items,meta=cached.consume(step,policy)
            choices=[]
            for n,item,slot in zip(items,meta,batches[step]['slots'],strict=True):
                uid=item['id']; row=rows[uid]
                if (identity(n,source)!=row['request_key'] or int(n['labels'].long().argmax())!=row['gold'] or
                        population[uid]['audio_sha256']!=row['audio_sha256']):
                    raise ValueError('native input, audio or gold differs')
                n.update(request_id=f"{rank}:{n['request_id']}",stage='tpo',gold=row['gold'],
                         paired=False,confusion=False,choice_mode='noul')
                x=dict(n,choice_mode='open')
                compiler.apply(x,cached._request(population[uid]['audio']),row,'open',source,indices=slot['slate'])
                choices.append(x)
            return items,choices,meta,batches[step]['slots']
        yield consume
