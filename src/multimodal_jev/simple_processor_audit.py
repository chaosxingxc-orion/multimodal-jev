"""Check complete direct-Final processor coverage against frozen epoch slots."""
from collections import Counter
import json
from pathlib import Path
from .baseline import file_hash
from .slurp_training import content_hash


def verify_records(schedule_root, records_path, proof, population, task):
    from .slurp_intent_structure import compile_intent
    schedule_root = Path(schedule_root)
    manifest = json.loads((schedule_root/'MANIFEST.json').read_text())
    if (proof['schedule_sha256'] != file_hash(schedule_root/'MANIFEST.json')
            or proof['release_sha256'] != manifest['manifest_sha256']
            or manifest.get('task_mode') != 'direct_final'
            or proof['records_sha256'] != file_hash(records_path)
            or not proof['passed'] or proof['context_overflow']):
        raise ValueError('matching successful simple schedule processor required')
    by_id = {r['id']: r for r in population}
    total=0; lengths=[]; seen={}; counts=Counter()
    with Path(records_path).open() as stream:
        for epoch in manifest['epochs']:
            path=schedule_root/f'epoch-{epoch["epoch"]}.json'
            if file_hash(path) != epoch['file_sha256']:
                raise ValueError('epoch identity changed')
            batches=json.loads(path.read_text())['batches']
            if content_hash(batches) != epoch['batches_sha256']:
                raise ValueError('epoch content changed')
            ids=[]
            for batch_index,batch in enumerate(batches):
                for slot_index,slot in enumerate(batch):
                    if slot['padding']:
                        continue
                    raw=stream.readline()
                    if not raw:
                        raise ValueError('missing processor record')
                    record=json.loads(raw);row=by_id[slot['id']];ids.append(row['id'])
                    expected=dict(epoch=epoch['epoch'],batch=batch_index,slot=slot_index,id=row['id'],
                                  view='direct',variant='native',audio_sha256=row['audio_sha256'],candidate_counts=[60])
                    if any(record.get(k)!=v for k,v in expected.items()) or slot['view']!='direct' or slot['variant']!='native':
                        raise ValueError('processor coordinate or waveform differs')
                    if record['input_sha256'] != content_hash(compile_intent(row['audio'],task,'D').model_input()):
                        raise ValueError('processor task input differs')
                    length=record['input_tokens'];positions=record['positions']
                    if (type(length)!=int or not 0<length<=proof['context_limit']
                            or len(positions)!=1 or len(positions[0])!=1
                            or not 0<=positions[0][0]<length
                            or record['input_ids']['shape'] != [1,length]):
                        raise ValueError('untruncated complete single-Final tensor required')
                    if row['id'] in seen:
                        previous=seen[row['id']]
                        keys=('input_tokens','positions','input_sha256','input_ids','audio_sha256','samples')
                        if record.get('reused') is not True or any(record[k]!=previous[k] for k in keys):
                            raise ValueError('reused processor input changed across epochs')
                    else:
                        seen[row['id']]=record
                    total+=1;lengths.append(length);counts['direct']+=1
            if len(ids)!=len(by_id) or set(ids)!=by_id.keys():
                raise ValueError('epoch does not cover each real source exactly once')
        if stream.readline():
            raise ValueError('extra processor records')
    if (total!=manifest['total_nonpadding_recording_uses'] or total!=proof['requests']
            or len(seen)!=proof['unique_input_views'] or dict(counts)!=proof['views']
            or max(lengths)!=proof['maximum_tokens'] or min(lengths)!=proof['minimum_tokens']):
        raise ValueError('processor summary differs from complete ledger')
    return dict(passed=True,requests=total,unique_input_views=len(seen),maximum_tokens=max(lengths),
        independently_matched_every_schedule_coordinate=True,records_sha256=file_hash(records_path),
        scope='Independent full ledger/schedule/input binding; not repeated processor or NPU execution.')
